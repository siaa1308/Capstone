#!/usr/bin/env python3
"""Explain the selected adaptive federated global model's final predictions."""
from __future__ import annotations

import argparse
import copy
import io
import json
import socket
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch

XAI_ROOT = Path(__file__).resolve().parent
if str(XAI_ROOT) not in sys.path:
    sys.path.insert(0, str(XAI_ROOT))
from run_updated import BANKS, DATASET, REPO_ROOT, git_blob, load_cached_engine, normalize_checkpoint_for_engine, sha256
from src.model_catalog import filter_catalog, model_catalog_rows, verify_saved_metrics
from src.transaction_story import build_transaction_story, canonical_frames, completed_history

SCHEMA = "adaptive-federated-single-model-group-shap-v2"


def cached_explanation_matches(report, cache_key, checkpoint_hash):
    """Return whether a cached explanation is safe to serve for this model and score."""
    if not isinstance(report, dict):
        return False
    feature = report.get("feature_explanation")
    prediction = report.get("prediction")
    verification = report.get("verification")
    model = report.get("model")
    if not all(isinstance(value, dict) for value in (feature, prediction, verification, model)):
        return False
    target, score = feature.get("target_probability"), prediction.get("probability")
    return (report.get("cache_key") == cache_key
            and verification.get("passed") is True
            and model.get("checkpoint_sha256") == checkpoint_hash
            and isinstance(target, (int, float)) and np.isfinite(target)
            and isinstance(score, (int, float)) and np.isfinite(score)
            and abs(target - score) <= 1e-6)


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class AdaptiveFederatedDashboard:
    def __init__(self, seed: int = 42, background_size: int = 32):
        self.seed, self.background_size = seed, background_size
        if seed not in (42, 52, 62) or background_size < 1:
            raise ValueError("invalid model run or explanation settings")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.lock = threading.RLock()
        self.cli = load_cached_engine("origin/sia")
        import xai.temporal as temporal
        self.temporal = temporal
        self.prepared = XAI_ROOT / "outputs/updated/sia_context_harmonized" / f"seed_{seed}"
        self.output = XAI_ROOT / "outputs/updated/adaptive_federated" / f"seed_{seed}"
        self.output.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / "tmp").mkdir(exist_ok=True)
        self.temp_context = tempfile.TemporaryDirectory(prefix="xai_adaptive_model_", dir=REPO_ROOT / "tmp")
        self.checkpoint = Path(self.temp_context.name) / "global_model.pt"
        self.preprocessing = self.prepared / "preprocessing.json"
        self.preprocessing_hash = sha256(self.preprocessing.read_bytes())
        self.rows, self.adapters, self.probabilities, self.thresholds = {}, {}, {}, {}
        self.story_frames, self.story_cache, self.account_ids = {}, {}, {}
        self._load_model()

    def _load_model(self):
        base = f"artifacts/research_adaptive_fcl/warm_start/federated/context_harmonized/seed_{self.seed}"
        raw = git_blob("origin/sia", base + "/global_model.pt")
        metrics_blob = git_blob("origin/sia", base + "/metrics.json")
        metrics = json.loads(metrics_blob)
        prepared_manifest = json.loads((self.prepared / "xai_run_manifest.json").read_text(encoding="utf-8"))
        self.checkpoint_hash = sha256(raw)
        if prepared_manifest["checkpoint_sha256"] != self.checkpoint_hash:
            raise ValueError("selected global checkpoint differs from prepared prediction artifacts")
        expected_hash = json.loads(self.preprocessing.read_text(encoding="utf-8"))["checkpoint_sha256"]
        staged = raw if sha256(raw) == expected_hash else normalize_checkpoint_for_engine(raw)
        if sha256(staged) != expected_hash:
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            stream = io.BytesIO()
            torch.save(payload, stream)
            staged = stream.getvalue()
        self.engine_checkpoint_hash = sha256(staged)
        if self.engine_checkpoint_hash != expected_hash:
            raise ValueError("selected checkpoint cannot reproduce preprocessing's exact checkpoint hash")
        self.checkpoint.write_bytes(staged)
        source = git_blob("origin/sia", "src/gnn/causal_temporal_graphsage.py")
        if sha256(source) != sha256((XAI_ROOT / "src/gnn/causal_temporal_graphsage.py").read_bytes()):
            raise ValueError("inference model source does not match the selected model's branch")
        self.reference_hashes, banks = {}, {}
        expected_banks = {row["bank"]: row for row in metrics["per_bank"]}
        for bank in BANKS:
            reference = self.prepared / "local" / bank / "reference_predictions.npz"
            self.reference_hashes[bank] = sha256(reference.read_bytes())
            with np.load(reference) as arrays:
                probabilities = np.asarray(arrays["validation"], dtype=np.float64)
                if len(arrays.get("testing", [])):
                    raise ValueError("the validation dashboard requires test-free prediction artifacts")
            adapter = self.temporal.TemporalAdapter(DATASET, self.checkpoint, bank, "federated",
                                                    self.preprocessing, reference)
            frame = adapter.frames["validation"]
            threshold = float(expected_banks[bank]["threshold"])
            if abs(float(adapter.threshold) - threshold) > 1e-12:
                raise ValueError(f"{bank}: XAI threshold differs from the selected model's saved threshold")
            rows = model_catalog_rows(bank, frame["txn_id"].astype(str).tolist(), frame["timestamp"].astype(str).tolist(),
                                      frame["y"].to_numpy(), probabilities, threshold=threshold)
            parity = verify_saved_metrics(frame["y"].to_numpy(), probabilities, threshold,
                                          expected_banks[bank]["validation"])
            self.adapters[bank], self.probabilities[bank] = adapter, probabilities
            self.rows[bank], self.thresholds[bank] = rows, threshold
            banks[bank] = {"validation_transactions": len(rows), "known_fraud": sum(row["label_value"] for row in rows),
                           "flagged": sum(row["flagged"] for row in rows),
                           "missed_fraud": sum(row["status"] == "missed_fraud" for row in rows),
                           "threshold": threshold, "validation_parity": parity}
        self.manifest = {
            "schema": SCHEMA, "family": "adaptive_federated_global_model", "seed": self.seed,
            "model_count": 1, "ensemble": False, "best_round": metrics["best_round"],
            "training_rounds": metrics["configuration"]["rounds"], "source_ref": "origin/sia",
            "checkpoint_path": f"origin/sia:{base}/global_model.pt", "checkpoint_sha256": self.checkpoint_hash,
            "inference_checkpoint_sha256": self.engine_checkpoint_hash,
            "metrics_path": f"origin/sia:{base}/metrics.json", "metrics_sha256": sha256(metrics_blob),
            "preprocessing_sha256": self.preprocessing_hash, "reference_prediction_sha256": self.reference_hashes,
            "evaluation_split": "validation", "test_loaded": False, "privacy_mode": "none",
            "threshold_policy": "saved per-bank threshold from the selected adaptive federated model's metrics.json; no dashboard threshold fitting",
            "banks": banks,
            "implementation_sha256": {str(path.relative_to(XAI_ROOT)): sha256(path.read_bytes()) for path in (
                XAI_ROOT / "serve_dashboard.py", XAI_ROOT / "src/model_catalog.py",
                XAI_ROOT / "src/counterfactuals.py", XAI_ROOT / "src/transaction_story.py",
                XAI_ROOT / "src/grouped_features.py",
                XAI_ROOT / "__pycache__/explanation_engine.cpython-312.pyc",
                XAI_ROOT / "__pycache__/temporal.cpython-312.pyc",
                XAI_ROOT / "__pycache__/graph_explainer.cpython-312.pyc")},
        }
        (self.output / "dashboard_catalog_manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")

    def catalog(self, bank, view, query, page, page_size):
        if bank not in BANKS:
            raise ValueError("unknown bank")
        rows = filter_catalog(self.rows[bank], view, query)
        page, page_size = max(0, page), min(200, max(1, page_size))
        return {"bank": bank, "view": view, "page": page, "page_size": page_size, "total": len(rows),
                "threshold": self.thresholds[bank], "threshold_policy": self.manifest["threshold_policy"],
                "counts": self.manifest["banks"][bank], "transactions": rows[page * page_size:(page + 1) * page_size]}

    def _raw_report_valid(self, report, bank, transaction_id, index):
        provenance = report.get("provenance", {})
        prediction = report.get("prediction", {})
        return (str(report.get("transaction_id")) == transaction_id and report.get("bank") == bank
                and report.get("split") == "validation"
                and provenance.get("checkpoint_sha256") == self.engine_checkpoint_hash
                and provenance.get("preprocessing_sha256") == self.preprocessing_hash
                and report.get("verification", {}).get("passed") is True
                and report.get("feature_explanation", {}).get("background_size") == self.background_size
                and abs(float(prediction.get("threshold", -1)) - self.thresholds[bank]) <= 1e-12
                and abs(float(prediction.get("probability", -1)) - float(self.probabilities[bank][index])) <= 1e-6)

    def _raw_report(self, bank, transaction_id, index):
        paths = [self.output / "raw_reports" / bank / f"validation_{index:08d}" / "explanation.json",
                 self.prepared / "local" / bank / "reports" / f"validation_{index:08d}" / "explanation.json",
                 XAI_ROOT / "outputs/updated/sia_ensemble/42_52_62/member_cache" / f"seed_{self.seed}" / "local" / bank / "reports" / f"validation_{index:08d}" / "explanation.json"]
        for path in paths:
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8"))
                if self._raw_report_valid(report, bank, transaction_id, index):
                    return report
        engine = self.cli.ExplanationEngine(self.adapters[bank], self.cli.XAIConfig(
            background_size=self.background_size, seed=self.seed))
        report = engine.explain("validation", index)
        if not self._raw_report_valid(report, bank, transaction_id, index):
            raise ValueError("new explanation does not match the selected model, saved cutoff, or full-stream prediction")
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        paths[0].write_text(json.dumps(report, indent=2, default=json_default) + "\n", encoding="utf-8")
        return report

    def _attach_story(self, bank, transaction_id, report):
        key = (bank, transaction_id, report["cache_key"])
        if key not in self.story_cache:
            adapter = self.adapters[bank]
            index = int(adapter.index("validation", transaction_id))
            if bank not in self.story_frames:
                self.story_frames[bank] = canonical_frames(adapter)
                maps = [pd.read_csv(DATASET / split / bank / "node_map.csv.gz") for split in ("training", "validation")]
                account_values = pd.concat([mapping["account_id"] for mapping in maps], ignore_index=True).drop_duplicates().tolist()
                self.account_ids[bank] = {index: str(value) for index, value in enumerate(account_values)}
                for split, mapping in zip(("training", "validation"), maps):
                    local_accounts = mapping.set_index("node_id")["account_id"].astype(str)
                    for endpoint in ("src_node", "dst_node"):
                        raw_ids = adapter.frames[split][endpoint].map(local_accounts).tolist()
                        model_ids = self.story_frames[bank][split][endpoint].map(self.account_ids[bank]).tolist()
                        if raw_ids != model_ids:
                            del self.story_frames[bank]
                            raise ValueError("dataset account IDs do not match model stream identities")
            frames = self.story_frames[bank]
            history = completed_history(frames["training"], frames["validation"], index, int(adapter.batch_size))
            self.story_cache[key] = build_transaction_story(frames["validation"].iloc[index], history, report,
                                                           account_ids=self.account_ids[bank])
        report["transaction_story"] = self.story_cache[key]
        evaluation_path = self.output / "xai_evaluation.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if evaluation.get("model") == self.checkpoint_hash:
                report["evaluation_checks"] = {
                    name: next((item for item in evaluation.get(name, [])
                        if item["bank"] == bank and str(item["transaction_id"]) == transaction_id), None)
                    for name in ("stability", "faithfulness", "coverage", "temporal_replay")}
        return report

    def explain(self, bank, transaction_id):
        if bank not in BANKS:
            raise ValueError("unknown bank")
        row = next((row for row in self.rows[bank] if row["transaction_id"] == transaction_id), None)
        if row is None:
            raise ValueError("transaction is not in this bank's validation catalog")
        key_inputs = {"schema": SCHEMA, "bank": bank, "transaction_id": transaction_id, "seed": self.seed,
                      "checkpoint_sha256": self.checkpoint_hash, "preprocessing_sha256": self.preprocessing_hash,
                      "reference_prediction_sha256": self.reference_hashes[bank], "threshold": self.thresholds[bank],
                      "background_size": self.background_size, "implementation_sha256": self.manifest["implementation_sha256"]}
        key = sha256(json.dumps(key_inputs, sort_keys=True).encode())
        path = self.output / "local" / bank / "reports" / f"validation_{transaction_id}" / "explanation.json"
        with self.lock:
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8"))
                if cached_explanation_matches(report, key, self.checkpoint_hash):
                    return self._attach_story(bank, transaction_id, report)
            adapter = self.adapters[bank]
            index = int(adapter.index("validation", transaction_id))
            raw = copy.deepcopy(self._raw_report(bank, transaction_id, index))
            xai_probability = float(raw["prediction"]["probability"])
            if bool(xai_probability >= self.thresholds[bank]) != row["flagged"]:
                raise ValueError("XAI score and saved model score disagree at the decision threshold")
            state, _, _ = adapter.context("validation", index, 8)
            predictor = adapter.predictor(state, "validation", index)
            from src.grouped_features import explain_groups
            grouped = explain_groups(predictor, adapter.frames["validation"].iloc[index],
                adapter.frames["training"], background_size=self.background_size, seed=self.seed)
            if abs(grouped["target_probability"] - row["probability"]) > 1e-6:
                raise ValueError("group SHAP does not target the displayed global-model score")
            raw["legacy_feature_explanation"] = raw["feature_explanation"]
            raw["feature_explanation"] = grouped
            from src.counterfactuals import constrained_amount_counterfactuals
            counterfactual = constrained_amount_counterfactuals(adapter.frames["validation"].iloc[index],
                adapter.frames["training"], predictor, threshold=self.thresholds[bank])
            raw.update({"schema_version": 3, "evaluation_label": row["label"],
                        "model": {"family": "adaptive_federated_global_model", "seed": self.seed,
                                  "best_round": self.manifest["best_round"], "model_count": 1,
                                  "checkpoint_sha256": self.checkpoint_hash},
                        "prediction": {"probability": row["probability"], "threshold": row["threshold"],
                                       "class": "fraud" if row["flagged"] else "not_fraud",
                                       "score_type": "saved adaptive federated global model probability",
                                       "threshold_selection": self.manifest["threshold_policy"]},
                        "counterfactual_explanation": counterfactual,
                        "xai_provenance": self.manifest, "cache_key": key, "cache_key_inputs": key_inputs})
            raw["verification"].update({"model_saved_catalog_error": abs(xai_probability - row["probability"]),
                                         "explained_checkpoint_sha256": self.checkpoint_hash,
                                         "saved_threshold_used": True, "model_count": 1})
            raw["limitations"] = [item for item in raw.get("limitations", [])
                                   if item != "No combined federated-continual checkpoint or formal differential privacy is implemented."]
            raw["limitations"].append("This is the selected adaptive federated continual model checkpoint; it was trained without formal differential privacy.")
            self._attach_story(bank, transaction_id, raw)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(raw, indent=2, default=json_default) + "\n", encoding="utf-8")
            temporary.replace(path)
            return raw


class LocalDashboardServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        def send_data(self, data, content_type="application/json", status=200):
            body = data if isinstance(data, bytes) else json.dumps(data, default=json_default).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            params = parse_qs(url.query)
            try:
                if url.path in {"/", "/dashboard.js", "/dashboard.css"}:
                    name = "dashboard.html" if url.path == "/" else url.path.lstrip("/")
                    content_type = {"dashboard.html": "text/html", "dashboard.js": "text/javascript", "dashboard.css": "text/css"}[name]
                    self.send_data((XAI_ROOT / "static" / name).read_bytes(), content_type + "; charset=utf-8")
                elif url.path == "/api/meta":
                    self.send_data({"banks": BANKS, "manifest": app.manifest})
                elif url.path == "/api/catalog":
                    self.send_data(app.catalog(params.get("bank", [""])[0], params.get("view", ["flagged"])[0],
                        params.get("q", [""])[0], int(params.get("page", ["0"])[0]), int(params.get("page_size", ["75"])[0])))
                elif url.path == "/api/explanation":
                    self.send_data(app.explain(params.get("bank", [""])[0], params.get("transaction_id", [""])[0]))
                else:
                    self.send_data({"error": "not found"}, status=404)
            except Exception as exc:
                self.send_data({"error": str(exc)}, status=500)

        def log_message(self, fmt, *args):
            print("[dashboard] " + fmt % args, flush=True)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, choices=(42, 52, 62), default=42,
                        help="one saved federated training run; never averages models")
    parser.add_argument("--background-size", type=int, default=32)
    args = parser.parse_args()
    app = AdaptiveFederatedDashboard(args.seed, args.background_size)
    server = LocalDashboardServer((args.host, args.port), handler_for(app))
    print(f"Adaptive federated dashboard ready: http://{args.host}:{args.port}/", flush=True)
    print(json.dumps({"model": app.manifest["checkpoint_path"], "best_round": app.manifest["best_round"],
                      "banks": app.manifest["banks"]}, indent=2), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.temp_context.cleanup()


if __name__ == "__main__":
    main()
