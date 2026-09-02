#!/usr/bin/env python3
"""Generate an interactive, sampled local-bank transaction graph.

The complete bank graphs are too large for a browser force layout. This tool
therefore keeps all suspicious transactions first, then fills the remaining
display budget with the highest-value transactions whose endpoints fit within
the configured node limit. Source data is never modified.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPOSITORY_ROOT / "data" / "final_temporal_dataset"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "graph_visualizations"
BANKS = ("JPMorgan_Chase", "Wells_Fargo", "Citi", "Fifth_Third_Bancorp", "Key_Bank")
SPLITS = ("training", "validation", "testing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--bank", choices=BANKS, default="Citi")
    parser.add_argument("--split", choices=SPLITS, default="testing")
    parser.add_argument("--max-nodes", type=int, default=150)
    parser.add_argument("--max-edges", type=int, default=300)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_nodes < 2:
        raise ValueError("--max-nodes must be at least 2")
    if args.max_edges < 1:
        raise ValueError("--max-edges must be at least 1")


def load_graph(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges = pd.read_csv(folder / "edge_list.csv.gz")
    truth = pd.read_csv(folder / "ground_truth.csv.gz", usecols=["txn_id", "y"])
    node_map = pd.read_csv(folder / "node_map.csv.gz")

    required_edges = {"txn_id", "src_node", "dst_node", "amount", "timestamp"}
    if missing := required_edges - set(edges.columns):
        raise ValueError(f"edge_list.csv.gz is missing columns: {sorted(missing)}")
    if not {"node_id", "account_id"}.issubset(node_map.columns):
        raise ValueError("node_map.csv.gz must contain node_id and account_id")

    labeled = edges.merge(truth, on="txn_id", how="left", validate="one_to_one")
    if labeled["y"].isna().any():
        raise ValueError("Some graph edges do not have a ground-truth label")
    labeled["y"] = labeled["y"].astype(int)
    return labeled, node_map, truth


def select_display_edges(edges: pd.DataFrame, max_nodes: int, max_edges: int) -> pd.DataFrame:
    ranked = edges.assign(
        _amount=pd.to_numeric(edges["amount"], errors="coerce").fillna(0.0).abs()
    ).sort_values(["y", "_amount", "timestamp"], ascending=[False, False, True])

    selected: list[int] = []
    nodes: set[int] = set()
    for index, edge in ranked.iterrows():
        endpoints = {int(edge["src_node"]), int(edge["dst_node"])}
        if len(nodes | endpoints) > max_nodes:
            continue
        selected.append(index)
        nodes.update(endpoints)
        if len(selected) >= max_edges:
            break

    if not selected:
        raise ValueError("No edges fit within the requested display limits")
    return edges.loc[selected].sort_values("timestamp").reset_index(drop=True)


def short_account(value: object) -> str:
    text = str(value)
    return text if len(text) <= 14 else f"{text[:6]}…{text[-5:]}"


def graph_payload(edges: pd.DataFrame, node_map: pd.DataFrame) -> dict[str, object]:
    account_by_node = {
        int(row.node_id): short_account(row.account_id)
        for row in node_map[["node_id", "account_id"]].itertuples(index=False)
    }
    degree: dict[int, int] = {}
    for row in edges.itertuples(index=False):
        degree[int(row.src_node)] = degree.get(int(row.src_node), 0) + 1
        degree[int(row.dst_node)] = degree.get(int(row.dst_node), 0) + 1

    nodes = [
        {"id": node, "label": account_by_node.get(node, str(node)), "degree": count}
        for node, count in sorted(degree.items())
    ]
    links = []
    for row in edges.itertuples(index=False):
        amount = float(row.amount) if pd.notna(row.amount) else 0.0
        links.append({
            "source": int(row.src_node),
            "target": int(row.dst_node),
            "txn_id": str(row.txn_id),
            "timestamp": str(row.timestamp),
            "amount": amount if np.isfinite(amount) else 0.0,
            "currency": str(getattr(row, "currency", "")),
            "payment_format": str(getattr(row, "payment_format", "")),
            "suspicious": bool(row.y),
        })
    return {"nodes": nodes, "links": links}


def render_html(payload: dict[str, object], bank: str, split: str, total_edges: int) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = f"{bank.replace('_', ' ')} — {split.title()} graph"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; background: Canvas; color: CanvasText; }}
    header {{ padding: 16px 20px 8px; }}
    h1 {{ margin: 0 0 6px; font-size: 1.25rem; }}
    .meta {{ opacity: .75; font-size: .9rem; }}
    .controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; padding: 8px 20px; }}
    label {{ display: flex; gap: 6px; align-items: center; }}
    #graph {{ width: 100%; height: 680px; border-top: 1px solid color-mix(in srgb, CanvasText 18%, transparent); }}
    .link {{ stroke: color-mix(in srgb, CanvasText 28%, transparent); }}
    .link.suspicious {{ stroke: #d62728; }}
    .node {{ fill: #3977b8; stroke: Canvas; stroke-width: 1.5px; cursor: pointer; }}
    .node.has-suspicious {{ fill: #d62728; }}
    .tooltip {{ position: absolute; display: none; max-width: 340px; padding: 9px 11px; border-radius: 6px;
      background: Canvas; color: CanvasText; border: 1px solid color-mix(in srgb, CanvasText 25%, transparent);
      box-shadow: 0 3px 14px #0004; pointer-events: none; font-size: .85rem; }}
    .legend {{ margin-left: auto; font-size: .85rem; opacity: .8; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="meta">Showing {len(payload['nodes']):,} accounts and {len(payload['links']):,} sampled transactions from {total_edges:,} total transactions.</div>
  </header>
  <div class="controls">
    <label><input id="suspicious" type="checkbox"> Fraud transactions only</label>
    <button id="restart" type="button">Restart layout</button>
    <span class="legend">Blue = regular account · Red = touches a ground-truth fraud transaction</span>
  </div>
  <svg id="graph" role="img" aria-label="Interactive sampled transaction network"></svg>
  <div id="tooltip" class="tooltip" role="tooltip"></div>
  <script>
    const original = {data};
    const svg = d3.select("#graph"), tip = d3.select("#tooltip");
    let simulation;
    function draw(onlySuspicious = false) {{
      svg.selectAll("*").remove();
      if (simulation) simulation.stop();
      const links = original.links.filter(d => !onlySuspicious || d.suspicious).map(d => ({{...d}}));
      const ids = new Set(links.flatMap(d => [d.source, d.target]));
      const nodes = original.nodes.filter(d => ids.has(d.id)).map(d => ({{...d}}));
      const suspiciousNodes = new Set(links.filter(d => d.suspicious).flatMap(d => [d.source, d.target]));
      const width = svg.node().clientWidth, height = svg.node().clientHeight;
      const root = svg.append("g");
      svg.call(d3.zoom().scaleExtent([0.2, 8]).on("zoom", e => root.attr("transform", e.transform)));
      const amounts = links.map(d => Math.abs(d.amount));
      const edgeWidth = d3.scaleSqrt().domain(d3.extent(amounts).map((d,i) => d ?? i)).range([0.6, 4]);
      const link = root.append("g").selectAll("line").data(links).join("line")
        .attr("class", d => d.suspicious ? "link suspicious" : "link")
        .attr("stroke-width", d => edgeWidth(Math.abs(d.amount)))
        .on("mousemove", (e,d) => tip.style("display","block").style("left",`${{e.pageX+12}}px`).style("top",`${{e.pageY+12}}px`)
          .html(`<b>${{d.suspicious ? "Ground-truth fraud" : "Regular"}} transaction</b><br>Amount: ${{d.amount.toLocaleString()}} ${{d.currency}}<br>Time: ${{d.timestamp}}<br>Format: ${{d.payment_format}}<br>Txn: ${{d.txn_id}}`))
        .on("mouseleave", () => tip.style("display","none"));
      const node = root.append("g").selectAll("circle").data(nodes).join("circle")
        .attr("class", d => suspiciousNodes.has(d.id) ? "node has-suspicious" : "node")
        .attr("r", d => Math.min(13, 4 + Math.sqrt(d.degree)))
        .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended))
        .on("mousemove", (e,d) => tip.style("display","block").style("left",`${{e.pageX+12}}px`).style("top",`${{e.pageY+12}}px`)
          .html(`<b>Account ${{d.label}}</b><br>Sampled degree: ${{d.degree}}<br>Local node: ${{d.id}}`))
        .on("mouseleave", () => tip.style("display","none"));
      simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(55).strength(.18))
        .force("charge", d3.forceManyBody().strength(-65))
        .force("center", d3.forceCenter(width/2, height/2))
        .force("collision", d3.forceCollide().radius(d => Math.min(16, 7 + Math.sqrt(d.degree))))
        .on("tick", () => {{
          link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
          node.attr("cx", d => d.x).attr("cy", d => d.y);
        }});
      function dragstarted(e,d) {{ if(!e.active) simulation.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; }}
      function dragged(e,d) {{ d.fx=e.x; d.fy=e.y; }}
      function dragended(e,d) {{ if(!e.active) simulation.alphaTarget(0); d.fx=null; d.fy=null; }}
    }}
    document.getElementById("suspicious").addEventListener("change", e => draw(e.target.checked));
    document.getElementById("restart").addEventListener("click", () => simulation && simulation.alpha(1).restart());
    draw();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    validate_args(args)
    folder = args.dataset_dir / args.split / args.bank
    if not folder.is_dir():
        raise FileNotFoundError(f"Graph folder does not exist: {folder}")

    edges, node_map, _ = load_graph(folder)
    selected = select_display_edges(edges, args.max_nodes, args.max_edges)
    payload = graph_payload(selected, node_map)
    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.bank}_{args.split}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(payload, args.bank, args.split, len(edges)), encoding="utf-8")
    suspicious = int(selected["y"].sum())
    print(f"Wrote {output.resolve()}")
    print(f"Displayed nodes={len(payload['nodes'])}, edges={len(selected)}, suspicious_edges={suspicious}")


if __name__ == "__main__":
    main()
