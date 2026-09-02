# Historical experiment archive

This directory preserves superseded experiments for auditability. Nothing here is an active final method or a recommendation for new runs.

- `original_fedavg_final/`: the original poor FedAvg final result and its checkpoints.
- `legacy_five_bank/`: historical five-bank artifacts. Citi and Fifth Third Bancorp evidence is preserved intact.
- `legacy_federated/`: early federated variants.
- `validation_search/`: earlier Local/CL/FedAvg/FedProx searches.
- `corrected_development/`: validation-only experiments used to diagnose and correct the temporal and federated pipelines.

The final comparison uses only `artifacts/final_evaluation/{local,cl,corrected_fedavg}`. See `docs/REPOSITORY_MANIFEST.md` and `docs/FINAL_EVALUATION_REPORT.md`.

