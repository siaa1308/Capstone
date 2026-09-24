# Adaptive continual and federated AML artifacts

See the [completed results](../../docs/ADAPTIVE_FCL_RESULTS.md) for tables, ablations, limitations, and reproduction commands, and the [method description](../../docs/ADAPTIVE_FCL_RESEARCH.md) for prior work.

| Directory | Contents |
|---|---|
| `warm_start/` | Continuation study from the historical frozen checkpoints; selected method `context_harmonized` |
| `warm_start/exploratory_september/` | Frozen selected model and matched control, three-seed September evaluation |
| `global_rng/` | From-scratch federated study and standalone continual-learning studies |
| `global_rng/local_exploratory_september/` | Context-replay CL and matched uniform-replay control, three-seed September evaluation |
| `federated/` | Initial per-client RNG diagnostic; not the primary comparison |
| `original_protocol/` | Validation-only original-trainer reproduction diagnostic |

Run directories preserve checkpoints, manifests, source snapshots, validation metrics, and progress histories. Logs and unsuccessful variants are retained. September was previously inspected in the project: all new September results are exploratory, not an untouched confirmatory test.

The original freeze records and manifests retain absolute paths from the execution machine as provenance. On another checkout, resolve the suffix starting at `artifacts/` or `data/` against your repository root. Do not rewrite existing freeze records or claim a rerun reproduces the original freeze hash. New runs must use new output directories; follow the reproduction commands in the report.
