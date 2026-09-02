# AML Paper Evaluation Summary

Primary metric: PR-AUC. Active cohort: JPMorgan Chase, Wells Fargo, and Key Bank. Citi and Fifth Third Bancorp are excluded from new experiments based on prior development PR-AUC and remain in historical artifacts for auditability. Results are development diagnostics until the final configuration is frozen and evaluated once on the held-out test period.

## Leakage-safe tabular XGBoost

_Not yet run: `artifacts/xgboost_paper_baseline/metrics.json`._

## Static GraphSAGE

_No saved results for the active three-bank cohort._

## Local temporal GNN

| Bank | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| JPMorgan_Chase | 0.559 | 0.998 | 0.220 | 0.737 | 0.332 |
| Wells_Fargo | 0.257 | 0.902 | 0.153 | 0.361 | 0.170 |
| Key_Bank | 0.333 | 0.885 | 0.528 | 0.304 | 0.368 |
| **Macro mean** | 0.383 | 0.928 | 0.300 | 0.467 | 0.290 |

Macro standard deviation across banks: pr_auc=0.128, roc_auc=0.050, precision=0.163, recall=0.192, f1=0.086.

## FedAvg temporal GNN

| Bank | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| JPMorgan_Chase | 0.377 | 0.995 | 0.364 | 0.632 | 0.462 |
| Wells_Fargo | 0.172 | 0.959 | 0.368 | 0.292 | 0.326 |
| Key_Bank | 0.335 | 0.894 | 0.722 | 0.289 | 0.413 |
| **Macro mean** | 0.295 | 0.949 | 0.485 | 0.404 | 0.400 |

Macro standard deviation across banks: pr_auc=0.088, roc_auc=0.042, precision=0.168, recall=0.161, f1=0.056.

## Federated continual GNN + replay

| Bank | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| JPMorgan_Chase | 0.280 | 0.976 | 0.186 | 0.421 | 0.258 |
| Wells_Fargo | 0.155 | 0.900 | 0.000 | 0.000 | 0.000 |
| Key_Bank | 0.354 | 0.897 | 0.750 | 0.333 | 0.462 |
| **Macro mean** | 0.263 | 0.925 | 0.312 | 0.251 | 0.240 |

Macro standard deviation across banks: pr_auc=0.082, roc_auc=0.037, precision=0.319, recall=0.181, f1=0.189.
