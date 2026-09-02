# Active-cohort non-IID audit

All statistics below use training data only. They diagnose heterogeneity; they do not use validation or test labels.

| bank           |   transactions |   positives |   fraud_rate |   nodes |   edges_per_node |   amount_mean |   amount_std |
|:---------------|---------------:|------------:|-------------:|--------:|-----------------:|--------------:|-------------:|
| JPMorgan_Chase |         126423 |          65 |  0.000514147 |   24906 |          5.07601 |       3561.88 |      68932.6 |
| Wells_Fargo    |          50346 |          80 |  0.001589    |   11292 |          4.45855 |       2988.52 |      42062.8 |
| Key_Bank       |          22412 |         100 |  0.0044619   |    5314 |          4.21754 |       3112.69 |      84932.2 |

Transaction-volume max/min ratio: 5.641.
Fraud-rate max/min ratio: 8.678.

Pairwise Jensen-Shannon divergences and client-update norms are recorded in `non_iid_audit.json`.
