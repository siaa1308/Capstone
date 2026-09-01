# EDA Report

Tables were computed from the full final dataset. Expensive plots use a reproducible stratified sample with random seed `42` and sample size `99999`.

## Class Imbalance

The positive class is rare across all selected banks and splits. See `eda_tables/label_distribution_by_split.csv` and `eda_tables/fraud_rate_by_bank.csv`.

## Temporal And Bank Drift

Use `eda_tables/split_bank_distribution.csv`, `transactions_by_day.csv`, `transactions_by_week.csv`, and `fraud_transactions_over_time.csv` to compare training, validation, and testing distributions. Bank-level differences are visible in transaction volume, fraud rate, graph size, and amount summaries, so graph and tabular evaluation should report split and bank-level metrics instead of only pooled metrics.

## Included Tables

- total transactions by bank
- total positive labels by bank
- fraud rate by bank
- label distribution by split
- amount and amount-by-label distributions
- payment-format, currency, and safe transaction-type distributions
- transactions by day, week, and month
- fraud transactions over time
- source and destination account activity
- account-degree, sent-count, and received-count distributions
- graph node and edge counts
