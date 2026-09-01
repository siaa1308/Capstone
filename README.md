# Project Context — Federated Continual Learning for Privacy-Preserving AML

> Current optimized three-bank results and protocol caveats are in
> `docs/OPTIMIZATION_REPORT.md`; non-IID evidence is in
> `artifacts/optimized_fedavg/diagnostics/NON_IID_AUDIT.md`.

You are working on my B.Tech CSE/AI-ML capstone project.

## 1. Project Title

**Federated Continual Learning (FCL) with Explainable AI (XAI) for Privacy-Preserving Anti-Money Laundering**

The project is being developed as a research-oriented capstone, not just a normal ML application.

The core idea is:

**Financial Transaction Data → Graph Construction → Graph Neural Network → Continual Learning → Federated Learning → Privacy Preservation → Explainable AI → AML/Fraud Detection**

The goal is to build a modular prototype that demonstrates how multiple financial institutions could collaboratively improve AML/fraud detection without sharing their raw transaction data.

---

# 2. Problem We Are Solving

Traditional AML/fraud detection systems have several limitations:

1. Rule-based systems are easy to bypass when fraud patterns evolve.
2. Centralized ML requires financial institutions to share/store sensitive data centrally.
3. Fraud datasets are extremely imbalanced.
4. Fraud patterns change over time, causing concept drift.
5. Conventional ML models often forget previous patterns when trained on new data.
6. Transaction relationships between accounts are difficult to capture using purely tabular models.
7. Black-box models make it difficult for analysts to understand why a transaction was flagged.
8. Financial institutions have non-IID data distributions.

Our proposed system addresses these problems using:

* Graph Neural Networks
* Federated Learning
* Continual Learning
* Explainable AI
* Differential Privacy
* Secure aggregation
* Real-time/streaming architecture

The project documents describe this as an integrated framework combining **Federated Learning + Graph Modeling + Continual Learning + XAI + Privacy-Preserving Mechanisms**.

---

# 3. Important Scope Clarification

Do NOT try to implement every technology mentioned in our architecture at once.

This is a student research project.

The implementation priority is:

### Priority 1 — Core ML

* Dataset preprocessing
* Graph construction
* GraphSAGE/GNN
* Fraud/AML prediction
* Centralized baseline

### Priority 2 — Federated Learning

* Multiple simulated banks/clients
* Local training
* Model update sharing
* FedAvg
* Comparison against centralized training

### Priority 3 — Continual Learning

* Sequential/time-based training
* Replay buffer
* New fraud patterns
* Evaluation of catastrophic forgetting

### Priority 4 — Privacy

* Differential Privacy using Opacus where practical
* Secure/encrypted communication as a prototype
* Secure aggregation where feasible

### Priority 5 — XAI

* Explain predictions
* Feature/relation importance
* Human-readable reason codes

### Priority 6 — Real-Time Architecture

* Kafka/Flink can be implemented as an MVP/simulation if full deployment is too heavy.

Do NOT sacrifice the core research experiments just to build unnecessary infrastructure.

---

# 4. Dataset

The primary dataset for the current implementation is the **IBM AML synthetic transaction dataset**.

The dataset contains millions of financial transactions and is highly imbalanced, with laundering transactions representing approximately 0.1% of all transactions in our processed version.

Important files include:

* `transactions_master.csv.gz`
* `model_feature_columns.json`
* `edge_list.csv.gz`
* `node_map.csv.gz`
* `account_node_features.csv.gz`

The dataset has been transformed into a graph representation.

Current approximate graph scale from our experiments:

* ~397k nodes
* ~2.82M transaction edges
* ~24 node features in the updated graph representation
* Fraud/laundering nodes are extremely rare.

The exact dimensions should always be obtained from the actual files/configuration rather than hardcoded.

---

# 5. Dataset Structure

The transaction dataset contains information such as:

* transaction amount
* transaction day
* currency
* received amount
* receiving currency
* payment format
* initial balance of sender
* ending balance of sender
* initial balance of receiver
* ending balance of receiver
* transaction type
* historical transaction statistics
* source/destination entities
* timestamps
* laundering labels

Examples of historical graph/transaction features include:

* `src_prev_txn_count`
* `dst_prev_txn_count`
* `dst_prev_amount_sum`
* `dst_prev_amount_mean`

The target is the laundering/fraud label.

---

# 6. VERY IMPORTANT — Data Leakage

We have already encountered serious data leakage during experimentation.

Some columns/features were found to contain direct or indirect information about the target.

Potential leakage columns that have previously been removed include:

```text
y
laundering_type
edge_label
Is_APP_Fraud
Is_Cheque_Fraud
APP_Fraudster_ID
Cheque_Fraudster_ID
APP_Fraud_Sequence_Number
txn_id
src_id
dst_id
Transaction_Date
Transaction_Time
timestamp
```

Additionally, **transaction_type_raw** was found to be suspicious/highly correlated with the laundering label in earlier experiments.

Do NOT blindly use any feature just because it improves the metric.

If a feature could encode the target, future information, laundering-generation logic, fraudster identity, or label-derived information, investigate it before using it.

The objective is a scientifically defensible model, not artificially high AUC.

---

# 7. Previous Modeling Results

We have already experimented with classical ML and GraphSAGE.

Some earlier centralized tabular results were suspiciously high:

* ROC-AUC around 0.999+
* PR-AUC around 0.998+

These were investigated because of possible leakage.

After removing suspicious leakage, performance dropped substantially.

More realistic experiments produced results around:

* ROC-AUC ≈ 0.984
* PR-AUC ≈ 0.416

A GraphSAGE experiment produced approximately:

* ROC-AUC ≈ 0.923
* PR-AUC ≈ 0.169

These numbers are NOT necessarily the final results.

They are historical checkpoints.

Never claim that these are final project results.

Always run the current code and report the actual results.

---

# 8. Graph Modeling

The project uses graph-based learning because AML is fundamentally relational.

Entities can be represented as nodes and transactions/relationships as edges.

The conceptual architecture supports a heterogeneous graph containing entities such as:

### Nodes

* Account
* Transaction
* Device
* Merchant

### Relationships

* Account → Transaction
* Account → Device
* Account ↔ Account
* shared device/IP
* transaction relationships

Our current IBM implementation is primarily focused on the available transaction/account graph.

Do NOT unnecessarily force a heterogeneous graph architecture if the current dataset representation does not support it cleanly.

Start with a robust homogeneous/account transaction graph if that is what the current data supports.

Then extend toward heterogeneous GraphSAGE only when justified.

---

# 9. Current GNN

Our primary GNN architecture is **GraphSAGE using PyTorch Geometric**.

Typical structure:

```text
Input Node Features
        ↓
SAGEConv
        ↓
Activation
        ↓
Dropout
        ↓
SAGEConv
        ↓
Prediction Head
```

The exact architecture should remain configurable.

Use:

* PyTorch
* PyTorch Geometric
* SAGEConv
* NeighborLoader or other scalable sampling methods where required

Because the graph contains hundreds of thousands of nodes and millions of edges, do NOT unnecessarily perform full-batch message passing if it causes memory problems.

---

# 10. Centralized Baseline

Before federated learning, we need a centralized baseline.

At minimum, maintain:

### Classical baselines

* Logistic Regression where practical
* Random Forest
* XGBoost

### Deep/Graph baseline

* GraphSAGE

The purpose is to establish reference performance.

The final research comparison should answer:

> Does federated continual graph learning provide competitive performance while providing privacy and adaptability benefits?

Do NOT compare models using only accuracy.

Because the dataset is extremely imbalanced, prioritize:

* PR-AUC
* ROC-AUC
* Precision
* Recall
* F1
* confusion matrix
* minority-class performance

Accuracy alone is misleading.

---

# 11. Federated Learning

The final architecture should simulate multiple financial institutions.

For example:

```text
Bank A ── Local Graph/Data ── Local GraphSAGE
                              ↓
Bank B ── Local Graph/Data ── Local GraphSAGE
                              ↓
Bank C ── Local Graph/Data ── Local GraphSAGE
                              ↓
                         Federated Server
                              ↓
                           FedAvg
                              ↓
                       Global Model
                              ↓
                 Distributed back to banks
```

Use **Flower (FLWR)** for federated learning where practical.

Start with:

**FedAvg**

Then optionally investigate:

**FedProx**

The clients should NOT send raw transaction data to the server.

Only model parameters/updates should be exchanged.

---

# 12. Simulated Banks

We do not have actual private datasets from multiple banks.

Therefore, the federated setup will simulate banks by partitioning the IBM dataset.

The active experimental cohort contains three simulated clients: JPMorgan Chase,
Wells Fargo, and Key Bank. Citi and Fifth Third Bancorp remain in the processed
dataset and historical artifacts but are excluded from new model runs based on
their weaker development PR-AUC. This selection must be disclosed as a
development-stage cohort decision rather than presented as a test-independent
scientific result.

Possible strategies:

### IID partition

Randomly divide data.

### Non-IID partition

Create heterogeneous clients based on:

* transaction distributions
* account groups
* time periods
* transaction types
* class distributions

The **non-IID experiment is important** because real banks will naturally have different transaction distributions.

Do not claim that the dataset represents actual banks.

Explicitly describe these as simulated federated clients.

---

# 13. Continual Learning

Continual Learning is one of the major research components.

Fraud patterns evolve over time.

We therefore want:

```text
Time Period 1
     ↓
Train
     ↓
Time Period 2
     ↓
Train on new data
     ↓
Retain knowledge from Period 1
     ↓
Time Period 3
     ↓
Continue learning
```

The primary continual learning strategy is:

## Replay-Based Continual Learning

Maintain a replay buffer containing selected historical fraud/important samples.

During new-task training:

```text
New Data + Replay Data
        ↓
Local Training
        ↓
Updated Model
```

The goal is to reduce **catastrophic forgetting**.

The replay buffer should be configurable.

Do not implement an unnecessarily complex continual-learning algorithm unless there is a research reason.

---

# 14. Continual Learning Evaluation

We need to measure whether continual learning actually works.

Do not simply train sequentially and claim success.

Evaluate:

### Before new task

Performance on old data.

### After learning new task

Performance on:

* new data
* old data

Compare:

```text
Without Replay
vs
With Replay
```

Measure forgetting.

For example:

```text
Forgetting = Old Performance Before Update
             -
             Old Performance After Update
```

The exact metric can be refined during implementation.

The important research question is:

> Does replay-based continual learning preserve previous AML knowledge while adapting to new patterns?

---

# 15. Federated + Continual Learning

The final research system combines both.

Conceptually:

```text
                GLOBAL MODEL
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
     Bank A        Bank B        Bank C
        ↓            ↓            ↓
   New local      New local     New local
     data           data          data
        ↓            ↓            ↓
   Replay CL      Replay CL     Replay CL
        ↓            ↓            ↓
   Local model    Local model   Local model
        └────────────┼────────────┘
                     ↓
              Privacy Layer
                     ↓
                FedAvg Server
                     ↓
               GLOBAL MODEL
```

This loop repeats.

---

# 16. Privacy-Preserving Layer

Privacy is part of the research contribution.

Potential components:

### Differential Privacy

Use:

**Opacus**

to introduce gradient clipping and noise.

The objective is to demonstrate that model updates can be protected from revealing individual data contributions.

Do not claim formal privacy guarantees unless the implementation actually supports and measures them.

### Secure Aggregation

The conceptual architecture uses secure aggregation so that the server receives an aggregate rather than directly inspecting each client's update.

If a full production implementation is too complex, build a research prototype and clearly document the limitation.

### Encryption

For communication:

* TLS/HTTPS
* gRPC + TLS where applicable

Encryption of model updates may be simulated/prototyped using standard cryptographic libraries if required.

Do NOT pretend that simple AES encryption alone constitutes a complete secure aggregation protocol.

---

# 17. Explainable AI

Every suspicious prediction should ideally have an explanation.

Possible tools:

* SHAP
* LIME
* GraphLIME
* custom graph-based explanations

For the initial implementation, choose the simplest method that works correctly with our model.

The output should ideally look conceptually like:

```text
Prediction: Suspicious

Risk Score: 0.91

Reason Codes:
- Unusual transaction amount
- Receiver has abnormal transaction history
- Strong relationship with previously suspicious entities
- Significant deviation from historical account behavior
```

Do not fabricate explanations.

Every explanation must actually be derived from model/features/graph evidence.

---

# 18. Streaming Architecture

The final system is intended to support near-real-time transaction processing.

Conceptual flow:

```text
Incoming Transaction
        ↓
Kafka
        ↓
Stream Processing
        ↓
Feature Processing
        ↓
Graph Update
        ↓
GNN Inference
        ↓
Risk Score
        ↓
XAI
        ↓
Decision / Alert
        ↓
Database
```

Technologies proposed:

* Kafka
* Apache Flink
* FastAPI
* PostgreSQL

However, infrastructure should be implemented progressively.

If Kafka/Flink makes the research pipeline unnecessarily difficult, create a clean simulation first and make the streaming layer modular.

The research experiments must remain reproducible without requiring a huge distributed cluster.

---

# 19. Backend

Use:

**FastAPI**

for APIs.

Potential endpoints:

```text
POST /transaction
POST /predict
GET /prediction/{id}
GET /model/status
GET /model/metrics
POST /federated/train
GET /federated/status
GET /explanation/{transaction_id}
```

These are examples, not strict requirements.

Keep APIs modular and clean.

---

# 20. Database

Preferred:

**PostgreSQL**

Possible data:

* transactions
* prediction results
* alerts
* model versions
* training rounds
* client status
* explanation records

Redis is optional.

Do not introduce Redis unless there is an actual need.

---

# 21. Frontend

The proposed frontend can use:

**React.js**

or, for a faster MVP:

**Streamlit**

The dashboard should eventually show:

* transaction details
* risk score
* fraud/AML prediction
* explanation/reason codes
* model version
* federated training status
* client/bank status
* metrics
* continual-learning status
* alerts

Do not spend excessive time on frontend styling before the ML pipeline works.

---

# 22. Proposed Repository Structure

Keep the repository modular.

A reasonable structure is:

```text
project/
│
├── configs/
│   └── config.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── graph_builder.py
│   │   └── dataset.py
│   │
│   ├── models/
│   │   ├── graphsage.py
│   │   └── classifiers.py
│   │
│   ├── training/
│   │   ├── centralized.py
│   │   ├── continual.py
│   │   └── federated.py
│   │
│   ├── privacy/
│   │   ├── differential_privacy.py
│   │   └── secure_aggregation.py
│   │
│   ├── explainability/
│   │   └── explainer.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   ├── federated/
│   │   ├── client.py
│   │   └── server.py
│   │
│   └── streaming/
│       ├── producer.py
│       └── consumer.py
│
├── scripts/
│   ├── preprocess.py
│   ├── build_graph.py
│   ├── train_centralized.py
│   ├── train_federated.py
│   └── evaluate.py
│
├── tests/
│
├── checkpoints/
├── logs/
├── requirements.txt
├── README.md
└── .gitignore
```

The actual repository structure may already differ.

**Do not restructure the repository blindly.**

First inspect the existing repository and preserve working code.

---

# 23. Coding Rules

This is extremely important.

Before modifying anything:

1. Inspect the existing repository.
2. Understand the current data flow.
3. Identify what is already working.
4. Run existing scripts/tests where possible.
5. Do not rewrite functioning components unnecessarily.

When implementing a new component:

* make it modular
* make paths configurable
* avoid hardcoded machine-specific paths
* use configuration files
* add useful logging
* save checkpoints
* make experiments reproducible
* set random seeds
* document assumptions
* avoid unnecessary dependencies

Do not create five different versions of the same model.

Prefer one clean implementation with configuration options.

---

# 24. Experimental Discipline

This is a research project.

Every experiment should record:

* dataset/split
* random seed
* model architecture
* hyperparameters
* training epochs
* client count
* federated rounds
* aggregation method
* continual-learning configuration
* replay-buffer size
* privacy parameters
* evaluation metrics

Never optimize solely for the highest metric.

If a result looks suspiciously high, investigate leakage.

---

# 25. Evaluation Strategy

The final evaluation should compare progressively:

### Experiment 1

Classical centralized ML

### Experiment 2

Centralized GraphSAGE

### Experiment 3

Federated GraphSAGE

### Experiment 4

Federated GraphSAGE + Continual Learning

### Experiment 5

Federated GraphSAGE + Continual Learning + Privacy

### Experiment 6

Add XAI

The exact experiments may change based on feasibility.

The final paper should be able to answer:

1. Does graph modeling improve AML detection?
2. Does federated learning maintain competitive performance?
3. How does non-IID data affect FL?
4. Does continual learning reduce forgetting?
5. What is the cost of privacy mechanisms?
6. Can the model provide meaningful explanations?
7. Can the architecture support near-real-time inference?

---

# 26. Metrics

Primary:

* PR-AUC
* ROC-AUC
* Precision
* Recall
* F1

Secondary:

* confusion matrix
* training time
* inference latency
* communication overhead
* convergence rounds
* client-to-client performance variance
* forgetting score
* memory usage where useful

For fraud detection, **PR-AUC is particularly important** because the positive class is extremely rare.

---

# 27. Research Positioning

The novelty is NOT that any one of these technologies is individually new.

The research contribution is the integrated framework:

**Graph-based AML + Federated Learning + Continual Learning + Privacy + Explainability**

with emphasis on:

* non-IID financial data
* evolving fraud patterns
* severe class imbalance
* privacy
* graph relationships
* explainability

The project documents explicitly position the system around this integrated architecture.

---

# 28. Technologies

Current planned stack:

### Programming

Python 3.10+

### ML

PyTorch
NumPy
Pandas

### Graph ML

PyTorch Geometric
GraphSAGE

### Federated Learning

Flower
FedAvg
FedProx

### Continual Learning

Custom replay buffer
Incremental PyTorch training

### Privacy

Opacus
Differential Privacy
Secure aggregation prototype
TLS

### Streaming

Kafka
Apache Flink

### Backend

FastAPI
Uvicorn

### Database

PostgreSQL

### Frontend

React.js or Streamlit

### Deployment

Docker / Docker Compose

### Explainability

SHAP / LIME / GraphLIME

Do not add technologies just because they sound impressive.

---

# 29. Current Development Philosophy

We are building this incrementally.

The correct order is:

```text
Dataset
   ↓
Clean preprocessing
   ↓
Leakage-free baseline
   ↓
Graph
   ↓
GraphSAGE
   ↓
Evaluation
   ↓
Federated Learning
   ↓
Continual Learning
   ↓
Privacy
   ↓
XAI
   ↓
Streaming/API
   ↓
Frontend
```

Do not jump directly to the complete architecture.

At every stage, make sure the previous stage works.

---

# 30. How You Should Work With Me

I am a student developing this project and I will frequently ask you to modify/create/debug individual files.

When I ask you to implement something:

1. Inspect the repository first.
2. Understand existing code.
3. Tell me briefly what you found.
4. Implement the smallest correct change.
5. Do not modify unrelated files.
6. Give me the exact commands to run.
7. Help me interpret the output.
8. Only move to the next architectural component after the current component is verified.

If something is scientifically questionable, tell me.

If something is likely leakage, tell me.

If a metric looks suspicious, tell me.

If a proposed architecture is unnecessarily complicated for our capstone, tell me.

Do not blindly follow my assumptions if they conflict with the actual code/data.

---

# 31. Most Important Rule

**The goal is a credible, reproducible research prototype — NOT artificially high metrics and NOT an unnecessarily complicated production system.**

Every architectural decision should support the research question:

> Can Federated Continual Learning with graph-based modeling provide adaptive, privacy-preserving, explainable AML detection across heterogeneous financial institutions without sharing raw transaction data?

Build toward answering that question experimentally.

Start by inspecting the current repository and identifying exactly what has already been implemented before making any changes.
