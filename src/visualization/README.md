# AML graph visualization

## Step 1: local-bank graph explorer

Generate a sampled interactive graph for one bank and temporal split:

```powershell
python src/visualization/local_graph_explorer.py --bank Citi --split testing
```

The generated HTML is written under `artifacts/graph_visualizations/`. Open it
in a browser, drag or zoom the network, hover over accounts and transactions,
and use the checkbox to isolate ground-truth fraud transactions.

The tool prioritizes ground-truth fraud edges and then high-value regular edges.
It intentionally limits the browser graph; it never changes the source dataset.

Available banks are `JPMorgan_Chase`, `Wells_Fargo`, `Citi`,
`Fifth_Third_Bancorp`, and `Key_Bank`. Available splits are `training`,
`validation`, and `testing`.

Use `--max-nodes` and `--max-edges` to adjust the display budget:

```powershell
python src/visualization/local_graph_explorer.py --bank Key_Bank --split validation --max-nodes 250 --max-edges 500
```

## Step 2: combined three-bank flow graph

Generate a bank-level combined graph for one temporal split:

```powershell
python src/visualization/combined_bank_graph.py --split testing
```

The graph combines JPMorgan Chase, Wells Fargo, and Key Bank, verifies that
transaction IDs do not overlap, and aggregates destinations outside those three into
`Other Banks`. Edge width can represent transaction count, total amount, or
suspicious transaction count. This is intentionally a bank-level view; it does
not merge or display account-level nodes.

Both visualizations use only the dataset graph and `ground_truth.csv.gz` labels.
They do not read model checkpoints, predictions, training metrics, or experiment
artifacts.

## Step 3: September fraud-case explorer

Build the standalone offline case explorer (Python standard library only):

```sh
python3 src/visualization/fraud_case_explorer.py
```

Open `artifacts/graph_visualizations/fraud_cases/index.html`. No server, CDN,
installation, model checkpoint or internet connection is required. The adjacent
`cases.json` contains the same evidence for reproducibility. Source data and
model artifacts are not modified.

The explorer covers **every September ground-truth positive**: 132 cases across
all five retained banks, including 88 in the current JPMorgan Chase / Wells Fargo /
Key Bank cohort. Filter by bank/cohort or search transaction/account ID, select a
case, then click its fan-in, fan-out or cycle buttons. Drag nodes, zoom, inspect
an edge's transaction features, and export the case evidence as JSON or its graph
as SVG for a presentation.

### Evidence definitions and scope

- Accounts are identified by **bank + account ID**; graph node indices from
  different partitions are never merged. Directed arrows indicate payment flow.
- Windows are 1, 7 or 30 days. The default includes strictly earlier transactions
  and the selected transaction; simultaneous other events are excluded. The
  retrospective option includes the same number of days before **and after** the
  selected transaction. All windows are clipped to available September data.
- Fan-in/out requires at least **three distinct counterparties** at either selected
  endpoint, excludes self-transfers, and uses both transaction labels. It is an
  endpoint pattern and need not include the selected transaction itself.
- Directed cycles must contain the selected transaction, have 2–4 edges, and use
  distinct accounts except for closing the loop. Search returns up to three
  examples with a 20,000-edge traversal budget; reaching either cap is disclosed.
  Strict chronological order around the cycle is evaluated separately. Structural
  connectivity does not prove a flow of the same funds.
- Amount percentile uses strictly earlier September transactions from the same
  source bank, currency and payment format. The sender comparison uses strictly
  earlier outgoing transactions in the same currency. Sample sizes are shown;
  there is no mixed-currency aggregation or implied fraud threshold.
- The graph displays at most 90 edges, preserving cycle examples and sampling
  fan witnesses before nearest endpoint transactions. Full motif counts and all
  motif witness records are retained in the JSON export. Not all context edges
  are retained in the export; the UI reports the full endpoint transaction count.
- Only five source-bank partitions are available. Outgoing flows from external
  banks may be missing. Absence of a detected motif is not proof that none exists.

**Interpretation:** red edges are dataset labels, not predictions. The prepared
files contain no laundering-type reason. Motifs and amount comparisons describe
observable activity, not the cause of a label or model feature attribution. The
GNN uses amount, currencies and payment format as edge features and timing for
memory updates; the explorer's motifs and percentiles are investigative summaries.
This post hoc visualization does not retrain, tune, or reevaluate the model.

Checks:

```sh
python3 -m unittest discover -s tests -p test_fraud_case_explorer.py
```

Tests cover distinct counterparties, self-transfers, directed and chronological
cycles, causal window boundaries, currency baselines, display sampling and search
cap disclosure.
