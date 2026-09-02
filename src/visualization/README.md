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
