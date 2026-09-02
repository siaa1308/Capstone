#!/usr/bin/env python3
"""Generate an interactive combined three-bank AML transaction graph.

The three active banks are source-bank partitions in the final dataset. This
tool combines those partitions for one temporal split, validates transaction
uniqueness, and aggregates destination institutions outside the selected three
into a single "Other Banks" node. It does not construct an account-level graph.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPOSITORY_ROOT / "data" / "final_temporal_dataset"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "graph_visualizations"
BANKS = ("JPMorgan_Chase", "Wells_Fargo", "Key_Bank")
SPLITS = ("training", "validation", "testing")
BANK_DISPLAY = {
    "JPMorgan_Chase": "JPMorgan Chase",
    "Wells_Fargo": "Wells Fargo",
    "Key_Bank": "Key Bank",
}
OTHER_BANKS = "Other Banks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", choices=SPLITS, default="testing")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_combined_transactions(dataset_dir: Path, split: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    transaction_columns = ["txn_id", "src_bank_id", "dst_bank_id", "amount"]
    for bank in BANKS:
        folder = dataset_dir / split / bank
        transactions = pd.read_csv(folder / "transactions.csv.gz", usecols=transaction_columns)
        truth = pd.read_csv(folder / "ground_truth.csv.gz", usecols=["txn_id", "y"])
        labeled = transactions.merge(truth, on="txn_id", how="left", validate="one_to_one")
        if labeled["y"].isna().any():
            raise ValueError(f"{split}/{bank} contains transactions without labels")
        expected_source = BANK_DISPLAY[bank]
        unexpected = set(labeled["src_bank_id"].dropna().astype(str)) - {expected_source}
        if unexpected:
            raise ValueError(
                f"{split}/{bank} contains unexpected source banks: {sorted(unexpected)}"
            )
        frames.append(labeled)

    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined[combined["txn_id"].duplicated(keep=False)]["txn_id"]
    if not duplicates.empty:
        examples = duplicates.astype(str).head(5).tolist()
        raise ValueError(f"Duplicate txn_id values across bank partitions: {examples}")
    combined["y"] = combined["y"].astype(int)
    combined["amount"] = pd.to_numeric(combined["amount"], errors="coerce").fillna(0.0).abs()
    return combined


def aggregate_flows(transactions: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected_names = set(BANK_DISPLAY.values())
    frame = transactions.copy()
    frame["source"] = frame["src_bank_id"].astype(str)
    frame["target"] = frame["dst_bank_id"].astype(str).where(
        frame["dst_bank_id"].astype(str).isin(selected_names), OTHER_BANKS
    )
    frame["suspicious_amount"] = frame["amount"] * frame["y"]

    grouped = (
        frame.groupby(["source", "target"], as_index=False, sort=True)
        .agg(
            transaction_count=("txn_id", "size"),
            total_amount=("amount", "sum"),
            suspicious_count=("y", "sum"),
            suspicious_amount=("suspicious_amount", "sum"),
        )
    )
    grouped["fraud_rate"] = grouped["suspicious_count"] / grouped["transaction_count"]

    names = [*BANK_DISPLAY.values(), OTHER_BANKS]
    nodes = [{"id": name, "external": name == OTHER_BANKS} for name in names]
    links = [
        {
            "source": str(row.source),
            "target": str(row.target),
            "transaction_count": int(row.transaction_count),
            "total_amount": float(row.total_amount),
            "suspicious_count": int(row.suspicious_count),
            "suspicious_amount": float(row.suspicious_amount),
            "fraud_rate": float(row.fraud_rate),
        }
        for row in grouped.itertuples(index=False)
    ]
    return nodes, links


def render_html(
    nodes: list[dict[str, object]],
    links: list[dict[str, object]],
    split: str,
    total_transactions: int,
    total_suspicious: int,
) -> str:
    payload = json.dumps({"nodes": nodes, "links": links}, separators=(",", ":")).replace("</", "<\\/")
    title = f"Combined three-bank flows — {split.title()}"
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
    header {{ padding: 16px 20px 6px; }}
    h1 {{ margin: 0 0 6px; font-size: 1.25rem; }}
    .meta {{ opacity: .75; font-size: .9rem; }}
    .controls {{ display: flex; gap: 18px; align-items: center; flex-wrap: wrap; padding: 10px 20px; }}
    label {{ display: flex; gap: 7px; align-items: center; }}
    select {{ font: inherit; }}
    #graph {{ width: 100%; height: 700px; border-top: 1px solid color-mix(in srgb, CanvasText 18%, transparent); }}
    .bank-node {{ fill: #3977b8; stroke: Canvas; stroke-width: 2px; cursor: pointer; }}
    .bank-node.external {{ fill: #777; }}
    .bank-label {{ fill: CanvasText; font-size: 13px; text-anchor: middle; pointer-events: none; }}
    .flow {{ fill: none; stroke: #3977b8; stroke-opacity: .38; cursor: pointer; }}
    .flow.has-suspicious {{ stroke: #d62728; stroke-opacity: .68; }}
    .tooltip {{ position: absolute; display: none; max-width: 360px; padding: 9px 11px; border-radius: 6px;
      background: Canvas; color: CanvasText; border: 1px solid color-mix(in srgb, CanvasText 25%, transparent);
      box-shadow: 0 3px 14px #0004; pointer-events: none; font-size: .85rem; }}
    .legend {{ margin-left: auto; opacity: .78; font-size: .85rem; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="meta">{total_transactions:,} transactions · {total_suspicious:,} ground-truth fraud cases · external destinations grouped as Other Banks</div>
  </header>
  <div class="controls">
    <label>Edge width
      <select id="metric">
        <option value="transaction_count">Transaction count</option>
        <option value="total_amount">Total amount</option>
        <option value="suspicious_count">Fraud count</option>
      </select>
    </label>
    <label><input id="suspicious-only" type="checkbox"> Flows containing fraud only</label>
    <span class="legend">Red = flow contains ground-truth fraud · Gray node = destinations outside the selected three</span>
  </div>
  <svg id="graph" role="img" aria-label="Combined directed transaction flows among three selected banks"></svg>
  <div id="tooltip" class="tooltip" role="tooltip"></div>
  <script>
    const data = {payload};
    const svg = d3.select("#graph"), tip = d3.select("#tooltip");
    function draw() {{
      svg.selectAll("*").remove();
      const width = svg.node().clientWidth, height = svg.node().clientHeight;
      const metric = document.getElementById("metric").value;
      const suspiciousOnly = document.getElementById("suspicious-only").checked;
      const links = data.links.filter(d => !suspiciousOnly || d.suspicious_count > 0);
      const cx = width / 2, cy = height / 2, radius = Math.min(width, height) * .34;
      const nodeById = new Map();
      data.nodes.forEach((d, i) => {{
        const angle = -Math.PI / 2 + i * Math.PI * 2 / data.nodes.length;
        nodeById.set(d.id, {{...d, x: cx + radius*Math.cos(angle), y: cy + radius*Math.sin(angle)}});
      }});
      const values = links.map(d => d[metric]);
      const maxValue = d3.max(values) || 1;
      const edgeWidth = d3.scaleSqrt().domain([0, maxValue]).range([1, 18]);
      const totalByBank = new Map(data.nodes.map(d => [d.id, 0]));
      data.links.forEach(d => {{
        totalByBank.set(d.source, totalByBank.get(d.source) + d.transaction_count);
        totalByBank.set(d.target, totalByBank.get(d.target) + d.transaction_count);
      }});
      const nodeSize = d3.scaleSqrt().domain([0, d3.max([...totalByBank.values()]) || 1]).range([24, 48]);
      const defs = svg.append("defs");
      defs.append("marker").attr("id","arrow").attr("viewBox","0 -5 10 10").attr("refX",10).attr("refY",0)
        .attr("markerWidth",5).attr("markerHeight",5).attr("orient","auto").append("path").attr("d","M0,-5L10,0L0,5").attr("fill","context-stroke");
      function path(d) {{
        const s=nodeById.get(d.source), t=nodeById.get(d.target);
        if (s.id === t.id) {{
          const r=nodeSize(totalByBank.get(s.id));
          return `M${{s.x-r*.65}},${{s.y-r*.65}} C${{s.x-r*2.5}},${{s.y-r*2.8}} ${{s.x+r*2.5}},${{s.y-r*2.8}} ${{s.x+r*.65}},${{s.y-r*.65}}`;
        }}
        const mx=(s.x+t.x)/2, my=(s.y+t.y)/2;
        const dx=t.x-s.x, dy=t.y-s.y, bend=.13;
        return `M${{s.x}},${{s.y}} Q${{mx-dy*bend}},${{my+dx*bend}} ${{t.x}},${{t.y}}`;
      }}
      svg.append("g").selectAll("path").data(links).join("path")
        .attr("class",d=>d.suspicious_count>0?"flow has-suspicious":"flow")
        .attr("d",path).attr("stroke-width",d=>edgeWidth(d[metric])).attr("marker-end","url(#arrow)")
        .on("mousemove",(e,d)=>tip.style("display","block").style("left",`${{e.pageX+12}}px`).style("top",`${{e.pageY+12}}px`)
          .html(`<b>${{d.source}} → ${{d.target}}</b><br>Transactions: ${{d.transaction_count.toLocaleString()}}<br>Total amount: ${{d.total_amount.toLocaleString(undefined,{{maximumFractionDigits:2}})}}<br>Fraud cases: ${{d.suspicious_count.toLocaleString()}}<br>Fraud rate: ${{(d.fraud_rate*100).toFixed(4)}}%`))
        .on("mouseleave",()=>tip.style("display","none"));
      const ng=svg.append("g").selectAll("g").data([...nodeById.values()]).join("g").attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
      ng.append("circle").attr("class",d=>d.external?"bank-node external":"bank-node")
        .attr("r",d=>nodeSize(totalByBank.get(d.id)))
        .on("mousemove",(e,d)=>tip.style("display","block").style("left",`${{e.pageX+12}}px`).style("top",`${{e.pageY+12}}px`)
          .html(`<b>${{d.id}}</b><br>Incident flow transactions: ${{totalByBank.get(d.id).toLocaleString()}}`))
        .on("mouseleave",()=>tip.style("display","none"));
      ng.append("text").attr("class","bank-label").attr("y",d=>nodeSize(totalByBank.get(d.id))+20)
        .each(function(d) {{
          const words=d.id.split(" "), line1=words.slice(0,2).join(" "), line2=words.slice(2).join(" ");
          d3.select(this).append("tspan").attr("x",0).text(line1);
          if(line2) d3.select(this).append("tspan").attr("x",0).attr("dy",15).text(line2);
        }});
    }}
    document.getElementById("metric").addEventListener("change",draw);
    document.getElementById("suspicious-only").addEventListener("change",draw);
    window.addEventListener("resize",draw);
    draw();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    transactions = load_combined_transactions(args.dataset_dir, args.split)
    nodes, links = aggregate_flows(transactions)
    output = args.output or DEFAULT_OUTPUT_DIR / f"combined_three_banks_{args.split}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_html(
            nodes,
            links,
            args.split,
            len(transactions),
            int(transactions["y"].sum()),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output.resolve()}")
    print(
        f"Combined transactions={len(transactions)}, suspicious={int(transactions['y'].sum())}, "
        f"bank_nodes={len(nodes)}, aggregated_flows={len(links)}"
    )


if __name__ == "__main__":
    main()
