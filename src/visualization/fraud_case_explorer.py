#!/usr/bin/env python3
"""Build an offline, descriptive September fraud-case explorer (not model attribution)."""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
ACTIVE = {'JPMorgan_Chase', 'Wells_Fargo', 'Key_Bank'}


def read_csv(path):
    with gzip.open(path, 'rt', newline='') as handle:
        return list(csv.DictReader(handle))


def load_transactions(dataset):
    transactions, seen = [], set()
    for folder in sorted((dataset / 'testing').iterdir()):
        if not folder.is_dir():
            continue
        labels = read_csv(folder / 'ground_truth.csv.gz')
        truth = {r['txn_id']: int(r['y']) for r in labels}
        if len(truth) != len(labels) or set(truth.values()) - {0, 1}:
            raise ValueError(f'Invalid or duplicate labels: {folder}')
        rows = read_csv(folder / 'transactions.csv.gz')
        if {r['txn_id'] for r in rows} != set(truth):
            raise ValueError(f'Transaction/label coverage mismatch: {folder}')
        for r in rows:
            txn = r['txn_id']
            if txn in seen:
                raise ValueError(f'Duplicate transaction ID: {txn}')
            seen.add(txn)
            stamp = datetime.fromisoformat(r['timestamp'])
            if (stamp.year, stamp.month) != (2025, 9):
                raise ValueError(f'Not September 2025: {txn}')
            amount = float(r['amount'])
            if not math.isfinite(amount):
                raise ValueError(f'Nonfinite amount: {txn}')
            # Bank-qualified identifiers avoid merging distinct accounts across banks.
            src = r['src_bank_id'] + ' | ' + r['src_id']
            dst = r['dst_bank_id'] + ' | ' + r['dst_id']
            transactions.append(dict(id=txn, t=r['timestamp'], seconds=(stamp-datetime(2025,9,1)).total_seconds(),
                src=src, dst=dst, bank=folder.name, amount=amount, currency=r['currency'],
                received=r['Amount_Received'], receivingCurrency=r['Receiving_Currency'],
                format=r['payment_format'], fraud=truth[txn]))
    return sorted(transactions, key=lambda e: (e['seconds'], e['id']))


def cycle_paths(focal, edges, max_edges=4, limit=3, budget=20000):
    """Find simple directed cycles containing focal; bounded search is disclosed."""
    if focal['src'] == focal['dst']:
        return [], False  # self-transfers are not multi-account cycles
    outgoing = defaultdict(list)
    for e in edges:
        if e['id'] != focal['id'] and e['src'] != e['dst']:
            outgoing[e['src']].append(e)
    for rows in outgoing.values():
        rows.sort(key=lambda e: (abs(e['seconds']-focal['seconds']), e['id']))
    paths, steps, capped = [], 0, False

    def visit(node, path, visited):
        nonlocal steps, capped
        if len(paths) >= limit:
            capped = True
            return
        for edge in outgoing[node]:
            steps += 1
            if steps > budget:
                capped = True
                return
            if edge['dst'] == focal['src']:
                paths.append([focal] + path + [edge])
                if len(paths) >= limit:
                    capped = True
                    return
            elif len(path) + 2 < max_edges and edge['dst'] not in visited:
                visit(edge['dst'], path + [edge], visited | {edge['dst']})
                if steps > budget or len(paths) >= limit:
                    return
    visit(focal['dst'], [], {focal['src'], focal['dst']})
    return paths, capped


def chronological_cycle(path):
    # A cycle can start at any account; strict order excludes simultaneous events.
    times = [e['seconds'] for e in path]
    return any(all(a < b for a,b in zip(times[i:]+times[:i], (times[i:]+times[:i])[1:]))
               for i in range(len(times)))


def analyze_case(focal, window, display_limit=90):
    endpoints = {focal['src'], focal['dst']}
    incident = [e for e in window if e['src'] in endpoints or e['dst'] in endpoints]
    motifs = []
    for node in sorted(endpoints):
        for kind in ('Fan-in', 'Fan-out'):
            relevant = [e for e in incident if e['src'] != e['dst'] and
                        (e['dst'] == node if kind == 'Fan-in' else e['src'] == node)]
            peer = 'src' if kind == 'Fan-in' else 'dst'
            peers = sorted({e[peer] for e in relevant})
            if len(peers) >= 3:
                # One witness per counterparty, closest to focal time.
                witness = {}
                for e in sorted(relevant, key=lambda x: (abs(x['seconds']-focal['seconds']), x['id'])):
                    witness.setdefault(e[peer], e['id'])
                motifs.append(dict(kind=kind, node=node, peers=len(peers), transactions=len(relevant),
                                   edges=list(witness.values())))
    paths, capped = cycle_paths(focal, window)
    for path in paths:
        motifs.append(dict(kind='Cycle', edges=[e['id'] for e in path],
                           chronological=chronological_cycle(path),
                           route=[path[0]['src']] + [e['dst'] for e in path]))
    by_id = {e['id']: e for e in window}
    chosen = {focal['id']}
    # Preserve complete small-cycle evidence; sample large fan witnesses fairly.
    for m in motifs:
        if m['kind'] == 'Cycle':
            chosen.update(m['edges'])
    fans = [m for m in motifs if m['kind'] != 'Cycle']
    for i in range(max((len(m['edges']) for m in fans), default=0)):
        for m in fans:
            if i < len(m['edges']) and len(chosen) < display_limit:
                chosen.add(m['edges'][i])
    for e in sorted(incident, key=lambda e: (abs(e['seconds']-focal['seconds']), e['id'])):
        if len(chosen) >= display_limit:
            break
        chosen.add(e['id'])
    selected = sorted(chosen, key=lambda key: (by_id[key]['seconds'], key))
    return dict(edges=selected, incidentCount=len(incident), motifs=motifs, cycleSearchCapped=capped,
                windowCount=len(window), displayLimited=len(set(e['id'] for e in incident) | chosen) > len(chosen))


def build_payload(transactions):
    times = [e['seconds'] for e in transactions]
    cases, retained = [], set()
    for focal in transactions:
        if not focal['fraud']:
            continue
        earlier = transactions[:bisect.bisect_left(times, focal['seconds'])]
        peer = [e['amount'] for e in earlier if e['bank'] == focal['bank'] and
                e['currency'] == focal['currency'] and e['format'] == focal['format']]
        own = [e['amount'] for e in earlier if e['src'] == focal['src'] and e['currency'] == focal['currency']]
        baseline = dict(peerCount=len(peer), ownCount=len(own),
            percentile=100*sum(x <= focal['amount'] for x in peer)/len(peer) if peer else None,
            peerMedian=median(peer) if peer else None, ownMedian=median(own) if own else None)
        views = {}
        for days in (1,7,30):
            for mode in ('prior', 'around'):
                lo = bisect.bisect_left(times, focal['seconds']-days*86400)
                # Prior mode excludes simultaneous transactions, except selected transaction.
                hi = bisect.bisect_left(times, focal['seconds']) if mode == 'prior' else bisect.bisect_right(times, focal['seconds']+days*86400)
                window = transactions[lo:hi]
                if mode == 'prior':
                    window = window + [focal]
                view = analyze_case(focal, window)
                views[f'{days}-{mode}'] = view
                retained.update(view['edges'])
                # Keep all motif evidence available for export even when graph is sampled.
                for motif in view['motifs']:
                    retained.update(motif['edges'])
        cases.append(dict(id=focal['id'], bank=focal['bank'], baseline=baseline, views=views))
    counts = {bank:sum(e['fraud'] for e in transactions if e['bank']==bank) for bank in sorted({e['bank'] for e in transactions})}
    return dict(cases=cases, edges={e['id']:e for e in transactions if e['id'] in retained},
                totalTransactions=len(transactions), counts=counts, activeBanks=sorted(ACTIVE),
                period='September 2025', methodology='Descriptive ground-truth case review; no model attribution or inferred label cause.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT/'data/final_temporal_dataset')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'artifacts/graph_visualizations/fraud_cases')
    args = parser.parse_args()
    payload = build_payload(load_transactions(args.dataset_dir))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(',', ':'), allow_nan=False)
    (args.output_dir/'cases.json').write_text(data)
    template = (Path(__file__).parent/'templates/fraud_cases.html').read_text()
    (args.output_dir/'index.html').write_text(template.replace('__CASE_DATA__', data.replace('<','\\u003c')))
    print(json.dumps(dict(cases=len(payload['cases']), counts=payload['counts'], output=str(args.output_dir/'index.html')), indent=2))

if __name__ == '__main__':
    main()
