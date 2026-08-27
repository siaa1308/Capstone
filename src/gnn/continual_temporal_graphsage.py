#!/usr/bin/env python3
"""Local continual-learning experiment: June → July with optional replay.

Replay samples are drawn only from June training events (all positives first, then
seeded negatives). No validation/test labels or events participate in replay.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from causal_temporal_graphsage import (BANKS, DEFAULT_DATASET, DEFAULT_OUTPUT, CausalTemporalGraphSAGE, Events,
    batches, build_bank_data, metric_block, sampled_loss, score_stream, set_seed)

def args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", choices=("all",*BANKS), default="all"); p.add_argument("--dataset-dir",type=Path,default=DEFAULT_DATASET)
    p.add_argument("--epochs-per-task",type=int,default=20); p.add_argument("--replay-size",type=int,default=2000)
    p.add_argument("--negative-ratio",type=int,default=20); p.add_argument("--hidden-channels",type=int,default=64)
    p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="cpu")
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT.parent/"continual_temporal_graphsage"); return p.parse_args()

def subset(e: Events, mask: torch.Tensor) -> Events:
    return Events(e.src[mask],e.dst[mask],e.edge_attr[mask],e.timestamp[mask],e.labels[mask])

def concat(a: Events,b: Events)->Events:
    order=torch.argsort(torch.cat((a.timestamp,b.timestamp)),stable=True)
    return Events(*(torch.cat((x,y))[order] for x,y in zip((a.src,a.dst,a.edge_attr,a.timestamp,a.labels),(b.src,b.dst,b.edge_attr,b.timestamp,b.labels))))

def train_task(model, static, events, optimizer, cfg, seed):
    gen=torch.Generator(device=static.device).manual_seed(seed)
    for _ in range(cfg.epochs_per_task):
        model.train(); state=model.initial_state(static)
        for batch in batches(events,cfg.batch_size):
            optimizer.zero_grad(); logits,state=model.score_and_update(state,batch)
            loss=sampled_loss(logits,batch.labels,cfg.negative_ratio,1.0,gen)
            if loss is not None: loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            state=state.detached()

def evaluate(model,static,events,batch_size):
    probs,labels,_=score_stream(model,model.initial_state(static),events,batch_size)
    # Fixed 0.5 threshold is used only for a like-for-like retention diagnostic;
    # no June-after-July labels are used to tune this threshold.
    return metric_block(labels,probs,0.5)

def run_bank(cfg,bank,device):
    seed=cfg.seed+BANKS.index(bank)*10000; set_seed(seed)
    static,streams,_,_=build_bank_data(cfg.dataset_dir,bank); static=static.to(device); train=streams["training"].to(device)
    # Dataset training span is June–July 2025; 31 days after the first event is July 1.
    june=subset(train,train.timestamp < 31*86400); july=subset(train,train.timestamp >= 31*86400)
    model=CausalTemporalGraphSAGE(static.shape[1],train.edge_attr.shape[1],cfg.hidden_channels,.25).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=1e-4)
    train_task(model,static,june,opt,cfg,seed); before=evaluate(model,static,june,cfg.batch_size)
    positive=torch.where(june.labels==1)[0]; negative=torch.where(june.labels==0)[0]; keep_neg=max(0,cfg.replay_size-len(positive))
    gen=torch.Generator(device=device).manual_seed(seed); picked=torch.cat((positive,negative[torch.randperm(len(negative),generator=gen,device=device)[:keep_neg]]))
    replay=subset(june,picked); task2=concat(replay,july) if cfg.replay_size else july
    train_task(model,static,task2,opt,cfg,seed+1); after=evaluate(model,static,june,cfg.batch_size)
    return {"bank":bank,"june_events":len(june),"july_events":len(july),"replay_events":len(replay) if cfg.replay_size else 0,
            "june_before_july":before,"june_after_july":after,"forgetting":before["pr_auc"]-after["pr_auc"]}

def main():
    cfg=args(); cfg.dataset_dir=cfg.dataset_dir.resolve(); device=torch.device(cfg.device); selected=BANKS if cfg.bank=="all" else (cfg.bank,)
    results=[run_bank(cfg,b,device) for b in selected]; cfg.output_dir.mkdir(parents=True,exist_ok=True); (cfg.output_dir/"continual_metrics.json").write_text(json.dumps(results,indent=2)+"\n")
    for r in results:
        before, after = r["june_before_july"], r["june_after_july"]
        print(f"{r['bank']}: June PR-AUC {before['pr_auc']:.5f} → {after['pr_auc']:.5f}; "
              f"precision {before['precision']:.5f} → {after['precision']:.5f}; "
              f"recall {before['recall']:.5f} → {after['recall']:.5f}; forgetting={r['forgetting']:.5f}")
if __name__=="__main__": main()
