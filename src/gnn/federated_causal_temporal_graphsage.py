#!/usr/bin/env python3
"""FedAvg simulation for five local Causal Temporal GraphSAGE clients.

Only model state dictionaries are averaged. Transaction streams, labels, account
memories, and client-specific evaluation state remain local to each simulated bank.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import torch
from causal_temporal_graphsage import (BANKS, DEFAULT_DATASET, DEFAULT_OUTPUT, CausalTemporalGraphSAGE, batches,
    build_bank_data, choose_threshold, fit_shared_feature_encoders, metric_block, sampled_loss, score_stream, set_seed)

def parse():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--rounds",type=int,default=3); p.add_argument("--local-epochs",type=int,default=1)
 p.add_argument("--dataset-dir",type=Path,default=DEFAULT_DATASET); p.add_argument("--hidden-channels",type=int,default=64); p.add_argument("--batch-size",type=int,default=512)
 p.add_argument("--negative-ratio",type=int,default=20); p.add_argument("--learning-rate",type=float,default=1e-3); p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="cpu")
 p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT.parent/"federated_causal_temporal_graphsage"); return p.parse_args()
def local_train(model,static,events,cfg,seed):
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=1e-4); gen=torch.Generator(device=static.device).manual_seed(seed)
 for _ in range(cfg.local_epochs):
  model.train(); state=model.initial_state(static)
  for b in batches(events,cfg.batch_size):
   opt.zero_grad(); logits,state=model.score_and_update(state,b); loss=sampled_loss(logits,b.labels,cfg.negative_ratio,1.,gen)
   if loss is not None: loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
   state=state.detached()
def fedavg(states,weights):
 total=float(sum(weights)); return {key:sum(s[key].float()*(w/total) for s,w in zip(states,weights)) for key in states[0]}
def main():
 cfg=parse(); cfg.dataset_dir=cfg.dataset_dir.resolve(); device=torch.device(cfg.device); set_seed(cfg.seed)
 node_enc,edge_enc=fit_shared_feature_encoders(cfg.dataset_dir); clients={}
 for bank in BANKS:
  static,streams,_,_=build_bank_data(cfg.dataset_dir,bank,node_enc,edge_enc); clients[bank]=(static.to(device),{k:v.to(device) for k,v in streams.items()})
 first=next(iter(clients.values())); global_model=CausalTemporalGraphSAGE(first[0].shape[1],first[1]['training'].edge_attr.shape[1],cfg.hidden_channels,.25).to(device)
 history=[]
 for rnd in range(1,cfg.rounds+1):
  states=[]; weights=[]
  for i,(bank,(static,streams)) in enumerate(clients.items()):
   local=copy.deepcopy(global_model); local_train(local,static,streams['training'],cfg,cfg.seed+rnd*100+i); states.append({k:v.detach().cpu() for k,v in local.state_dict().items()}); weights.append(len(streams['training']))
  global_model.load_state_dict(fedavg(states,weights)); history.append({"round":rnd,"client_weights":dict(zip(BANKS,weights))}); print(f"Completed FedAvg round {rnd}/{cfg.rounds}")
 results=[]
 for bank,(static,streams) in clients.items():
  state=global_model.initial_state(static); _,_,state=score_stream(global_model,state,streams['training'],cfg.batch_size); vp,vy,state=score_stream(global_model,state,streams['validation'],cfg.batch_size); tp,ty,_=score_stream(global_model,state,streams['testing'],cfg.batch_size); threshold=choose_threshold(vy,vp)
  results.append({"bank":bank,"threshold":threshold,"validation":metric_block(vy,vp,threshold),"testing":metric_block(ty,tp,threshold)})
 cfg.output_dir.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":global_model.state_dict(),"args":vars(cfg)},cfg.output_dir/"global_model.pt"); (cfg.output_dir/"metrics.json").write_text(json.dumps({"rounds":history,"per_bank":results},indent=2)+"\n")
 for r in results: print(f"{r['bank']}: test PR-AUC={r['testing']['pr_auc']:.5f}")
if __name__=='__main__': main()
