"""Experimental bank/task tail-ranking guard; not a novelty or privacy guarantee.

Clients reconstruct causal memory for each candidate using training events only.
The server needs one scalar risk per bank/task, not event-level predictions.
This offline implementation requires completed training history to remain available.
"""
from __future__ import annotations

import copy
import numpy as np
import torch

from causal_temporal_graphsage import batches


def tail_ranking_risk(logits, labels, negative_ratio=20):
    """Mean softplus(negative - positive) against the hardest training negatives."""
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    if logits.ndim != 1 or logits.shape != labels.shape or not np.isfinite(logits).all():
        raise ValueError("Expected finite aligned one-dimensional logits and labels")
    if negative_ratio < 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected binary labels and a positive negative ratio")
    positives, negatives = logits[labels == 1], logits[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    count = min(len(negatives), negative_ratio * len(positives))
    hardest = np.partition(negatives, len(negatives) - count)[-count:]
    # Chunk positives to bound temporary pairwise storage.
    total = sum(np.logaddexp(0, hardest[None, :] - chunk[:, None]).sum()
                for chunk in np.array_split(positives, max(1, (len(positives) + 63) // 64)))
    return float(total / (len(positives) * count))


@torch.no_grad()
def training_task_risks(model, clients, batch_size):
    """Each candidate rebuilds its own memory; July inherits its June memory."""
    model.eval()
    result = {}
    for bank, (static, streams, boundary) in clients.items():
        events = streams["training"]  # Never access validation or testing.
        state = model.initial_state(static)
        values = []
        for batch in batches(events, batch_size):
            logits, state = model.score_and_update(state, batch)
            values.append(logits.cpu().numpy())
        logits = np.concatenate(values)
        labels = events.labels.cpu().numpy()
        june = events.timestamp.cpu().numpy() < boundary
        for task, mask in (("june", june), ("july", ~june)):
            risk = tail_ranking_risk(logits[mask], labels[mask])
            if risk is not None:
                result[f"{bank}/{task}"] = risk
    return result


def select_guarded_step(old, proposed, evaluate, tolerance=0.05, steps=(1.0, 0.5, 0.25)):
    """Largest step with <= tolerance relative risk increase in EVERY valid task.

    The zero step is an exact fallback. This is an empirical training-risk check,
    not a bound on future AP or on unseen data. No gradients are transmitted.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    if not steps or any(not 0 < a <= 1 for a in steps) or list(steps) != sorted(set(steps), reverse=True):
        raise ValueError("steps must be unique, descending, and in (0, 1]")
    reference = evaluate(old)
    if not reference or any(not np.isfinite(v) or v < 0 for v in reference.values()):
        raise ValueError("Guard needs finite, nonnegative training risks")
    trials = []
    for alpha in steps:
        candidate = {key: old[key] + alpha * (proposed[key] - old[key]) for key in old}
        risks = evaluate(candidate)
        if risks.keys() != reference.keys():
            raise ValueError("Guard task support changed across candidates")
        increases = [(risks[key] - value) / max(value, 1e-8) for key, value in reference.items()]
        finite = all(np.isfinite(v) for v in risks.values())
        accepted = finite and max(increases) <= tolerance
        trials.append({"step": alpha, "risks": risks, "accepted": accepted})
        if accepted:
            return candidate, {"accepted_step": alpha, "reference_risks": reference, "trials": trials}
    return {key: value.clone() for key, value in old.items()}, {
        "accepted_step": 0.0, "reference_risks": reference, "trials": trials}


def guarded_aggregation(model, clients, old, proposed, batch_size, tolerance):
    probe = copy.deepcopy(model)
    def evaluate(state):
        probe.load_state_dict(state, strict=True)
        return training_task_risks(probe, clients, batch_size)
    return select_guarded_step(old, proposed, evaluate, tolerance)
