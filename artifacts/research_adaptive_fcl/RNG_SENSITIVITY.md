# RNG sensitivity diagnostic

The initial screen uses per-client reseeding. Its seed-42 baseline is 0.4869 August macro PR-AUC, substantially below the historical validation result. These runs are retained as a sensitivity study, not the principal improvement comparison. The main study under `global_rng/` restores the original global RNG policy and has a unit test verifying exact client update equivalence for uniform replay. No September data was used to make this decision.
