# ScaleMAC-RL v0.6.2

This release shifts the single-seed experiment toward higher UE fairness and shorter worst-case service gaps while preserving goodput as the primary positive objective.

- changes the primary starvation definition to **no successful delivery** for at least 64 consecutive slots;
- logs scheduling starvation separately, so a failed HARQ transmission no longer counts as successful service;
- adds explicit `max_wait_slots`, scheduling wait, near-deadline rate, and short-term Jain fairness KPIs;
- changes positive reward weights to throughput `0.50`, fairness `0.35`, and service `0.15`;
- combines cumulative and EWMA Jain fairness for a more responsive training signal;
- adds a non-saturating maximum-wait risk penalty with a 60-slot target;
- constrains validation by starvation, P99 wait, minimum Jain fairness, and the single worst UE wait;
- adds fairness and maximum-wait Lagrange multipliers;
- fixes checkpoint provenance by saving the exact validated model before rollback;
- ranks `best_lowest_violation.pt` lexicographically by starvation, maximum wait, P99 wait, fairness, then reward;
- exports `checkpoint_manifest.csv` with the selected update and reason;
- keeps the default single-seed budget at 200,192 environment steps with tqdm monitoring;
- source archives remain source-only and never include `artifacts/`.
