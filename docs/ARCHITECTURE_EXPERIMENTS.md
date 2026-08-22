# Architecture experiments for end-to-end PPO scheduling

## Current reference

The current policy is a permutation-equivariant shared MLP:

1. encode every UE with the same MLP;
2. aggregate the candidate set with mean/max pooling;
3. score every UE using its local embedding and global context;
4. use the action projector only for a valid Top-64, exact-273-PRB grant.

This is the correct reference because UEs form an unordered set rather than a natural
spatial grid.

## Recurrent PPO (recommended first)

A useful recurrent design is **shared per-UE GRU + set pooling**, not one LSTM over a
flattened 1,200-UE vector.

- each UE keeps a small recurrent hidden state;
- the same GRU parameters are shared across all UEs;
- hidden states are indexed by persistent UE ID;
- current per-UE hidden states are pooled for the critic and global actor context;
- PPO uses truncated backpropagation through time.

Expected benefit: better representation of service cycles, repeated HARQ outcomes,
and hidden temporal trends. It is most likely to improve tail wait and reduce the
number of training steps required to discover non-starving behavior.

Risk: the existing observation already contains EWMA throughput, successful-delivery
wait, and HARQ history. Recurrence may be redundant unless it improves validation
results under the same parameter and training budget.

## One-dimensional CNN (experimental second)

A CNN across the raw UE-ID axis is not valid because neighboring UE IDs are not
physically or statistically adjacent. Convolution becomes meaningful only after a
stable deterministic ordering, for example by:

- successful-delivery wait;
- throughput deficit;
- candidate urgency score;
- a lexicographic combination of HARQ, wait, deficit, and CQI.

A sorted 1-D CNN can then learn local patterns between neighboring ranks at O(Nk)
cost. However, sorting changes identity positions between slots and can hide
permutation errors. The experiment must compare against the set-MLP with identical
candidate ordering, parameter count, seeds, and environment steps.

## Recommended order

1. Finish the fixed-weight PPO-only + small-guard curve at reserves 0, 4, 8, 12, 16.
2. Fine-tune at most two non-dominated guard sizes from the same PPO-only checkpoint.
3. Train a recurrent-set PPO from random initialization.
4. Train a sorted-CNN PPO from random initialization.
5. Consider CNN + GRU only after both individual ablations show independent benefit.

## Fair comparison contract

Every architecture should use:

- the same 128-candidate filter first;
- the same random initialization seeds;
- 300,032 environment steps;
- the same PPO hyperparameters unless separately ablated;
- the same unified evaluation protocol and five evaluation seeds;
- matched parameter counts where practical;
- goodput, Jain fairness, starvation, P99/max wait, inference latency, balanced score,
  and worst KPI gap.

Only after an architecture is stable with 128 candidates should it be exposed to all
1,200 UEs. A full-UE failure otherwise cannot be separated from a reward-design or
exploration failure.

## Pure-RL roadmap after Round 20

The project should prefer policies learned from random initialization over PF/oracle
behavior cloning when making the main research claim. Expert warm-start remains a
sanity/control experiment because it can place PPO directly inside a scheduler-like
basin and make the apparent RL improvement difficult to attribute to PPO itself.

Before changing to multi-agent control, the current priority is:

1. Equal4 reward with service-aware retention and train/deploy alignment;
2. expose scheduler-owned per-UE scheduling history (`time_since_schedule`, recent
   schedule-rate deficit and rank) so schedule-frequency reward is Markov/locally
   observable to the actor;
3. if necessary, split priority and PRB-demand representation paths while keeping
   PPO end-to-end and teacher-free;
4. test curriculum/mixed UE counts from random initialization.

### Future hierarchical MAPPO design

If centralized full-UE PPO remains fragile after the pure-PPO studies, the planned
next architecture is hierarchical MAPPO rather than 1,200 independent agents:

```text
1200 UEs
   -> 12 x ~100 UE groups   (also screen 16 x ~75)
   -> global coordinator allocates per-group service/PRB quotas
   -> parameter-shared local actors rank/select UEs inside each group
   -> centralized critic observes the whole cell during training
```

The grouping must not be a fixed partition by UE ID. Candidate grouping signals
include QoS/slice, queue load, service wait, reported CQI, and/or position. Group
membership or quota changes require hysteresis/stability so slot-to-slot cluster
churn does not create another non-stationary learning problem. Coordinator quotas
must conserve total Top-K service opportunities and PRBs so lightly loaded groups
cannot strand resources while another group starves.

This remains PPO-family RL: the coordinator and local shared actors are learned,
not replaced by PF/oracle scheduling rules. It is intentionally deferred until the
single-agent PPO baseline has been pushed as far as practical so the MAPPO gain can
be attributed to hierarchy/action decomposition rather than an unfinished baseline.
