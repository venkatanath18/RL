# Phase 3: Learning with a safety budget

Phase 2 learned a replacement policy from sampled transitions. Phase 3 extends
that learner's state from machine condition `s` to `(s, z)`, where `z` is the
remaining safety budget. It compares ordinary Q-learning with a penalty-based,
budget-augmented learner on the same economic model.

## File

`constrained_q_learning.py` organizes the original `constraint.py` experiment.
It preserves its transition law, costs, exploration schedule and Q updates,
adds a reproducible seed and command-line training limits, fixes the invalid
budget-3 debug lookup, and reports all replacement intervals. The original
file remains available in the parent research workspace.

## Model

- Conditions are numbered 1–50; 0 is an unused table entry.
- Action `0` = REPLACE; action `1` = CONTINUE. Smaller Q-values are preferred.
- Operating cost increases linearly from 990 to 9,990. Replacement costs
  9,000 + 990 and samples the next condition using the state-1 transition law.
- The economic discount factor is 0.7.
- Continuing in conditions 35–50 spends one safety unit; other actions spend zero.
- The budget starts at 2 and replacement restores it to 2.
- An attempted overspend is counted as a violation, receives cost 50,000
  instead of the operating cost, and stores the next budget as zero.

The economic costs are ten times Phase 2's refined configuration. This common
scaling preserves the unconstrained optimal policy, but scales its Q-values.
Phase 3 also changes the exploration floor from 0.01 to 0.1 and increases the
training length.

Both learners use cost-minimizing Q-learning:

```text
Q(x,a) <- Q(x,a) + alpha * [cost + 0.7 * min Q(x_next,.) - Q(x,a)]
alpha = visits(x,a)^(-0.6)
```

For the baseline, `x=s`; for the augmented learner, `x=(s,z)`. Augmented episodes
start at random valid conditions and budgets for simulator coverage. They do
not represent physically starting every cycle with a new machine.

## Run

From the `RL` directory:

```bash
python3 phase3/constrained_q_learning.py
```

The default runs 100,000 episodes of 500 steps **for each learner**, totaling
100 million sampled transitions. A quick execution check is:

```bash
python3 phase3/constrained_q_learning.py --episodes 20 --steps-per-episode 50
```

Use `--seed` to change the default seed of 42. Short runs check execution only;
their policies are not convergence results. Output includes inclusive replacement
intervals, first replacement states, budget-dependent actions and training
violation counts.

## Interpretation and benchmark

This is a Sauté-inspired, per-cycle resource model with finite penalties and
budget clipping. It is not a literal implementation of the paper's cumulative
safety-state dynamics. Both actions remain available during exploration and
in the Bellman minimum, including actions that overspend. Consequently this
learner does **not** guarantee safe training. Clipping the budget does not erase
a violation, and zero observed rollout violations would not prove safety.

The existing deterministic review in `review/RESEARCH_REWIND.md` (in the parent
workspace) reports these exact-model replacement regions:

| Model / remaining budget | Replacement conditions |
|---|---|
| Unconstrained | 47–50 |
| Augmented, z=2 | 46–50 |
| Augmented, z=1 | 45–50 |
| Augmented, z=0 | 35–50 |

These are value-iteration benchmarks, not claimed outputs of a new training
run. Sampled learning can differ, especially near action ties. Full intervals
are printed because a learned policy need not have a single threshold.
