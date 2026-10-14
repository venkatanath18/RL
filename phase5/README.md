# Phase 5

This phase compares four machine-replacement models:

1. baseline;
2. budget depletion;
3. signed constraint costs from `positive_cost.py`;
4. signed constraint costs from `positive_costs_plus.py`.

The common parameters are

\[
N=70,\qquad z\in\{0,1,\ldots,12\},\qquad \gamma=0.7.
\]

The actions are

- `R` = REPLACE;
- `C` = CONTINUE.

Continuation changes the budget according to

\[
z'=\min\{12,z-l(s,C)\}.
\]

`C` is unavailable when `z-l(s,C)<0`. Replacement does not change the
budget:

\[
z'=z \quad\text{for } R.
\]

## Value iteration

Run:

```bash
cd phase5
python3 value_iteration.py
```

The program prints one table for each model. Every policy entry is printed as
`C` or `R` for every state and budget value.

For the plus model at `z=0`, value iteration gives two replacement regions:

\[
35\text{--}42 \quad\text{and}\quad 67\text{--}70.
\]

## Q-learning

Run:

```bash
python3 q_learning.py
```

The Q-learning parameters are constants at the top of the file. Training is
checked every 2,000 episodes and stops after three consecutive checks satisfy:

- every feasible state-budget-action entry has at least 200 visits;
- agreement with the value-iteration policy is at least 98%;
- at most 2% of policy entries changed since the preceding check.

The final learned policy is printed using the same `C` and `R` table.
