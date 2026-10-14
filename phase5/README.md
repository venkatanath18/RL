# Persistent-budget replacement MDP

This experiment compares four finite discounted MDPs using the same physical
states, transition matrix and economic costs.

## Model

\[
S=\{1,\ldots,70\},\qquad A=\{R,C\},\qquad
Z=\{0,\ldots,12\},\qquad \gamma=0.7.
\]

`R` means **REPLACE** and `C` means **CONTINUE**. For physical state `s`,

\[
c(s,C)=990+\frac{9000(s-1)}{69},\qquad c(s,R)=9000+c(1,C)=9990.
\]

The transition probabilities are

\[
p(j\mid i)=\frac{70+i-j}{70^2+70i-70(71)/2}.
\]

Continuation uses the constraint cost `l(s,C)`:

\[
z'=\min\{12,z-l(s,C)\}.
\]

`C` is feasible only when `z-l(s,C) >= 0`. Thus a positive constraint cost
reduces `z`, a negative constraint cost increases `z`, and a zero constraint
cost leaves `z` unchanged. Replacement preserves the current value:

\[
z'=z \quad\text{when }a=R.
\]

The four cases are:

1. `baseline`: no constraint state.
2. `depleting`: `l(s,C)=1` for `s>=35`, otherwise `0`.
3. `replenishing`: `l=-1` on states 1–20, `l=0` on 21–34, and `l=1` on 35–70.
4. `replenishing_plus`: `l=-2` on 1–10, `l=-1` on 11–20,
   `l=0` on 21–34, `l=1` on 35–42, and `l=-1` on 43–70.

## Run value iteration

```bash
cd phase5
python3 value_iteration.py
```

The console prints every physical state and every budget value using `C` and
`R`. In the plus model at `z=0`, the exact policy has two replacement regions:
states 35–42 and states 67–70.

## Run Q-learning

```bash
python3 q_learning.py --seeds 42 43 44
```

Q-learning samples every feasible `(s,z,a)` pair. The step size is

\[
\alpha_n(s,z,a)=N_n(s,z,a)^{-0.8}.
\]

The exponent satisfies the Robbins–Monro conditions. Training stops after
three consecutive checkpoints satisfy all of these finite numerical criteria:

- every feasible pair has at least 200 visits;
- policy agreement with value iteration is at least 98%;
- the maximum relative policy-value gap is at most 2%;
- at most 2% of policy entries change from the preceding checkpoint.

These criteria determine the number of episodes; `100000` is only an upper
limit.

## Generate the combined report

```bash
python3 compare_results.py
```

The numerical tables, JSON files, NumPy arrays and figures are written to
`results_n70_b12/`.
