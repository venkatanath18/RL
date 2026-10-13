# Phase 1: Exact machine-replacement benchmarks

Phase 1 solves the finite machine-replacement Markov decision process with
known transition probabilities. The condition is `s = 1, ..., 15`; action `0`
resets/replaces the machine and action `1` continues operating it. The objective
minimizes discounted operating and replacement cost with discount factor `0.7`.

The Python implementation in `first.py` computes the policy with policy
iteration and prints the value function. It also plots the value and policy
when a graphical backend is available. `first.cpp` is a compiled value/policy
iteration implementation of the same 15-state benchmark. `first1.cpp` is a
small numerical check of the transition-tail ordering for `N=10`.

## Model

The operating cost is linear from 99 at state 1 to 999 at state 15. Replacement
cost is 990. For a current state `i` and future state `j`, the transition weight
is

```text
p(j | i) = (N + i - j) / (N² + N i - N(N+1)/2)
```

The reset action uses the state-1 transition distribution in the Bellman
update, matching the convention in the C++ benchmark. The Python script's
policy-iteration path uses the same reset-state continuation convention.

## Run

From the `RL` directory:

```bash
python3 phase1/first.py
c++ -std=c++17 -O2 phase1/first.cpp -o /tmp/phase1_first
/tmp/phase1_first
c++ -std=c++17 -O2 phase1/first1.cpp -o /tmp/phase1_check
/tmp/phase1_check
```

The Python plot requires a graphical backend; use `MPLBACKEND=Agg` for a
headless run. The benchmark output is recorded in [RESULTS.md](RESULTS.md).

## Interpretation

Both main implementations produce the same threshold policy: continue in
states 1–13 and replace in states 14–15. Their values agree to the precision
printed by the C++ program. This is the exact-model baseline used to motivate
the sampled learning experiment in Phase 2.

`first1.cpp` is a floating-point diagnostic, not a proof. With its tolerance of
`1e-18`, it reports a difference of about `1.11e-16` caused by long-double
rounding in a comparison of quantities that are theoretically ordered/equal
at that precision. It exits with `done` when no numerical discrepancy exceeds
the chosen tolerance.
