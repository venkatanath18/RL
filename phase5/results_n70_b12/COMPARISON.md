# Persistent-budget comparison

All cases share the physical transition law, economic costs and state count. Replacement preserves z; maximum budget is 12.

## Value iteration

| Model | V(1,full budget) | Iterations | Bellman residual |
|---|---:|---:|---:|
| baseline | 11181.503278161 | 83 | 6.37e-10 |
| depleting | 11181.568108365 | 84 | 5.13e-10 |
| replenishing | 11181.505462497 | 83 | 6.44e-10 |
| replenishing_plus | 11181.503278161 | 83 | 6.42e-10 |

The baseline has no budget constraint. Its value is repeated across budget slices for comparison.

### All replacement intervals

| z | baseline | depleting | replenishing | replenishing_plus |
|---:|---|---|---|---|
| 0 | [[65, 70]] | [[35, 70]] | [[35, 70]] | [[35, 42], [67, 70]] |
| 1 | [[65, 70]] | [[57, 70]] | [[59, 70]] | [[65, 70]] |
| 2 | [[65, 70]] | [[62, 70]] | [[63, 70]] | [[65, 70]] |
| 3 | [[65, 70]] | [[64, 70]] | [[64, 70]] | [[65, 70]] |
| 4 | [[65, 70]] | [[64, 70]] | [[65, 70]] | [[65, 70]] |
| 5 | [[65, 70]] | [[65, 70]] | [[65, 70]] | [[65, 70]] |
| 6 | [[65, 70]] | [[65, 70]] | [[65, 70]] | [[65, 70]] |
| 7 | [[65, 70]] | [[65, 70]] | [[65, 70]] | [[65, 70]] |
| 8 | [[65, 70]] | [[65, 70]] | [[65, 70]] | [[65, 70]] |
| 9 | [[65, 70]] | [[65, 70]] | [[65, 70]] | [[65, 70]] |
| 10 | [[65, 70]] | [[65, 70]] | [[65, 70]] | [[65, 70]] |
| 11 | [[65, 70]] | [[65, 70]] | [[65, 70]] | [[65, 70]] |
| 12 | [[65, 70]] | [[65, 70]] | [[65, 70]] | [[65, 70]] |

## Q-learning comparisons

| Model | Seed | Status | Episodes | Agreement | Max Q error | Worst policy value gap | Initial value gap |
|---|---:|---|---:|---:|---:|---:|---:|
| baseline | 42 | tolerances_met | 8000 | 100.00% | 0.619% | 0.000000 | 0.000000000 |
| baseline | 43 | tolerances_met | 8000 | 100.00% | 0.627% | 0.000000 | 0.000000000 |
| baseline | 44 | tolerances_met | 8000 | 100.00% | 0.440% | 0.000000 | 0.000000000 |
| depleting | 42 | tolerances_met | 10000 | 98.13% | 2.968% | 213.569164 | 2.371883168 |
| depleting | 43 | tolerances_met | 12000 | 98.90% | 2.742% | 207.405824 | 1.221461678 |
| depleting | 44 | tolerances_met | 12000 | 99.34% | 2.433% | 134.298554 | 0.124905697 |
| replenishing_plus | 42 | tolerances_met | 14000 | 98.68% | 2.626% | 196.459132 | 2.572077599 |
| replenishing_plus | 43 | tolerances_met | 12000 | 98.90% | 2.756% | 195.290098 | 0.447256808 |
| replenishing_plus | 44 | tolerances_met | 16000 | 99.34% | 2.448% | 192.443750 | 0.000001154 |
| replenishing | 42 | tolerances_met | 12000 | 98.90% | 2.659% | 212.717771 | 1.600553649 |
| replenishing | 43 | tolerances_met | 12000 | 98.68% | 2.821% | 240.229978 | 1.112062977 |
| replenishing | 44 | tolerances_met | 12000 | 99.34% | 2.575% | 252.226825 | 0.083353254 |

Stopping means the declared finite tolerances passed for three consecutive checkpoints; it is not a proof of asymptotic convergence. Every feasible action requires minimum visitation. The worst policy gap covers every state-budget pair, not just the initial state.

## Interpretation

- At discount 0.7, distant costs receive little weight. A full budget of 12 makes initial discounted costs nearly identical even when low-budget policies differ.
- In the depletion-only model, the budget can never recover: replacement no longer replenishes it. At z=0, risky continuation is infeasible indefinitely, but replacement remains available.
- Signed-cost models can replenish through continuation in designated states, capped at 12.
- The plus cost pattern and all comparison models use N=70.
- Zero violations are enforced by the known-cost feasible-action rule. They do not demonstrate that Q-learning independently discovered safe exploration.

The trajectory figure shows individual simulated paths, not estimated average cycle lengths.

## Artifacts

- `policies.png`: exact policies for all budgets.
- `initial_costs.png`: economic costs as initial budget varies.
- `learning.png`: sampled-learning diagnostics across seeds.
- `budget_trajectories.png`: 500-decision budget examples.
- `value_iteration.json`, `*_vi.npz`: exact reports and arrays.
- `*_q_seed*.json`, `*_q_seed*.npz`: convergence histories, policies, Q tables and visits.
