# Phase 4: Q-learning with budget replenishment

Phase 3 introduced a remaining safety budget into the state. Its continuation
actions could consume budget or leave it unchanged. Phase 4 also allows
negative safety costs: these replenish the budget, up to its capacity.

This folder organizes two existing experiments into one standalone program:

| Command-line variant | Original workspace file | States N | Capacity d |
|---|---|---:|---:|
| `basic` | `positive_cost.py` | 50 | 2 |
| `plus` | `positive_costs_plus.py` | 70 | 5 |

The original names are historical. Both models use **signed** safety costs,
not exclusively positive safety costs. The negative values change the
mathematical constraint and must be stated explicitly.

## 1. Physical MDP and economic objective

Following Sauté's notation, let

$$\mathcal M=\langle\mathcal S,\mathcal A,P,c,\gamma_c\rangle,
\quad \mathcal S=\{1,\ldots,N\},\quad\mathcal A=\{0,1\}.$$

Action 0 is REPLACE; action 1 is CONTINUE. Index 0 in the implementation's
tables is unused. The operating cost is

$$h(s)=990+\frac{s-1}{N-1}(9990-990).$$

With replacement charge $K=9000$,

$$c(s,0)=K+h(1)=9990,\qquad c(s,1)=h(s),\qquad\gamma_c=0.7.$$

Define

$$p_N(j\mid i)=\frac{N+i-j}{N^2+Ni-N(N+1)/2}.$$

The physical transition kernel is

$$P(j\mid s,0)=p_N(j\mid1),\qquad P(j\mid s,1)=p_N(j\mid s).$$

Replacement samples from the state-1 transition distribution; it does not
force the next observed condition to equal 1. Every next condition has positive
probability, so the model permits jumps to both better and worse conditions.

The unconstrained comparator minimizes

$$J_{\rm task}(\pi)=\mathbb E^\pi\left[\sum_{t=0}^{\infty}
\gamma_c^t c(s_t,a_t)\right].$$

## 2. Augmented state and signed safety cost

$$\widetilde{\mathcal S}=\mathcal S\times\mathcal Z,
\qquad\mathcal Z=\{0,\ldots,d\},\qquad x_t=(s_t,z_t).$$

Here $z_t$ is the currently available budget, not the total number of risky
operations since replacement. Replenishment can permit more risky operations
over a cycle than the initial capacity $d$.

Replacement has safety cost $l(s,0)=0$. Continuation costs are:

| State interval | Basic l(s,1) | Effect on budget |
|---|---:|---|
| 1–20 | -1 | Restore one unit, capped at 2 |
| 21–34 | 0 | No change |
| 35–50 | +1 | Spend one unit |

| State interval | Plus l(s,1) | Effect on budget |
|---|---:|---|
| 1–10 | -2 | Restore two units, capped at 5 |
| 11–20 | -1 | Restore one unit, capped at 5 |
| 21–34 | 0 | No change |
| 35–42 | +1 | Spend one unit |
| 43–70 | -1 | Restore one unit, capped at 5 |

The plus variant deliberately assigns replenishment to high deterioration
states. This is a synthetic modeling choice, not a validated physical claim.
Changing N also changes the operating-cost slope and the transition kernel;
differences between variants cannot be attributed solely to replenishment.

## 3. Implemented budget update and violation penalty

Define the raw continuation budget $r_t=z_t-l(s_t,1)$. The stored update is

$$\widehat F(s,z,0)=d,\qquad
\widehat F(s,z,1)=\min\{d,\max\{0,z-l(s,1)\}\}.$$

The violation indicator is evaluated **before** clipping:

$$v_t=\mathbf1\{a_t=1,\ z_t-l(s_t,1)<0\}.$$

With penalty $n=50000$, the cost observed by the augmented learner is

$$\widehat c_n(s,z,a)=
\begin{cases}n,&a=1\text{ and }l(s,1)>z,\\
c(s,a),&\text{otherwise}.\end{cases}$$

Thus the penalty replaces the operating cost on the violating step. The
physical transition still occurs, the next budget is stored as zero, and
learning continues. There is no terminal failure state or action mask.

The augmented transition kernel is

$$\widehat P((s',z')\mid(s,z),a)
=P(s'\mid s,a)\mathbf1\{z'=\widehat F(s,z,a)\}.$$

The implemented optimization surrogate minimizes the expected discounted sum
of $\widehat c_n$. Since stored budgets are clipped, checking their
nonnegativity alone cannot establish safety: the raw violation indicators matter.

## 4. Q-learning

For the baseline, $x=s$ and the observed cost is $c$. For the augmented learner,
$x=(s,z)$ and the observed cost is $\widehat c_n$. Both use

$$Q(x_t,a_t)\leftarrow Q(x_t,a_t)+\alpha_t
\left[c_t+\gamma_c\min_b Q(x_{t+1},b)-Q(x_t,a_t)\right],$$

$$\alpha_t=N_t(x_t,a_t)^{-0.6}.$$

The visit count includes the current update. Exploration is epsilon-greedy,
starting at 0.5 and updated after each episode by
$\epsilon\leftarrow\max(0.1,0.9995\epsilon)$.
Both actions remain available even when continuation overspends.

Baseline episodes begin at state 1. Augmented episodes begin at a uniformly
sampled valid state and budget, to improve simulator coverage. These random
starts are not claims about a physical machine's initial condition.

The two learners each use a fresh random generator with the same supplied
seed. Their trajectories need not match because actions and random-number
consumption differ. Greedy-policy extraction chooses CONTINUE on exact ties,
matching the original scripts.

## 5. Run

Requires Python 3 and its standard library only. From the repository root:

```bash
python3 phase4/replenishing_q_learning.py --variant basic
python3 phase4/replenishing_q_learning.py --variant plus
```

Defaults match the source files as inspected when Phase 4 was prepared:

| Variant | Episodes per learner | Steps per episode | Total transitions, both learners |
|---|---:|---:|---:|
| basic | 100,000 | 500 | 100,000,000 |
| plus | 200,000 | 500 | 200,000,000 |

For an execution check:

```bash
python3 phase4/replenishing_q_learning.py --variant basic --episodes 20 --steps-per-episode 50
python3 phase4/replenishing_q_learning.py --variant plus --episodes 20 --steps-per-episode 50
```

To save a reproducible run, including Q tables, visit counts and policies:

```bash
python3 phase4/replenishing_q_learning.py --variant basic --episodes 2000 --steps-per-episode 50 --seed 42 --output /tmp/phase4_basic.json
```

Short runs check execution, not convergence. The program reports all inclusive
replacement intervals, the full policy by budget, exploratory violation counts,
and the number of unaffordable actions in the extracted greedy policy.

## 6. Existing deterministic benchmarks

Earlier value-iteration review of these model definitions gave:

| Variant | Policy / budget | Replacement states |
|---|---|---|
| basic | Unconstrained | 47–50 |
| basic | z=2 | 47–50 |
| basic | z=1 | 46–50 |
| basic | z=0 | 35–50 |
| plus | Unconstrained | 65–70 |
| plus | z=1,2,3,4,5 | 65–70 |
| plus | z=0 | 35–42 and 65–70 |

These are historical deterministic benchmarks, not claimed outputs of the
short validation runs for this commit. Q-learning may produce different
finite-training policies. The review solver remains in the parent research
workspace as `review/verify_models.py`; it is not a dependency of this folder.

The plus model at z=0 illustrates why a first-replacement threshold alone
is insufficient: it replaces at 35–42, continues at 43–64, then replaces
again at 65–70. High-state replenishment explains the middle continuation band.

## 7. Relation to Sauté RL and limitations

Sauté defines nonnegative safety costs and the recursion

$$z_{t+1}=\frac{z_t-l(s_t,a_t)}{\gamma_l}.$$

These implementations instead use signed costs, capped replenishment,
replacement resets, and immediate overspending penalties with clipping.
They are Sauté-inspired resource models, not literal implementations of
the paper's dynamics or a direct application of its equivalence theorems.

Negative costs mean later replenishment can offset earlier expenditure.
Consequently a final total-cost bound and a requirement that the budget never
becomes negative are distinct. The capped budget has its own dynamics.

The exploration floor permits continued training violations. A large finite
penalty and a finite training run do not prove almost-sure safety or Q-learning
convergence. The reported violation count concerns training; the greedy-action
check concerns the extracted policy. These are different measurements.

Reference: A. Sootla et al., *Sauté RL: Almost Surely Safe Reinforcement Learning
Using State Augmentation*, ICML 2022,
[arXiv:2202.06558](https://arxiv.org/abs/2202.06558).

## 8. Organization changes from the original scripts

The economic models, signed costs, cap/reset rules, exploration schedule and
learning updates are preserved. The two duplicated implementations are
consolidated into a `Model` class and a shared trainer. Command-line parameters,
seeded random generators, input checks and optional JSON output are added.
Stale descriptions of the risky region and the plus script's broken bitwise
debug condition are omitted. Full replacement intervals replace reliance on
a first-crossing summary. The original workspace scripts are unchanged.
