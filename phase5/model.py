"""Shared physical model. Replacement preserves the remaining budget.

Array indices are zero-based; displayed physical states are 1..N.
The baseline has one dummy budget layer. Constrained models have layers 0..12.
"""
from dataclasses import dataclass, asdict
import numpy as np

CASES = ('baseline', 'depleting', 'replenishing', 'replenishing_plus')


@dataclass(frozen=True)
class Config:
    case: str
    states: int = 70
    capacity: int = 12
    gamma: float = 0.7
    replacement_charge: float = 9000.0


class Model:
    def __init__(self, config):
        if config.case not in CASES or config.states < 43 or config.capacity < 1:
            raise ValueError('Unsupported case, states < 43, or capacity < 1')
        if not 0 < config.gamma < 1:
            raise ValueError('gamma must lie in (0,1)')
        self.config = config
        n = config.states
        self.layers = 1 if config.case == 'baseline' else config.capacity + 1
        self.cost = np.linspace(990.0, 9990.0, n)
        s = np.arange(1, n + 1)
        self.p = (n + s[:, None] - s[None, :]) / (
            n*n + n*s[:, None] - n*(n+1)/2)
        self.cdf = np.cumsum(self.p, axis=1).tolist()
        for row in self.cdf:
            row[-1] = 1.0
        if config.case == 'baseline':
            self.safety = np.zeros(n, dtype=int)
        elif config.case == 'depleting':
            self.safety = (s >= 35).astype(int)
        elif config.case == 'replenishing':
            self.safety = np.where(s <= 20, -1, np.where(s <= 34, 0, 1))
        else:
            self.safety = np.select([s <= 10, s <= 20, s <= 34, s <= 42],
                                    [-2, -1, 0, 1], default=-1)
        z = np.arange(self.layers)
        raw = z[None, :] - self.safety[:, None]
        self.feasible = np.ones((n, self.layers, 2), dtype=bool)
        self.feasible[:, :, 1] = raw >= 0
        # Invalid actions are masked before this array can be used.
        self.next_z = np.clip(raw, 0, self.layers - 1)
        self.replace_cost = config.replacement_charge + self.cost[0]

    def q_from_value(self, value):
        """Bellman action costs; only this model-based method uses P explicitly."""
        expected = self.p @ value
        q = np.empty(value.shape + (2,))
        # Replacement draws s' ~ P(.|1) and retains current z.
        q[:, :, 0] = self.replace_cost + self.config.gamma * expected[0, :]
        q[:, :, 1] = self.cost[:, None] + self.config.gamma * np.take_along_axis(
            expected, self.next_z, axis=1)
        q[~self.feasible] = np.inf
        return q

    def transition(self, s, z, action, uniform):
        """One sampled observation; raises before any unaffordable action."""
        from bisect import bisect_right
        if not (0 <= s < self.config.states and 0 <= z < self.layers
                and action in (0, 1) and 0 <= uniform < 1):
            raise ValueError('Invalid state, budget, action, or random draw')
        if not self.feasible[s, z, action]:
            raise ValueError('CONTINUE would make the budget negative')
        source = 0 if action == 0 else s
        ns = bisect_right(self.cdf[source], uniform)
        nz = z if action == 0 else int(self.next_z[s, z])
        cost = self.replace_cost if action == 0 else float(self.cost[s])
        return ns, nz, cost


def regions(actions):
    result, start = [], None
    for s, a in enumerate([*actions, 1], start=1):
        if a == 0 and start is None:
            start = s
        elif a != 0 and start is not None:
            result.append([start, s-1])
            start = None
    return result


def evaluate_policy(model, policy, tolerance=1e-9):
    """Discounted value of a fixed policy, with a residual check."""
    if not np.take_along_axis(model.feasible, policy[..., None], axis=2).all():
        raise ValueError('Infeasible policy')
    v = np.zeros(policy.shape)
    for _ in range(10000):
        q = model.q_from_value(v)
        new = np.take_along_axis(q, policy[..., None], axis=2)[..., 0]
        if np.max(np.abs(new-v)) < tolerance:
            return new
        v = new
    raise RuntimeError('Policy evaluation failed to converge')


def solve(model, tolerance=1e-9):
    v = np.zeros((model.config.states, model.layers))
    for iteration in range(1, 10001):
        new = model.q_from_value(v).min(axis=2)
        v, delta = new, float(np.max(np.abs(new-v)))
        if delta < tolerance:
            break
    else:
        raise RuntimeError('VI did not converge')
    q = model.q_from_value(v)
    policy = q.argmin(axis=2)
    residual = float(np.max(np.abs(q.min(axis=2)-v)))
    vp = evaluate_policy(model, policy)
    report = dict(config=asdict(model.config), iterations=iteration,
                  bellman_residual=residual,
                  value_error_bound=residual/(1-model.config.gamma),
                  policy_evaluation_error=float(np.max(np.abs(vp-v))),
                  initial_state=1, initial_budget=0 if model.layers == 1 else model.config.capacity,
                  initial_value=float(v[0, -1]),
                  replacement_regions={str(z): regions(policy[:, z].tolist())
                                       for z in range(model.layers)},
                  unsafe_actions=int(np.sum(~np.take_along_axis(
                      model.feasible, policy[..., None], axis=2))),
                  value_budget_monotonicity_violations=int(np.sum(np.diff(v, axis=1)>1e-7)),
                  continuation_nesting_violations=int(np.sum(
                      (policy[:, :-1] == 1) & (policy[:, 1:] == 0))))
    return v, q, policy, report
