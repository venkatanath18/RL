"""Phase 4: signed safety costs and capped budget replenishment.

Reproduces the models in positive_cost.py and positive_costs_plus.py.
Only Python's standard library is required. No safe-action mask is used.
"""

import argparse
import json
import random
from pathlib import Path


class Model:
    def __init__(self, variant):
        if variant not in ('basic', 'plus'):
            raise ValueError('variant must be basic or plus')
        self.variant = variant
        self.n, self.budget = (50, 2) if variant == 'basic' else (70, 5)
        self.gamma = 0.7
        self.penalty = 50000.0
        self.replacement_cost = 9000.0
        self.states = list(range(1, self.n + 1))
        self.costs = [0.0] + [990.0 + (s - 1) * 9000.0 / (self.n - 1)
                              for s in self.states]
        self.weights = {
            s: [(self.n + s - j) /
                (self.n**2 + self.n*s - self.n*(self.n + 1)/2)
                for j in self.states]
            for s in self.states
        }

    def safety_cost(self, state, action):
        if action == 0:
            return 0
        if self.variant == 'basic':
            return -1 if state <= 20 else (0 if state <= 34 else 1)
        if state <= 10:
            return -2
        if state <= 20:
            return -1
        if state <= 34:
            return 0
        return 1 if state <= 42 else -1

    def step(self, state, budget, action, rng, augmented=True):
        if state not in self.states or not 0 <= budget <= self.budget:
            raise ValueError('state or budget outside the model domain')
        if action not in (0, 1):
            raise ValueError('action must be 0 or 1')
        source = 1 if action == 0 else state
        next_state = rng.choices(self.states, weights=self.weights[source])[0]
        cost = (self.replacement_cost + self.costs[1]
                if action == 0 else self.costs[state])
        if not augmented:
            return next_state, 0, cost, False
        if action == 0:
            return next_state, self.budget, cost, False
        raw_budget = budget - self.safety_cost(state, action)
        violated = raw_budget < 0
        next_budget = min(self.budget, max(0, raw_budget))
        return next_state, next_budget, self.penalty if violated else cost, violated


def choose_action(q, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice([0, 1])
    if q[0] == q[1]:
        return rng.choice([0, 1])
    return 0 if q[0] < q[1] else 1


def train(model, episodes, steps, seed, augmented):
    rng = random.Random(seed)
    budgets = model.budget + 1 if augmented else 1
    q = [[[0.0, 0.0] for _ in range(budgets)] for _ in range(model.n + 1)]
    visits = [[[0, 0] for _ in range(budgets)] for _ in range(model.n + 1)]
    epsilon, violations = 0.5, 0
    for _ in range(episodes):
        state = rng.randint(1, model.n) if augmented else 1
        budget = rng.randint(0, model.budget) if augmented else 0
        for _ in range(steps):
            action = choose_action(q[state][budget], epsilon, rng)
            ns, nz, cost, violated = model.step(state, budget, action, rng, augmented)
            visits[state][budget][action] += 1
            alpha = visits[state][budget][action] ** -0.6
            target = cost + model.gamma * min(q[ns][nz])
            q[state][budget][action] += alpha * (target - q[state][budget][action])
            violations += int(violated)
            state, budget = ns, nz
        epsilon = max(0.1, epsilon * 0.9995)
    return q, visits, violations


def replacement_regions(policy):
    regions, start = [], None
    for state, action in enumerate([*policy, 1], start=1):
        if action == 0 and start is None:
            start = state
        elif action != 0 and start is not None:
            regions.append([start, state - 1])
            start = None
    return regions


def policy_at(q, budget):
    # Preserve the original scripts' continuation tie-break at extraction.
    return [0 if row[budget][0] < row[budget][1] else 1 for row in q[1:]]


def positive_integer(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['basic', 'plus'], default='basic')
    parser.add_argument('--episodes', type=positive_integer,
                        help='Default: basic=100000; plus=200000')
    parser.add_argument('--steps-per-episode', type=positive_integer, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, help='Optional JSON report including Q tables')
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = 100000 if args.variant == 'basic' else 200000
    model = Model(args.variant)
    print(f'Variant={args.variant}, N={model.n}, d={model.budget}', flush=True)
    print('Training unconstrained learner...', flush=True)
    baseline, base_visits, _ = train(model, args.episodes, args.steps_per_episode,
                                    args.seed, False)
    print('Training replenishing-budget learner...', flush=True)
    q, visits, violations = train(model, args.episodes, args.steps_per_episode,
                                  args.seed, True)
    base_policy = policy_at(baseline, 0)
    policies = {str(z): policy_at(q, z) for z in range(model.budget + 1)}
    regions = {z: replacement_regions(p) for z, p in policies.items()}
    unsafe = [[s, z] for z in range(model.budget + 1) for s in model.states
              if policies[str(z)][s-1] == 1 and model.safety_cost(s, 1) > z]
    print('Actions: 0=REPLACE; 1=CONTINUE. State indices start at 1.')
    print('Baseline replacement intervals:', replacement_regions(base_policy))
    for z in range(model.budget, -1, -1):
        print(f'z={z}: replacement intervals={regions[str(z)]}')
    print('Exploratory training violations:', violations)
    print('Unaffordable greedy CONTINUE decisions:', len(unsafe))
    print('state | l(s,1) | baseline | ' + ' | '.join(f'z={z}' for z in policies))
    for s in model.states:
        print(f'{s:5} | {model.safety_cost(s, 1):6} | {base_policy[s-1]:8} | '
              + ' | '.join(str(p[s-1]) for p in policies.values()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(variant=args.variant, states=model.n, maximum_budget=model.budget,
                      episodes=args.episodes, steps_per_episode=args.steps_per_episode,
                      seed=args.seed, baseline_policy=base_policy, policies=policies,
                      replacement_regions=regions, training_violations=violations,
                      unsafe_greedy_states=unsafe, baseline_q=baseline, augmented_q=q,
                      baseline_visits=base_visits, augmented_visits=visits)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
