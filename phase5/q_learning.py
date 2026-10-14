"""Sample-based Q-learning with monitored stopping, separate from VI.

Each update samples a state-budget-action pair from the finite augmented MDP.
This guarantees continuing exploration of every feasible pair. Exact P and VI
are used only at checkpoints to evaluate the learned policy; they never enter
the Q-learning update.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from model import CASES, Config, Model, solve, evaluate_policy, regions


def diagnostics(model, q, visits, exact_v, exact_q, exact_policy):
    q = np.asarray(q)
    visits = np.asarray(visits)
    feasible = model.feasible
    policy = q.argmin(axis=2)
    learned_v = q.min(axis=2)
    backed_up = model.q_from_value(learned_v)
    scale = float(np.max(np.abs(exact_q[feasible])))
    policy_v = evaluate_policy(model, policy)
    regret = np.take_along_axis(exact_q, policy[..., None], axis=2)[..., 0] - exact_q.min(axis=2)
    return dict(
        minimum_visits=int(visits[feasible].min()),
        relative_q_error=float(np.max(np.abs(q[feasible]-exact_q[feasible]))/scale),
        relative_bellman_residual=float(np.max(np.abs(q[feasible]-backed_up[feasible]))/scale),
        policy_agreement=float(np.mean(policy == exact_policy)),
        maximum_policy_value_gap=float(np.max(policy_v-exact_v)),
        relative_policy_value_gap=float(np.max(policy_v-exact_v)/np.max(exact_v)),
        initial_policy_value=float(policy_v[0, -1]),
        initial_value_gap=float(policy_v[0, -1]-exact_v[0, -1]),
        maximum_action_regret=float(np.max(regret)),
        unsafe_actions=int(np.sum(~np.take_along_axis(feasible, policy[...,None], axis=2))),
    )


def train(model, args, seed, exact):
    rng = random.Random(seed)
    n, layers = model.config.states, model.layers
    q = np.where(model.feasible, 0.0, np.inf).tolist()
    visits = np.zeros((n,layers,2), dtype=int).tolist()
    continue_feasible = model.feasible[:, :, 1].tolist()
    vstar, qstar, pstar, _ = exact
    history, streak, previous_policy = [], 0, None
    for episode in range(1, args.max_episodes+1):
        # Uniform sampling gives every feasible (s,z,a) pair infinite visits
        # as the number of episodes tends to infinity.
        for _ in range(args.steps):
            s, z = rng.randrange(n), rng.randrange(layers)
            action = rng.randrange(2) if continue_feasible[s][z] else 0
            row = q[s][z]
            ns, nz, cost = model.transition(s,z,action,rng.random())
            visits[s][z][action] += 1
            alpha = visits[s][z][action] ** -args.alpha_exponent
            row[action] += alpha * (cost + model.config.gamma*min(q[ns][nz])-row[action])
        if episode % args.check_every == 0 or episode == args.max_episodes:
            check = diagnostics(model,q,visits,vstar,qstar,pstar)
            policy = np.asarray(q).argmin(axis=2)
            check['policy_change_fraction'] = (1.0 if previous_policy is None else
                                               float(np.mean(policy != previous_policy)))
            previous_policy = policy.copy()
            passed = (check['minimum_visits'] >= args.min_visits
                      and check['relative_policy_value_gap'] <= args.policy_tolerance
                      and check['policy_agreement'] >= args.minimum_policy_agreement
                      and check['policy_change_fraction'] <= args.change_tolerance
                      and check['unsafe_actions'] == 0)
            streak = streak + 1 if passed else 0
            check.update(episode=episode, transitions=episode*args.steps,
                         passes_criteria=bool(passed), consecutive_passes=streak)
            history.append(check)
            print(f'{model.config.case} seed={seed} episode={episode} '
                  f'Qerr={check["relative_q_error"]:.4%} '
                  f'policy_gap={check["relative_policy_value_gap"]:.4%} '
                  f'agreement={check["policy_agreement"]:.2%} streak={streak}', flush=True)
            if streak >= args.patience:
                break
    status = 'tolerances_met' if streak >= args.patience else 'max_episodes_reached'
    policy = np.asarray(q).argmin(axis=2)
    report = dict(case=model.config.case, seed=seed, status=status,
                  episodes=episode, training_transitions=episode*args.steps,
                  training_violations=0, history=history,
                  replacement_regions={str(z): regions(policy[:,z].tolist()) for z in range(layers)})
    return np.asarray(q), np.asarray(visits), policy, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=int, default=70)
    parser.add_argument('--capacity', type=int, default=12)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--max-episodes', type=int, default=100000)
    parser.add_argument('--check-every', type=int, default=2000)
    parser.add_argument('--min-visits', type=int, default=200)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--alpha-exponent', type=float, default=0.8)
    parser.add_argument('--policy-tolerance', type=float, default=0.02)
    parser.add_argument('--minimum-policy-agreement', type=float, default=0.98)
    parser.add_argument('--change-tolerance', type=float, default=0.02)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent/'results_n70_b12')
    args = parser.parse_args()
    if min(args.steps,args.max_episodes,args.check_every,args.min_visits,args.patience)<=0:
        parser.error('Training counts must be positive')
    if not 0.5 < args.alpha_exponent <= 1:
        parser.error('alpha-exponent must lie in (0.5, 1]')
    if not all(0 <= x <= 1 for x in (args.policy_tolerance,
                                     args.minimum_policy_agreement,
                                     args.change_tolerance)):
        parser.error('policy tolerances and agreement must lie in [0, 1]')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for case in args.cases:
        model = Model(Config(case,args.states,args.capacity))
        exact = solve(model)
        for seed in args.seeds:
            q, visits, policy, report = train(model,args,seed,exact)
            report['configuration'] = vars(args) | {'output_dir':str(args.output_dir)}
            stem = f'{case}_q_seed{seed}'
            np.savez(args.output_dir/f'{stem}.npz',q=q,visits=visits,policy=policy)
            (args.output_dir/f'{stem}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
