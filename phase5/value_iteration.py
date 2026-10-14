"""Solve all four models and print their policies as C/R tables."""
import argparse
import json
from pathlib import Path
import numpy as np
from model import CASES, Config, Model, solve


NAMES = {
    'baseline': 'BASELINE: NO BUDGET CONSTRAINT',
    'depleting': 'BUDGET-DEPLETING MODEL',
    'replenishing': 'POSITIVE-COST MODEL',
    'replenishing_plus': 'POSITIVE-COST-PLUS MODEL',
}


def action_letter(action):
    return 'R' if action == 0 else 'C'


def print_policy_table(case, model, policy):
    print('\n' + '=' * 92)
    print(NAMES[case])
    print('=' * 92)
    if case == 'baseline':
        print('state | policy')
        print('------+-------')
        for state in range(model.config.states):
            print(f'{state + 1:5} |   {action_letter(policy[state, 0])}')
        return

    budget_header = ' | '.join(f'z={z:>2}' for z in range(model.layers))
    print(f'state | l(s,C) | {budget_header}')
    print('-' * (15 + 7 * model.layers))
    for state in range(model.config.states):
        actions = ' | '.join(f'  {action_letter(policy[state, z])}'
                             for z in range(model.layers))
        print(f'{state + 1:5} | {model.safety[state]:6} | {actions}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', type=int, default=70)
    parser.add_argument('--capacity', type=int, default=12)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent/'results_n70_b12')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('C = CONTINUE    R = REPLACE')
    print(f'Number of states = {args.states}    Maximum budget = {args.capacity}')
    reports = {}
    for case in CASES:
        model = Model(Config(case, args.states, args.capacity))
        v, q, policy, report = solve(model)
        np.savez(args.output_dir/f'{case}_vi.npz', values=v, q=q, policy=policy)
        reports[case] = report
        print_policy_table(case, model, policy)
        print(f'Value iteration converged in {report["iterations"]} iterations.')
    (args.output_dir/'value_iteration.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(f'\nSaved numerical results in: {args.output_dir}')


if __name__ == '__main__':
    main()
