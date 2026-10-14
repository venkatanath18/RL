"""Combine VI and Q-learning output into a report and scientific figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from model import CASES, Config, Model, regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,
                        default=Path(__file__).parent/'results_n70_b12')
    args = parser.parse_args()
    out = args.output_dir
    exact = json.loads((out/'value_iteration.json').read_text())
    learned = [json.loads(p.read_text()) for p in sorted(out.glob('*_q_seed*.json'))]
    for r in learned:
        cfg = exact[r['case']]['config']
        if (r['configuration']['states'],r['configuration']['capacity']) != (cfg['states'],cfg['capacity']):
            raise ValueError('Mismatched model configurations in output directory')
    cmap = ListedColormap(['#cc6633','#208c86'])
    fig, axes = plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,case in zip(axes.flat,CASES):
        policy = np.load(out/f'{case}_vi.npz')['policy']
        capacity = exact[case]['config']['capacity']
        if case == 'baseline':
            policy = np.repeat(policy,capacity+1,axis=1)
        ax.imshow(policy.T,origin='lower',aspect='auto',cmap=cmap,vmin=0,vmax=1,
                  extent=(0.5,policy.shape[0]+0.5,-0.5,capacity+0.5),interpolation='nearest')
        ax.set_title(case.replace('_',' ').title())
        ax.set_xlabel('Machine state (1-based)')
        ax.set_ylabel('Remaining budget z')
    fig.suptitle('Value iteration: replacement preserves z\nOrange = REPLACE; green = CONTINUE',fontsize=13)
    fig.tight_layout(rect=(0,0,1,0.93))
    fig.savefig(out/'policies.png',dpi=180)
    plt.close(fig)

    fig,ax = plt.subplots(figsize=(9,5))
    for case in CASES:
        v = np.load(out/f'{case}_vi.npz')['values']
        if case == 'baseline':
            ax.axhline(v[0,0],color='black',linestyle='--',label='Baseline')
        else:
            ax.plot(np.arange(v.shape[1]),v[0],marker='o',label=case.replace('_',' '))
    ax.set(xlabel='Initial budget z (machine state 1)',ylabel='Expected discounted monetary cost',
           title='Value iteration: cost versus initial budget')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out/'initial_costs.png',dpi=180); plt.close(fig)

    if learned:
        fig,axes = plt.subplots(1,2,figsize=(12,4.5))
        colors = dict(zip(CASES,['#444444','#cc6633','#208c86','#7656a4']))
        for r in learned:
            h = r['history']; x = [c['transitions'] for c in h]
            label = f'{r["case"]}, seed {r["seed"]}'
            axes[0].plot(x,[100*c['relative_q_error'] for c in h],color=colors[r['case']],alpha=.7,label=label)
            axes[1].plot(x,[100*c['policy_agreement'] for c in h],color=colors[r['case']],alpha=.7)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Maximum relative Q error (%)')
        axes[1].set_ylabel('Agreement with VI policy (%)')
        for ax in axes:
            ax.set_xlabel('Sampled transitions'); ax.grid(alpha=.2)
        axes[0].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(out/'learning.png',dpi=180); plt.close(fig)

    # One reproducible 500-decision trajectory per exact policy. The shared
    # uniform stream couples physical transitions, not an equality of paths.
    uniforms = np.random.default_rng(2026).random(500)
    trajectory_report = {}
    fig,ax = plt.subplots(figsize=(10,4.5))
    for case in CASES[1:]:
        model = Model(Config(**exact[case]['config']))
        policy = np.load(out/f'{case}_vi.npz')['policy']
        s,z = 0,model.config.capacity
        trace, replacements = [z],0
        for u in uniforms:
            a = int(policy[s,z]); replacements += a == 0
            s,z,_ = model.transition(s,z,a,float(u)); trace.append(z)
        trajectory_report[case] = dict(seed=2026,steps=500,replacements=replacements,
                                       minimum_budget=min(trace),final_budget=z,
                                       first_zero_step=trace.index(0) if 0 in trace else None)
        ax.step(range(len(trace)),trace,where='post',label=case.replace('_',' '),alpha=.8)
    ax.set(xlabel='Decision step',ylabel='Remaining budget',
           title='One illustrative trajectory under each VI policy (start z=12)')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(out/'budget_trajectories.png',dpi=180); plt.close(fig)
    (out/'trajectory_examples.json').write_text(json.dumps(trajectory_report,indent=2)+'\n')

    lines = ['# Persistent-budget comparison','',
             'All cases share the physical transition law, economic costs and state count. '
             'Replacement preserves z; maximum budget is '+str(exact['depleting']['config']['capacity'])+'.',
             '', '## Value iteration', '',
             '| Model | V(1,full budget) | Iterations | Bellman residual |',
             '|---|---:|---:|---:|']
    for case,r in exact.items():
        lines.append(f'| {case} | {r["initial_value"]:.9f} | {r["iterations"]} | {r["bellman_residual"]:.3g} |')
    lines += ['', 'The baseline has no budget constraint. Its value is repeated across budget slices for comparison.',
              '', '### All replacement intervals', '', '| z | baseline | depleting | replenishing | replenishing_plus |',
              '|---:|---|---|---|---|']
    for z in range(exact['depleting']['config']['capacity']+1):
        cells = [str(exact[c]['replacement_regions'][str(0 if c=='baseline' else z)]) for c in CASES]
        lines.append('| '+str(z)+' | '+' | '.join(cells)+' |')
    lines += ['', '## Q-learning comparisons', '',
              '| Model | Seed | Status | Episodes | Agreement | Max Q error | Worst policy value gap | Initial value gap |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in learned:
        h = r['history'][-1]
        lines.append(f'| {r["case"]} | {r["seed"]} | {r["status"]} | {r["episodes"]} | '
                     f'{h["policy_agreement"]:.2%} | {h["relative_q_error"]:.3%} | '
                     f'{h["maximum_policy_value_gap"]:.6f} | {h["initial_value_gap"]:.9f} |')
    lines += ['', 'Stopping means the declared finite tolerances passed for three consecutive checkpoints; '
              'it is not a proof of asymptotic convergence. Every feasible action requires minimum visitation. '
              'The worst policy gap covers every state-budget pair, not just the initial state.',
              '', '## Interpretation', '',
              '- At discount 0.7, distant costs receive little weight. A full budget of 12 makes '
              'initial discounted costs nearly identical even when low-budget policies differ.',
              '- In the depletion-only model, the budget can never recover: replacement no longer replenishes it. '
              'At z=0, risky continuation is infeasible indefinitely, but replacement remains available.',
              '- Signed-cost models can replenish through continuation in designated states, capped at 12.',
              '- The plus cost pattern and all comparison models use N=70.',
              '- Zero violations are enforced by the known-cost feasible-action rule. They do not '
              'demonstrate that Q-learning independently discovered safe exploration.',
              '', 'The trajectory figure shows individual simulated paths, not estimated average cycle lengths.',
              '', '## Artifacts', '',
              '- `policies.png`: exact policies for all budgets.',
              '- `initial_costs.png`: economic costs as initial budget varies.',
              '- `learning.png`: sampled-learning diagnostics across seeds.',
              '- `budget_trajectories.png`: 500-decision budget examples.',
              '- `value_iteration.json`, `*_vi.npz`: exact reports and arrays.',
              '- `*_q_seed*.json`, `*_q_seed*.npz`: convergence histories, policies, Q tables and visits.']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print('Combined report and figures saved:',out/'COMPARISON.md')


if __name__ == '__main__':
    main()
