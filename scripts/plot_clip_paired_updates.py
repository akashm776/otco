"""Plot the audited paired updates from committed data; no GPU needed."""

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'experiment_results/clip_paired_updates_2026-09/paired_updates.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True, layout='constrained')
    for ax, step in zip(axes, [100, 1001]):
        ax.axhline(0, color='#333333', lw=1)
        for arm, label, color, marker in [('uniform_top8', 'Synthetic top-8', '#D55E00', 'o'),
                                          ('hardest_real', 'Hardest real', '#0072B2', 's')]:
            selected = sorted([r for r in rows if r['checkpoint_step'] == step and r['arm'] == arm], key=lambda r: r['trial'])
            x = [r['trial'] + 1 for r in selected]
            y = [r['incremental_heldout_loss'] * 1e6 for r in selected]
            ax.scatter(x, y, label=label, color=color, marker=marker, s=30)
            ax.axhline(statistics.fmean(y), color=color, ls='--', lw=1, alpha=.65)
        synthetic = [r['incremental_heldout_loss'] * 1e6 for r in rows if r['checkpoint_step'] == step and r['arm'] == 'uniform_top8']
        ax.set(title=f'{step:,} completed training updates', xlabel='Fixed training batch (not training time)',
               xticks=[1, 4, 8, 12, 16], ylim=(-40, 45))
        ax.text(.02, .96, f'Synthetic: {sum(x < 0 for x in synthetic)}/16 beneficial\nMean: {statistics.fmean(synthetic):+.2f}',
                transform=ax.transAxes, va='top', color='#A54100', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Extra held-out loss vs native update (× 10⁻⁶)\nBelow zero = incremental benefit')
    axes[1].legend(loc='lower right', frameon=False)
    fig.suptitle('Synthetic updates help early, slightly hurt later in this one-step probe\nOne checkpoint trajectory • same 16 training batches • dashed lines = means', fontsize=12)
    for suffix in ['png', 'svg']:
        target = ROOT / f'docs/figures/clip_paired_update_effects.{suffix}'
        fig.savefig(target, dpi=180)
        if suffix == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
