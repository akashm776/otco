"""Plot all paired inputs and seed-level means from committed result records."""

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def main():
    reference = ROOT / 'experiment_results/clip_paired_updates_2026-09'
    replication = ROOT / 'experiment_results/clip_paired_seeds_2026-09'
    rows = {}
    for seed in [42, 123, 456]:
        directory = reference if seed == 42 else replication / f'seed_{seed}/results/paired'
        rows[seed] = [json.loads(line) for line in (directory / 'paired_updates.jsonl').read_text().splitlines()]
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, layout='constrained')
    for ax, step in zip(axes, [100, 1001]):
        ax.axhline(0, color='#333333', lw=1)
        for arm, label, color, offset in [('uniform_top8', 'Synthetic top-8', '#D55E00', -.13),
                                         ('hardest_real', 'Hardest real', '#0072B2', .13)]:
            means = []
            for index, seed in enumerate([42, 123, 456]):
                chosen = sorted([r for r in rows[seed] if r['checkpoint_step'] == step and r['arm'] == arm], key=lambda r: r['trial'])
                values = [r['incremental_heldout_loss'] * 1e6 for r in chosen]
                means.append(statistics.fmean(values))
                ax.scatter([index + offset + (i-7.5)*.008 for i in range(16)], values, s=13, color=color, alpha=.4)
            ax.scatter([i + offset for i in range(3)], means, marker='D', s=55, color=color,
                       edgecolors='white', linewidth=.6, label=label, zorder=3)
        ax.set(title=f'{step:,} completed training updates', xticks=[0, 1, 2],
               xticklabels=['42 (reference)', '123', '456'], xlabel='Training seed', ylim=(-45, 45), xlim=(-.5, 2.5))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Extra held-out loss vs native update (× 10⁻⁶)\nBelow zero = incremental benefit')
    axes[0].legend(loc='upper right', frameon=False, fontsize=9)
    fig.suptitle('Early benefit repeats; later effects vary by training seed', fontsize=14)
    fig.supxlabel('Diamonds = seed means; small dots = 16 fixed input batches per seed (not independent seeds)', fontsize=9)
    for suffix in ['png', 'svg']:
        target = ROOT / f'docs/figures/clip_paired_seed_replication.{suffix}'
        fig.savefig(target, dpi=180)
        if suffix == 'svg':
            target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
