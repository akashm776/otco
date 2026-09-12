import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / 'docs/figures'
rows = json.loads((root / 'experiment_results/clip_curriculum_2026-09/early_pulse_performance.json').read_text())
baseline = {r['step']: r for r in rows if r['arm'] == 'baseline'}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for arm, label, color in [('uniform_top8','Synthetic pulse','#df7b12'), ('hardest_real','Real-negative pulse','#26945a')]:
    selected = [r for r in rows if r['arm'] == arm]
    for ax, metric in zip(axes, ['canonical_avg_r1_percent','species_top1_percent']):
        x = [r['step'] for r in selected]
        y = [r[metric] - baseline[r['step']][metric] for r in selected]
        ax.plot(x, y, marker='.', color=color, label=label)
        ax.annotate(f'{y[-1]:+.3f} pp', (x[-1], y[-1]), xytext=(-6, 9 if arm == 'uniform_top8' else -15),
                    textcoords='offset points', ha='right', color=color, fontsize=9)
for ax, title in zip(axes, ['Canonical average R@1: change vs baseline', 'Species top-1: change vs baseline']):
    ax.axhline(0, color='black', lw=.8)
    ax.axvspan(100, 200, color='gray', alpha=.13)
    ax.set(title=title, xlabel='Completed updates', ylabel='Difference (percentage points)')
    ax.grid(alpha=.15)
axes[0].legend(fontsize=9)
fig.suptitle('Early-pulse pilot: one seed, descriptive differences—not confidence intervals')
for suffix in ['png','svg']:
    fig.savefig(out / f'clip_early_pulse_differences.{suffix}', dpi=180)
    if suffix == 'svg':
        path = out / f'clip_early_pulse_differences.{suffix}'
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
plt.close(fig)
