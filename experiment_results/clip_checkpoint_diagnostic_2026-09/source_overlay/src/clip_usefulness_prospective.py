"""Frozen-rule evaluation on new native training seeds; no gate-driven training."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

import torch
import yaml

from src import clip_train, clip_paired_updates as paired
from src.clip_gradient_stages import ROOT
from src.clip_paired_intermediate import IntermediateObserver
from src.clip_paired_seed_replication import CheckpointObserver, source_inventory

SEEDS = [789, 2026, 31415]
STEPS = [100, 250, 500, 750, 1001]
PARTS = ['shuffle_seed_42', 'shuffle_seed_123', 'shuffle_seed_4242']
RULE_NAMES = ['full_gradient_alignment', 'checkpoint_step']


def load_study():
    study = yaml.safe_load((ROOT / 'configs/clip_usefulness_prospective.yaml').read_text())
    frozen = json.loads((ROOT / study['frozen_rules']).read_text())
    if study['training_seeds'] != SEEDS or study['checkpoint_steps'] != STEPS:
        raise ValueError('Changed prospective seed/stage design')
    if set(SEEDS) & set(frozen['development_seeds']) or study['refit_on_new_seeds']:
        raise ValueError('Development seeds must not enter prospective evaluation')
    if frozen['rules'] != {
        'full_gradient_alignment': dict(feature='full_gradient_alignment', threshold=0.030847286945687016, lower_is_helpful=False),
        'checkpoint_step': dict(feature='checkpoint_step', threshold=175.0, lower_is_helpful=True)}:
        raise ValueError('Frozen rules changed')
    return study, frozen


def protocols(seed, source_name):
    study, _ = load_study()
    if seed not in SEEDS:
        raise ValueError('Not a prospective seed')
    config = clip_train.load_training_config(ROOT / study['baseline_config'])
    config['experiment']['seed'] = config['training']['seed'] = seed
    config['diagnostics']['separate_projection_gradient_steps'] = 0
    protocol = yaml.safe_load((ROOT / study['paired_protocol']).read_text())
    protocol.update(reference_run=protocol['source_run'], source_run=source_name, training_seed=seed,
                    diagnostic_caption_seed=42, checkpoint_steps=STEPS)
    audit = json.loads((ROOT / 'experiment_results/clip_paired_updates_2026-09/audit.json').read_text())
    protocol['fixed_input_sha256'] = {name: audit['files'][name]['sha256'] for name in
                                     ['training_batches.json', 'heldout_partitions.json']}
    paired.validate_protocol(protocol)
    return study, config, protocol


class ProspectiveObserver(IntermediateObserver):
    def __init__(self, source, study, protocol):
        CheckpointObserver.__init__(self, source, study, protocol)
        # New seeds have no historical endpoint state to reproduce. All five
        # observations still save state/feature hashes for exact paired replay.
        self.references = {}

    def __call__(self, *, model, epoch, global_step):
        super().__call__(model=model, epoch=epoch, global_step=global_step)
        # The four diagnostic snapshots are closed now; latest.pt at 1001 is
        # saved later by clip_train and is backed up after train_seed returns.
        if global_step in STEPS[:-1] and getattr(self, 'on_checkpoint', None) is not None:
            self.on_checkpoint()


def train_seed(source, seed, *, on_checkpoint=None):
    study, config, protocol = protocols(seed, source.name)
    source.mkdir(parents=True, exist_ok=False)
    observer = ProspectiveObserver(source, study, protocol)
    observer.on_checkpoint = on_checkpoint
    summary = clip_train.run(config, output_directory=source / 'results/baseline/training',
        checkpoint_directory=source / 'checkpoints/baseline', observer=observer,
        stop_after_epochs=study['stop_after_epochs'])
    if observer.seen != STEPS or summary['execution']['completed_updates'] != 1001:
        raise AssertionError('Incomplete baseline trajectory')
    clip_train.write_json(source / 'results/baseline/completion.json',
        dict(status='complete', training_seed=seed, completed_updates=1001, checkpoint_steps=STEPS))
    del observer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    source_inventory(source)
    return protocol


def state_records(source, seed):
    directory = source / 'results/paired'
    read = lambda p: json.loads(p.read_text())
    completion = read(directory / 'completion.json')
    rows = [json.loads(line) for line in (directory / 'paired_updates.jsonl').read_text().splitlines()]
    lookup = {(r['checkpoint_step'], r['trial'], r['arm']): r for r in rows}
    expected = {(s,t,a) for s in STEPS for t in range(16) for a in ['baseline','uniform_top8','hardest_real']}
    if (completion.get('status') != 'complete' or completion.get('rows') != 240
            or completion.get('checkpoint_steps') != STEPS or len(rows) != 240 or set(lookup) != expected):
        raise AssertionError('Incomplete five-state paired evaluation')
    provenance = read(directory / 'checkpoint_provenance.json')
    if len(provenance) != 5 or sorted(r['step'] for r in provenance) != STEPS:
        raise AssertionError('Missing state provenance')
    by_step = {r['step']: r for r in provenance}
    for state in provenance:
        if state.get('optimizer_reset_checks') != 48 or not all(state.get(k) is True for k in
            ['no_update_reencoding_exact','native_update_and_evaluation_replay_exact','source_state_immutable','frozen_parameters_unchanged']):
            raise AssertionError('Failed reset/replay checks')
        observed = read(source / f'results/baseline/diagnostics/{state["step"]:06d}_pulse_{state["step"]}/report.json')
        if (observed['training_seed'] != seed or observed['features_and_scale_sha256'] != state['features_sha256']
                or observed['initial_state_sha256'] != state['initial_state_sha256']
                or observed['learning_rates'] != state['learning_rates']):
            raise AssertionError('Checkpoint state does not match its training observation')
    for row in rows:
        native = lookup[row['checkpoint_step'],row['trial'],'baseline']
        losses = row['heldout_loss']
        if set(losses) != set(PARTS+['sequential']) or any(len(v) != 16 or not all(math.isfinite(x) for x in v) for v in losses.values()):
            raise AssertionError('Missing or nonfinite held-out batch losses')
        measured = paired.primary_mean(losses, PARTS)
        initial = paired.primary_mean(by_step[row['checkpoint_step']]['initial_heldout_loss'], PARTS)
        checks = [(row['heldout_mean'], measured), (row['heldout_change'], measured-initial),
                  (row['incremental_heldout_loss'], measured-native['heldout_mean']),
                  (row['incremental_train_loss'], row['train_after']-native['train_after'])]
        if not all(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=0,abs_tol=1e-12) for a,b in checks):
            raise AssertionError('Invalid paired arithmetic')
    states = []
    for step in STEPS:
        selected = [lookup[step,t,'uniform_top8'] for t in range(16)]
        if not all(math.isfinite(r['full_gradient_alignment']) for r in selected):
            raise AssertionError('Nonfinite alignment')
        states.append(dict(training_seed=seed, checkpoint_step=step,
            full_gradient_alignment=stats.fmean(r['full_gradient_alignment'] for r in selected),
            effect=stats.fmean(r['incremental_heldout_loss'] for r in selected),
            beneficial_trials=sum(r['incremental_heldout_loss'] < 0 for r in selected),
            native_heldout_change=stats.fmean(lookup[step,t,'baseline']['heldout_change'] for t in range(16)),
            synthetic_heldout_change=stats.fmean(r['heldout_change'] for r in selected)))
    return states


def score(states, frozen, band=1e-6):
    if len(states) != 15 or {(s['training_seed'],s['checkpoint_step']) for s in states} != {(s,t) for s in SEEDS for t in STEPS}:
        raise AssertionError('Exactly fifteen new seed/stage states required')
    predictions, scores = [], []
    for name in RULE_NAMES:
        rule = frozen['rules'][name]
        for seed in SEEDS:
            chosen = [r for r in states if r['training_seed'] == seed]
            for row in chosen:
                if not math.isfinite(row['effect']) or not math.isfinite(row[rule['feature']]):
                    raise AssertionError('Nonfinite state')
                # Prediction uses the frozen rule and predictor value only.
                label = (row[rule['feature']] <= rule['threshold']) == rule['lower_is_helpful']
                predictions.append(dict(predictor=name, **row, actual_helpful=row['effect'] < 0,
                                        predicted_helpful=label, near_zero=abs(row['effect']) <= band))
            selected = [p for p in predictions if p['predictor'] == name and p['training_seed'] == seed]
            truth = [p['actual_helpful'] for p in selected]
            recalls = [sum(p['predicted_helpful'] == label for p in selected if p['actual_helpful'] == label)/truth.count(label)
                       for label in [False,True] if label in truth]
            robust = [p for p in selected if not p['near_zero']]
            scores.append(dict(predictor=name, training_seed=seed, count=5,
                correct=sum(p['predicted_helpful'] == p['actual_helpful'] for p in selected),
                balanced_accuracy=stats.fmean(recalls) if len(recalls) == 2 else None,
                non_near_zero_correct=sum(p['predicted_helpful'] == p['actual_helpful'] for p in robust),
                non_near_zero_count=len(robust)))
    differences = []
    for seed in SEEDS:
        a,b = [next(r for r in scores if r['training_seed'] == seed and r['predictor'] == name) for name in RULE_NAMES]
        differences.append(None if a['balanced_accuracy'] is None or b['balanced_accuracy'] is None
                           else a['balanced_accuracy']-b['balanced_accuracy'])
    return dict(predictions=predictions, per_seed_scores=scores,
        per_seed_balanced_accuracy_differences=dict(zip(map(str,SEEDS),differences)),
        primary_mean_balanced_accuracy_difference=stats.fmean(differences) if all(d is not None for d in differences) else None,
        note='Null balanced accuracy if a seed has only one outcome class. No refitting, significance claim, or automatic curriculum selection.')


def plot(directory, states, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for seed,color in zip(SEEDS,['#0072B2','#D55E00','#009E73']):
        rows = sorted([r for r in states if r['training_seed'] == seed], key=lambda r:r['checkpoint_step'])
        axes[0].plot(STEPS,[r['effect']*1e6 for r in rows],marker='o',label=f'Seed {seed}',color=color)
    axes[0].axhline(0,color='black',lw=.8)
    axes[0].set(xlabel='Completed updates',ylabel='Extra held-out loss × 10⁻⁶ (lower better)',xticks=STEPS)
    axes[0].legend()
    for offset,name,color in [(-.18,RULE_NAMES[0],'#0072B2'),(.18,RULE_NAMES[1],'#999999')]:
        vals = [next(r['balanced_accuracy'] for r in report['per_seed_scores'] if r['predictor']==name and r['training_seed']==seed) for seed in SEEDS]
        axes[1].bar([i+offset for i in range(3)],[float('nan') if v is None else v for v in vals],width=.36,label=name,color=color)
    axes[1].set(xticks=range(3),xticklabels=map(str,SEEDS),ylim=(0,1),xlabel='New training seed',ylabel='Balanced accuracy (missing if one class)')
    axes[1].legend()
    fig.suptitle('Frozen-rule prospective seed test; no refitting\nLines connect measured states only; no confidence intervals')
    for suffix in ['png','svg']:
        fig.savefig(directory / f'prospective_usefulness.{suffix}',dpi=180)
    plt.close(fig)


def verify_completion(output):
    study,frozen = load_study()
    if json.loads((output/'results/protocol.json').read_text()) != study:
        raise AssertionError('Run protocol changed')
    marker = json.loads((output/'results/completion.json').read_text())
    if marker != dict(status='complete',training_seeds=SEEDS,checkpoint_steps=STEPS,total_branch_rows=720,evaluated_states=15,rules_refitted=False):
        raise AssertionError('Incomplete prospective experiment')
    if json.loads((output/'results/frozen_rules.json').read_text()) != frozen:
        raise AssertionError('Rules differ from frozen development artifact')
    states = [r for seed in SEEDS for r in state_records(output/f'seed_{seed}',seed)]
    if score(states,frozen,study['near_zero_absolute_band']) != json.loads((output/'results/prediction_report.json').read_text()):
        raise AssertionError('Prediction report does not reproduce')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-directory',required=True)
    output = Path(parser.parse_args().output_directory)
    output.mkdir(parents=True,exist_ok=False)
    results = output/'results'
    results.mkdir()
    study,frozen = load_study()
    clip_train.write_json(results/'protocol.json',study)
    clip_train.write_json(results/'frozen_rules.json',frozen)
    print('FROZEN BEFORE TRAINING:',json.dumps(frozen['rules']),flush=True)
    print('FROZEN ARTIFACT SHA256:',hashlib.sha256((ROOT/study['frozen_rules']).read_bytes()).hexdigest(),flush=True)
    states = []
    for seed in SEEDS:
        print(f'=== NEW SEED {seed}: baseline and 240 paired branches ===',flush=True)
        source = output/f'seed_{seed}'
        protocol = train_seed(source,seed)
        paired.run(source,source/'results/paired',protocol)
        states.extend(state_records(source,seed))
        clip_train.write_json(results/'progress.json',dict(last_completed_seed=seed,evaluated_states=len(states)))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = score(states,frozen,study['near_zero_absolute_band'])
    clip_train.write_json(results/'states.json',states)
    clip_train.write_json(results/'prediction_report.json',report)
    plot(results,states,report)
    clip_train.write_json(results/'completion.json',dict(status='complete',training_seeds=SEEDS,
        checkpoint_steps=STEPS,total_branch_rows=720,evaluated_states=15,rules_refitted=False))
    verify_completion(output)
    print('COMPLETE: 720 new branches; frozen-rule primary difference:',report['primary_mean_balanced_accuracy_difference'],flush=True)


if __name__ == '__main__':
    main()
