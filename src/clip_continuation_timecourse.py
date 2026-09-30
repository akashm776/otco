"""Fresh matched streams with a fixed, observational gradient/report time course.

No optimizer trials, adaptive gates, report-selected checkpoints, or early stops.
The parent saved-checkpoint implementation and its executed bundle stay unchanged.
"""
import gc
import json
import math
from pathlib import Path
import statistics

import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_train
from src.clip_gated_training import fixed_probe_batches, measure_alignment, load_protocol as gate_protocol
from src.clip_paired_updates import cosine

ARMS = ["native", "sustained"]


def load_protocol():
    extension = json.loads((d.ROOT / 'configs/clip_continuation_timecourse.json').read_text())
    expected = dict(experiment='clip_continuation_timecourse_v1',
        parent_protocol='configs/clip_checkpoint_diagnostic.json',
        training_seeds=[789, 2026, 31415], checkpoint_steps=[100, 250, 500, 750],
        stream_seeds=[2026092601, 2026092602], arms=ARMS, horizon=50,
        observation_steps=[0, 1, 5, 10, 25, 50], coefficient=0.06005351588542947,
        early_step=100, later_steps=[250, 500, 750],
        primary='seed_mean_later_minus_early_sustained_minus_native_report_loss_at_50',
        probe_definition='historical_16_batch_alignment_and_meta_vs_fixed_first_probe_auxiliary',
        probe_controls_treatment=False, report_controls_treatment=False, rules_refitted=False,
        independent_test_set=False, literal_training_loader_resume=False, optimizer_step_budget=2400)
    if extension != expected:
        raise ValueError('Frozen continuation protocol changed')
    parent = d.load_protocol()
    p = {**parent, **extension}
    p['previous_branch_data_seed'] = parent['branch_data_seed']
    del p['branch_data_seed']
    return p


def make_plans(training, p):
    plans = {}
    for seed in p['training_seeds']:
        for step in p['checkpoint_steps']:
            previous = d.branch_plan(training, seed, step,
                {**p, 'branch_data_seed': p['previous_branch_data_seed']})
            seen = {d.state_digest(previous)}
            for stream in p['stream_seeds']:
                plan = d.branch_plan(training, seed, step, {**p, 'branch_data_seed': stream})
                stamp = d.state_digest(plan)
                if stamp in seen:
                    raise AssertionError('Repeated continuation stream')
                seen.add(stamp)
                plans[f'{seed}:{step}:{stream}'] = plan
    return plans


class GradientReprobe:
    """Fixed training probes + meta set; no report object or optimizer access.

Historical score is mean of 16 batch cosines, NOT cosine of mean gradients.
Meta score uses the auxiliary gradient from the fixed first historical batch.
No AdamW update-utility claim is made for either gradient-only measurement.
"""
    def __init__(self, batches, config, meta, gate):
        self.batches, self.config, self.meta, self.gate = batches, config, meta, gate

    def measure(self, model):
        historical = measure_alignment(model, self.batches, self.config, self.gate['probe_branch_seed'])
        with d.observe(model):
            clip_train.set_global_seed(self.gate['probe_branch_seed'])
            native, auxiliary = d.training_gradients(model, self.batches[0], self.config)
            meta_gradient, meta_loss = self.meta.gradient(model)
        values = dict(historical_mean_cosine=historical['mean_cosine'],
            first_fixed_batch_cosine=cosine(native, auxiliary),
            meta_fixed_aux_cosine=cosine(meta_gradient, auxiliary), meta_native_loss=meta_loss)
        if any(v is None or not math.isfinite(v) for v in values.values()):
            raise ValueError('Degenerate or nonfinite re-probe')
        return dict(**values, historical_batch_cosines=historical['batch_cosines'],
            historical_threshold=self.gate['alignment_threshold'],
            historical_above_threshold=values['historical_mean_cosine'] > self.gate['alignment_threshold'],
            diagnostic_only=True)


def observation(model, optimizer, scheduler, probe, report, commit_probe, label):
    before = d.live_digest(model, optimizer, scheduler)
    gradients = d.state_digest([p.grad for p in model.parameters()])
    modes = [m.training for m in model.modules()]
    measured = probe.measure(model)
    if d.live_digest(model, optimizer, scheduler) != before:
        raise AssertionError('Re-probe changed training state')
    commit_probe(label, measured)  # Durable barrier before the corresponding outcome.
    if (d.live_digest(model, optimizer, scheduler) != before
            or d.state_digest([p.grad for p in model.parameters()]) != gradients
            or [m.training for m in model.modules()] != modes):
        raise AssertionError('Re-probe persistence changed training state')
    evaluated = report.evaluate(model)
    if (d.live_digest(model, optimizer, scheduler) != before
            or d.state_digest([p.grad for p in model.parameters()]) != gradients
            or [m.training for m in model.modules()] != modes):
        raise AssertionError('Observation/commit changed state, modes, or gradients')
    return dict(probe=measured, report=evaluated, training_state_sha256=before)


def run_pair(model, optimizer, scheduler, initial, batches, config, probe, report,
             *, coefficient, observation_steps, commit_probe, save_point=lambda *_: None,
             progress=lambda _: None):
    times = list(observation_steps)
    if not batches or times != sorted(set(times)) or times[0] != 0 or times[-1] != len(batches):
        raise ValueError('Observation grid must include zero and the fixed horizon')
    source_hash = d.state_digest(initial)
    frozen = d.state_digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
    d.restore(model, optimizer, scheduler, initial)
    common = observation(model, optimizer, scheduler, probe, report, commit_probe, ('common', 0))
    if report.evaluate(model) != common['report'] or d.live_digest(model, optimizer, scheduler) != common['training_state_sha256']:
        raise AssertionError('No-update report replay changed')
    save_point('common', 0, common)
    if d.live_digest(model, optimizer, scheduler) != common['training_state_sha256']:
        raise AssertionError('Common-point persistence changed training state')
    points, traces, starts = {}, {}, {}
    for arm in ARMS:
        d.restore(model, optimizer, scheduler, initial)
        starts[arm] = d.live_digest(model, optimizer, scheduler)
        if starts[arm] != common['training_state_sha256']:
            raise AssertionError('Branch start differs')
        points[arm], traces[arm] = {'0': common}, []
        for offset, batch in enumerate(batches):
            trace = d.step_update(model, optimizer, scheduler, batch, config,
                                  coefficient if arm == 'sustained' else 0.)
            trace['offset'] = offset
            traces[arm].append(trace)
            if arm == 'sustained' and any(trace[k] != traces['native'][offset][k]
                    for k in ('learning_rates', 'rng_after_sha256')):
                raise AssertionError('Unmatched branch LR/RNG stream')
            completed = offset + 1
            if completed in times:
                point = observation(model, optimizer, scheduler, probe, report,
                                    commit_probe, (arm, completed))
                points[arm][str(completed)] = point
                save_point(arm, completed, point)
                # Backup callbacks are also forbidden from perturbing continuation.
                if d.live_digest(model, optimizer, scheduler) != point['training_state_sha256']:
                    raise AssertionError('Point persistence changed training state')
                progress(f'{arm}: update {completed}/{len(batches)}')
    if d.state_digest(initial) != source_hash:
        raise AssertionError('Source checkpoint mutated')
    if d.state_digest({n: p for n, p in model.named_parameters() if not p.requires_grad}) != frozen:
        raise AssertionError('Frozen parameters changed')
    differences = {}
    for t in times:
        native, treated = (points[arm][str(t)] for arm in ARMS)
        differences[str(t)] = dict(
            report_loss=treated['report']['native_loss']-native['report']['native_loss'],
            pool_r1_pp=treated['report']['pool_retrieval']['mean_r1_percent']-
                       native['report']['pool_retrieval']['mean_r1_percent'],
            partition_loss={key: statistics.fmean(values)-statistics.fmean(native['report']['partition_losses'][key])
                            for key, values in treated['report']['partition_losses'].items()},
            historical_cosine=treated['probe']['historical_mean_cosine']-native['probe']['historical_mean_cosine'],
            meta_aux_cosine=treated['probe']['meta_fixed_aux_cosine']-native['probe']['meta_fixed_aux_cosine'])
    return dict(points=points, traces=traces, differences=differences,
        audit=dict(source_state_sha256=source_hash, branch_start_hashes=starts,
            source_immutable=True, frozen_parameters_unchanged=True, observations_preserve_state=True,
            exact_no_update_report_replay=True, matched_rng_and_lrs=True,
            probes_control_treatment=False, actual_optimizer_steps=2*len(batches)))


def summarize(rows, p):
    expected = {(s, t, u) for s in p['training_seeds'] for t in p['checkpoint_steps'] for u in p['stream_seeds']}
    keys = {(r['training_seed'], r['checkpoint_step'], r['stream_seed']) for r in rows}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError('Incomplete or duplicate seed/state/stream grid')
    per_seed = []
    metrics = ['report_loss', 'pool_r1_pp', 'historical_cosine', 'meta_aux_cosine']
    for seed in p['training_seeds']:
        by_step = {str(step): {str(t): {metric: statistics.fmean(
            r['differences'][str(t)][metric] for r in rows
            if r['training_seed'] == seed and r['checkpoint_step'] == step)
            for metric in metrics} for t in p['observation_steps']} for step in p['checkpoint_steps']}
        course = {}
        for t in p['observation_steps']:
            early = by_step[str(p['early_step'])][str(t)]['report_loss']
            later = statistics.fmean(by_step[str(step)][str(t)]['report_loss'] for step in p['later_steps'])
            course[str(t)] = dict(early_report_loss=early, later_report_loss=later, later_minus_early=later-early)
        per_seed.append(dict(training_seed=seed, per_starting_step=by_step, early_later_timecourse=course))
    mean_course = {str(t): {k: statistics.fmean(s['early_later_timecourse'][str(t)][k] for s in per_seed)
                          for k in ['early_report_loss', 'later_report_loss', 'later_minus_early']}
                   for t in p['observation_steps']}
    return dict(primary=p['primary'], primary_value=mean_course[str(p['horizon'])]['later_minus_early'],
        seed_mean_timecourse=mean_course, per_seed=per_seed, pairs=len(rows), branches=2*len(rows),
        training_seed_count=len(per_seed), streams_per_state=len(p['stream_seeds']),
        optimizer_steps=sum(r['audit']['actual_optimizer_steps'] for r in rows),
        interpretation='Treatment minus native; negative loss helps. Average streams within states, '
        'later states within seeds, then seeds equally. Reused states/data, correlated checkpoints and '
        'streams; no independent-test, significance, gate-superiority, or readiness-consumption claim. '
        'Re-probes are gradients on fixed probes, not actual AdamW usefulness trials.')


def run(source, directory, sync, p=None):
    from transformers import AutoProcessor
    p = load_protocol() if p is None else p
    source, directory = Path(source), Path(directory)
    d.write_json(directory/'protocol.json', p)
    clip_train.set_global_seed(42)
    config = clip_train.load_training_config(d.ROOT/'configs/hf_cub200_clip_vit_b32_baseline.yaml')
    config['runtime']['num_workers'] = 0
    device, info = clip_train.resolve_device(config['runtime'])
    processor = AutoProcessor.from_pretrained(config['model']['checkpoint'])
    data = clip_train.build_clip_training_data(config, processor, pin_memory=False)
    training = data.train_dataset.source_indices
    holdout = json.loads((d.ROOT/config['dataset']['diagnostic_holdout_indices']).read_text())
    if len(training) != 4970 or len(data.train_loader) != 77:
        raise AssertionError('Training exclusions/horizon changed')
    pools = d.split_pool(training, holdout, p)
    meta_batches, meta_records = d.build_batches(data, pools['meta'], seed=0, epoch=0, canonical=True)
    report_batches, report_records = d.build_batches(data, pools['report'], seed=0, epoch=0, canonical=True)
    train_keys = {data.train_dataset.grouped_split.groups[i].image_key for i in training}
    meta_keys, report_keys = ({r['image_key'] for r in records} for records in [meta_records, report_records])
    if len(meta_keys) != 512 or len(report_keys) != 512 or meta_keys & report_keys or train_keys & (meta_keys | report_keys):
        raise AssertionError('Train/meta/report image identities overlap')
    conditions = d.report_conditions(p)
    meta, report = d.MetaPool(meta_batches), d.ReportPool(report_batches, conditions)
    d.write_json(directory/'data_roles.json', dict(meta=meta_records, report=report_records,
        train_source_indices=training, report_partitions=conditions, disjoint_by_image_key=True,
        reused_historical_diagnostic_pool=True, independent_test_set=False))
    plans = make_plans(training, p)
    d.write_json(directory/'branch_plans.json', plans)
    d.write_json(directory/'training_config.json', config)
    d.write_json(directory/'device.json', info)
    gate = gate_protocol()
    probe = GradientReprobe(fixed_probe_batches(data, gate, directory), config, meta, gate)
    sync()  # Seal all stream identities, data roles and observation times before updates.
    model = d.CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
    policy = d.configure_clip_trainable_parameters(model, config['model']['trainable_policy'])
    if (policy['trainable_parameter_count'], policy['trainable_tensor_count']) != (10895617, 35):
        raise AssertionError('Source trainable policy differs')
    d.write_json(directory/'trainable_policy.json', policy)
    optimizer = d.build_clip_optimizer(model, config['optimizer'])
    scheduler = clip_train.build_scheduler(optimizer, warmup_steps=config['scheduler']['warmup_steps'],
                                           total_steps=p['scheduler_horizon'])
    rows = []
    for seed in p['training_seeds']:
        for step in p['checkpoint_steps']:
            relative = d.checkpoint_path(seed, step)
            path = source/relative
            if d.checksum(path) != p['checkpoints'][relative]:
                raise AssertionError(f'Checkpoint checksum mismatch: {relative}')
            initial = torch.load(path, map_location='cpu', weights_only=False)
            if (initial['training_seed'], initial['completed_updates'], initial['arm']) != (
                    seed, step, 'baseline' if step == 100 else 'alignment_gated'):
                raise AssertionError('Checkpoint identity mismatch')
            if initial['scheduler']['last_epoch'] != step:
                raise AssertionError('Saved scheduler horizon differs')
            for stream in p['stream_seeds']:
                folder = directory/f'seed_{seed}'/f'step_{step:06d}'/f'stream_{stream}'
                folder.mkdir(parents=True, exist_ok=False)
                batches, records = d.build_batches(data, [i for b in plans[f'{seed}:{step}:{stream}'] for i in b],
                    seed=seed, epoch=initial['epoch'], canonical=False)
                d.write_json(folder/'training_batches.json', dict(records=records, caption_epoch=initial['epoch'],
                    batch_size=64, literal_source_loader_continuation=False))
                sync()
                def commit(label, value):
                    arm, t = label
                    d.write_json(folder/f'probe_{arm}_{t:03d}.json', value)
                    sync()
                def save(arm, t, value):
                    d.write_json(folder/f'point_{arm}_{t:03d}.json', value)
                    sync()
                label = f'seed={seed} state={step} stream={stream}'
                print('CONTINUATION_PAIR:', label, flush=True)
                result = run_pair(model, optimizer, scheduler, initial, batches, config, probe, report,
                    coefficient=p['coefficient'], observation_steps=p['observation_steps'],
                    commit_probe=commit, save_point=save,
                    progress=lambda msg: print(f'TIMECOURSE: {label} {msg}', flush=True))
                result.update(training_seed=seed, checkpoint_step=step, stream_seed=stream,
                    source_checkpoint=relative, source_checkpoint_sha256=p['checkpoints'][relative])
                d.write_json(folder/'result.json', result)
                rows.append(result)
                sync()
                del batches, records, result
                gc.collect()
            del initial
            gc.collect()
            torch.cuda.empty_cache()
    summary = summarize(rows, p)
    if summary['optimizer_steps'] != p['optimizer_step_budget']:
        raise AssertionError('Bounded optimization budget violated')
    d.write_json(directory/'comparison.json', summary)
    d.write_json(directory/'completion.json', dict(status='complete', pairs=24, branches=48,
        optimizer_steps=2400, trial_updates=0, observation_steps=p['observation_steps'],
        diagnostic_only_reprobes=True, streams_committed_before_updates=True))
    sync()
    return summary
