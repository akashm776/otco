"""Frozen closed-loop alignment gate versus dose-matched randomized timing.

The random comparator receives only the gate's total exposure, never its scores
or reporting outcomes. It is a conditional timing control, not a deployable gate.
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

ARMS = ['gated', 'native', 'sustained', 'random_matched']


def load_protocol():
    extension = json.loads((d.ROOT/'configs/clip_policy_test.json').read_text())
    expected = dict(experiment='clip_policy_test_v1', training_seeds=[789, 2026, 31415],
        checkpoint_steps=[100, 500], stream_seeds=[2026093002, 2026093003],
        previous_stream_seeds=[2026092402, 2026092601, 2026092602], arms=ARMS, horizon=50,
        gate_refresh_offsets=[0, 10, 25], observation_steps=[0, 10, 25, 50],
        alignment_threshold=0.030847286945687016, permutation_seed=2026093004,
        primary='seed_mean_gated_minus_random_matched_report_loss_at_50',
        optimizer_step_budget=2400, rules_refitted=False, report_controls_treatment=False,
        independent_test_set=False, exploratory_followup=True)
    if extension != expected or extension['alignment_threshold'] != gate_protocol()['alignment_threshold']:
        raise ValueError('Frozen policy-test protocol changed')
    return {**d.load_protocol(), **extension}


def make_plans(training, p):
    plans, permutations = {}, {}
    for seed in p['training_seeds']:
        for step in p['checkpoint_steps']:
            seen = {d.state_digest(d.branch_plan(training, seed, step, {**p, 'branch_data_seed': u}))
                    for u in p['previous_stream_seeds']}
            for stream in p['stream_seeds']:
                key = f'{seed}:{step}:{stream}'
                plan = d.branch_plan(training, seed, step, {**p, 'branch_data_seed': stream})
                digest = d.state_digest(plan)
                if digest in seen:
                    raise AssertionError('Repeated continuation stream')
                seen.add(digest)
                plans[key] = plan
                permutations[key] = d.ranked(range(p['horizon']),
                    f"policy-timing-v1:{p['permutation_seed']}:{key}")
    return plans, permutations


def matched_schedule(actions, permutation):
    if (not actions or any(type(x) is not bool for x in actions)
            or sorted(permutation) != list(range(len(actions)))):
        raise ValueError('Invalid exposure schedule or permutation')
    selected = set(permutation[:sum(actions)])
    return [i in selected for i in range(len(actions))]


def state_stamp(model, optimizer, scheduler):
    return (d.live_digest(model, optimizer, scheduler),
            d.state_digest([p.grad for p in model.parameters()]),
            tuple(m.training for m in model.modules()))


def guarded(model, optimizer, scheduler, callback):
    before = state_stamp(model, optimizer, scheduler)
    value = callback()
    if state_stamp(model, optimizer, scheduler) != before:
        raise AssertionError('Observation or persistence changed training state, gradients, or modes')
    return value


class AlignmentGate:
    """Training-only archived probes. No reporting or meta pool access."""
    def __init__(self, batches, config, gate):
        self.batches, self.config, self.gate = batches, config, gate

    def measure(self, model):
        return measure_alignment(model, self.batches, self.config, self.gate['probe_branch_seed'])


def run_pair(model, optimizer, scheduler, initial, batches, config, probe, report, *,
             coefficient, threshold, refresh_offsets, observation_steps, permutation,
             commit, save_point=lambda *_: None, progress=lambda _: None):
    horizon = len(batches)
    if (not batches or list(refresh_offsets) != sorted(set(refresh_offsets))
            or refresh_offsets[0] != 0 or refresh_offsets[-1] >= horizon
            or list(observation_steps) != sorted(set(observation_steps))
            or observation_steps[0] != 0 or observation_steps[-1] != horizon
            or not math.isfinite(threshold) or not math.isfinite(coefficient) or coefficient <= 0):
        raise ValueError('Invalid fixed policy grid')
    matched_schedule([False]*horizon, permutation)  # Validate before any update.
    source_hash = d.state_digest(initial)
    frozen_hash = d.state_digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
    points, traces, actions, starts, decisions = {}, {}, {}, {}, []
    initial_report = None
    for arm in ARMS:
        d.restore(model, optimizer, scheduler, initial)
        starts[arm] = d.live_digest(model, optimizer, scheduler)
        if starts[arm] != starts['gated']:
            raise AssertionError('Branch initial states differ')
        schedule = (matched_schedule(actions['gated'], permutation) if arm == 'random_matched'
                    else [arm == 'sustained']*horizon)
        if arm != 'gated':
            guarded(model, optimizer, scheduler, lambda: commit(f'schedule_{arm}',
                dict(actions=schedule, dose=sum(schedule), derived_only_from_gate_dose=arm == 'random_matched')))
        points[arm], traces[arm], actions[arm] = {}, [], []
        active = False
        for offset in range(horizon+1):
            if arm == 'gated' and offset in refresh_offsets:
                measured = guarded(model, optimizer, scheduler, lambda: probe.measure(model))
                score = measured['mean_cosine']
                if not math.isfinite(score):
                    raise ValueError('Nonfinite alignment; no default gate action')
                active = score > threshold
                decision = dict(completed_updates=offset, measurement=measured,
                    threshold=threshold, active=active, state_sha256=d.live_digest(model, optimizer, scheduler))
                # Decision durable BEFORE its report and all affected optimizer steps.
                guarded(model, optimizer, scheduler,
                    lambda: commit(f'decision_{offset:03d}', decision))
                decisions.append(decision)
            if offset in observation_steps:
                evaluated = guarded(model, optimizer, scheduler, lambda: report.evaluate(model))
                if offset == 0:
                    if initial_report is None:
                        initial_report = evaluated
                    elif evaluated != initial_report:
                        raise AssertionError('No-update report replay differs')
                point = dict(report=evaluated, state_sha256=d.live_digest(model, optimizer, scheduler))
                points[arm][str(offset)] = point
                guarded(model, optimizer, scheduler, lambda: save_point(arm, offset, point))
                guarded(model, optimizer, scheduler, lambda: progress(f'{arm}: {offset}/{horizon}'))
            if offset == horizon:
                break
            action = active if arm == 'gated' else schedule[offset]
            actions[arm].append(action)
            trace = d.step_update(model, optimizer, scheduler, batches[offset], config,
                                  coefficient if action else 0.)
            traces[arm].append(trace)
            if arm != 'gated' and any(trace[k] != traces['gated'][offset][k]
                                     for k in ['learning_rates', 'rng_after_sha256']):
                raise AssertionError('Unmatched branch LR/RNG')
    if sum(actions['gated']) != sum(actions['random_matched']):
        raise AssertionError('Auxiliary exposure differs')
    if (d.state_digest(initial) != source_hash or d.state_digest(
            {n: p for n, p in model.named_parameters() if not p.requires_grad}) != frozen_hash):
        raise AssertionError('Source or frozen parameters mutated')
    differences = {}
    for time in observation_steps:
        q = str(time)
        gated = points['gated'][q]['report']
        differences[q] = {}
        for control in ARMS[1:]:
            other = points[control][q]['report']
            differences[q][control] = dict(report_loss=gated['native_loss']-other['native_loss'],
                pool_r1_pp=gated['pool_retrieval']['mean_r1_percent']-other['pool_retrieval']['mean_r1_percent'],
                partition_loss={k: statistics.fmean(v)-statistics.fmean(other['partition_losses'][k])
                                for k, v in gated['partition_losses'].items()})
    return dict(points=points, traces=traces, actions=actions, decisions=decisions, differences=differences,
        exposure={a: sum(v) for a, v in actions.items()},
        random_schedule_equals_gate=actions['random_matched'] == actions['gated'],
        audit=dict(actual_optimizer_steps=4*horizon, branch_start_hashes=starts, source_immutable=True,
            frozen_parameters_unchanged=True, observations_preserve_state=True, exact_no_update_report_replay=True,
            matched_rng_and_lrs=True, exposure_matched=True, reports_control_treatment=False))


def summarize(rows, p):
    expected = {(s, k, u) for s in p['training_seeds'] for k in p['checkpoint_steps'] for u in p['stream_seeds']}
    if len(rows) != len(expected) or {(r['training_seed'], r['checkpoint_step'], r['stream_seed']) for r in rows} != expected:
        raise ValueError('Incomplete or duplicate policy grid')
    per_seed = []
    for seed in p['training_seeds']:
        by_step = {}
        for step in p['checkpoint_steps']:
            group = [r for r in rows if (r['training_seed'], r['checkpoint_step']) == (seed, step)]
            by_step[str(step)] = {str(time): {control: {metric: statistics.fmean(
                r['differences'][str(time)][control][metric] for r in group)
                for metric in ['report_loss', 'pool_r1_pp']} for control in ARMS[1:]}
                for time in p['observation_steps']}
        per_seed.append(dict(training_seed=seed, per_starting_step=by_step,
            primary=statistics.fmean(by_step[str(k)][str(p['horizon'])]['random_matched']['report_loss']
                                     for k in p['checkpoint_steps'])))
    return dict(primary=p['primary'], primary_value=statistics.fmean(s['primary'] for s in per_seed),
        per_seed=per_seed, pairs=len(rows), branches=4*len(rows), training_seed_count=len(per_seed),
        optimizer_steps=sum(r['audit']['actual_optimizer_steps'] for r in rows),
        exposure=[dict(training_seed=r['training_seed'], checkpoint_step=r['checkpoint_step'],
            stream_seed=r['stream_seed'], counts=r['exposure'],
            random_schedule_equals_gate=r['random_schedule_equals_gate']) for r in rows],
        interpretation='Gated minus comparator; negative loss helps, positive R1 helps. '
        'Average streams within checkpoints, checkpoints within seeds, then three seeds equally. '
        'Random timing is conditional on gate exposure; one precommitted permutation per pair. '
        'Equal total dose is guaranteed only at update 50, not intermediate reports. '
        'Reused checkpoints/report data and post-hoc policy choice: exploratory, not independent confirmation. '
        'No mediation or OT-weighting superiority claim. A timing advantage alone need not beat native training.')


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
    report_batches, report_records = d.build_batches(data, pools['report'], seed=0, epoch=0, canonical=True)
    train_keys = {data.train_dataset.grouped_split.groups[i].image_key for i in training}
    meta_keys = {data.train_dataset.grouped_split.groups[i].image_key for i in pools['meta']}
    report_keys = {r['image_key'] for r in report_records}
    if len(meta_keys) != 512 or len(report_keys) != 512 or meta_keys & report_keys or train_keys & (meta_keys | report_keys):
        raise AssertionError('Train/meta/report image identities overlap')
    conditions = d.report_conditions(p)
    report = d.ReportPool(report_batches, conditions)
    d.write_json(directory/'data_roles.json', dict(report=report_records, meta_reserved_unused=pools['meta'],
        train_source_indices=training, report_partitions=conditions, disjoint_by_image_key=True,
        historical_probes_are_training_data=True, probe_continuation_overlap_allowed=True,
        reused_historical_diagnostic_pool=True, independent_test_set=False))
    plans, permutations = make_plans(training, p)
    d.write_json(directory/'branch_plans.json', plans)
    d.write_json(directory/'timing_permutations.json', permutations)
    d.write_json(directory/'training_config.json', config)
    d.write_json(directory/'device.json', info)
    gate = gate_protocol()
    probe = AlignmentGate(fixed_probe_batches(data, gate, directory), config, gate)
    sync()  # All streams, permutations, reporting partitions and rule fixed before updates.
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
            if (initial['training_seed'], initial['completed_updates'], initial['arm'], initial['scheduler']['last_epoch']) != (
                    seed, step, 'baseline' if step == 100 else 'alignment_gated', step):
                raise AssertionError('Checkpoint identity/scheduler mismatch')
            for stream in p['stream_seeds']:
                key = f'{seed}:{step}:{stream}'
                folder = directory/f'seed_{seed}'/f'step_{step:06d}'/f'stream_{stream}'
                folder.mkdir(parents=True, exist_ok=False)
                batches, records = d.build_batches(data, [i for b in plans[key] for i in b],
                    seed=seed, epoch=initial['epoch'], canonical=False)
                d.write_json(folder/'training_batches.json', dict(records=records, caption_epoch=initial['epoch'],
                    batch_size=64, literal_source_loader_continuation=False))
                sync()
                def commit(label, value):
                    d.write_json(folder/f'{label}.json', value)
                    sync()
                def save(arm, offset, value):
                    d.write_json(folder/f'point_{arm}_{offset:03d}.json', value)
                    sync()
                label = f'seed={seed} state={step} stream={stream}'
                print('POLICY_PAIR:', label, flush=True)
                result = run_pair(model, optimizer, scheduler, initial, batches, config, probe, report,
                    coefficient=p['coefficient'], threshold=p['alignment_threshold'],
                    refresh_offsets=p['gate_refresh_offsets'], observation_steps=p['observation_steps'],
                    permutation=permutations[key], commit=commit, save_point=save,
                    progress=lambda message: print(f'POLICY: {label} {message}', flush=True))
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
        raise AssertionError('Bounded policy budget violated')
    d.write_json(directory/'comparison.json', summary)
    d.write_json(directory/'completion.json', dict(status='complete', pairs=12, branches=48,
        optimizer_steps=2400, trial_updates=0, observation_steps=p['observation_steps'],
        exposure_matched=True, reports_control_treatment=False, streams_committed_before_updates=True))
    sync()
    return summary
