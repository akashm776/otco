"""Rollback-only one-update tests on reproduced native/treated histories.

The estimand is marginal auxiliary utility from each CURRENT optimizer state,
not cumulative sustained-vs-native performance. Trials never enter continuation.
"""
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import statistics

import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_continuation_timecourse as t
from src import clip_train
from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol
from src.clip_paired_updates import cosine


def load_protocol():
    parent = t.load_protocol()
    extension = json.loads((d.ROOT/'configs/clip_marginal_utility.json').read_text())
    expected = dict(experiment='clip_marginal_utility_v1',
        parent_protocol='configs/clip_continuation_timecourse.json',
        training_seeds=[789, 2026, 31415], checkpoint_steps=[100, 500],
        stream_seeds=[2026092601, 2026092602], arms=t.ARMS, horizon=50,
        observation_steps=[0, 10, 25, 50], trial_batch_count=2, trial_data_seed=2026093001,
        primary_time=25,
        primary='seed_mean_treated_history_minus_native_history_marginal_report_loss_at_25',
        continuation_updates=1200, trial_updates=336, optimizer_step_budget=1536,
        exploratory_followup=True, reference_file='configs/clip_marginal_utility_reference.json',
        rules_refitted=False, report_controls_treatment=False, independent_test_set=False)
    if extension != expected:
        raise ValueError('Frozen marginal-utility protocol changed')
    return {**parent, **extension}


def trial_plan(training, continuation, historical, seed, step, stream, p):
    """Two fixed batches per pair, unused by that pair's continuation/probes."""
    candidates = set(training) - {i for b in continuation for i in b} - set(historical)
    needed = p['trial_batch_count'] * p['batch_size']
    if len(candidates) < needed:
        raise ValueError('Insufficient disjoint trial training images')
    selected = d.ranked(candidates, f"marginal-v1:{p['trial_data_seed']}:{seed}:{step}:{stream}")[:needed]
    return [selected[i:i+p['batch_size']] for i in range(0, needed, p['batch_size'])]


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


def snapshot(model, optimizer, scheduler):
    return cpu_copy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), **d.rng_state(),
        gradients=[p.grad for p in model.parameters()], modes=[m.training for m in model.modules()]))


def restore_snapshot(model, optimizer, scheduler, saved):
    d.restore(model, optimizer, scheduler, saved)
    for parameter, gradient in zip(model.parameters(), saved['gradients']):
        parameter.grad = None if gradient is None else gradient.to(parameter.device).clone()
    for module, mode in zip(model.modules(), saved['modes']):
        module.training = mode


def assert_snapshot(model, optimizer, scheduler, saved):
    if d.state_digest(snapshot(model, optimizer, scheduler)) != d.state_digest(saved):
        raise AssertionError('Audit changed live state, gradients, or modes')


def marginal_audit(model, optimizer, scheduler, batches, config, meta, probe, report,
                   *, coefficient, commit_predictions):
    """2 updates/batch; gradient predictions committed before ANY report access.

Trials use the actual clipped AdamW step, LR scheduler and logit clamp. Restore
the full state even if a probe, report or durable-write callback fails.
"""
    saved = snapshot(model, optimizer, scheduler)
    saved_hash = d.state_digest(saved)
    anchor_hash = d.live_digest(model, optimizer, scheduler)
    try:
        historical = probe.measure(model)
        meta_gradient, meta_loss = meta.gradient(model)
        scores = []
        for batch in batches:
            native, auxiliary = d.training_gradients(model, batch, config)
            values = dict(raw_cosine=cosine(native, auxiliary), meta_cosine=cosine(meta_gradient, auxiliary))
            if any(v is None or not math.isfinite(v) for v in values.values()):
                raise ValueError('Nonfinite or degenerate marginal prediction')
            scores.append(dict(**values, decisions=dict(raw=values['raw_cosine'] > 0,
                meta=values['meta_cosine'] > 0, historical=historical['historical_above_threshold'])))
        assert_snapshot(model, optimizer, scheduler, saved)
        predictions = dict(historical=historical, meta_native_loss=meta_loss, batches=scores,
                           anchor_state_sha256=anchor_hash, trial_rng_sha256=d.state_digest(d.rng_state()))
        commit_predictions(predictions)
        assert_snapshot(model, optimizer, scheduler, saved)
        trials = []
        for index, batch in enumerate(batches):
            outcomes, traces, hashes = {}, {}, {}
            for action in ['native', 'auxiliary']:
                restore_snapshot(model, optimizer, scheduler, saved)
                assert_snapshot(model, optimizer, scheduler, saved)
                traces[action] = d.step_update(model, optimizer, scheduler, batch, config,
                                               coefficient if action == 'auxiliary' else 0.)
                hashes[action] = d.live_digest(model, optimizer, scheduler)
                after = snapshot(model, optimizer, scheduler)
                outcomes[action] = report.evaluate(model)
                assert_snapshot(model, optimizer, scheduler, after)
                del after
            if any(traces['native'][k] != traces['auxiliary'][k] for k in ['learning_rates', 'rng_after_sha256']):
                raise AssertionError('Marginal actions have unmatched LR/RNG')
            n, a = outcomes['native'], outcomes['auxiliary']
            trials.append(dict(batch_index=index, reports=outcomes, traces=traces, state_hashes=hashes,
                report_loss=a['native_loss']-n['native_loss'],
                pool_r1_pp=a['pool_retrieval']['mean_r1_percent']-n['pool_retrieval']['mean_r1_percent'],
                partition_loss={k: statistics.fmean(v)-statistics.fmean(n['partition_losses'][k])
                                for k, v in a['partition_losses'].items()}))
        if d.state_digest(saved) != saved_hash:
            raise AssertionError('Rollback snapshot mutated')
        result = dict(predictions=predictions, trials=trials,
            mean_marginal_report_loss=statistics.fmean(x['report_loss'] for x in trials),
            mean_marginal_pool_r1_pp=statistics.fmean(x['pool_r1_pp'] for x in trials),
            trial_updates=2*len(batches), matched_actions=True, state_restored=True)
    finally:
        restore_snapshot(model, optimizer, scheduler, saved)
        assert_snapshot(model, optimizer, scheduler, saved)
    return result


def run_pair(model, optimizer, scheduler, initial, batches, trial_batches, config, meta, probe, report,
             *, coefficient, observation_steps, commit_predictions, save_point=lambda *_: None,
             expected_hashes=None, progress=lambda _: None):
    times = list(observation_steps)
    if not trial_batches or not batches or times != sorted(set(times)) or times[0] != 0 or times[-1] != len(batches):
        raise ValueError('Invalid fixed marginal grid')
    source_hash = d.state_digest(initial)
    frozen = d.state_digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
    points, traces = {}, {}

    def audit(arm, offset):
        current = d.live_digest(model, optimizer, scheduler)
        label = f'{arm}:{offset}'
        if expected_hashes is not None and current != expected_hashes[label]:
            raise AssertionError(f'Continuation replay differs from September 29 at {label}; stop, do not reinterpret as replication')
        point = marginal_audit(model, optimizer, scheduler, trial_batches, config, meta, probe, report,
            coefficient=coefficient, commit_predictions=lambda value: commit_predictions((arm, offset), value))
        point['anchor_state_sha256'] = current
        before_save = snapshot(model, optimizer, scheduler)
        save_point(arm, offset, point)
        assert_snapshot(model, optimizer, scheduler, before_save)
        progress(f'{arm}: audited {offset}/{len(batches)}')
        assert_snapshot(model, optimizer, scheduler, before_save)
        return point

    d.restore(model, optimizer, scheduler, initial)
    common = audit('common', 0)
    for arm in t.ARMS:
        d.restore(model, optimizer, scheduler, initial)
        if d.live_digest(model, optimizer, scheduler) != common['anchor_state_sha256']:
            raise AssertionError('History start differs')
        points[arm], traces[arm] = {'0': common}, []
        for offset, batch in enumerate(batches, 1):
            trace = d.step_update(model, optimizer, scheduler, batch, config, coefficient if arm == 'sustained' else 0.)
            traces[arm].append(trace)
            if arm == 'sustained' and any(trace[k] != traces['native'][offset-1][k]
                                          for k in ['learning_rates', 'rng_after_sha256']):
                raise AssertionError('Histories have unmatched LR/RNG')
            if offset in times:
                points[arm][str(offset)] = audit(arm, offset)
    if d.state_digest(initial) != source_hash or d.state_digest(
            {n: p for n, p in model.named_parameters() if not p.requires_grad}) != frozen:
        raise AssertionError('Source or frozen parameters mutated')
    trial_count = (1+2*(len(times)-1))*2*len(trial_batches)
    return dict(points=points, traces=traces,
        history_contrast={str(time): points['sustained'][str(time)]['mean_marginal_report_loss']
                                    - points['native'][str(time)]['mean_marginal_report_loss'] for time in times},
        audit=dict(continuation_updates=2*len(batches), trial_updates=trial_count,
            actual_optimizer_steps=2*len(batches)+trial_count, trials_rolled_back=True,
            matched_rng_and_lrs=True, source_immutable=True, frozen_parameters_unchanged=True,
            exact_reference_replay=expected_hashes is not None, reports_control_treatment=False))


def summarize(rows, p):
    expected = {(s, k, u) for s in p['training_seeds'] for k in p['checkpoint_steps'] for u in p['stream_seeds']}
    if len(rows) != len(expected) or {(r['training_seed'], r['checkpoint_step'], r['stream_seed']) for r in rows} != expected:
        raise ValueError('Incomplete or duplicate marginal grid')
    per_seed = []
    for seed in p['training_seeds']:
        by_step = {}
        for step in p['checkpoint_steps']:
            group = [r for r in rows if (r['training_seed'], r['checkpoint_step']) == (seed, step)]
            course = {}
            for time in p['observation_steps']:
                q = str(time)
                effects = {arm: statistics.fmean(r['points'][arm][q]['mean_marginal_report_loss'] for r in group)
                           for arm in t.ARMS}
                course[q] = dict(marginal_report_loss=effects,
                    treated_minus_native_history=effects['sustained']-effects['native'],
                    marginal_pool_r1_pp={arm: statistics.fmean(r['points'][arm][q]['mean_marginal_pool_r1_pp'] for r in group)
                                         for arm in t.ARMS})
            by_step[str(step)] = course
        per_seed.append(dict(training_seed=seed, per_starting_step=by_step,
            primary=statistics.fmean(by_step[str(step)][str(p['primary_time'])]['treated_minus_native_history']
                                     for step in p['checkpoint_steps'])))
    descriptives = {}
    for rule in ['historical', 'raw', 'meta']:
        checks = []
        for r in rows:
            anchors = [r['points']['native']['0']] + [r['points'][arm][str(time)]
                for arm in t.ARMS for time in p['observation_steps'] if time]
            for anchor in anchors:
                checks.extend((score['decisions'][rule], trial['report_loss'])
                              for score, trial in zip(anchor['predictions']['batches'], anchor['trials']))
        outside = [(a, v) for a, v in checks if abs(v) > p['sensitivity_absolute_band']]
        descriptives[rule] = dict(trials=len(checks), correct=sum(a == (v < 0) for a, v in checks),
            outside_band=len(outside), correct_outside_band=sum(a == (v < 0) for a, v in outside),
            mean_decision_regret=statistics.fmean((v if a else 0.)-min(v, 0.) for a, v in checks),
            warning='Pooled correlated descriptive trials, NOT independent replications or a fitted gate')
    return dict(primary=p['primary'], primary_value=statistics.fmean(x['primary'] for x in per_seed),
        per_seed=per_seed, predictor_descriptives=descriptives, pairs=len(rows), training_seed_count=len(per_seed),
        continuation_updates=sum(r['audit']['continuation_updates'] for r in rows),
        trial_updates=sum(r['audit']['trial_updates'] for r in rows),
        optimizer_steps=sum(r['audit']['actual_optimizer_steps'] for r in rows),
        interpretation='Negative marginal loss means the NEXT auxiliary step helps versus a native step. '
        'Positive primary means auxiliary marginal utility is worse after sustained history at update 25. '
        'All trial branches roll back; batch identities are matched across histories/times. '
        'Exploratory reused histories/data; three seed units; no mediation, optimal policy, independent-test or OT-superiority claim.')


def run(source, directory, sync, p=None):
    from transformers import AutoProcessor
    p = load_protocol() if p is None else p
    source, directory = Path(source), Path(directory)
    reference = json.loads((d.ROOT/p['reference_file']).read_text())
    d.write_json(directory/'protocol.json', p)
    d.write_json(directory/'replay_reference.json', reference)
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
    plans = t.make_plans(training, p)
    historical_ids = {i for batch in json.loads((d.ROOT/
        'experiment_results/clip_paired_updates_2026-09/training_source_indices.json').read_text()) for i in batch}
    trials = {}
    for seed in p['training_seeds']:
        for step in p['checkpoint_steps']:
            for stream in p['stream_seeds']:
                key = f'{seed}:{step}:{stream}'
                if d.state_digest(plans[key]) != reference['pairs'][key]['plan_sha256']:
                    raise AssertionError('Replay continuation identities differ')
                trials[key] = trial_plan(training, plans[key], historical_ids, seed, step, stream, p)
    d.write_json(directory/'branch_plans.json', plans)
    d.write_json(directory/'trial_plans.json', trials)
    d.write_json(directory/'training_config.json', config)
    d.write_json(directory/'device.json', info)
    gate = gate_protocol()
    probe_batches = fixed_probe_batches(data, gate, directory)
    probe = t.GradientReprobe(probe_batches, config, meta, gate)
    sync()  # Freeze all trial/continuation identities before training or outcomes.
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
                trial_batches, trial_records = d.build_batches(data, [i for b in trials[key] for i in b],
                    seed=seed, epoch=initial['epoch'], canonical=False)
                continuation_keys = {r['image_key'] for r in records}
                trial_keys = {r['image_key'] for r in trial_records}
                historical_keys = {data.train_dataset.grouped_split.groups[i].image_key for i in historical_ids}
                if len(trial_keys) != 128 or trial_keys & (continuation_keys | historical_keys | meta_keys | report_keys):
                    raise AssertionError('Trial image identities overlap excluded roles')
                d.write_json(folder/'training_batches.json', dict(records=records, trial_records=trial_records,
                    caption_epoch=initial['epoch'], batch_size=64, trials_disjoint_from_continuation_and_probes=True))
                sync()
                def commit(label, value):
                    arm, offset = label
                    d.write_json(folder/f'predictions_{arm}_{offset:03d}.json', value)
                    sync()
                def save(arm, offset, value):
                    d.write_json(folder/f'marginal_{arm}_{offset:03d}.json', value)
                    sync()
                label = f'seed={seed} state={step} stream={stream}'
                print('MARGINAL_PAIR:', label, flush=True)
                result = run_pair(model, optimizer, scheduler, initial, batches, trial_batches, config, meta, probe, report,
                    coefficient=p['coefficient'], observation_steps=p['observation_steps'],
                    commit_predictions=commit, save_point=save,
                    expected_hashes=reference['pairs'][key]['anchor_hashes'],
                    progress=lambda message: print(f'MARGINAL: {label} {message}', flush=True))
                result.update(training_seed=seed, checkpoint_step=step, stream_seed=stream,
                    source_checkpoint=relative, source_checkpoint_sha256=p['checkpoints'][relative])
                d.write_json(folder/'result.json', result)
                rows.append(result)
                sync()
                del batches, records, trial_batches, trial_records, result
                gc.collect()
            del initial
            gc.collect()
            torch.cuda.empty_cache()
    summary = summarize(rows, p)
    if (summary['optimizer_steps'], summary['continuation_updates'], summary['trial_updates']) != (1536, 1200, 336):
        raise AssertionError('Bounded marginal budget violated')
    d.write_json(directory/'comparison.json', summary)
    d.write_json(directory/'completion.json', dict(status='complete', pairs=12, history_branches=24,
        audited_anchors=84, matched_trial_pairs=168, continuation_updates=1200, trial_updates=336,
        optimizer_steps=1536, trial_steps_rolled_back=True, exact_reference_replay=True,
        observation_steps=p['observation_steps'], reports_control_treatment=False))
    sync()
    return summary
