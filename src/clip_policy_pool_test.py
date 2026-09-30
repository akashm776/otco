"""Exact recorded-policy replay, full-pool loss and fixed negative-pool sensitivity."""
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_policy_test as policy
from src import clip_train
from scripts.analyze_clip_policy_robustness import SPEC, checked_json


def load_protocol():
    extension = json.loads((d.ROOT/'configs/clip_policy_pool_test.json').read_text())
    expected = dict(experiment='clip_policy_pool_test_v1', policy_source_run=SPEC['source_run'],
        policy_backup_manifest_sha256=SPEC['backup_manifest_sha256'],
        training_seeds=[789, 2026, 31415], checkpoint_steps=[100, 500],
        stream_seeds=[2026093002, 2026093003], arms=policy.ARMS, horizon=50,
        replay_check_offsets=[0, 10, 25, 50], report_size=512,
        new_partition_seed_start=2026093010, new_partition_count=32, partition_batch_size=64,
        primary='seed_mean_gated_minus_random_matched_full512_loss_at_50', optimizer_step_budget=2400,
        gate_decisions_recomputed=False, rules_refitted=False, independent_test_set=False,
        exploratory_followup=True, save_endpoint_embeddings=True)
    if extension != expected:
        raise ValueError('Frozen evaluation-pool protocol changed')
    return {**d.load_protocol(), **extension}


def make_conditions(p):
    if p['report_size'] % p['partition_batch_size'] or p['new_partition_count'] != 32:
        raise ValueError('Invalid reporting grid')
    conditions, seen = {}, set()
    for seed in range(p['new_partition_seed_start'], p['new_partition_seed_start']+p['new_partition_count']):
        if seed in p['report_partition_seeds']:
            raise ValueError('Reused reporting partition seed')
        order = d.ranked(range(p['report_size']), f'report-partition-v1:{seed}')
        batches = [order[i:i+p['partition_batch_size']] for i in range(0, len(order), p['partition_batch_size'])]
        # Batch and within-batch ordering do not define a new negative pool.
        grouping = tuple(sorted(tuple(sorted(batch)) for batch in batches))
        if grouping in seen:
            raise ValueError('Duplicate negative-pool grouping')
        seen.add(grouping)
        conditions[str(seed)] = dict(batches=batches)
    old = d.report_conditions(p)
    for value in old.values():
        grouping = tuple(sorted(tuple(sorted(batch)) for batch in value['batches']))
        if grouping in seen:
            raise ValueError('New grouping duplicates an original partition')
    return conditions


def validate_encoded(encoded, size=None):
    images, texts, scale = encoded
    if (images.ndim != 2 or texts.shape != images.shape or len(images) < 2
            or (size is not None and len(images) != size)
            or images.dtype != torch.float32 or texts.dtype != torch.float32
            or images.device.type != 'cpu' or texts.device.type != 'cpu'
            or not torch.isfinite(images).all() or not torch.isfinite(texts).all()
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid cached endpoint features or scale')


def evaluate_encoded(encoded, conditions):
    validate_encoded(encoded)
    images, texts, scale = encoded
    count = len(images)
    for value in conditions.values():
        flat = [i for b in value['batches'] for i in b]
        if sorted(flat) != list(range(count)) or any(not b for b in value['batches']):
            raise ValueError('Partition must cover each paired row exactly once')
        if len({len(b) for b in value['batches']}) != 1:
            raise ValueError('Unequal reporting batch sizes')
    scores = texts.double() @ images.double().T
    full_loss = float(d.native_clip_contrastive_loss(scores * scale))
    truth = torch.arange(count)
    retrieval = dict(text_to_image_r1_percent=100*float((scores.argmax(1) == truth).double().mean()),
                     image_to_text_r1_percent=100*float((scores.argmax(0) == truth).double().mean()))
    retrieval['mean_r1_percent'] = statistics.fmean(retrieval.values())
    losses = d.partition_losses(encoded, conditions)
    if not all(math.isfinite(x) for x in [full_loss, *[v for batch in losses.values() for v in batch]]):
        raise ValueError('Nonfinite endpoint evaluation')
    return dict(full_pool_loss=full_loss, pool_retrieval=retrieval, partition_losses=losses,
                partition_means={k: statistics.fmean(v) for k, v in losses.items()},
                features_sha256=d.feature_hash(encoded), learned_logit_scale=scale)


def save_cache(path, encoded, row_order_sha256):
    """Numerical NPZ only; no pickles, captions, images, model or optimizer weights."""
    validate_encoded(encoded)
    path = Path(path)
    images, texts, scale = encoded
    with path.open('xb') as handle:
        np.savez_compressed(handle, images=images.numpy(), texts=texts.numpy(),
                            scale=np.asarray(scale, dtype=np.float64))
    metadata = dict(schema='clip_endpoint_features_v1', rows=len(images), dimension=images.shape[1],
        dtype='float32', feature_sha256=d.feature_hash(encoded),
        npz_sha256=d.checksum(path), row_order_sha256=row_order_sha256)
    load_cache(path, metadata)  # Closed-file checksum and exact feature round trip.
    d.write_json(path.with_suffix('.json'), metadata)
    return metadata


def load_cache(path, metadata):
    if d.checksum(path) != metadata['npz_sha256']:
        raise ValueError('Endpoint cache checksum mismatch')
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'images', 'texts', 'scale'} or arrays['scale'].shape != ():
            raise ValueError('Unexpected cache layout')
        encoded = (torch.from_numpy(arrays['images'].copy()), torch.from_numpy(arrays['texts'].copy()),
                   float(arrays['scale']))
    validate_encoded(encoded, metadata['rows'])
    if encoded[0].shape[1] != metadata['dimension'] or d.feature_hash(encoded) != metadata['feature_sha256']:
        raise ValueError('Endpoint cache feature hash/shape mismatch')
    return encoded


def replay_branch(model, optimizer, scheduler, initial, batches, config, reference, arm, coefficient,
                  *, checkpoint_offsets, progress=lambda _: None):
    actions = reference['actions'][arm]
    if len(actions) != len(batches) or any(type(v) is not bool for v in actions):
        raise ValueError('Invalid recorded treatment schedule')
    if len(reference['traces'][arm]) != len(batches):
        raise ValueError('Incomplete reference traces')
    source_hash = d.state_digest(initial)
    frozen_hash = d.state_digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
    d.restore(model, optimizer, scheduler, initial)
    def check(offset):
        current = d.live_digest(model, optimizer, scheduler)
        if current != reference['points'][arm][str(offset)]['state_sha256']:
            raise AssertionError(f'Exact policy replay mismatch: {arm} offset {offset}; no new evaluation accepted')
        return current
    hashes, traces = {'0': check(0)}, []
    for offset, batch in enumerate(batches, 1):
        trace = d.step_update(model, optimizer, scheduler, batch, config,
                              coefficient if actions[offset-1] else 0.)
        expected = reference['traces'][arm][offset-1]
        if any(trace[k] != expected[k] for k in ['coefficient', 'learning_rates', 'rng_after_sha256']):
            raise AssertionError(f'Recorded action/LR/RNG differs: {arm} offset {offset}')
        traces.append(trace)
        if offset in checkpoint_offsets:
            hashes[str(offset)] = check(offset)
            policy.guarded(model, optimizer, scheduler, lambda: progress(f'{arm}: replay verified {offset}/{len(batches)}'))
    final_hash = check(len(batches))
    if (d.state_digest(initial) != source_hash or frozen_hash != d.state_digest(
            {n: p for n, p in model.named_parameters() if not p.requires_grad})):
        raise AssertionError('Source or frozen parameters changed')
    return dict(actual_optimizer_steps=len(batches), exact_reference_replay=True,
                source_immutable=True, frozen_parameters_unchanged=True, gate_decisions_recomputed=False,
                checked_state_hashes=hashes, final_state_sha256=final_hash, traces=traces)


def encode_verified(model, optimizer, scheduler, batches, original_report, old_conditions):
    def encode():
        with d.observe(model):
            return d.encode_cached(model, batches)
    encoded = policy.guarded(model, optimizer, scheduler, encode)
    if d.feature_hash(encoded) != original_report['features_sha256']:
        raise AssertionError('Archived endpoint feature hash differs; stop before new evaluation')
    measured = evaluate_encoded(encoded, old_conditions)
    if (measured['partition_losses'] != original_report['partition_losses']
            or measured['pool_retrieval'] != original_report['pool_retrieval']
            or statistics.fmean(x for v in measured['partition_losses'].values() for x in v)
                != original_report['native_loss']):
        raise AssertionError('Original reporting endpoint did not replay exactly')
    return encoded


def summarize(rows, p):
    expected = {(s, k, u) for s in p['training_seeds'] for k in p['checkpoint_steps'] for u in p['stream_seeds']}
    if len(rows) != len(expected) or {(r['training_seed'], r['checkpoint_step'], r['stream_seed']) for r in rows} != expected:
        raise ValueError('Incomplete or duplicate endpoint grid')
    partition_names = list(make_conditions(p))
    comparisons = {}
    for control in policy.ARMS[1:]:
        per_seed = []
        for seed in p['training_seeds']:
            states = {}
            for step in p['checkpoint_steps']:
                group = [r for r in rows if (r['training_seed'], r['checkpoint_step']) == (seed, step)]
                def contrast(row, metric, part=None):
                    a, b = row['endpoints']['gated'], row['endpoints'][control]
                    if part is not None:
                        return a['partition_means'][part]-b['partition_means'][part]
                    if metric == 'pool_r1_pp':
                        return a['pool_retrieval']['mean_r1_percent']-b['pool_retrieval']['mean_r1_percent']
                    return a[metric]-b[metric]
                states[str(step)] = dict(
                    full_pool_loss=statistics.fmean(contrast(r, 'full_pool_loss') for r in group),
                    pool_r1_pp=statistics.fmean(contrast(r, 'pool_r1_pp') for r in group),
                    partitions={part: statistics.fmean(contrast(r, None, part) for r in group) for part in partition_names})
            per_seed.append(dict(training_seed=seed, per_starting_step=states,
                full_pool_loss=statistics.fmean(x['full_pool_loss'] for x in states.values()),
                pool_r1_pp=statistics.fmean(x['pool_r1_pp'] for x in states.values()),
                partitions={part: statistics.fmean(x['partitions'][part] for x in states.values()) for part in partition_names}))
        partition_effects = {part: statistics.fmean(x['partitions'][part] for x in per_seed) for part in partition_names}
        vals = list(partition_effects.values())
        comparisons[control] = dict(full_pool_loss=statistics.fmean(x['full_pool_loss'] for x in per_seed),
            pool_r1_pp=statistics.fmean(x['pool_r1_pp'] for x in per_seed), per_seed=per_seed,
            partition_effects=partition_effects,
            partition_sensitivity=dict(mean=statistics.fmean(vals), min=min(vals), max=max(vals),
                favorable=sum(v<0 for v in vals), tied=sum(v==0 for v in vals), unfavorable=sum(v>0 for v in vals),
                note='32 paired negative-pool groupings on the SAME images, not independent seed replications or a confidence interval'))
    return dict(primary=p['primary'], primary_value=comparisons['random_matched']['full_pool_loss'],
        comparisons=comparisons, pairs=len(rows), branches=4*len(rows), training_seed_count=len(p['training_seeds']),
        optimizer_steps=sum(v['actual_optimizer_steps'] for r in rows for v in r['replay'].values()),
        interpretation='Gated minus control at the exact archived endpoints. Negative loss helps; positive R1 helps. '
        'Full-512 loss uses a different negative pool from batch-64 loss: compare paired directions, not absolute magnitudes. '
        'Average streams within checkpoints, checkpoints within seeds, seeds equally. Three reused seeds; '
        'no independent-test, statistical significance, new training replication, fitted-policy or OT-superiority claim.')


def run(source, directory, sync, reference_files, provenance, p=None):
    from transformers import AutoProcessor
    p = load_protocol() if p is None else p
    source, directory = Path(source), Path(directory)
    read = lambda name: checked_json(reference_files[name])
    archived_p = read('protocol.json')
    conditions = make_conditions(p)
    old_conditions = read('data_roles.json')['report_partitions']
    d.write_json(directory/'protocol.json', p)
    d.write_json(directory/'new_report_partitions.json', conditions)
    d.write_json(directory/'original_report_partitions.json', old_conditions)
    d.write_json(directory/'policy_reference_provenance.json', provenance)
    clip_train.set_global_seed(42)
    config = clip_train.load_training_config(d.ROOT/'configs/hf_cub200_clip_vit_b32_baseline.yaml')
    config['runtime']['num_workers'] = 0
    if config != read('training_config.json'):
        raise AssertionError('Training configuration differs from policy run')
    device, info = clip_train.resolve_device(config['runtime'])
    processor = AutoProcessor.from_pretrained(config['model']['checkpoint'])
    data = clip_train.build_clip_training_data(config, processor, pin_memory=False)
    training = data.train_dataset.source_indices
    holdout = json.loads((d.ROOT/config['dataset']['diagnostic_holdout_indices']).read_text())
    if len(training) != 4970 or len(data.train_loader) != 77:
        raise AssertionError('Training population changed')
    pools = d.split_pool(training, holdout, p)
    report_batches, records = d.build_batches(data, pools['report'], seed=0, epoch=0, canonical=True)
    roles = read('data_roles.json')
    if records != roles['report'] or list(training) != roles['train_source_indices'] or pools['meta'] != roles['meta_reserved_unused']:
        raise AssertionError('Training/report/meta identities or reporting captions differ')
    train_keys = {data.train_dataset.grouped_split.groups[i].image_key for i in training}
    meta_keys = {data.train_dataset.grouped_split.groups[i].image_key for i in pools['meta']}
    report_keys = {r['image_key'] for r in records}
    if len(meta_keys) != 512 or len(report_keys) != 512 or meta_keys & report_keys or train_keys & (meta_keys | report_keys):
        raise AssertionError('Train/meta/report image identities overlap')
    row_order = [hashlib.sha256(r['image_key'].encode()).hexdigest() for r in records]
    row_order_sha = hashlib.sha256(json.dumps(row_order).encode()).hexdigest()
    d.write_json(directory/'report_row_order.json', dict(image_key_sha256=row_order, row_order_sha256=row_order_sha,
        canonical_first_captions=True, rows=512, meta_reserved_unused=True, independent_test_set=False))
    plans, permutations = policy.make_plans(training, archived_p)
    if plans != read('branch_plans.json') or permutations != read('timing_permutations.json'):
        raise AssertionError('Archived continuation plans differ')
    d.write_json(directory/'branch_plans.json', plans)
    d.write_json(directory/'training_config.json', config)
    d.write_json(directory/'device.json', info)
    sync()  # Commit all evaluation groupings before the first replay or new outcome.
    model = d.CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
    trainable = d.configure_clip_trainable_parameters(model, config['model']['trainable_policy'])
    if (trainable['trainable_parameter_count'], trainable['trainable_tensor_count']) != (10895617, 35):
        raise AssertionError('Source trainable policy differs')
    optimizer = d.build_clip_optimizer(model, config['optimizer'])
    scheduler = clip_train.build_scheduler(optimizer, warmup_steps=config['scheduler']['warmup_steps'], total_steps=p['scheduler_horizon'])
    rows = []
    for seed in p['training_seeds']:
        for step in p['checkpoint_steps']:
            relative = d.checkpoint_path(seed, step)
            path = source/relative
            if d.checksum(path) != p['checkpoints'][relative]:
                raise AssertionError('Source checkpoint checksum differs')
            initial = torch.load(path, map_location='cpu', weights_only=False)
            if (initial['training_seed'], initial['completed_updates'], initial['arm'], initial['scheduler']['last_epoch']) != (
                    seed, step, 'baseline' if step == 100 else 'alignment_gated', step):
                raise AssertionError('Checkpoint identity/scheduler differs')
            for stream in p['stream_seeds']:
                key = f'{seed}:{step}:{stream}'
                relative_folder = f'seed_{seed}/step_{step:06d}/stream_{stream}'
                folder = directory/relative_folder
                folder.mkdir(parents=True, exist_ok=False)
                reference = read(relative_folder+'/result.json')
                d.write_json(folder/'reference.json', reference)
                batches, batch_records = d.build_batches(data, [i for b in plans[key] for i in b],
                    seed=seed, epoch=initial['epoch'], canonical=False)
                if batch_records != read(relative_folder+'/training_batches.json')['records']:
                    raise AssertionError('Replay training image/caption identities differ')
                row = dict(training_seed=seed, checkpoint_step=step, stream_seed=stream,
                    source_checkpoint_sha256=p['checkpoints'][relative], endpoints={}, replay={}, caches={})
                sync()
                for arm in p['arms']:
                    label = f'{key}:{arm}'
                    print('POOL_REPLAY:', label, flush=True)
                    audit = replay_branch(model, optimizer, scheduler, initial, batches, config, reference, arm,
                        p['coefficient'], checkpoint_offsets=p['replay_check_offsets'],
                        progress=lambda msg: print(f'POOL_REPLAY: {key} {msg}', flush=True))
                    encoded = encode_verified(model, optimizer, scheduler, report_batches,
                        reference['points'][arm]['50']['report'], old_conditions)
                    validate_encoded(encoded, p['report_size'])
                    metadata = policy.guarded(model, optimizer, scheduler,
                        lambda: save_cache(folder/f'features_{arm}.npz', encoded, row_order_sha))
                    endpoint = policy.guarded(model, optimizer, scheduler, lambda: evaluate_encoded(encoded, conditions))
                    audit.update(original_features_exact=True, original_reports_exact=True,
                                 encoding_preserves_state=True, cache_round_trip_exact=True)
                    row['replay'][arm], row['endpoints'][arm], row['caches'][arm] = audit, endpoint, metadata
                    d.write_json(folder/f'endpoint_{arm}.json', dict(replay=audit, evaluation=endpoint, cache=metadata))
                    policy.guarded(model, optimizer, scheduler, sync)
                    del encoded
                d.write_json(folder/'result.json', row)
                rows.append(row)
                sync()
                del batches, batch_records, reference, row
                gc.collect()
            del initial
            gc.collect()
            torch.cuda.empty_cache()
    comparison = summarize(rows, p)
    if comparison['optimizer_steps'] != p['optimizer_step_budget']:
        raise AssertionError('Replay budget differs')
    d.write_json(directory/'comparison.json', comparison)
    d.write_json(directory/'completion.json', dict(status='complete', pairs=12, branches=48,
        optimizer_steps=2400, trial_updates=0, endpoint_caches=48, reporting_partitions=32,
        primary_full_pool_size=512, exact_policy_replay=True, original_features_exact=True,
        original_reports_exact=True, gate_decisions_recomputed=False))
    sync()
    return comparison
