"""Bounded counterfactual branches from authenticated on-policy saved states.

Predictors are committed before any report outcomes. No predictor controls any
branch, and no threshold is fitted. The old excluded pool is reused explicitly.
"""

from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
import torch

from colabs.backup_clip_run import checksum
from model.clip_backend import CLIPEncoderBackend
from model.clip_training import (build_clip_optimizer, configure_clip_trainable_parameters,
                                 native_clip_contrastive_loss)
from src import clip_train
from src.clip_early_pulse import auxiliary_losses, state_digest
from src.clip_gradient_stages import ROOT, observational_state
from src.clip_paired_updates import cosine, flat_grad, encode_cached, partition_losses, feature_hash
from src.clip_training_data import CLIPCUBPairDataset

ARMS = ['native', 'pulse', 'sustained']


def write_json(path, value):
    """Closed, finite JSON only; used before synchronous backup barriers."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checkpoint_path(seed, step):
    arm = 'shared' if step == 100 else 'alignment_gated'
    return f'seed_{seed}/{arm}/checkpoints/step_{step:06d}.pt'


def load_protocol():
    p = json.loads((ROOT / 'configs/clip_checkpoint_diagnostic.json').read_text())
    expected = dict(training_seeds=[789, 2026, 31415], checkpoint_steps=[100, 250, 500, 750],
                    arms=ARMS, horizon=50, batch_size=64, coefficient=0.06005351588542947,
                    scheduler_horizon=3850, meta_size=512, report_size=512,
                    rules_refitted=False, independent_test_set=False,
                    literal_training_loader_resume=False)
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Bounded diagnostic protocol changed')
    paths = {checkpoint_path(seed, step) for seed in p['training_seeds'] for step in p['checkpoint_steps']}
    if set(p['checkpoints']) != paths:
        raise ValueError('Expected all 12 fixed source checkpoints')
    return p


def ranked(items, salt):
    return sorted(items, key=lambda i: hashlib.sha256(f'{salt}:{i}'.encode()).hexdigest())


def split_pool(training, holdout, p):
    if len(set(training)) != len(training) or len(set(holdout)) != len(holdout):
        raise ValueError('Duplicate source identities')
    if set(training) & set(holdout) or len(holdout) != p['meta_size'] + p['report_size']:
        raise ValueError('Invalid excluded pool or training overlap')
    ordered = ranked(holdout, f"meta-report-v1:{p['split_seed']}")
    return {'meta': ordered[:p['meta_size']], 'report': ordered[p['meta_size']:]}


def branch_plan(training, seed, step, p):
    n = p['horizon'] * p['batch_size']
    if len(set(training)) < n:
        raise ValueError('Not enough distinct training images for the bounded branch')
    ordered = ranked(training, f"branch-v1:{p['branch_data_seed']}:{seed}:{step}")[:n]
    return [ordered[i:i+p['batch_size']] for i in range(0, n, p['batch_size'])]


def active(arm, offset):
    if arm not in ARMS or offset < 0:
        raise ValueError('Invalid branch or offset')
    return arm == 'sustained' or (arm == 'pulse' and offset == 0)


def rng_state():
    return dict(python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore(model, optimizer, scheduler, initial):
    model.load_state_dict(initial['model'], strict=True)
    # A CPU optimizer can alias checkpoint tensors without this copy.
    optimizer.load_state_dict(deepcopy(initial['optimizer']))
    scheduler.load_state_dict(deepcopy(initial['scheduler']))
    optimizer.zero_grad(set_to_none=True)
    random.setstate(initial['python_rng'])
    np.random.set_state(initial['numpy_rng'])
    torch.set_rng_state(initial['torch_rng'])
    if len(initial['cuda_rng']) != torch.cuda.device_count():
        raise ValueError('Checkpoint CUDA RNG device count differs')
    if initial['cuda_rng']:
        torch.cuda.set_rng_state_all(initial['cuda_rng'])
    model.train()


@contextmanager
def observe(model):
    """Preserve buffers as well as RNG, modes, and accumulated gradients."""
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    try:
        with observational_state(model):
            yield
    finally:
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])


def trainable_vector(model):
    return torch.cat([p.detach().float().flatten().cpu() for p in model.parameters() if p.requires_grad])


def live_digest(model, optimizer, scheduler):
    return state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), **rng_state()))


def step_update(model, optimizer, scheduler, batch, config, coefficient):
    """Same bf16 loss, gradient clip, AdamW, clamp, then LR step as trainer."""
    device = next(model.parameters()).device
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with clip_train._autocast(device, config['training']['mixed_precision']):
        out = model(batch['pixel_values'].to(device), clip_train._text_batch(batch, device))
        native = native_clip_contrastive_loss(out.logits)
        auxiliary = auxiliary_losses(out)['uniform_top8'] if coefficient else None
        loss = native if auxiliary is None else native + coefficient * auxiliary
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, config['optimizer']['max_gradient_norm'],
                                        error_if_nonfinite=True)
    lrs = [g['lr'] for g in optimizer.param_groups]
    optimizer.step()
    with torch.no_grad():
        model.clip_model.logit_scale.clamp_(max=math.log(100.0))
    scheduler.step()
    return dict(native_loss=float(native.detach()), total_loss=float(loss.detach()),
                gradient_norm_before_clip=float(norm), learning_rates=lrs,
                coefficient=coefficient, rng_after_sha256=state_digest(rng_state()))


def training_gradients(model, batch, config):
    device = next(model.parameters()).device
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    with observe(model):
        model.train()
        with clip_train._autocast(device, config['training']['mixed_precision']):
            out = model(batch['pixel_values'].to(device), clip_train._text_batch(batch, device))
            native = native_clip_contrastive_loss(out.logits)
            auxiliary = auxiliary_losses(out)['uniform_top8']
        return flat_grad(native, parameters).cpu(), flat_grad(auxiliary, parameters).cpu()


class MetaPool:
    """Only meta images enter autograd. FP32 encoders, FP64 logits/reduction."""
    def __init__(self, batches):
        self.batches = batches

    def gradient(self, model):
        device = next(model.parameters()).device
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        result = torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64)
        losses = []
        with observe(model):
            for batch in self.batches:
                images = model.encode_images(batch['pixel_values'].to(device))
                texts = model.encode_texts(clip_train._text_batch(batch, device))
                loss = native_clip_contrastive_loss(texts.double() @ images.double().T
                                                    * model.get_logit_scale().double())
                result += flat_grad(loss, parameters).cpu().double() / len(self.batches)
                losses.append(float(loss.detach()))
        if not torch.isfinite(result).all() or not all(map(math.isfinite, losses)):
            raise ValueError('Nonfinite meta gradient/loss')
        return result, statistics.fmean(losses)


class ReportPool:
    """Read-only report evaluation; never passed into the predictor function."""
    def __init__(self, batches, conditions):
        self.batches, self.conditions = batches, conditions

    def evaluate(self, model):
        with observe(model):
            encoded = encode_cached(model, self.batches)
        losses = partition_losses(encoded, self.conditions)
        images, texts, _ = encoded
        scores = texts.double() @ images.double().T
        truth = torch.arange(len(images))
        retrieval = {'text_to_image_r1_percent': 100 * float((scores.argmax(1) == truth).double().mean()),
                     'image_to_text_r1_percent': 100 * float((scores.argmax(0) == truth).double().mean())}
        retrieval['mean_r1_percent'] = statistics.fmean(retrieval.values())
        result = dict(native_loss=statistics.fmean(x for v in losses.values() for x in v),
                      partition_losses=losses, pool_retrieval=retrieval, features_sha256=feature_hash(encoded))
        if not math.isfinite(result['native_loss']):
            raise ValueError('Nonfinite report loss')
        return result


def predictors(model, optimizer, scheduler, initial, batch, config, meta, coefficient):
    """Two exact trial updates; returns sealed scores, not report-derived choices."""
    restore(model, optimizer, scheduler, initial)
    before = live_digest(model, optimizer, scheduler)
    origin = trainable_vector(model)
    native_g, auxiliary_g = training_gradients(model, batch, config)
    meta_g, meta_before = meta.gradient(model)
    if live_digest(model, optimizer, scheduler) != before:
        raise AssertionError('Gradient probes changed training state')
    step_update(model, optimizer, scheduler, batch, config, 0.)
    native_hash = live_digest(model, optimizer, scheduler)
    native_vector = trainable_vector(model)
    meta_native_g, meta_native = meta.gradient(model)
    if live_digest(model, optimizer, scheduler) != native_hash:
        raise AssertionError('Native-state meta probe changed training state')
    restore(model, optimizer, scheduler, initial)
    if live_digest(model, optimizer, scheduler) != before:
        raise AssertionError('Trial branches did not start identically')
    step_update(model, optimizer, scheduler, batch, config, coefficient)
    auxiliary_hash = live_digest(model, optimizer, scheduler)
    delta = (trainable_vector(model) - native_vector).double()
    # No differentiation through AdamW: this is a finite-displacement Taylor score.
    s0, sn = float(meta_g @ delta), float(meta_native_g @ delta)
    scores = dict(first_batch_native_aux_cosine=cosine(native_g, auxiliary_g),
                  meta_aux_cosine=cosine(meta_g, auxiliary_g),
                  adamw_s0=s0, adamw_sn=sn, incremental_update_norm=float(delta.norm()),
                  native_update_norm=float((native_vector-origin).double().norm()),
                  meta_loss_before=meta_before, meta_loss_native_step1=meta_native)
    if any(v is None or not math.isfinite(v) for v in scores.values()):
        raise ValueError('Degenerate/nonfinite predictor; refusing to label it')
    scores['decisions'] = dict(raw_cosine=scores['first_batch_native_aux_cosine'] > 0,
                              meta_cosine=scores['meta_aux_cosine'] > 0,
                              adamw_s0=s0 < 0, adamw_sn=sn < 0)
    return scores, {'native': native_hash, 'auxiliary': auxiliary_hash}, before


def run_state(model, optimizer, scheduler, initial, batches, config, meta, report,
              *, coefficient, commit_predictions, historical_alignment=None, progress=lambda _: None):
    """150 branch updates + 2 trial updates; exact step-one replays are asserted."""
    source_hash = state_digest(initial)
    frozen_before = state_digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
    scores, first_hashes, initial_hash = predictors(model, optimizer, scheduler, initial,
                                                   batches[0], config, meta, coefficient)
    if historical_alignment is not None:
        scores['historical_probe'] = historical_alignment
        scores['decisions']['frozen_alignment'] = historical_alignment['active']
    commit_predictions(scores)  # Durable barrier BEFORE report access, not just a comment.
    restore(model, optimizer, scheduler, initial)
    initial_report = report.evaluate(model)
    if report.evaluate(model) != initial_report:
        raise AssertionError('No-update report replay differs')
    endpoints, traces, first_reports, starts = {}, {}, {}, {}
    for arm in ARMS:
        restore(model, optimizer, scheduler, initial)
        starts[arm] = live_digest(model, optimizer, scheduler)
        if starts[arm] != initial_hash:
            raise AssertionError('Branch initial model/optimizer/scheduler/RNG differs')
        trace = []
        for offset, batch in enumerate(batches):
            row = step_update(model, optimizer, scheduler, batch, config,
                              coefficient if active(arm, offset) else 0.)
            row['offset'] = offset
            trace.append(row)
            if arm != 'native' and any(row[k] != traces['native'][offset][k]
                                      for k in ['learning_rates', 'rng_after_sha256']):
                raise AssertionError('Branch LR or stochastic stream mismatch')
            if offset == 0:
                expected = first_hashes['native' if arm == 'native' else 'auxiliary']
                if live_digest(model, optimizer, scheduler) != expected:
                    raise AssertionError('Actual branch does not replay its trial update exactly')
                first_reports[arm] = report.evaluate(model)
                if live_digest(model, optimizer, scheduler) != expected:
                    raise AssertionError('Report observer mutated training state')
        endpoint_hash = live_digest(model, optimizer, scheduler)
        endpoints[arm] = report.evaluate(model)
        if live_digest(model, optimizer, scheduler) != endpoint_hash:
            raise AssertionError('Endpoint report observer mutated training state')
        endpoints[arm]['training_state_sha256'] = endpoint_hash
        traces[arm] = trace
        progress(arm)
    if first_reports['pulse'] != first_reports['sustained']:
        raise AssertionError('Identical first auxiliary steps have different outcomes')
    if state_digest(initial) != source_hash:
        raise AssertionError('Source checkpoint mutated')
    frozen_after = state_digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
    if frozen_before != frozen_after:
        raise AssertionError('Frozen parameters changed')
    differences = {}
    for arm in ['pulse', 'sustained']:
        differences[arm] = dict(
            step1_loss=first_reports[arm]['native_loss']-first_reports['native']['native_loss'],
            step50_loss=endpoints[arm]['native_loss']-endpoints['native']['native_loss'],
            step50_pool_r1_pp=endpoints[arm]['pool_retrieval']['mean_r1_percent']
                                - endpoints['native']['pool_retrieval']['mean_r1_percent'])
    return dict(predictors=scores, initial_report=initial_report, first_reports=first_reports,
                endpoints=endpoints, differences=differences, traces=traces,
                audit=dict(source_state_sha256=source_hash, branch_start_hashes=starts,
                           first_step_hashes=first_hashes, exact_trial_replay=True,
                           exact_no_update_report_replay=True, matched_rng_and_lrs=True,
                           frozen_parameters_unchanged=True, source_immutable=True,
                           actual_optimizer_steps=3*len(batches)+2))


def build_batches(data, indices, *, seed, epoch, canonical):
    dataset = CLIPCUBPairDataset(data.train_dataset.grouped_split, data.train_dataset.species_ids,
        source_indices=indices, seed=seed,
        caption_view='canonical_first_caption' if canonical else 'epoch_random')
    dataset.set_epoch(epoch)
    batches, records = [], []
    for start in range(0, len(indices), 64):
        raw = [dataset[i] for i in range(start, min(start+64, len(dataset)))]
        batches.append(data.train_loader.collate_fn(raw))
        records.extend({k: r[k] for k in ['source_index', 'image_key', 'caption', 'caption_index', 'species_id']}
                       for r in raw)
    return batches, records


def report_conditions(p):
    conditions = {}
    for seed in p['report_partition_seeds']:
        order = ranked(range(p['report_size']), f'report-partition-v1:{seed}')
        conditions[str(seed)] = {'batches': [order[i:i+64] for i in range(0, len(order), 64)]}
    return conditions


def summarize(rows, p):
    keys = {(r['training_seed'], r['checkpoint_step']) for r in rows}
    expected = {(s, t) for s in p['training_seeds'] for t in p['checkpoint_steps']}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError('Incomplete or duplicate diagnostic states')
    by_seed = []
    for seed in p['training_seeds']:
        selected = [r for r in rows if r['training_seed'] == seed]
        by_seed.append(dict(training_seed=seed, differences={
            arm: {metric: statistics.fmean(r['differences'][arm][metric] for r in selected)
                  for metric in ['step1_loss', 'step50_loss', 'step50_pool_r1_pp']}
            for arm in ['pulse', 'sustained']}))
    predictions = {}
    for name in rows[0]['predictors']['decisions']:
        predictions[name] = {}
        for target, arm, metric in [('immediate', 'pulse', 'step1_loss'),
                                    ('pulse50', 'pulse', 'step50_loss'),
                                    ('sustained50', 'sustained', 'step50_loss')]:
            effects = [r['differences'][arm][metric] for r in rows]
            decisions = [r['predictors']['decisions'][name] for r in rows]
            selected = [i for i, effect in enumerate(effects) if abs(effect) > p['sensitivity_absolute_band']]
            correct = lambda i: decisions[i] == (effects[i] < 0)
            predictions[name][target] = dict(states=len(rows), correct_sign=sum(correct(i) for i in range(len(rows))),
                outside_sensitivity_band=len(selected), correct_outside_band=sum(correct(i) for i in selected),
                mean_decision_regret=statistics.fmean(abs(e) if not correct(i) else 0. for i,e in enumerate(effects)))
    return dict(per_seed=by_seed, seed_mean={
        arm: {metric: statistics.fmean(r['differences'][arm][metric] for r in by_seed)
              for metric in ['step1_loss', 'step50_loss', 'step50_pool_r1_pp']}
        for arm in ['pulse', 'sustained']}, predictor_descriptives=predictions,
        states=len(rows), training_seed_count=len(by_seed), optimizer_steps=sum(r['audit']['actual_optimizer_steps'] for r in rows),
        interpretation='Exploratory reused data; 3 training seeds, correlated checkpoints, one branch stream/state. '
                       'No threshold fitting, significance, fresh-test, or guaranteed-benefit claim. '
                       'Loss differences negative = helpful; pool retrieval is 512-image canonical only.')


def run(source, directory, sync, p=None):
    from transformers import AutoProcessor
    from src.clip_gated_training import fixed_probe_batches, load_protocol as gate_protocol, measure_alignment
    p = load_protocol() if p is None else p
    source, directory = Path(source), Path(directory)
    write_json(directory / 'protocol.json', p)
    clip_train.set_global_seed(42)  # Match source cuDNN flags; saved RNG is restored per branch.
    config = clip_train.load_training_config(ROOT / 'configs/hf_cub200_clip_vit_b32_baseline.yaml')
    config['runtime']['num_workers'] = 0
    device, device_info = clip_train.resolve_device(config['runtime'])
    processor = AutoProcessor.from_pretrained(config['model']['checkpoint'])
    data = clip_train.build_clip_training_data(config, processor, pin_memory=False)
    holdout = json.loads((ROOT / config['dataset']['diagnostic_holdout_indices']).read_text())
    training = data.train_dataset.source_indices
    if len(training) != 4970 or len(data.train_loader) != 77:
        raise AssertionError('Source dataset/training horizon changed')
    pools = split_pool(training, holdout, p)
    meta_batches, meta_records = build_batches(data, pools['meta'], seed=0, epoch=0, canonical=True)
    report_batches, report_records = build_batches(data, pools['report'], seed=0, epoch=0, canonical=True)
    train_keys = {data.train_dataset.grouped_split.groups[i].image_key for i in training}
    meta_keys, report_keys = ({r['image_key'] for r in records} for records in [meta_records, report_records])
    if len(meta_keys) != 512 or len(report_keys) != 512 or meta_keys & report_keys or train_keys & (meta_keys | report_keys):
        raise AssertionError('Image-key overlap or duplicate evaluation images')
    conditions = report_conditions(p)
    write_json(directory/'data_roles.json', dict(meta=meta_records, report=report_records,
        train_source_indices=training, report_partitions=conditions, disjoint_by_image_key=True,
        reused_historical_diagnostic_pool=True, independent_test_set=False))
    meta, report = MetaPool(meta_batches), ReportPool(report_batches, conditions)
    plans = {f'{seed}:{step}': branch_plan(training, seed, step, p)
             for seed in p['training_seeds'] for step in p['checkpoint_steps']}
    write_json(directory/'branch_plans.json', plans)
    write_json(directory/'training_config.json', config)
    write_json(directory/'device.json', device_info)
    frozen_gate = gate_protocol()
    probe_batches = fixed_probe_batches(data, frozen_gate, directory)
    sync()  # Commit identities/partitions/plans before loading or branching a state.
    model = CLIPEncoderBackend.from_pretrained(config['model']['checkpoint']).to(device)
    policy = configure_clip_trainable_parameters(model, config['model']['trainable_policy'])
    if (policy['trainable_parameter_count'], policy['trainable_tensor_count']) != (10895617, 35):
        raise AssertionError('Trainable parameter space differs from saved experiment')
    write_json(directory/'trainable_policy.json', policy)
    optimizer = build_clip_optimizer(model, config['optimizer'])
    scheduler = clip_train.build_scheduler(optimizer, warmup_steps=config['scheduler']['warmup_steps'],
                                           total_steps=p['scheduler_horizon'])
    rows = []
    for seed in p['training_seeds']:
        for step in p['checkpoint_steps']:
            relative = checkpoint_path(seed, step)
            path = source / relative
            if checksum(path) != p['checkpoints'][relative]:
                raise AssertionError(f'Unauthenticated checkpoint: {relative}')
            # Only previously audited checkpoints with embedded SHA256 pins may be unpickled.
            initial = torch.load(path, map_location='cpu', weights_only=False)
            expected_arm = 'baseline' if step == 100 else 'alignment_gated'
            if (initial['training_seed'], initial['completed_updates'], initial['arm']) != (seed, step, expected_arm):
                raise AssertionError('Checkpoint identity mismatch')
            if initial['scheduler']['last_epoch'] != step:
                raise AssertionError('Checkpoint LR state has wrong completed-update count')
            restore(model, optimizer, scheduler, initial)
            folder = directory / f'seed_{seed}' / f'step_{step:06d}'
            folder.mkdir(parents=True, exist_ok=False)
            plan = plans[f'{seed}:{step}']
            batches, records = build_batches(data, [i for batch in plan for i in batch],
                                             seed=seed, epoch=initial['epoch'], canonical=False)
            write_json(folder/'training_batches.json', dict(records=records, caption_epoch=initial['epoch'],
                batch_size=64, literal_source_loader_continuation=False))
            before_probe = live_digest(model, optimizer, scheduler)
            measured = measure_alignment(model, probe_batches, config, frozen_gate['probe_branch_seed'])
            if live_digest(model, optimizer, scheduler) != before_probe:
                raise AssertionError('Historical probe changed source state')
            measured['threshold'] = frozen_gate['alignment_threshold']
            measured['active'] = measured['mean_cosine'] > measured['threshold']
            def commit(scores):
                write_json(folder/'predictions.json', scores)
                sync()
            print(f'DIAGNOSTIC_STATE: seed={seed} saved_updates={step}', flush=True)
            result = run_state(model, optimizer, scheduler, initial, batches, config, meta, report,
                coefficient=p['coefficient'], commit_predictions=commit, historical_alignment=measured,
                progress=lambda arm: print(f'BRANCH_COMPLETE: seed={seed} step={step} arm={arm}', flush=True))
            result.update(training_seed=seed, checkpoint_step=step, source_checkpoint=relative,
                          source_checkpoint_sha256=p['checkpoints'][relative])
            write_json(folder/'result.json', result)
            rows.append(result)
            sync()
            del initial, batches, records, result
            gc.collect()
            torch.cuda.empty_cache()
    summary = summarize(rows, p)
    if summary['optimizer_steps'] != 1824:
        raise AssertionError('Unexpected optimization budget')
    write_json(directory/'comparison.json', summary)
    write_json(directory/'completion.json', dict(status='complete', states=12, branches=36,
        branch_updates=1800, trial_updates=24, optimizer_steps=1824, predictions_committed_before_reporting=True))
    sync()
    return summary
