"""Frozen online alignment policy versus three randomized, equal-dose schedules.

Only the reference's training gate decisions determine the matched dose. No
evaluation results enter randomization, dose choice, or objective selection.
"""

import gc
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch

from colabs.exposure_checkpoint_retention import ARMS, SEEDS, required_paths, save_checkpoint
from model.clip_training import CLIPLossOutput, native_clip_contrastive_loss
from src import clip_gated_training as gated
from src.clip_early_pulse import auxiliary_losses, state_digest

ROOT = gated.ROOT
RANDOM_ARMS = ARMS[1:]


def load_protocol():
    parent = gated.load_protocol()  # validates the original frozen threshold/loss/horizon
    followup = json.loads((ROOT / 'configs/clip_exposure_matched.json').read_text())
    if (followup['training_seeds'] != SEEDS or followup['arms'] != ARMS
            or followup['randomization_seed'] != 202609230
            or followup['schedule_algorithm'] != 'sha256_rank_without_replacement_v1'
            or followup['eligible_objective_steps'] != [100, 1000]
            or followup['checkpoint_retention'] != 'exposure_hybrid_v1'
            or followup['primary_contrast'] != 'alignment_minus_mean_of_all_three_random_controls_within_seed'
            or followup['rules_refitted'] or not followup['exposure_matched']
            or followup['compute_matched'] or followup['independent_test_set']):
        raise ValueError('Frozen exposure protocol changed')
    return {**parent, **followup}


def rank_orders(seed, protocol):
    """Platform-stable pseudorandom ranks, with no global RNG consumption."""
    if seed not in SEEDS:
        raise ValueError('Unknown training seed')
    start, end = protocol['eligible_objective_steps']
    return {arm: sorted(range(start, end + 1), key=lambda step: (
        hashlib.sha256(f"otco-dose-v1:{protocol['randomization_seed']}:{seed}:{arm}:{step}".encode()).digest(), step))
        for arm in RANDOM_ARMS}


def matched_schedules(seed, reference_audit, protocol):
    gated.verify_decisions('alignment_gated', reference_audit, protocol)
    count = reference_audit['active_updates']
    if not 0 < count < 901:
        raise ValueError('Degenerate dose: no distinct equal-dose timing comparison is possible; stopping')
    orders = rank_orders(seed, protocol)
    schedules = {arm: sorted(order[:count]) for arm, order in orders.items()}
    original = tuple(d['objective_step'] for d in reference_audit['decisions'] if d['active'])
    identities = {tuple(s) for s in schedules.values()} | {original}
    if len(identities) != 4:
        raise ValueError('Duplicate reference/control schedules; refuse selective resampling')
    return {'seed': seed, 'active_updates': count, 'schedules': schedules,
            'source': 'current_seed_alignment_gate_decisions_only',
            'randomization_seed': protocol['randomization_seed'],
            'reference_decisions_sha256': decision_digest(reference_audit['decisions'])}


def decision_digest(decisions):
    return hashlib.sha256(json.dumps(decisions, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ScheduledObjective(torch.nn.Module):
    def __init__(self, active_steps, protocol):
        super().__init__()
        if (len(active_steps) != len(set(active_steps)) or any(type(s) is not int or not 100 <= s <= 1000 for s in active_steps)):
            raise ValueError('Invalid scheduled objective steps')
        self.active_steps = frozenset(active_steps)
        self.protocol, self.decisions = protocol, []
        self.alignment = None

    def forward(self, output, *, species_ids, step):
        if step != len(self.decisions) or not 0 <= step < 1001:
            raise ValueError('Missing/repeated objective step')
        active = step in self.active_steps
        native = native_clip_contrastive_loss(output.logits)
        auxiliary = auxiliary_losses(output)['uniform_top8'] if active else None
        weighted = self.protocol['uniform_top8_coefficient'] * auxiliary if active else None
        total = native + weighted if active else native
        self.decisions.append({'objective_step': step, 'active': active})
        return CLIPLossOutput(total, native, auxiliary, weighted, {
            'clip_loss': float(native.detach()), 'total_loss': float(total.detach()),
            'gate_active': int(active),
            'alpha_effective': self.protocol['uniform_top8_coefficient'] if active else 0.,
        })


class ExposureObserver(gated.RolloutObserver):
    """Reuse data/probe setup and exact state snapshot; separate retention layout."""
    def __call__(self, *, model, epoch, global_step):
        if global_step not in self.protocol['checkpoint_steps']:
            return
        state = self.snapshot(epoch, global_step)
        if global_step == 100:
            hashes = {key: state_digest(value) for key, value in state.items()}
            reference = self.shared / 'common_prefix_hashes.json'
            if self.arm == 'alignment_gated':
                gated.clip_train.write_json(reference, hashes)
            elif json.loads(reference.read_text()) != hashes:
                raise AssertionError('Native prefix model/optimizer/RNG differs across arms')
            gated.clip_train.write_json(self.directory / 'prefix_hashes.json', hashes)
        if self.arm == 'alignment_gated' and global_step in self.protocol['alignment_refresh_steps']:
            before = state_digest(state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            measured = gated.measure_alignment(model, self.batches, self.config, self.protocol['probe_branch_seed'])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            measured.update(completed_updates=global_step, seconds=time.perf_counter() - started)
            if state_digest(self.snapshot(epoch, global_step)) != before:
                raise AssertionError('Gate probe mutated training state')
            self.objective.alignment = measured['mean_cosine']
            measured['active_until_next_refresh'] = gated.gate_active(
                self.arm, global_step, self.protocol, measured['mean_cosine'])
            self.probes.append(measured)
            print('GATE_PROBE:', json.dumps(measured), flush=True)
        state['gate_state'] = {'alignment': self.objective.alignment, 'probes': self.probes,
                              'objective_decisions': self.objective.decisions,
                              'scheduled_active_steps': sorted(getattr(self.objective, 'active_steps', []))}
        state['arm'] = self.arm
        save_checkpoint(self.shared.parent, state, seed=self.config['training']['seed'],
                        arm=self.arm, step=global_step)
        self.write_audit()
        self.sync()


def verify_audit(arm, audit, protocol, schedule=None):
    if arm == 'alignment_gated':
        gated.verify_decisions(arm, audit, protocol)
        return
    if arm not in RANDOM_ARMS or schedule is None:
        raise ValueError('Missing random schedule')
    if audit['probes'] or [d['objective_step'] for d in audit['decisions']] != list(range(1001)):
        raise AssertionError('Invalid control audit')
    expected = set(schedule)
    if len(expected) != len(schedule) or not expected <= set(range(100, 1001)):
        raise AssertionError('Invalid control schedule')
    if any(d['active'] != (d['objective_step'] in expected) for d in audit['decisions']):
        raise AssertionError('Actual control differs from committed schedule')
    if audit['active_updates'] != len(expected):
        raise AssertionError('Exposure count mismatch')


def summarize(directory, protocol):
    directory = Path(directory)
    rows, contrasts = [], []
    for seed in SEEDS:
        shared = directory / f'seed_{seed}'
        reference = json.loads((shared / 'alignment_gated/gate_audit.json').read_text())
        plan = json.loads((shared / 'matched_schedules.json').read_text())
        if plan != matched_schedules(seed, reference, protocol):
            raise AssertionError('Schedule plan changed')
        prefix = json.loads((shared / 'common_prefix_hashes.json').read_text())
        for arm in ARMS:
            path = shared / arm
            if json.loads((path / 'prefix_hashes.json').read_text()) != prefix:
                raise AssertionError('Prefix audit mismatch')
            audit = json.loads((path / 'gate_audit.json').read_text())
            verify_audit(arm, audit, protocol, plan['schedules'].get(arm))
            if audit['active_updates'] != plan['active_updates']:
                raise AssertionError('Not exposure matched')
            summary = json.loads((path / 'training/summary.json').read_text())
            if summary['execution']['completed_updates'] != 1001:
                raise AssertionError('Incomplete arm')
            evaluation = summary['final']['evaluation']
            rows.append({'seed': seed, 'arm': arm,
                'canonical_avg_r1_percent': statistics.fmean(evaluation['canonical_retrieval'][d]['r_at_1']
                    for d in ['text_to_image', 'image_to_text']),
                'species_top1_percent': 100 * evaluation['species']['top_1_accuracy'],
                'active_updates': audit['active_updates'],
                'probe_seconds': sum(p['seconds'] for p in audit['probes']),
                'wall_seconds': json.loads((path / 'timing.json').read_text())['wall_seconds']})
        current = [r for r in rows if r['seed'] == seed]
        aligned, controls = current[0], current[1:]
        contrasts.append({'seed': seed,
            'alignment_minus_mean_random_r1_pp': aligned['canonical_avg_r1_percent'] - statistics.fmean(r['canonical_avg_r1_percent'] for r in controls),
            'alignment_minus_mean_random_species_pp': aligned['species_top1_percent'] - statistics.fmean(r['species_top1_percent'] for r in controls),
            'random_r1_range_percent': [min(r['canonical_avg_r1_percent'] for r in controls), max(r['canonical_avg_r1_percent'] for r in controls)],
            'alignment_minus_each_random_r1_pp': {r['arm']: aligned['canonical_avg_r1_percent'] - r['canonical_avg_r1_percent'] for r in controls}})
    report = {'rows': rows, 'per_seed_primary_contrasts': contrasts,
        'mean_alignment_minus_mean_random_r1_pp': statistics.fmean(r['alignment_minus_mean_random_r1_pp'] for r in contrasts),
        'mean_alignment_minus_mean_random_species_pp': statistics.fmean(r['alignment_minus_mean_random_species_pp'] for r in contrasts),
        'training_seed_count': 3, 'random_controls_per_seed': 3,
        'interpretation': 'Exploratory equal-dose timing control on reused CUB test data/seeds. Controls are nested within seeds, not nine independent training seeds. Uniform step randomization changes burst structure as well as timing; not compute matched. No significance/novelty claim.'}
    gated.clip_train.write_json(directory / 'comparison.json', report)
    return report


def run(directory, sync):
    protocol = load_protocol()
    directory = Path(directory)
    write = gated.clip_train.write_json
    write(directory / 'protocol.json', protocol)
    # Commit all ranks before any model training; dose truncation uses only the
    # current reference's gate decisions. No outcome-dependent rerolls/seeds.
    write(directory / 'randomization_commitment.json', {
        'algorithm': protocol['schedule_algorithm'], 'randomization_seed': protocol['randomization_seed'],
        'rank_orders': {str(seed): rank_orders(seed, protocol) for seed in SEEDS},
        'selection': 'first K ranks, K=current reference active_updates; never read evaluation outcomes'})
    scratch = Path(tempfile.mkdtemp(prefix='.otco_exposure_work_', dir=directory.parent)) / 'checkpoints'
    write(directory / 'checkpoint_retention.json', {
        'policy': 'exposure_hybrid_v1', 'permanent_checkpoints': required_paths(),
        'rolling_checkpoint': 'rolling_checkpoint.pt', 'rolling_refresh_steps': protocol['checkpoint_steps'],
        'local_only_trainer_scratch': str(scratch), 'automatic_mid_epoch_resume': False,
        'best_epoch_weights_retained': False})
    sync()
    for seed in SEEDS:
        shared = directory / f'seed_{seed}'
        shared.mkdir(exist_ok=False)
        plan = None
        for arm in ARMS:
            path = shared / arm
            path.mkdir()
            config = gated.clip_train.load_training_config(ROOT / protocol['baseline_config'])
            config['experiment']['seed'] = config['training']['seed'] = seed
            config['experiment']['name'] = f'clip_exposure_matched_{seed}_{arm}'
            config['diagnostics']['separate_projection_gradient_steps'] = 0
            schedule = None if arm == 'alignment_gated' else plan['schedules'][arm]
            factory = (lambda _: gated.GatedObjective('alignment_gated', protocol)) if schedule is None else (lambda _, s=schedule: ScheduledObjective(s, protocol))
            observer = ExposureObserver(path, shared, arm, protocol, sync)
            print(f'ROLLOUT: seed={seed} arm={arm}', flush=True)
            started = time.perf_counter()
            summary = gated.clip_train.run(config, output_directory=path / 'training', checkpoint_directory=scratch,
                observer=observer, stop_after_epochs=13, objective_factory=factory)
            if summary['execution']['completed_updates'] != 1001:
                raise AssertionError('Incomplete rollout')
            observer.write_audit()
            audit = json.loads((path / 'gate_audit.json').read_text())
            verify_audit(arm, audit, protocol, schedule)
            write(path / 'timing.json', {'wall_seconds': time.perf_counter() - started,
                                       'includes_evaluation_and_synchronous_backup': True})
            if arm == 'alignment_gated':
                plan = matched_schedules(seed, audit, protocol)
                write(shared / 'matched_schedules.json', plan)
                print('DOSE_MATCH:', seed, plan['active_updates'], 'active updates per arm', flush=True)
            sync()  # Publish schedule before any control runs, fail closed.
            del observer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report = summarize(directory, protocol)
    write(directory / 'completion.json', {'status': 'complete', 'seeds': SEEDS, 'arms': ARMS,
        'trajectories': 12, 'required_state_checkpoints': 24, 'rolling_checkpoint_files': 1,
        'checkpoint_retention': 'exposure_hybrid_v1', 'rules_refitted': False,
        'exposure_matched': True, 'compute_matched': False})
    sync()
    return report
