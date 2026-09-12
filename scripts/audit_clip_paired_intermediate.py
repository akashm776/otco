import sys
sys.modules['readline'] = None
from pathlib import Path
import hashlib
import json
import math
import statistics as stats
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from colabs.run_clip_paired_intermediate import verify_completion
from src.clip_paired_intermediate import reference_directory
from src.clip_paired_updates import compare_endpoint_losses, checkpoint_relative_path

import argparse
parser = argparse.ArgumentParser(description="Audit the completed clip_paired_intermediate_20260912T131926_801067Z ZIP; no training.")
parser.add_argument("archive", type=Path)
parser.add_argument("--output-directory", type=Path, required=True)
args = parser.parse_args()
ARCHIVE = args.archive
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == "c34004cc190bf42bfe45e89fed137750b79ee0af7294ddcc306c1124f3eb4004", "Unexpected archive bytes"
RUN = "clip_paired_intermediate_20260912T131926_801067Z"
TEMP = args.output_directory
TEMP.mkdir(parents=True, exist_ok=False)
DATA = TEMP / RUN
read = lambda path: json.loads(path.read_text())
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
PARTS = ['shuffle_seed_42', 'shuffle_seed_123', 'shuffle_seed_4242']
mean = lambda losses, parts=PARTS: stats.fmean(x for p in parts for x in losses[p])
def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=0,abs_tol=1e-12), (a,b)

with zipfile.ZipFile(ARCHIVE) as bundle:
    names = bundle.namelist()
    assert len(names) == len(set(names)) == 101 and bundle.testzip() is None
    assert all(n.startswith(RUN + '/') and '..' not in Path(n).parts and not n.endswith('.pt') for n in names)
    bundle.extractall(TEMP)
    assert all((TEMP / n).read_bytes() == bundle.read(n) for n in names)
verify_completion(DATA)
manifest = read(DATA / 'results/run_manifest.json')
assert manifest['status'] == 'complete'
assert manifest['source_commit'] == 'cce20072fe61be22bfcfd3b17ac8605d8bdce78b'
REF = ROOT / 'experiment_results/clip_paired_updates_2026-09'
assert manifest['packages'] == read(REF / 'run_manifest.json')['packages']
original_protocol = read(REF / 'protocol.json')
original_audit = read(REF / 'audit.json')
combined = read(DATA / 'results/per_seed_summary.json')
assert len(combined) == 30
details, endpoints, identities = [], [], []
for seed in [42,123,456]:
    source = DATA / f'seed_{seed}'
    directory = source / 'results/paired'
    protocol = read(directory / 'protocol.json')
    assert protocol['training_seed'] == seed and protocol['diagnostic_caption_seed'] == 42
    for key in ['coefficients','training_batches','batch_size','selection_seed','branch_seed','caption_epoch','primary_partitions']:
        assert protocol[key] == original_protocol[key]
    for name in ['training_batches.json','heldout_partitions.json']:
        assert digest(directory / name) == original_audit['files'][name]['sha256']
    inventory = read(source / 'backup_manifest.json')['verified_files']
    for name, entry in inventory.items():
        if not name.endswith('.pt'):
            assert digest(source / name) == entry['sha256'] and (source/name).stat().st_size == entry['bytes']
    config = yaml.safe_load((source / 'results/baseline/training/resolved_config.yaml').read_text())
    assert config['training']['seed'] == config['experiment']['seed'] == seed
    assert config['ot']['enabled'] is False and config['training']['epochs'] == 50
    assert read(source / 'results/baseline/training/summary.json')['execution'] == dict(completed_epochs=13,completed_updates=1001,scheduler_horizon_updates=3850,planned_short_run=True)
    metrics = [json.loads(s) for s in (source / 'results/baseline/training/metrics.jsonl').read_text().splitlines()]
    assert [r['global_step'] for r in metrics] == [77*i for i in range(14)]
    calibration = read(source / 'results/calibration.json')
    assert calibration['recalibrated'] is False and calibration['coefficients'] == original_protocol['coefficients']
    holdout = read(source / 'results/baseline/diagnostics/diagnostic_holdout_indices.json')
    assert holdout == read(ROOT / 'configs/cub200_clip_diagnostic_holdout_indices.json')
    selected = [i for b in read(directory/'training_batches.json') for i in b['source_indices']]
    excluded = {i for b in calibration['batches'] for i in b['source_indices']}
    assert len(selected) == len(set(selected)) == 1024 and not set(selected)&(set(holdout)|excluded)
    rows = [json.loads(s) for s in (directory/'paired_updates.jsonl').read_text().splitlines()]
    lookup = {(r['checkpoint_step'],r['trial'],r['arm']):r for r in rows}
    states = {r['step']:r for r in read(directory/'checkpoint_provenance.json')}
    historic = reference_directory(seed)
    historical_states = {r['step']:r for r in read(historic/'checkpoint_provenance.json')}
    historical_rows = [json.loads(s) for s in (historic/'paired_updates.jsonl').read_text().splitlines()]
    for step, state in states.items():
        report = read(source / f'results/baseline/diagnostics/{step:06d}_pulse_{step}/report.json')
        assert state['checkpoint_sha256'] == inventory[checkpoint_relative_path(step)]['sha256']
        assert state['features_sha256'] == report['features_and_scale_sha256']
        assert state['initial_state_sha256'] == report['initial_state_sha256']
        assert state['learning_rates'] == report['learning_rates']
        identities.append(state['features_sha256'])
        if step in [100,1001]:
            assert report['historical_endpoint_state_exact'] is True
            for key in ['features_sha256','initial_state_sha256','learning_rates']:
                assert state[key] == historical_states[step][key]
            check = compare_endpoint_losses(rows,historical_rows,step,1e-8)
            assert check in read(directory/'endpoint_replay_checks.json')
            endpoints.append(dict(training_seed=seed,**check))
    for row in rows:
        native = lookup[row['checkpoint_step'],row['trial'],'baseline']
        close(row['heldout_mean'],mean(row['heldout_loss']))
        close(row['heldout_change'],row['heldout_mean']-mean(states[row['checkpoint_step']]['initial_heldout_loss']))
        close(row['train_before'],native['train_before'])
        close(row['coefficient'],original_protocol['coefficients'].get(row['arm'],0.))
        close(row['actual_update_difference_ratio'],row['actual_update_difference_norm']/native['actual_update_norm'])
    for summary in read(directory/'summary.json'):
        step, arm = summary['checkpoint_step'], summary['arm']
        chosen = [lookup[step,trial,arm] for trial in range(16)]
        values = [r['incremental_heldout_loss'] for r in chosen]
        close(summary['mean_incremental_heldout_loss'],stats.fmean(values))
        close(summary['median_incremental_heldout_loss'],stats.median(values))
        assert summary['range'] == [min(values),max(values)]
        assert summary['beneficial_trials'] == sum(v<0 for v in values)
        for key in ['incremental_train_loss','full_gradient_alignment','projection_gradient_alignment','actual_update_difference_ratio','actual_update_cosine_to_native']:
            close(summary['mean_'+key],stats.fmean(r[key] for r in chosen))
        assert dict(training_seed=seed,historical_endpoint_replay=step in [100,1001],**summary) in combined
        partition_effects = {}
        for part in PARTS+['sequential']:
            effects = [mean(r['heldout_loss'],[part])-mean(lookup[step,r['trial'],'baseline']['heldout_loss'],[part]) for r in chosen]
            partition_effects[part] = dict(mean=stats.fmean(effects),beneficial_trials=sum(e<0 for e in effects))
        details.append(dict(training_seed=seed,**summary,
            mean_native_heldout_change=stats.fmean(lookup[step,t,'baseline']['heldout_change'] for t in range(16)),
            mean_auxiliary_heldout_change=stats.fmean(r['heldout_change'] for r in chosen),
            mean_weighted_gradient_ratio=stats.fmean(r['weighted_gradient_ratio'] for r in chosen),
            partition_effects=partition_effects))
assert len(set(identities)) == 15
report = dict(run_id=RUN,archive_sha256=digest(ARCHIVE),zip_files=101,rows_verified=720,
    new_intermediate_rows=432,endpoint_replay_rows=288,all_checks_passed=True,
    checkpoint_note='No model checkpoints in ZIP. Runtime hashes and replay records verified against historical reports; no independent GPU re-encoding.',
    endpoint_checks=endpoints,details=details,
    files={str(p.relative_to(DATA)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in DATA.rglob('*') if p.is_file()})
(TEMP/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['details','files']},indent=2))
for r in details:
    print(r['training_seed'],r['checkpoint_step'],r['arm'],
          'effect_micro=',round(r['mean_incremental_heldout_loss']*1e6,4),'beneficial=',r['beneficial_trials'],
          'native_change=',round(r['mean_native_heldout_change'],9),'aux_change=',round(r['mean_auxiliary_heldout_change'],9),
          'partition_micro=',{k:round(v['mean']*1e6,4) for k,v in r['partition_effects'].items()})
