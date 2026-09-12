import sys
sys.modules['readline']=None
from pathlib import Path
import hashlib
import json
import math
import statistics as stats
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.clip_usefulness_prospective import verify_completion, SEEDS, STEPS, PARTS
from src.clip_paired_updates import checkpoint_relative_path

import argparse
parser = argparse.ArgumentParser(description="Audit the completed clip_usefulness_prospective_20260912T175040_066114Z ZIP; no training.")
parser.add_argument("archive", type=Path)
parser.add_argument("--output-directory", type=Path, required=True)
args = parser.parse_args()
ARCHIVE = args.archive
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == "854f27d954030775c2640a6ee87ec80e59ccacc1f9915773b5b07669762574d0", "Unexpected archive bytes"
RUN = "clip_usefulness_prospective_20260912T175040_066114Z"
TEMP = args.output_directory
TEMP.mkdir(parents=True, exist_ok=False)
DATA=TEMP/RUN
read=lambda p:json.loads(p.read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(ARCHIVE) as bundle:
    names=bundle.namelist()
    assert len(names)==len(set(names)) and bundle.testzip() is None
    assert all(n.startswith(RUN+'/') and '..' not in Path(n).parts and not n.endswith('.pt') for n in names)
    bundle.extractall(TEMP)
    assert all((TEMP/n).read_bytes()==bundle.read(n) for n in names)
verify_completion(DATA)
manifest=read(DATA/'results/run_manifest.json')
assert manifest['status']=='complete' and manifest['source_commit']=='d907f67968b4ead5e06d416b2f675283cdba2438'
assert manifest['training_seeds']==SEEDS and manifest['checkpoint_steps']==STEPS
REF=ROOT/'experiment_results/clip_paired_updates_2026-09'
assert manifest['packages']==read(REF/'run_manifest.json')['packages']
original=read(REF/'protocol.json')
original_audit=read(REF/'audit.json')
frozen=read(DATA/'results/frozen_rules.json')
assert frozen==read(ROOT/'configs/clip_usefulness_frozen_rules.json')
states=read(DATA/'results/states.json')
assert len(states)==15
report=read(DATA/'results/prediction_report.json')
effects=[]
features=[]
for seed in SEEDS:
    source=DATA/f'seed_{seed}'
    directory=source/'results/paired'
    protocol=read(directory/'protocol.json')
    assert protocol['training_seed']==seed and protocol['diagnostic_caption_seed']==42
    for key in ['coefficients','training_batches','batch_size','selection_seed','branch_seed','caption_epoch','primary_partitions']:
        assert protocol[key]==original[key]
    for name in ['training_batches.json','heldout_partitions.json']:
        assert digest(directory/name)==original_audit['files'][name]['sha256']
    inventory=read(source/'backup_manifest.json')['verified_files']
    for name,entry in inventory.items():
        if not name.endswith('.pt'):
            assert digest(source/name)==entry['sha256'] and (source/name).stat().st_size==entry['bytes']
    config=yaml.safe_load((source/'results/baseline/training/resolved_config.yaml').read_text())
    assert config['training']['seed']==config['experiment']['seed']==seed
    assert config['ot']['enabled'] is False and config['training']['epochs']==50
    assert read(source/'results/baseline/training/summary.json')['execution']==dict(completed_epochs=13,completed_updates=1001,scheduler_horizon_updates=3850,planned_short_run=True)
    metrics=[json.loads(s) for s in (source/'results/baseline/training/metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in metrics]==list(range(14)) and [r['global_step'] for r in metrics]==[77*i for i in range(14)]
    calibration=read(source/'results/calibration.json')
    assert calibration['recalibrated'] is False and calibration['coefficients']==original['coefficients']
    holdout=read(source/'results/baseline/diagnostics/diagnostic_holdout_indices.json')
    assert holdout==read(ROOT/'configs/cub200_clip_diagnostic_holdout_indices.json')
    selected=[i for b in read(directory/'training_batches.json') for i in b['source_indices']]
    excluded={i for b in calibration['batches'] for i in b['source_indices']}
    assert len(selected)==len(set(selected))==1024 and not set(selected)&(set(holdout)|excluded)
    for state in read(directory/'checkpoint_provenance.json'):
        assert state['checkpoint_sha256']==inventory[checkpoint_relative_path(state['step'])]['sha256']
        features.append(state['features_sha256'])
    rows=[json.loads(s) for s in (directory/'paired_updates.jsonl').read_text().splitlines()]
    lookup={(r['checkpoint_step'],r['trial'],r['arm']):r for r in rows}
    for row in rows:
        assert row['coefficient']==original['coefficients'].get(row['arm'],0.)
    for step in STEPS:
        chosen=[lookup[step,t,'uniform_top8'] for t in range(16)]
        state=next(r for r in states if r['training_seed']==seed and r['checkpoint_step']==step)
        assert state['effect']==stats.fmean(r['incremental_heldout_loss'] for r in chosen)
        assert state['full_gradient_alignment']==stats.fmean(r['full_gradient_alignment'] for r in chosen)
        partitions={}
        for p in PARTS+['sequential']:
            diffs=[stats.fmean(r['heldout_loss'][p])-stats.fmean(lookup[step,r['trial'],'baseline']['heldout_loss'][p]) for r in chosen]
            partitions[p]=dict(mean=stats.fmean(diffs),beneficial_trials=sum(d<0 for d in diffs))
        effects.append(dict(**state,partition_effects=partitions))
assert len(set(features))==15
# Independently recompute frozen-rule labels and macro per-seed balanced accuracy.
scores=[]
for name,rule in frozen['rules'].items():
    for seed in SEEDS:
        chosen=[r for r in states if r['training_seed']==seed]
        pairs=[(r['effect']<0, r[name]>rule['threshold'] if name=='full_gradient_alignment' else r[name]<=rule['threshold']) for r in chosen]
        tpr=sum(y and p for y,p in pairs)/sum(y for y,p in pairs)
        tnr=sum(not y and not p for y,p in pairs)/sum(not y for y,p in pairs)
        score=next(r for r in report['per_seed_scores'] if r['predictor']==name and r['training_seed']==seed)
        assert math.isclose(score['balanced_accuracy'],(tpr+tnr)/2,abs_tol=1e-15)
        assert score['correct']==sum(y==p for y,p in pairs)
        scores.append(score)
summary=[]
for name in frozen['rules']:
    chosen=[r for r in scores if r['predictor']==name]
    summary.append(dict(predictor=name,mean_seed_balanced_accuracy=stats.fmean(r['balanced_accuracy'] for r in chosen),
        correct=sum(r['correct'] for r in chosen),total=15,
        non_near_zero_correct=sum(r['non_near_zero_correct'] for r in chosen),non_near_zero_total=sum(r['non_near_zero_count'] for r in chosen)))
audit=dict(run_id=RUN,archive_sha256=digest(ARCHIVE),zip_files=len(names),rows_verified=720,
    states_verified=15,frozen_rules_exact=True,fixed_inputs_exact=True,same_package_versions=True,all_checks_passed=True,
    checkpoint_note='No .pt files in ZIP. Recorded runtime checks and inventory hashes verified; no independent offline GPU replay.',
    score_summary=summary,per_seed_scores=scores,primary_difference=report['primary_mean_balanced_accuracy_difference'],effects=effects,
    files={str(p.relative_to(DATA)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in DATA.rglob('*') if p.is_file()})
(TEMP/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ['effects','files']},indent=2))
for r in effects:
    print(r['training_seed'],r['checkpoint_step'],'effect_micro=',round(r['effect']*1e6,4),'beneficial=',r['beneficial_trials'],
          'partition_micro=',{k:round(v['mean']*1e6,4) for k,v in r['partition_effects'].items()})
