import hashlib
import json
from pathlib import Path

import pytest

from src.clip_paired_intermediate import verify_seed
from src.clip_usefulness_prospective import verify_completion
from scripts.analyze_clip_usefulness_predictors import analyze

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'experiment_results'
read=lambda p:json.loads(p.read_text())


@pytest.mark.parametrize('name',['clip_paired_intermediate_2026-09','clip_usefulness_prospective_2026-09'])
def test_archived_original_files_match_audit_and_no_captions_or_checkpoints(name):
    directory=EVIDENCE/name
    audit=read(directory/'audit.json')
    assert audit['all_checks_passed'] and audit['rows_verified']==720
    count=0
    for path in directory.rglob('*'):
        if not path.is_file() or path.name in ['README.md','audit.json']:
            continue
        assert path.name not in ['training_batches.json','heldout_partitions.json','diagnostic_holdout_indices.json','stdout.txt']
        assert path.suffix!='.pt'
        entry=audit['files'][str(path.relative_to(directory))]
        assert entry['bytes']==path.stat().st_size
        assert entry['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
        count+=1
    assert count>60


def test_archived_intermediate_rows_and_exact_endpoint_replays():
    directory=EVIDENCE/'clip_paired_intermediate_2026-09'
    study=read(directory/'results/protocol.json')
    for seed in [42,123,456]:
        assert len(verify_seed(directory/f'seed_{seed}/results/paired',study))==240
    audit=read(directory/'audit.json')
    assert len(audit['endpoint_checks'])==6
    assert all(c['maximum_absolute_loss_difference']==0 for c in audit['endpoint_checks'])


def test_archived_prospective_scores_and_limits():
    directory=EVIDENCE/'clip_usefulness_prospective_2026-09'
    verify_completion(directory)
    report=read(directory/'results/prediction_report.json')
    assert report['primary_mean_balanced_accuracy_difference']==pytest.approx(0.06944444444444442)
    for name in ['full_gradient_alignment','checkpoint_step']:
        rows=[r for r in report['per_seed_scores'] if r['predictor']==name]
        assert sum(r['correct'] for r in rows)==13
        assert sum(r['non_near_zero_correct'] for r in rows)==12
        assert sum(r['non_near_zero_count'] for r in rows)==13


def test_archived_screening_reproduces_and_inventory_matches():
    directory=EVIDENCE/'clip_usefulness_predictors_2026-09'
    for name,digest in read(directory/'inventory.json').items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    report=read(directory/'analysis.json')
    folds,predictions=analyze(report['states'])
    assert folds==report['folds'] and predictions==report['predictions']
    assert len(predictions)==105
    assert report['script_sha256']==hashlib.sha256((ROOT/'scripts/analyze_clip_usefulness_predictors.py').read_bytes()).hexdigest()
