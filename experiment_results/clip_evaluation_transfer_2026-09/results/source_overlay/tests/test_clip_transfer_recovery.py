import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from colabs import recover_clip_evaluation_transfer_one_cell as recovery


def fixture_archive(tmp_path):
    run_id = 'test_run'
    marker = dict(status='complete', training_seeds=[789,2026,31415],
                  checkpoint_steps=[100,250,500,750,1001], replay_branch_rows=720,
                  transfer_branch_rows=720, evaluated_states=15, rules_refitted=False)
    contents = {run_id+'/results/completion.json': json.dumps(marker).encode(),
                run_id+'/results/prediction_report.json': b'{"fixture":true}'}
    for seed in [789,2026,31415]:
        for name in ['common_step_100.pt','step_000250.pt','step_000500.pt','step_000750.pt','latest.pt','best_model.pt']:
            key = f'{run_id}/seed_{seed}/checkpoints/baseline/{name}'
            contents[key] = ('fixture only: '+key).encode()
    packed = dict(run_id=run_id, status='complete', bundle_id='fixture', files={
        name:dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) for name,data in contents.items()})
    path = tmp_path/'original.zip'
    with zipfile.ZipFile(path,'w') as bundle:
        for name,data in contents.items():
            bundle.writestr(name,data)
        bundle.writestr('PACKING_MANIFEST.json',json.dumps(packed))
    return path, contents


def test_reports_first_and_each_checkpoint_recover_exact_bytes(tmp_path):
    path, contents = fixture_archive(tmp_path)
    before = path.read_bytes()
    packed, checkpoints, reports = recovery.inspect_archive(path,'test_run')
    assert len(checkpoints)==18 and len(reports)==2
    for index, selected in enumerate([reports]+[[name] for name in checkpoints]):
        destination = tmp_path/f'part_{index}.zip'
        recovery.export_subset(path,packed,selected,destination)
        assert recovery.export_subset(path,packed,selected,destination)==destination
        with zipfile.ZipFile(destination) as part:
            for name in selected:
                assert part.read(name)==contents[name]
            assert set(part.namelist())==set(selected)|{'RECOVERY_MANIFEST.json'}
    assert path.read_bytes()==before


def test_bad_source_checksum_does_not_publish_download(tmp_path):
    path,_ = fixture_archive(tmp_path)
    packed,checkpoints,_ = recovery.inspect_archive(path,'test_run')
    packed['files'][checkpoints[0]]['sha256']='0'*64
    destination=tmp_path/'invalid.zip'
    with pytest.raises(ValueError,match='Source checksum'):
        recovery.export_subset(path,packed,[checkpoints[0]],destination)
    assert not destination.exists() and path.exists()


def test_wrong_run_or_missing_archive_stops_without_retraining(tmp_path):
    path,_=fixture_archive(tmp_path)
    with pytest.raises(ValueError,match='Wrong run'):
        recovery.inspect_archive(path,'different_run')
    with pytest.raises(FileNotFoundError,match='NOT retrain'):
        recovery.inspect_archive(tmp_path/'missing.zip','test_run')


def test_existing_corrupt_download_is_not_silently_replaced(tmp_path):
    path,_=fixture_archive(tmp_path)
    packed,_,reports=recovery.inspect_archive(path,'test_run')
    destination=tmp_path/'existing.zip'
    destination.write_bytes(b'broken file')
    with pytest.raises(zipfile.BadZipFile):
        recovery.export_subset(path,packed,reports,destination)
    assert destination.read_bytes()==b'broken file'


def test_recovery_contains_no_training_or_drive_calls():
    source=Path(recovery.__file__).read_text()
    for forbidden in ['drive.mount','/content/drive','subprocess','torch.load','train_seed','pip install']:
        assert forbidden not in source
