import hashlib
import itertools
import json
import zipfile

import pytest

from scripts import analyze_clip_policy_robustness as a


def rows():
    result = []
    for seed, step, stream in itertools.product(a.SPEC['seeds'], a.SPEC['checkpoints'], a.SPEC['streams']):
        parts = {a.SPEC['partitions'][0]: [1.]*8, a.SPEC['partitions'][1]: [-3.]*8}
        result.append(dict(seed=seed, checkpoint=step, stream=stream,
            contrasts={k: dict(loss=-1., r1_pp=0., batches=parts) for k in a.SPEC['controls']},
            exposure=25, random_equals_gate=False))
    return result


def test_exact_seed_sign_reference_and_small_n_resolution():
    result = a.sign_flip_reference([-1., -2., -3.])
    assert len(result['enumerated_means']) == 8
    assert result['observed'] == -2.
    assert result['two_sided_tail_fraction'] == .25
    assert a.sign_flip_reference([0., 0., 0.])['two_sided_tail_fraction'] == 1.
    with pytest.raises(ValueError):
        a.sign_flip_reference([])


def test_hierarchical_average_not_independent_pair_mean():
    records = [dict(seed=1, checkpoint=100, value=0.), dict(seed=1, checkpoint=500, value=2.),
               dict(seed=2, checkpoint=100, value=9.)]
    mean, seeds = a.paired_mean(records, lambda r: r['value'])
    assert seeds == {'1': 1., '2': 9.}
    assert mean == 5.


def test_partition_reversal_and_leave_outs_are_not_hidden():
    result = a.analyze(rows())['controls']['random_matched']
    assert result['mean'] == -1.
    assert result['partitions']['2026092403']['mean'] == 1.
    assert result['partitions']['2026092404']['mean'] == -3.
    assert result['partition_A_weight_zero_crossing'] == .75
    assert len(result['strata']) == 8
    assert len(result['paired_batch_deletions']) == 16
    assert all(v['mean'] == -1. for v in result['paired_batch_deletions'])
    assert all(v == -1. for group in result['leave_one_out'].values() for v in group.values())


def test_batch_removal_preserves_partition_weight_and_pairing():
    records = rows()
    for r in records:
        for c in r['contrasts'].values():
            c['batches']['2026092403'] = [8.]+[0.]*7
    result = a.analyze(records)['controls']['random_matched']
    deletion = next(x for x in result['paired_batch_deletions']
                    if x['partition'] == '2026092403' and x['omitted_batch'] == 0)
    assert deletion['mean'] == -1.5  # Mean(0 from seven batches, -3 from eight), not 15-batch mean.
    assert deletion['shift_from_original'] == -.5


def evidence(tmp_path):
    content = b'{"value": 1}\n'
    spec = {**a.SPEC, 'source_run': 'example'}
    marker = json.dumps(dict(run_id='example', status='verified_tree', files={
        'record.json': dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())})).encode()
    spec['backup_manifest_sha256'] = hashlib.sha256(marker).hexdigest()
    directory = tmp_path/'example'
    directory.mkdir()
    (directory/'record.json').write_bytes(content)
    (directory/'DRIVE_BACKUP_MANIFEST.json').write_bytes(marker)
    return directory, spec


def test_directory_and_zip_are_equivalent_and_read_only(tmp_path):
    directory, spec = evidence(tmp_path)
    archive = tmp_path/'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for p in directory.iterdir():
            z.write(p, 'example/'+p.name)
    left = a.verify_input(directory, spec)
    right = a.verify_input(archive, spec)
    assert left == right
    assert left[1]['verified_files'] == 1
    assert set(p.name for p in directory.iterdir()) == {'record.json', 'DRIVE_BACKUP_MANIFEST.json'}


@pytest.mark.parametrize('kind', ['content', 'manifest', 'extra', 'missing', 'symlink'])
def test_changed_or_incomplete_evidence_rejected(tmp_path, kind):
    directory, spec = evidence(tmp_path)
    if kind == 'content':
        (directory/'record.json').write_text('changed')
    elif kind == 'manifest':
        (directory/'DRIVE_BACKUP_MANIFEST.json').write_text('{}')
    elif kind == 'extra':
        (directory/'extra.json').write_text('{}')
    elif kind == 'missing':
        (directory/'record.json').unlink()
    else:
        (directory/'link').symlink_to(directory/'record.json')
    with pytest.raises(ValueError):
        a.verify_input(directory, spec)


def test_unsafe_zip_rejected(tmp_path):
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../outside.json', '{}')
    with pytest.raises(ValueError, match='Unsafe'):
        a.verify_input(archive)


@pytest.mark.parametrize('data', ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'])
def test_invalid_json_rejected(data):
    with pytest.raises(ValueError):
        a.checked_json(data)


def test_existing_output_never_overwritten(tmp_path):
    with pytest.raises(ValueError, match='overwrite'):
        a.main(tmp_path/'missing-source', tmp_path)


def test_report_carries_uncertainty_caveats():
    result = a.analyze(rows())
    result['provenance'] = {'verified_files': 334}
    text = a.render(result)
    assert 'NOT a confirmatory' in text
    assert 'same 512 images' in text
    assert 'No confidence interval' in text
    assert 'zero optimizer updates' in text
