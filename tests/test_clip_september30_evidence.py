import json
from pathlib import Path

import pytest

from scripts.archive_clip_september30_evidence import STUDIES, check_partial_overlap, contains_caption
from scripts.archive_clip_followup_evidence import verify

ROOT = Path(__file__).resolve().parents[1]/'experiment_results'


@pytest.mark.parametrize('name', STUDIES)
def test_retained_evidence_hashes_and_exclusions(name):
    target = ROOT/name
    verify(target)
    audit = json.loads((target/'export_audit.json').read_text())
    assert audit['experiment_complete'] == STUDIES[name][2]
    assert audit['manifest_sha256'] == STUDIES[name][1]
    assert audit['other_files_omitted']
    for relative in audit['files']:
        path = target/relative
        assert path.suffix not in {'.pt', '.pth', '.ckpt', '.zip', '.pdf'}
        if path.suffix == '.json':
            assert not contains_caption(json.loads(path.read_text()))


def test_partial_pairs_not_double_counted():
    actual = check_partial_overlap(ROOT)
    expected = json.loads((ROOT/'clip_marginal_utility_partial_2026-09/overlap_audit.json').read_text())
    assert actual == expected
    assert actual['matched_pair_count_in_analysis'] == 12
    assert actual['identical_partial_pairs'] == 6
    assert actual['partial_run_is_additional_evidence'] is False


@pytest.mark.parametrize('name,module', [
    ('clip_continuation_timecourse_2026-09', 'clip_continuation_timecourse'),
    ('clip_marginal_utility_2026-09', 'clip_marginal_utility'),
    ('clip_policy_test_2026-09', 'clip_policy_test'),
])
def test_archived_comparisons_recompute_exactly(name, module):
    from importlib import import_module
    target = ROOT/name
    p = json.loads((target/'protocol.json').read_text())
    rows = [json.loads((target/f'seed_{seed}/step_{step:06d}/stream_{stream}/result.json').read_text())
            for seed in p['training_seeds'] for step in p['checkpoint_steps'] for stream in p['stream_seeds']]
    assert import_module('src.'+module).summarize(rows, p) == json.loads((target/'comparison.json').read_text())


def test_nested_caption_detection():
    assert contains_caption({'batch': [{'caption': 'raw text'}]})
    assert not contains_caption({'caption_count': 512, 'report_loss': .5})
