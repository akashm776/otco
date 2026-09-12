"""Regression checks for the offline audit using the caption-free evidence."""

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import audit_clip_paired_updates as audit


EVIDENCE = Path(__file__).resolve().parents[1] / 'experiment_results/clip_paired_updates_2026-09'


def fixture_archive(tmp_path, monkeypatch, mutation=None):
    # This is a synthetic ZIP fixture, not a substitute for the authenticated
    # downloaded archive. Only this test overrides the production checksum.
    blobs = {name: (EVIDENCE / name).read_bytes() for name in
             ['paired_updates.jsonl', 'summary.json', 'checkpoint_provenance.json', 'protocol.json',
              'run_manifest.json', 'completion.json', 'heldout_partitions.json']}
    indices = json.loads((EVIDENCE / 'training_source_indices.json').read_text())
    blobs['training_batches.json'] = json.dumps([{'source_indices': batch} for batch in indices]).encode()
    for name in ['device.json', 'progress.json', 'stdout.txt', 'paired_update_effects.png', 'paired_update_effects.svg']:
        blobs[name] = b'fixture'
    if mutation:
        mutation(blobs)
    path = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(path, 'w') as bundle:
        for name, content in blobs.items():
            bundle.writestr(audit.RUN + '/' + name, content)
    monkeypatch.setattr(audit, 'ARCHIVE_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_recompute_committed_measurements(tmp_path, monkeypatch):
    report, _, indices = audit.audit(fixture_archive(tmp_path, monkeypatch))
    assert report['unique_branch_rows'] == 96
    assert len(indices) == 16
    saved = json.loads((EVIDENCE / 'audit.json').read_text())
    assert report['details'] == saved['details']


@pytest.mark.parametrize('failure', ['duplicate_branch', 'wrong_difference', 'missing_replay', 'duplicate_summary', 'wrong_coefficient'])
def test_reject_invalid_evidence(tmp_path, monkeypatch, failure):
    def mutate(blobs):
        if failure in ['duplicate_branch', 'wrong_difference', 'wrong_coefficient']:
            rows = [json.loads(line) for line in blobs['paired_updates.jsonl'].splitlines()]
            if failure == 'duplicate_branch':
                rows[-1] = rows[0]
            elif failure == 'wrong_difference':
                rows[1]['incremental_heldout_loss'] += .01
            else:
                rows[1]['coefficient'] = 0.
            blobs['paired_updates.jsonl'] = '\n'.join(json.dumps(row) for row in rows).encode()
        else:
            name = 'checkpoint_provenance.json' if failure == 'missing_replay' else 'summary.json'
            records = json.loads(blobs[name])
            if failure == 'missing_replay':
                records[0]['native_update_and_evaluation_replay_exact'] = False
            else:
                records[1] = records[0]
            blobs[name] = json.dumps(records).encode()
    with pytest.raises(AssertionError):
        audit.audit(fixture_archive(tmp_path, monkeypatch, mutate))


def test_reject_unexpected_archive_bytes(tmp_path):
    path = tmp_path / 'untrusted.zip'
    path.write_bytes(b'not the expected result archive')
    with pytest.raises(AssertionError, match='Unexpected archive bytes'):
        audit.audit(path)
