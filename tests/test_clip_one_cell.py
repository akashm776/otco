import json
from pathlib import Path

from colabs.clip_early_pulse_one_cell import completed, find_training_jobs, processes


def test_detects_training_but_not_loader_workers():
    args = ['python', '-m', 'src.clip_early_pulse', '--output-directory', '/content/otco_outputs/run']
    table = {10: {'parent': 1, 'args': ['python', '-u', '/content/control/runner.py']},
             20: {'parent': 10, 'args': args}, 21: {'parent': 20, 'args': args},
             22: {'parent': 20, 'args': args}}
    assert find_training_jobs(table) == [(10, Path('/content/otco_outputs/run'))]


def test_proc_reader_excludes_zombies_and_handles_spaces(tmp_path):
    for pid, state in [(20, 'S'), (21, 'Z')]:
        proc = tmp_path / str(pid)
        proc.mkdir()
        (proc / 'stat').write_text(f'{pid} (python worker) {state} 10 0 0')
        (proc / 'cmdline').write_bytes(b'python\0-m\0src.clip_early_pulse\0')
    assert processes(tmp_path) == {20: {'parent': 10, 'args': ['python', '-m', 'src.clip_early_pulse']}}


def test_completion_requires_full_protocol(tmp_path):
    path = tmp_path / 'completion.json'
    assert not completed(tmp_path)
    path.write_text('{')
    assert not completed(tmp_path)
    value = {'status': 'complete', 'completed_updates': 1001,
             'arms': ['baseline', 'uniform_top8', 'hardest_real']}
    path.write_text(json.dumps(value))
    assert completed(tmp_path)
    value['arms'].pop()
    path.write_text(json.dumps(value))
    assert not completed(tmp_path)
