"""Package frozen policy test plus a persistent subprocess transcript."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

from scripts.build_clip_continuation_timecourse_cell import BASE, FILES as PARENT_FILES

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(set(PARENT_FILES + [
    'configs/clip_policy_test.json', 'src/clip_policy_test.py',
    'colabs/run_clip_policy_test.py', 'colabs/policy_test_logging.py',
    'tests/test_clip_policy_test.py', 'docs/clip-policy-test.md',
]))


def build():
    sources = {name: (ROOT/name).read_text() for name in FILES}
    sources['policy_source_manifest.json'] = json.dumps({
        name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()}, indent=2)
    raw = json.dumps(sources, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    payload = base64.b64encode(zlib.compress(raw, 9)).decode()
    cell = '''"""ONE A100 Colab cell: frozen alignment policy versus matched timing controls.
12 matched sets / 48 branches / 2400 updates. No retraining or optimizer trials.
Uses clip_checkpoint_diagnostic_inputs.zip in MyDrive/OTCO/evaluation_transfer
or /content, or the original complete saved-checkpoint run folder on Drive.
Allow ~25 GiB recovery space and 15 GiB free afterward. No large new checkpoints.
Paste the ENTIRE file. Wait for DRIVE_SYNC_COMPLETE; no automatic retries.
Full subprocess output is saved separately in clip_policy_test_logs on Drive.
"""
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib

BASE_COMMIT = __BASE__
BUNDLE_SHA256 = __DIGEST__
PAYLOAD = __PAYLOAD__

def run_all():
    if not Path('/content').is_dir():
        raise RuntimeError('Run in Google Colab with one A100 GPU.')
    raw = zlib.decompress(base64.b64decode(PAYLOAD))
    if hashlib.sha256(raw).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError('Embedded bundle checksum mismatch')
    from google.colab import drive
    drive.mount('/content/drive')
    staging = Path(tempfile.mkdtemp(prefix='otco_policy_source_', dir='/content'))
    repo = staging/'repo'
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/akashm776/otco.git', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', BASE_COMMIT], check=True)
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != BASE_COMMIT:
        raise RuntimeError('Wrong pinned repository base')
    for name, source in json.loads(raw).items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('Unsafe source overlay path')
        target = repo/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('otco_policy_logging', repo/'colabs/policy_test_logging.py')
    logger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(logger)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    log = Path('/content/drive/MyDrive/OTCO/evaluation_transfer/clip_policy_test_logs')/(stamp+'.log')
    print('FULL_SUBPROCESS_LOG:', log, flush=True)
    try:
        logger.run_logged([sys.executable, '-u', str(repo/'colabs/run_clip_policy_test.py'),
                           '--bundle-id', BUNDLE_SHA256], repo, log)
    except BaseException:
        print('POLICY_TEST_FAILED: inspect the full log; do not restart automatically.')
        try:
            drive.flush_and_unmount()
            print('PARTIAL_DRIVE_SYNC_COMPLETE: this does NOT mean the experiment completed.')
        except BaseException as error:
            print('DRIVE_FLUSH_FAILED:', repr(error))
        raise
    drive.flush_and_unmount()
    print('DRIVE_SYNC_COMPLETE: policy-test results and log saved; Drive is now unmounted.')
    print('Download the new clip_policy_test run folder and its log. No large checkpoints were written.')

run_all()
'''
    cell = cell.replace('__BASE__', repr(BASE)).replace('__DIGEST__', repr(digest)).replace('__PAYLOAD__', repr(payload))
    target = ROOT/'colabs/clip_policy_test_drive_one_cell.py'
    target.write_text(cell)
    print(target, 'bundle_sha256=', digest)
    return cell


if __name__ == '__main__':
    build()
