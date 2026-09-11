"""Independent, standard-library-only backup worker for an existing Colab run.

Mount Google Drive in the notebook first (or later); this worker never creates
a fake local Drive directory. It does not interact with training or its RNGs.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(source, destination):
    """Publish a verified file via rename; retain the previous copy on failure."""
    before = signature(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.upload-', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        shutil.copyfile(source, temporary)
        if source.suffix == '.json':
            json.loads(temporary.read_text())
        elif source.suffix == '.jsonl':
            for line in temporary.read_text().splitlines():
                json.loads(line)
        if signature(source) != before:
            raise RuntimeError('Source changed while being copied')
        source_hash, copied_hash = checksum(source), checksum(temporary)
        if source_hash != copied_hash or signature(source) != before:
            raise RuntimeError('Source changed or backup checksum mismatch')
        os.replace(temporary, destination)
        return {'bytes': before[0], 'sha256': copied_hash}, before
    finally:
        temporary.unlink(missing_ok=True)


def backup_sources(output, checkpoint):
    if output.exists():
        for source in sorted(output.rglob('*')):
            if not source.is_file() or source.name.startswith('.'):
                continue
            # A completed projection report is written after stage tensor files.
            if source.suffix == '.pt' and not (source.parent / 'projection_probe.json').exists():
                continue
            yield source, Path('results') / source.relative_to(output)
    if checkpoint.exists():
        common = checkpoint / 'baseline/common_step_100.pt'
        # Calibration is saved only after the common checkpoint has closed.
        if common.exists() and (output / 'calibration.json').exists():
            yield common, Path('checkpoints/baseline/common_step_100.pt')
        for arm in ('baseline', 'uniform_top8', 'hardest_real'):
            latest = checkpoint / arm / 'latest.pt'
            # Do not copy an actively overwritten training checkpoint.
            if latest.exists() and (output / arm / 'completion.json').exists():
                yield latest, Path('checkpoints') / arm / 'latest.pt'


class RunBackup:
    def __init__(self, output, checkpoint, destination):
        self.output, self.checkpoint, self.destination = map(Path, (output, checkpoint, destination))
        self.copied_signatures, self.inventory = {}, {}

    def sync(self):
        copied, errors = 0, []
        for source, relative in backup_sources(self.output, self.checkpoint):
            key = str(relative)
            try:
                if self.copied_signatures.get(key) == signature(source):
                    continue
                entry, source_signature = copy_verified(source, self.destination / relative)
                self.copied_signatures[key], self.inventory[key] = source_signature, entry
                copied += 1
            except (OSError, ValueError, RuntimeError) as error:
                errors.append({'file': key, 'error': str(error)})
        manifest = {'run_id': self.output.name, 'utc': datetime.now(timezone.utc).isoformat(),
                    'copied_this_pass': copied, 'verified_files': self.inventory, 'errors': errors,
                    'checkpoint_policy': 'common step 100 and latest checkpoint of each completed arm; no arbitrary mid-epoch resume'}
        self.destination.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix='.manifest-', dir=self.destination)
        try:
            with os.fdopen(descriptor, 'w') as handle:
                json.dump(manifest, handle, indent=2)
            os.replace(name, self.destination / 'backup_manifest.json')
        finally:
            Path(name).unlink(missing_ok=True)
        return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-directory', required=True)
    parser.add_argument('--checkpoint-directory', required=True)
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if args.interval < 10:
        parser.error('Use an interval of at least 10 seconds')
    output = Path(args.output_directory)
    destination = Path('/content/drive/MyDrive/OTCO/early_pulse') / output.name
    worker = RunBackup(output, args.checkpoint_directory, destination)
    while True:
        # Recheck every pass: users can authorize Drive after training starts.
        if not os.path.ismount('/content/drive') or not Path('/content/drive/MyDrive').is_dir():
            print('BACKUP_WAITING_FOR_DRIVE: mount /content/drive in the notebook', flush=True)
        else:
            try:
                manifest = worker.sync()
                print(json.dumps({'backup_destination': str(destination),
                    'copied_this_pass': manifest['copied_this_pass'],
                    'verified_file_count': len(manifest['verified_files']),
                    'errors': manifest['errors'], 'utc': manifest['utc']}), flush=True)
                run_manifest = output / 'run_manifest.json'
                if run_manifest.exists() and not manifest['errors']:
                    status = json.loads(run_manifest.read_text()).get('status')
                    if status in ('complete', 'interrupted_or_failed'):
                        print('BACKUP_FINAL_SYNC_COMPLETE: ' + status, flush=True)
                        return
            except (OSError, ValueError, RuntimeError) as error:
                print('BACKUP_RETRY: ' + str(error), flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
