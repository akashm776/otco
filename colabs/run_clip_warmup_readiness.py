"""A100 Colab runner. Supply an immutable --source-commit when launching."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.source_commit):
        raise ValueError('Use the full immutable source commit')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                  '--format=csv,noheader,nounits'], text=True).splitlines()[0]
    if 'A100' not in gpu or int(gpu.split(',')[1].strip()) < 14000:
        raise RuntimeError(f'A100 required; found {gpu}')
    run_id = 'clip_warmup_readiness_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    repo = Path('/content') / ('otco_source_' + run_id)
    output = Path('/content/otco_outputs') / run_id
    checkpoint = Path('/content/otco_checkpoints') / run_id
    stdout = Path('/content') / (run_id + '_stdout.txt')
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/akashm776/otco.git', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', args.source_commit], check=True)
    os.chdir(repo)
    assert subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip() == args.source_commit
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'datasets>=2.21.0,<3.0.0',
                    'transformers==4.57.3', 'pyyaml>=6.0.1', 'pytest', 'matplotlib'], check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_clip_warmup_readiness.py',
                    'tests/test_clip_gradient_stages.py', 'tests/test_clip_training.py',
                    'tests/test_clip_negative_gradient_geometry.py',
                    'tests/test_clip_negative_gradient_geometry_randomized.py'], check=True)
    manifest = {'run_id': run_id, 'source_commit': args.source_commit, 'gpu': gpu,
                'python': sys.version, 'status': 'running',
                'packages': {name: importlib.metadata.version(name) for name in
                             ['torch','torchvision','transformers','datasets','numpy','PyYAML']}}
    command = [sys.executable, '-u', '-m', 'src.clip_warmup_readiness',
               '--output-directory', str(output), '--checkpoint-directory', str(checkpoint)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print('WARMUP_OUTPUT:', output, flush=True)
    try:
        with stdout.open('w') as log:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
        completed = json.loads((output / 'completion.json').read_text())
        assert completed['status'] == 'complete' and completed['completed_updates'] == 1001
        manifest['status'] = 'complete'
    except BaseException as error:
        manifest['status'] = 'interrupted_or_failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        output.mkdir(parents=True, exist_ok=True)
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        if stdout.exists():
            shutil.copyfile(stdout, output / 'stdout.txt')
        archive_path = Path('/content') / f"{run_id}_{manifest['status']}.zip"
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(output.parent))
        print('ARTIFACT_READY:', archive_path, flush=True)
        print('CHECKPOINT_DIRECTORY:', checkpoint, flush=True)


if __name__ == '__main__':
    main()
