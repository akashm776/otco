"""Paste into ONE NEW cell in the SAME Colab runtime. No training or Drive.

Downloads the small results first. Then select and download checkpoints one at
a time (~0.7 GB each). Keep the runtime until your browser has saved every file.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

RUN_ID = 'clip_evaluation_transfer_20260914T130457_447373Z'
ARCHIVE = Path('/content') / ('otco_transfer_' + RUN_ID) / (RUN_ID + '_complete.zip')


def stream_digest(handle):
    value = hashlib.sha256()
    for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
        value.update(chunk)
    return value.hexdigest()


def inspect_archive(archive, expected_run_id):
    if not archive.is_file():
        raise FileNotFoundError(f'Existing archive not found: {archive}. '
                                'Reconnect to the original runtime. This cell will NOT retrain.')
    with zipfile.ZipFile(archive) as bundle:
        packed = json.loads(bundle.read('PACKING_MANIFEST.json'))
        if packed['run_id'] != expected_run_id or packed['status'] != 'complete':
            raise ValueError('Wrong run or incomplete archive')
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != set(packed['files']) | {'PACKING_MANIFEST.json'}:
            raise ValueError('Archive inventory mismatch')
        for name, record in packed['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != expected_run_id:
                raise ValueError('Unsafe archive member path')
            if bundle.getinfo(name).file_size != record['bytes']:
                raise ValueError(f'Archive member size mismatch: {name}')
        required = [f'{expected_run_id}/seed_{seed}/checkpoints/baseline/{name}'
                    for seed in [789, 2026, 31415] for name in
                    ['common_step_100.pt', 'step_000250.pt', 'step_000500.pt', 'step_000750.pt', 'latest.pt']]
        if any(name not in packed['files'] for name in required):
            raise ValueError('Archive does not contain all 15 required checkpoints')
        marker = json.loads(bundle.read(expected_run_id + '/results/completion.json'))
        if marker != dict(status='complete', training_seeds=[789,2026,31415],
                          checkpoint_steps=[100,250,500,750,1001], replay_branch_rows=720,
                          transfer_branch_rows=720, evaluated_states=15, rules_refitted=False):
            raise ValueError('Study completion marker is invalid')
    checkpoints = sorted(name for name in packed['files'] if '/checkpoints/' in name and name.endswith('.pt'))
    reports = sorted(set(packed['files']) - set(checkpoints))
    return packed, required + [name for name in checkpoints if name not in required], reports


def verify_subset(path, expected):
    with zipfile.ZipFile(path) as bundle:
        manifest = json.loads(bundle.read('RECOVERY_MANIFEST.json'))
        if manifest != expected:
            raise ValueError('Existing recovery file belongs to different contents')
        if set(bundle.namelist()) != set(expected['files']) | {'RECOVERY_MANIFEST.json'}:
            raise ValueError('Recovery file inventory mismatch')
        for name, record in expected['files'].items():
            if bundle.getinfo(name).file_size != record['bytes']:
                raise ValueError('Recovery member size mismatch')
            with bundle.open(name) as handle:
                if stream_digest(handle) != record['sha256']:
                    raise ValueError(f'Recovery checksum mismatch: {name}')


def export_subset(archive, packed, selected, destination):
    """Copy selected members as bytes, retaining source paths and checking SHA256."""
    manifest = dict(run_id=packed['run_id'], bundle_id=packed['bundle_id'],
        scope='Only listed files are included; this is NOT the full checkpoint archive.',
        files={name:packed['files'][name] for name in selected})
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_subset(destination, manifest)
        return destination
    required_space = sum(record['bytes'] for record in manifest['files'].values()) + 64*1024**2
    if shutil.disk_usage(destination.parent).free < required_space:
        raise RuntimeError('Not enough runtime disk to package this download; original archive is unchanged')
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.recovery-', suffix='.partial', delete=False) as handle:
        temporary = Path(handle.name)
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(temporary, 'w', allowZip64=True) as target:
        for name in selected:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED if name.endswith('.pt') else zipfile.ZIP_DEFLATED
            value = hashlib.sha256()
            with source.open(name) as reader, target.open(info, 'w', force_zip64=True) as writer:
                for chunk in iter(lambda: reader.read(8*1024**2), b''):
                    value.update(chunk)
                    writer.write(chunk)
            if value.hexdigest() != manifest['files'][name]['sha256']:
                raise ValueError(f'Source checksum mismatch: {name}. Nothing has been deleted.')
        target.writestr('RECOVERY_MANIFEST.json', json.dumps(manifest, indent=2))
    verify_subset(temporary, manifest)
    temporary.replace(destination)
    return destination


def run_recovery(archive=ARCHIVE, run_id=RUN_ID):
    from google.colab import files
    import ipywidgets as widgets
    from IPython.display import display

    archive = Path(archive)
    packed, checkpoints, reports = inspect_archive(archive, run_id)
    print('Found the completed run. All 15 required checkpoint members are present.')
    print('Each requested download is hash-verified; the full archive is left unchanged.')
    print('First save the RESULTS ZIP. Then select one checkpoint at a time and click Download selected.')
    print('Wait for each browser download to FINISH before requesting the next. Requests are not proof of saved files.')
    destination = archive.parent / 'recovery_downloads'
    choices = [('RESULTS FIRST — records, scores, graphs, logs (no model checkpoints)', 'results')]
    choices.extend((name.removeprefix(run_id+'/') + f' ({packed["files"][name]["bytes"]/1e6:.0f} MB)', name)
                   for name in checkpoints)
    selected = widgets.Dropdown(options=choices, layout=widgets.Layout(width='95%'))
    button = widgets.Button(description='Download selected', button_style='primary',
                            layout=widgets.Layout(width='200px'))
    output = widgets.Output()

    def download_selected(_):
        button.disabled = True
        try:
            with output:
                output.clear_output(wait=True)
                if selected.value == 'results':
                    members, filename = reports, run_id + '_results_only.zip'
                else:
                    members = [selected.value]
                    relative = PurePosixPath(selected.value).relative_to(run_id)
                    filename = run_id + '_' + relative.parts[0] + '_' + relative.name.removesuffix('.pt') + '.zip'
                print('Packaging/verifying:', filename, flush=True)
                path = export_subset(archive, packed, members, destination/filename)
                print(f'Ready: {path.stat().st_size/1e6:.1f} MB\n{path}', flush=True)
                print('If automatic download fails, download this smaller ZIP through Colab Files.', flush=True)
                files.download(str(path))
        except Exception as error:
            with output:
                print(type(error).__name__ + ': ' + str(error))
        finally:
            button.disabled = False

    button.on_click(download_selected)
    display(selected, button, output)
    download_selected(None)


if __name__ == '__main__':
    run_recovery()
