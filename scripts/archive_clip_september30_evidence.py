"""Archive pinned small September 29–30 evidence, preserving partial-run status.

No weights, raw images, caption-bearing data, or personal PDFs enter this export.
Original downloads are read only. Partial results are never additional samples.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from scripts.archive_clip_followup_evidence import DATA_FILES, EXPORT_SUFFIXES, digest, safe_path, verify

STUDIES = {
    'clip_continuation_timecourse_2026-09': (
        'clip_continuation_timecourse_20260929T135631_790370Z',
        '5c5b35a05d384bc84af14aacc113152777cc59a5ebc90c0c051b0e58aa2066e2', True),
    'clip_marginal_utility_partial_2026-09': (
        'clip_marginal_utility_20260930T000856_646786Z',
        'c4fadc28345168e70029cd43594dab4790437e90725765944c2c74064c26f2a0', False),
    'clip_marginal_utility_2026-09': (
        'clip_marginal_utility_20260930T021908_849917Z',
        '5875f43f8e9fabc36af23260c6fd285b8888896fbae39e481c8b754ba6816406', True),
    'clip_policy_test_2026-09': (
        'clip_policy_test_20260930T050231_622869Z',
        '0d974b89ec799cc3a90ec384b13e8382f9864884ac9a33ff4d30740213c5e67e', True),
}


def read_archive(path, run, manifest_sha, complete):
    files = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len({i.filename for i in infos}) != len(infos):
            raise ValueError('Duplicate ZIP members')
        if sum(i.file_size for i in infos) > 256*1024**2:
            raise ValueError('Oversized evidence ZIP')
        for info in infos:
            safe_path(info.filename)
            if not info.filename.startswith(run+'/') or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Wrong root or symlink')
            if info.is_dir():
                continue
            if info.file_size > 32*1024**2:
                raise ValueError('Oversized evidence member')
            files[info.filename[len(run)+1:]] = archive.read(info)
    marker = 'DRIVE_BACKUP_MANIFEST.json'
    if hashlib.sha256(files[marker]).hexdigest() != manifest_sha:
        raise ValueError('Changed pinned manifest')
    manifest = json.loads(files[marker])
    if manifest['run_id'] != run or manifest['status'] != ('verified_tree' if complete else 'in_progress'):
        raise ValueError('Wrong identity/status')
    if set(files) != set(manifest['files']) | {marker}:
        raise ValueError('Missing or unmanifested files')
    for name, expected in manifest['files'].items():
        safe_path(name)
        if digest(files[name]) != {k: expected[k] for k in ['bytes', 'sha256']}:
            raise ValueError(f'Changed evidence: {name}')
    if complete:
        if json.loads(files['completion.json'])['status'] != 'complete':
            raise ValueError('Incomplete experiment')
    elif 'completion.json' in files:
        raise ValueError('Partial run unexpectedly has completion marker')
    return files


def contains_caption(value):
    if isinstance(value, dict):
        return 'caption' in value or any(contains_caption(v) for v in value.values())
    return isinstance(value, list) and any(contains_caption(v) for v in value)


def export(downloads, output, name, spec):
    run, manifest_sha, complete = spec
    target = output/name
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite: {target}')
    archives = sorted(downloads.glob(run+'*.zip'))
    if len(archives) != 1:
        raise ValueError(f'Need one complete ZIP for {run}')
    files = read_archive(archives[0], *spec)
    selected, omitted = {}, []
    for relative, data in files.items():
        path = PurePosixPath(relative)
        if path.name in DATA_FILES or path.suffix not in EXPORT_SUFFIXES:
            omitted.append(relative)
            continue
        if path.suffix == '.json' and contains_caption(json.loads(data)):
            raise ValueError(f'Unclassified caption-bearing evidence: {relative}')
        selected[relative] = data
    audit = dict(run_id=run, status='small_evidence_verified', experiment_complete=complete,
        source_backup_status='verified_tree' if complete else 'in_progress',
        archives=[dict(name=archives[0].name, **digest(archives[0].read_bytes()))],
        manifest_sha256=manifest_sha, source_members_verified=len(files)-1,
        other_files_omitted=sorted(omitted), files={k: digest(v) for k, v in selected.items()},
        scope='ZIP CRC and pinned manifest SHA256/size for every source file. Original files unchanged. '
              'No raw captions, images or checkpoint tensors retained. A verified partial export '
              'does not mean a completed experiment; Drive flush is separate from manifest verification.')
    target.mkdir(parents=True)
    for name, data in selected.items():
        destination = target/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (target/'export_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    verify(target)
    print(f'{run}: {len(files)-1} verified, {len(selected)} retained, {len(omitted)} omitted')


def check_partial_overlap(output):
    partial = output/'clip_marginal_utility_partial_2026-09'
    complete = output/'clip_marginal_utility_2026-09'
    results = sorted(partial.glob('seed_*/step_*/stream_*/result.json'))
    if len(results) != 6 or len(list(complete.glob('seed_*/step_*/stream_*/result.json'))) != 12:
        raise ValueError('Unexpected partial/complete pair counts')
    for path in results:
        if path.read_bytes() != (complete/path.relative_to(partial)).read_bytes():
            raise ValueError('Partial pair differs from completed rerun')
    return dict(partial_pairs=6, complete_pairs=12, identical_partial_pairs=6,
        partial_run_is_additional_evidence=False, matched_pair_count_in_analysis=12,
        note='Twelve matched sets are not twelve independent training seeds; there are three seeds.',
        result_files=[str(p.relative_to(partial)) for p in results])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--downloads', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('experiment_results'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and args.downloads is None:
        parser.error('--downloads required for export')
    for name, spec in STUDIES.items():
        if not args.verify_only:
            export(args.downloads, args.output_root, name, spec)
        verify(args.output_root/name)
    overlap = check_partial_overlap(args.output_root)
    target = args.output_root/'clip_marginal_utility_partial_2026-09/overlap_audit.json'
    if args.verify_only:
        if json.loads(target.read_text()) != overlap:
            raise ValueError('Overlap audit differs')
    else:
        target.write_text(json.dumps(overlap, indent=2)+'\n')
    print('Six partial pairs exactly reproduced; excluded from additional sample counts.')


if __name__ == '__main__':
    main()
