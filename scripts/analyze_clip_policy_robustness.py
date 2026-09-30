"""Standard-library-only, paired robustness audit of one completed policy run.

No training, model loading, new predictions, tuning, or independent-test claim.
All source files are authenticated before any result is interpreted.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import statistics as stats
import zipfile

SPEC = dict(experiment='clip_policy_robustness_v1',
    source_run='clip_policy_test_20260930T050231_622869Z',
    source_bundle='7b5253943005c684a1c730d9b2568acd0cc2c5883bdf630e49ca7271fb09cf73',
    backup_manifest_sha256='0d974b89ec799cc3a90ec384b13e8382f9864884ac9a33ff4d30740213c5e67e',
    seeds=[789, 2026, 31415], checkpoints=[100, 500], streams=[2026093002, 2026093003],
    partitions=['2026092403', '2026092404'], endpoint=50,
    controls=['native', 'sustained', 'random_matched'], optimizer_updates=0,
    analysis_selected_after_results=True, independent_test_set=False, rules_refitted=False)
MAX_FILE_BYTES = 32*1024**2
MAX_TOTAL_BYTES = 256*1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, label):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12),
            f'Arithmetic mismatch: {label}')


def safe_name(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name,
            f'Unsafe path: {name}')
    return p


def checked_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f'Nonfinite JSON: {value}')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def verify_input(path, spec=SPEC):
    """Read a directory or ZIP without extraction; refuse unsafe/changed evidence."""
    path = Path(path)
    files = {}
    if path.is_dir():
        require(not path.is_symlink(), 'Symlink input directory')
        total = 0
        for member in path.rglob('*'):
            require(not member.is_symlink(), 'Symlink evidence member')
            if not member.is_file():
                continue
            name = member.relative_to(path).as_posix()
            if member.name == '.DS_Store':
                continue
            safe_name(name)
            total += member.stat().st_size
            require(member.stat().st_size <= MAX_FILE_BYTES and total <= MAX_TOTAL_BYTES, 'Oversized evidence')
            files[name] = member.read_bytes()
    else:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), 'Duplicate ZIP members')
            require(sum(i.file_size for i in archive.infolist()) <= MAX_TOTAL_BYTES, 'Oversized ZIP')
            prefix = spec['source_run']+'/'
            for member in archive.infolist():
                safe_name(member.filename)
                require((member.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink ZIP member')
                require(member.filename.startswith(prefix), 'Wrong ZIP root')
                if member.is_dir():
                    continue
                require(member.file_size <= MAX_FILE_BYTES, 'Oversized ZIP member')
                name = member.filename[len(prefix):]
                files[name] = archive.read(member)  # CRC checked by zipfile during read.
    marker = 'DRIVE_BACKUP_MANIFEST.json'
    require(marker in files, 'Missing backup manifest')
    digest = hashlib.sha256(files[marker]).hexdigest()
    require(digest == spec['backup_manifest_sha256'], 'Wrong or changed pinned backup manifest')
    manifest = checked_json(files[marker])
    require(manifest['run_id'] == spec['source_run'] and manifest['status'] == 'verified_tree', 'Incomplete source backup')
    require(set(files) == set(manifest['files']) | {marker}, 'Missing or unmanifested evidence')
    for name, record in manifest['files'].items():
        safe_name(name)
        data = files[name]
        require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'],
                f'Checksum mismatch: {name}')
    return files, dict(run_id=manifest['run_id'], backup_manifest_sha256=digest,
        verified_files=len(manifest['files']), source_unchanged=True)


def extract_rows(files, spec=SPEC):
    read = lambda name: checked_json(files[name])
    run, completion = read('run_manifest.json'), read('completion.json')
    require(run['bundle_id'] == spec['source_bundle'] and run['status'] == 'complete_pending_drive_flush',
            'Wrong source bundle or run status')
    require(completion['status'] == 'complete' and completion['pairs'] == 12
            and completion['branches'] == 48 and completion['optimizer_steps'] == 2400, 'Incomplete grid')
    roles = read('data_roles.json')['report_partitions']
    require(set(roles) == set(spec['partitions']), 'Wrong reporting partitions')
    for value in roles.values():
        batches = value['batches']
        require(len(batches) == 8 and all(len(b) == 64 for b in batches)
                and sorted(i for b in batches for i in b) == list(range(512)), 'Invalid reporting batch membership')
    rows = []
    expected_paths = set()
    for seed, step, stream in itertools.product(spec['seeds'], spec['checkpoints'], spec['streams']):
        path = f'seed_{seed}/step_{step:06d}/stream_{stream}/result.json'
        expected_paths.add(path)
        row = read(path)
        require((row['training_seed'], row['checkpoint_step'], row['stream_seed']) == (seed, step, stream), 'Wrong pair identity')
        audit = row['audit']
        require(audit['actual_optimizer_steps'] == 200 and all(audit[k] is True for k in
            ['source_immutable', 'frozen_parameters_unchanged', 'observations_preserve_state',
             'exact_no_update_report_replay', 'matched_rng_and_lrs', 'exposure_matched'])
            and audit['reports_control_treatment'] is False, 'Failed source audit')
        require(sum(row['actions']['gated']) == sum(row['actions']['random_matched']), 'Unmatched exposure')
        require(all(len(v) == 50 for v in row['traces'].values()), 'Incomplete update traces')
        endpoint = str(spec['endpoint'])
        reports = {arm: row['points'][arm][endpoint]['report'] for arm in ['gated', *spec['controls']]}
        for arm, report in reports.items():
            require(set(report['partition_losses']) == set(spec['partitions']), 'Reporting partitions differ')
            require(all(len(v) == 8 for v in report['partition_losses'].values()), 'Missing reporting batches')
            close(report['native_loss'], stats.fmean(v for vs in report['partition_losses'].values() for v in vs), arm)
        contrasts = {}
        for control in spec['controls']:
            parts = {part: [g-c for g, c in zip(reports['gated']['partition_losses'][part],
                                               reports[control]['partition_losses'][part])]
                     for part in spec['partitions']}
            loss = stats.fmean(stats.fmean(v) for v in parts.values())
            close(loss, row['differences'][endpoint][control]['report_loss'], control)
            for part, values in parts.items():
                close(stats.fmean(values), row['differences'][endpoint][control]['partition_loss'][part], part)
            r1 = reports['gated']['pool_retrieval']['mean_r1_percent']-reports[control]['pool_retrieval']['mean_r1_percent']
            close(r1, row['differences'][endpoint][control]['pool_r1_pp'], 'R1')
            contrasts[control] = dict(loss=loss, r1_pp=r1, batches=parts)
        rows.append(dict(seed=seed, checkpoint=step, stream=stream, contrasts=contrasts,
                         exposure=row['exposure']['gated'], random_equals_gate=row['random_schedule_equals_gate']))
    require({name for name in files if name.endswith('/result.json')} == expected_paths, 'Extra/duplicate result grid')
    comparison = read('comparison.json')
    mean, seeds = paired_mean(rows, lambda r: r['contrasts']['random_matched']['loss'])
    close(mean, comparison['primary_value'], 'primary')
    require({v['training_seed'] for v in comparison['per_seed']} == set(spec['seeds']), 'Wrong seed summary grid')
    for item in comparison['per_seed']:
        close(seeds[str(item['training_seed'])], item['primary'], 'per-seed primary')
    return rows


def paired_mean(rows, value):
    """Equal streams within state, equal states within seed, equal seeds."""
    require(bool(rows), 'Empty contrast')
    seeds = {}
    for seed in sorted({r['seed'] for r in rows}):
        states = []
        for step in sorted({r['checkpoint'] for r in rows if r['seed'] == seed}):
            states.append(stats.fmean(value(r) for r in rows if (r['seed'], r['checkpoint']) == (seed, step)))
        seeds[str(seed)] = stats.fmean(states)
    return stats.fmean(seeds.values()), seeds


def sign_flip_reference(values):
    """Exact sign-symmetry reference, NOT a randomized-policy significance test."""
    require(bool(values) and len(values) <= 12, 'Invalid sign-reference size')
    observed = stats.fmean(values)
    distribution = [stats.fmean(s*v for s, v in zip(signs, values))
                    for signs in itertools.product([-1, 1], repeat=len(values))]
    tail = sum(abs(v) >= abs(observed)-1e-15 for v in distribution)
    return dict(seed_count=len(values), observed=observed, enumerated_means=distribution,
        two_sided_tail_fraction=tail/len(distribution), tail_count=tail,
        assumptions='Independent seed-level effects symmetric about zero under the null. '
        'Not justified by treatment randomization here; exploratory reference only.',
        warning='Three reused training seeds; eight sign patterns. Do not treat 12 pairs or 16 reporting batches as independent seeds.')


def analyze(rows, spec=SPEC):
    controls = {}
    for control in spec['controls']:
        effect = lambda r: r['contrasts'][control]['loss']
        mean, seeds = paired_mean(rows, effect)
        partitions = {part: dict(zip(['mean', 'per_seed'], paired_mean(rows,
            lambda r: stats.fmean(r['contrasts'][control]['batches'][part])))) for part in spec['partitions']}
        leave_out = {}
        for axis, values in [('seed', spec['seeds']), ('stream', spec['streams']), ('checkpoint', spec['checkpoints'])]:
            leave_out[axis] = {str(v): paired_mean([r for r in rows if r[axis] != v], effect)[0] for v in values}
        strata = []
        for part, step, stream in itertools.product(spec['partitions'], spec['checkpoints'], spec['streams']):
            selected = [r for r in rows if r['checkpoint'] == step and r['stream'] == stream]
            value, per_seed = paired_mean(selected, lambda r: stats.fmean(r['contrasts'][control]['batches'][part]))
            strata.append(dict(partition=part, checkpoint=step, stream=stream, mean=value, per_seed=per_seed))
        deletions = []
        # Delete the same paired batch index across all arms/seeds/states/streams
        # within ONE partition; retain equal weight for the two partitions.
        for part in spec['partitions']:
            for index in range(8):
                def deleted(row):
                    batches = row['contrasts'][control]['batches']
                    return stats.fmean(stats.fmean(v for j, v in enumerate(values) if k != part or j != index)
                                       for k, values in batches.items())
                value, per_seed = paired_mean(rows, deleted)
                deletions.append(dict(partition=part, omitted_batch=index, mean=value, per_seed=per_seed,
                                      shift_from_original=value-mean))
        a, b = [partitions[k]['mean'] for k in spec['partitions']]
        crossing = -b/(a-b) if a != b else None
        if crossing is not None and not 0 <= crossing <= 1:
            crossing = None
        controls[control] = dict(mean=mean, per_seed=seeds,
            mean_r1_pp=paired_mean(rows, lambda r: r['contrasts'][control]['r1_pp'])[0],
            pair_wins=sum(effect(r)<0 for r in rows), pair_ties=sum(effect(r)==0 for r in rows),
            pair_losses=sum(effect(r)>0 for r in rows), partitions=partitions,
            leave_one_out=leave_out, strata=strata, paired_batch_deletions=deletions,
            partition_A_weight_zero_crossing=crossing,
            partition_weighting_note='w*A+(1-w)*B. Sensitivity only; do not choose w to improve the outcome.',
            seed_sign_symmetry_reference=sign_flip_reference(list(seeds.values())))
    return dict(protocol=spec, pairs=len(rows), training_seed_count=len(spec['seeds']), controls=controls,
        paired_rows=rows,
        limitations=[
            'Post-hoc diagnostic of reused states, seeds and reporting images, not independent confirmation.',
            'Both partitions contain the same 512 images; differences reflect contrastive batch grouping.',
            'Batch deletion is a paired influence diagnostic, not a new image-level or partition bootstrap.',
            'No confidence interval is claimed from only three reused seeds. Sign symmetry is an explicit assumption, not a design guarantee.',
            'New reporting partitions cannot be evaluated from batch-loss summaries; final embeddings or saved branch states would be needed.',
            'Loss changes and retrieval changes are separate endpoints. No threshold or policy selection is performed.'])


def render(result):
    primary = result['controls']['random_matched']
    lines = ['# Policy-test robustness diagnostic', '',
        'Completed offline: zero optimizer updates. Source inputs authenticated; no source files modified.', '',
        f"Source: `{result['protocol']['source_run']}`. Verified {result['provenance']['verified_files']} manifest-listed files.", '',
        '## Endpoint comparisons', '', 'Gated minus comparator at update 50; negative loss favors gating.', '',
        '| Comparator | Mean loss difference | R@1 difference (pp) | Pair wins / ties / losses |',
        '|---|---:|---:|---:|']
    for name, v in result['controls'].items():
        lines.append(f"| {name} | {v['mean']:+.9f} | {v['mean_r1_pp']:+.6f} | {v['pair_wins']} / {v['pair_ties']} / {v['pair_losses']} |")
    lines += ['', '## Primary timing contrast: sensitivity', '',
        '| Check | Loss difference |', '|---|---:|']
    for part, v in primary['partitions'].items():
        lines.append(f"| Reporting partition {part} | {v['mean']:+.9f} |")
    for axis, values in primary['leave_one_out'].items():
        for omitted, value in values.items():
            lines.append(f'| Omit {axis} {omitted} | {value:+.9f} |')
    values = [v['mean'] for v in primary['paired_batch_deletions']]
    ref = primary['seed_sign_symmetry_reference']
    lines += ['', f"Paired single-batch deletion range: {min(values):+.9f} to {max(values):+.9f}; "
              f"{sum(v < 0 for v in values)}/{len(values)} deletions retain a favorable sign.", '',
        f"Partition-A weighting zero crossing: {primary['partition_A_weight_zero_crossing']}. "
        'The original analysis weights the partitions equally; this diagnostic does not change those weights.', '',
        f"Seed sign-symmetry reference: {ref['tail_count']}/{len(ref['enumerated_means'])} patterns "
        f"are at least as extreme in absolute mean (fraction {ref['two_sided_tail_fraction']:.3f}). "
        'This is NOT a confirmatory randomized-treatment p-value or proof of no effect.', '',
        '## Interpretation', '',
        'Read the partition and stream effects alongside the averaged primary. A negative average '
        'does not establish robustness if these signs disagree. A timing advantage does not by itself '
        'establish improvement over native training. Keep all fixed strata; do not select favorable ones.', '',
        '## Limits', '']
    lines.extend('- '+v for v in result['limitations'])
    lines += ['', 'Full paired values, all strata and influence checks are in `analysis.json`. No captions, images, or model weights are exported.', '']
    return '\n'.join(lines)


def main(source, output):
    output = Path(output)
    require(not output.exists(), f'Refusing to overwrite analysis: {output}')
    files, provenance = verify_input(source)
    result = analyze(extract_rows(files))
    result['provenance'] = provenance
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'REPORT.md').write_text(render(result))
    inventory = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in output.iterdir() if p.is_file()}
    (output/'completion.json').write_text(json.dumps(dict(status='complete', optimizer_updates=0,
        source_run=SPEC['source_run'], files=inventory), indent=2)+'\n')
    print(render(result))
    print('OFFLINE_ANALYSIS_COMPLETE:', output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', required=True, help='Completed policy-test directory or downloaded ZIP')
    parser.add_argument('--output', required=True, help='New analysis directory; never overwritten')
    args = parser.parse_args()
    main(args.input, args.output)
