"""Export small, hash-checked September follow-up evidence; never load weights.

Original ZIPs remain untouched. This validates all non-weight members listed in
the run's manifest, but deliberately does NOT rehash checkpoint tensors. Dataset
captions and large binary artifacts are omitted from the public evidence export.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

STUDIES = {
    "clip_evaluation_transfer_2026-09": "clip_evaluation_transfer_20260922T001127_444974Z",
    "clip_gated_training_2026-09": "clip_gated_training_20260922T114312_608272Z",
    "clip_exposure_matched_2026-09": "clip_exposure_matched_20260923T114154_690094Z",
    "clip_checkpoint_diagnostic_2026-09": "clip_checkpoint_diagnostic_20260924T122020_801040Z",
}
DATA_FILES = {"training_batches.json", "heldout_partitions.json", "pool.json",
              "data_roles.json", "diagnostic_holdout_indices.json"}
WEIGHTS = {".pt", ".pth", ".ckpt"}
EXPORT_SUFFIXES = {".json", ".jsonl", ".yaml", ".py", ".md", ".png", ".svg"}
LIMIT = 10 * 1024 * 1024


def safe_path(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return p


def digest(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def export(downloads, output, name, run):
    target = output / name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {target}")
    archives = sorted(downloads.glob(run + "*.zip"))
    if not archives:
        raise FileNotFoundError(run)
    with ExitStack() as stack:
        members = {}
        for path in archives:
            z = stack.enter_context(zipfile.ZipFile(path))
            for info in z.infolist():
                safe_path(info.filename)
                if info.is_dir():
                    continue
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f"Symlink: {info.filename}")
                if info.filename in members:
                    raise ValueError(f"Duplicate member: {info.filename}")
                members[info.filename] = (z, info)
        manifest_name = ("PACKING_MANIFEST.json" if "PACKING_MANIFEST.json" in members
                         else run + "/DRIVE_BACKUP_MANIFEST.json")

        def read(key):
            z, info = members[key]
            if info.file_size > LIMIT:
                raise ValueError(f"Unexpected oversized evidence: {key}")
            return z.read(info)

        manifest_bytes = read(manifest_name)
        manifest = json.loads(manifest_bytes)
        if manifest["run_id"] != run or manifest["status"] not in {"complete", "verified_tree"}:
            raise ValueError("Unexpected run identity or manifest status")
        selected, omitted, weights, verified = {}, [], [], 0
        expected = {manifest_name}
        for key, record in manifest["files"].items():
            member = key if manifest_name == "PACKING_MANIFEST.json" else run + "/" + key
            relative = str(safe_path(member).relative_to(run))
            expected.add(member)
            if member not in members:
                raise FileNotFoundError(member)
            if PurePosixPath(relative).suffix in WEIGHTS:
                weights.append(relative)
                continue
            data = read(member)
            if digest(data) != {k: record[k] for k in ("bytes", "sha256")}:
                raise ValueError(f"Manifest hash/size mismatch: {member}")
            verified += 1
            p = PurePosixPath(relative)
            if p.name in DATA_FILES or p.suffix not in EXPORT_SUFFIXES:
                omitted.append(relative)
            else:
                selected[relative] = data
        if set(members) != expected:
            raise ValueError(f"Unmanifested archive members: {set(members) - expected}")
        selected[PurePosixPath(manifest_name).name] = manifest_bytes
        inventory = {key: digest(value) for key, value in selected.items()}
        audit = {
            "run_id": run, "status": "small_evidence_verified",
            "archives": [{"name": p.name, "bytes": p.stat().st_size} for p in archives],
            "manifest": {"name": manifest_name, **digest(manifest_bytes)},
            "non_weight_members_verified": verified,
            "checkpoint_hashes_rechecked_by_this_export": False,
            "weights_omitted": weights, "other_files_omitted": omitted,
            "files": inventory,
            "scope": "ZIP CRC and original manifest SHA256/size for all non-weight members. "
                     "Original manifests are provenance, not signatures or evidence of remote Drive flush. "
                     "Original checkpoints and caption-bearing data remain outside Git.",
        }
        target.mkdir(parents=True)
        for key, value in selected.items():
            destination = target / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)
        (target / "export_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(f"{name}: {verified} non-weight members verified; {len(selected)} exported; "
              f"{sum(map(len, selected.values())):,} bytes")


def verify(target):
    audit = json.loads((target / "export_audit.json").read_text())
    for name, expected in audit["files"].items():
        path = target / str(safe_path(name))
        if path.is_symlink() or digest(path.read_bytes()) != expected:
            raise ValueError(f"Changed evidence: {path}")
    print(f"Verified {target.name}: {len(audit['files'])} retained files")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--downloads", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("experiment_results"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only and args.downloads is None:
        parser.error("--downloads is required for export")
    for name, run in STUDIES.items():
        if not args.verify_only:
            export(args.downloads, args.output_root, name, run)
        verify(args.output_root / name)


if __name__ == "__main__":
    main()
