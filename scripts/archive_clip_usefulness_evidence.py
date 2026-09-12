"""Audit and export completed usefulness studies without captions/checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_clip_usefulness_predictors import load_archive, analyze


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    studies = [
        ('clip_paired_intermediate_2026-09', 'clip_paired_intermediate_20260912T131926_801067Z', 'audit_clip_paired_intermediate.py'),
        ('clip_usefulness_prospective_2026-09', 'clip_usefulness_prospective_20260912T175040_066114Z', 'audit_clip_usefulness_prospective.py')]
    targets = [args.output_root/name for name,_,_ in studies] + [args.output_root/'clip_usefulness_predictors_2026-09']
    if any(p.exists() for p in targets):
        raise FileExistsError('Choose an output root without existing study exports; never overwrite evidence')
    with tempfile.TemporaryDirectory(prefix='otco-evidence-export-') as temporary:
        for name,run,script in studies:
            audited = Path(temporary)/name
            subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(args.downloads/(run+'_complete.zip')),
                            '--output-directory',str(audited)],check=True)
            target = args.output_root/name
            target.mkdir(parents=True,exist_ok=False)
            for path in (audited/run).rglob('*'):
                if not path.is_file() or path.suffix not in {'.json','.jsonl','.yaml'}:
                    continue
                if path.name in {'training_batches.json','heldout_partitions.json','diagnostic_holdout_indices.json'}:
                    continue
                dest = target/path.relative_to(audited/run)
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,dest)
            shutil.copyfile(audited/'audit.json',target/'audit.json')
            for path in (audited/run/'results').glob('*'):
                if path.suffix in {'.png','.svg'}:
                    shutil.copyfile(path,target/'results'/path.name)
        screening = args.downloads/(studies[0][1]+'_complete_predictor_analysis')
        report = json.loads((screening/'analysis.json').read_text())
        archive = args.downloads/(studies[0][1]+'_complete.zip')
        states,manifest = load_archive(archive)
        folds,predictions = analyze(states)
        assert report['states']==states and report['folds']==folds and report['predictions']==predictions
        assert report['source_manifest']==manifest
        assert report['archive_sha256']==hashlib.sha256(archive.read_bytes()).hexdigest()
        assert report['script_sha256']==hashlib.sha256((ROOT/'scripts/analyze_clip_usefulness_predictors.py').read_bytes()).hexdigest()
        import csv
        for name,rows in [('states.csv',states),('predictions.csv',predictions)]:
            with (screening/name).open() as handle:
                assert list(csv.DictReader(handle))==[{k:str(v) for k,v in r.items()} for r in rows]
        target=targets[-1]
        target.mkdir(parents=True,exist_ok=False)
        inventory={}
        for name in ['analysis.json','states.csv','predictions.csv','README.md']:
            shutil.copyfile(screening/name,target/name)
            inventory[name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
        (target/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print('Verified exports:',*[str(p) for p in targets],sep='\n')


if __name__=='__main__':
    main()
