"""Regenerate usefulness graphs from committed numeric evidence; no training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.clip_paired_intermediate import plot as plot_intermediate
from src.clip_usefulness_prospective import plot as plot_prospective


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-directory',type=Path,required=True)
    output=parser.parse_args().output_directory
    output.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    intermediate=ROOT/'experiment_results/clip_paired_intermediate_2026-09/results'
    prospective=ROOT/'experiment_results/clip_usefulness_prospective_2026-09/results'
    plot_intermediate(output,read(intermediate/'per_seed_summary.json'))
    plot_prospective(output,read(prospective/'states.json'),read(prospective/'prediction_report.json'))
    print('Four PNG/SVG plots saved:',output)


if __name__=='__main__':
    main()
