#!/usr/bin/env python3
"""Prepare current public benchmark revisions using the recorded formatter."""
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('gsm8k', 'math500', 'humaneval', 'mbpp_sanitized')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/prepared')
    parser.add_argument('--snapshot', choices=('natural', 'serving'), default='natural')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(ROOT / 'code' / args.snapshot / 'src'))
    from gbv_experiments.data import prepare

    print('Preparing current dataset revisions; verify frozen hashes for an exact reproduction.')
    prepare(list(DATASETS), args.output, {'protocol': 'ddtree_counts', 'sample_seed': 0})


if __name__ == '__main__':
    main()
