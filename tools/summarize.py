#!/usr/bin/env python3
"""Summarize ab_bench raw runs: medians per experiment and case.

Usage:
  python3 tools/summarize.py [results_dir]

Reads LABEL-rN.json files (the output of tools/ab_run4.py) and prints a
markdown table of medians. Client-class numbers: TTFT and tok/s are measured
on the client, not inside the engine.
"""
import glob
import json
import os
import re
import statistics
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'v100-16gb-linux')

CASE_ORDER = ['short-1k', 'medium-8k', 'long-32k']


def label_of(doc, path):
    label = doc.get('label') or os.path.basename(path)
    return re.sub(r'(-r\d+)?(\.json)?$', '', label)


def load_runs(root):
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, '*-r*.json'))):
        if path.endswith('-summary.json'):
            continue
        try:
            with open(path, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f'skip {path}: {exc}', file=sys.stderr)
            continue
        label = label_of(doc, path)
        runs.setdefault(label, []).append(doc)
    return runs


def median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def fmt(value, digits=2):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def main():
    runs = load_runs(ROOT)
    if not runs:
        print(f'no raw runs in {ROOT}', file=sys.stderr)
        return 1

    print('| Experiment | n | ' + ' | '.join(
        f'{case} dec / TTFT' for case in CASE_ORDER) + ' |')
    print('|---|---:|' + '---:|' * len(CASE_ORDER))
    for label in sorted(runs):
        docs = runs[label]
        cells = []
        for case in CASE_ORDER:
            dec, ttft = [], []
            for doc in docs:
                for item in doc.get('cases', []):
                    if item.get('case') != case or 'error' in item:
                        continue
                    dec.append(item.get('decode_tok_s'))
                    ttft.append(item.get('ttft_s'))
            d, t = median(dec), median(ttft)
            cells.append(f'{fmt(d, 1)} / {fmt(t, 2)}s')
        print(f'| {label} | {len(docs)} | ' + ' | '.join(cells) + ' |')

    print()
    print('Medians over the runs of each label. Raw files: '
          f'`{os.path.relpath(ROOT).replace(os.sep, "/")}/`.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

