#!/usr/bin/env python3
"""Validate raw ab_bench JSON files under results/.

Usage:
  python3 tools/validate_results.py [results_dir]

Checks that every file parses and that every run has the fields the tables
rely on. A run with no generated tokens is an error: the harness used to turn
that into "1000 tok/s", which hid engine OOMs.
"""
import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'v100-16gb-linux')

REQUIRED_CASE = ('case', 'prompt_tokens', 'completion_tokens', 'ttft_s',
                 'prefill_tok_s', 'decode_tok_s')


def check_run(doc, path, problems):
    if 'cases' not in doc:
        problems.append(f'{path}: no "cases" list')
        return 0
    if not isinstance(doc['cases'], list) or not doc['cases']:
        problems.append(f'{path}: empty "cases"')
        return 0
    for item in doc['cases']:
        name = item.get('case', '?')
        if 'error' in item:
            problems.append(f'{path}: {name}: {item["error"]}')
            continue
        missing = [key for key in REQUIRED_CASE if key not in item]
        if missing:
            problems.append(f'{path}: {name}: missing {", ".join(missing)}')
            continue
        if not item.get('completion_tokens'):
            problems.append(f'{path}: {name}: completion_tokens is 0 (engine error?)')
    return len(doc['cases'])


def main():
    problems = []
    files = 0
    runs = 0
    for path in sorted(glob.glob(os.path.join(ROOT, '*.json'))):
        files += 1
        try:
            with open(path, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            problems.append(f'{path}: {exc}')
            continue
        if 'cases' in doc:
            runs += 1
            check_run(doc, path, problems)
        elif 'results' in doc:
            for run in doc['results']:
                runs += 1
                check_run(run, path, problems)
        else:
            problems.append(f'{path}: neither "cases" nor "results"')

    print(f'checked {files} files, {runs} runs')
    for problem in problems:
        print('FAIL', problem)
    if problems:
        print(f'{len(problems)} problem(s)')
        return 1
    print('ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
