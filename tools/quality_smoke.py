#!/usr/bin/env python3
"""Deterministic quality smoke: compare two engine configs on fixed prompts.

Usage:
  python3 tools/quality_smoke.py <base_url> <label> <out.json>

The prompts are short and closed-form (arithmetic, spelling, one Russian
question). The answer is scored with a word-boundary match, so "The answer is
4" passes for "4" while "14" does not. Run it against the same server before
and after a config change, then diff the two JSON files.
"""
import json
import re
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip('/') if len(sys.argv) > 1 else 'http://127.0.0.1:8080'
LABEL = sys.argv[2] if len(sys.argv) > 2 else 'quality'
OUT = sys.argv[3] if len(sys.argv) > 3 else f'/tmp/{LABEL}-quality.json'
MODEL = 'qwen3.8-flash-next-iq3_s'

PROMPTS = [
    ('What is 2+2? Answer with the number only.', '4'),
    ('What is 17*23? Answer with the number only.', '391'),
    ('What is 144/12? Answer with the number only.', '12'),
    ('What is 2**10? Answer with the number only.', '1024'),
    ('What is the square root of 81? Answer with the number only.', '9'),
    ('What is 15% of 200? Answer with the number only.', '30'),
    ('What is 100-37? Answer with the number only.', '63'),
    ('What is 9*9? Answer with the number only.', '81'),
    ('What is the capital of France? Answer with one word.', 'paris'),
    ('What is the capital of Japan? Answer with one word.', 'tokyo'),
    ('Which planet is the largest in the solar system? Answer with one word.', 'jupiter'),
    ('What is the chemical symbol for gold? Answer with the symbol only.', 'au'),
    ('How many days are in a leap year? Answer with the number only.', '366'),
    ('Reverse the string abc. Answer with the result only.', 'cba'),
    ('How many letters are in the word banana? Answer with the number only.', '6'),
    ('In Python, what is len([1,2,3])? Answer with the number only.', '3'),
    ("In Python, what is 'abc'[::-1]? Answer with the result only.", 'cba'),
    ('What is the binary representation of 5? Answer with the digits only.', '101'),
    ('What is 7 in binary? Answer with the digits only.', '111'),
    ('Столица России? Ответь одним словом.', 'москва'),
]


def post(payload, timeout=300):
    req = urllib.request.Request(
        BASE + '/v1/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json; charset=utf-8'},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def ask(prompt):
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 64,
        'temperature': 0,
        'reasoning_effort': 'none',
    }
    started = time.time()
    with post(payload) as response:
        doc = json.load(response)
    elapsed = round(time.time() - started, 2)
    choice = (doc.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    content = message.get('content') or ''
    reasoning = message.get('reasoning_content') or ''
    usage = doc.get('usage') or {}
    return {
        'content': content,
        'reasoning': reasoning[:200],
        'finish_reason': choice.get('finish_reason'),
        'prompt_tokens': usage.get('prompt_tokens'),
        'completion_tokens': usage.get('completion_tokens'),
        'elapsed_s': elapsed,
    }


def check(expected, answer):
    text = (answer or '').lower()
    if not text:
        return False
    if expected.isdigit():
        return re.search(r'(?<!\d)' + re.escape(expected) + r'(?!\d)', text) is not None
    if re.fullmatch(r'[a-z]+', expected) and len(expected) <= 4:
        return re.search(r'(?<![a-z])' + re.escape(expected) + r'(?![a-z])', text) is not None
    return expected in text


def main():
    results = []
    passed = 0
    for prompt, expected in PROMPTS:
        try:
            got = ask(prompt)
            ok = check(expected, got['content'])
        except Exception as exc:  # noqa: BLE001 - keep the suite running
            got = {'content': '', 'reasoning': '', 'error': f'{type(exc).__name__}: {exc}'}
            ok = False
        passed += int(ok)
        results.append({
            'prompt': prompt,
            'expected': expected,
            'pass': ok,
            **got,
        })
        print(f'{"PASS" if ok else "FAIL"}  {expected!r:>12}  {got.get("content", "")[:70]!r}',
              flush=True)

    doc = {
        'label': LABEL,
        'base': BASE,
        'model': MODEL,
        'started': time.strftime('%F %T'),
        'passed': passed,
        'total': len(PROMPTS),
        'score': f'{passed}/{len(PROMPTS)}',
        'results': results,
    }
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    print(f'score {passed}/{len(PROMPTS)} -> {OUT}')
    return 0 if passed == len(PROMPTS) else 0  # a quality smoke reports, it does not fail the build


if __name__ == '__main__':
    sys.exit(main())
