#!/usr/bin/env python3
"""A/B: the same prompt set against an OpenAI-compatible server.

Client-class numbers: TTFT and tok/s are measured on the client, streaming.
Usage: ab_bench.py <base_url> <label> [out.json]

A response with zero generated tokens is reported as an error. Without this,
an engine OOM looked like 1000 tok/s and silently poisoned the medians.
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip('/')
LABEL = sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else f'/tmp/opencode/ab-{LABEL}.json'

FILLER = (
    "The scheduler assigns each request to a worker queue, and the worker pulls "
    "the next task when it becomes idle. Latency measurements are collected at the "
    "edge of the network so that queueing delay is attributed to the correct hop. "
    "Every batch is validated against a schema before it is written to storage, and "
    "invalid records are routed to a quarantine table for later inspection. "
)

CASES = [
    ("short-1k", 1_000, 256),
    ("medium-8k", 8_000, 256),
    ("long-32k", 32_000, 256),
]


def make_prompt(target_tokens: int, tag: str) -> str:
    body = FILLER * max(1, int(target_tokens * 4 / len(FILLER)) + 1)
    return (
        f"[run {tag}]\n"
        "Read the following operational notes and then answer the question.\n\n"
        f"{body}\n"
        "Question: which component validates batches before storage, and what "
        "happens to invalid records? Answer in one sentence."
    )


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def stream_case(name, prompt, max_tokens):
    payload = {
        "model": "qwen3.8-flash-next-iq3_s",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    ttft = None
    chunks = 0
    usage = None
    try:
        with post("/v1/chat/completions", payload) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        if ttft is None:
                            ttft = time.time() - t0
                        chunks += 1
    except Exception as e:
        return {"case": name, "error": f"{type(e).__name__}: {e}"}
    total = time.time() - t0
    prompt_tokens = (usage or {}).get("prompt_tokens")
    completion_tokens = (usage or {}).get("completion_tokens") or chunks
    if not completion_tokens:
        return {
            "case": name,
            "error": "no generated tokens (engine error or empty response)",
            "total_s": round(total, 2),
        }
    decode_s = max(total - (ttft or total), 0.001)
    return {
        "case": name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft or 0, 2),
        "total_s": round(total, 2),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if (prompt_tokens and ttft) else None,
        "decode_tok_s": round(max(completion_tokens - 1, 1) / decode_s, 1),
    }


def main():
    results = {"label": LABEL, "base": BASE, "started": time.strftime("%F %T"), "cases": []}
    try:
        stream_case("warmup", "Say hello in five words.", 32)
    except Exception:
        pass
    for name, ptoks, mtoks in CASES:
        p = make_prompt(ptoks, name)
        r = stream_case(name, p, mtoks)
        results["cases"].append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
