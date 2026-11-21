#!/usr/bin/env python3
import json, subprocess, sys, os
BASE=sys.argv[1]
LABEL=sys.argv[2]
OUTDIR=sys.argv[3]
N=int(sys.argv[4]) if len(sys.argv)>4 else 4
BENCH=sys.argv[5] if len(sys.argv)>5 else os.path.join(os.path.dirname(__file__),"ab_bench.py")
os.makedirs(OUTDIR, exist_ok=True)
all=[]
for i in range(1,N+1):
    out=os.path.join(OUTDIR, f"{LABEL}-r{i}.json")
    cmd=[sys.executable, BENCH, BASE, f"{LABEL}-r{i}", out]
    print(f"== run {i}/{N} ==", flush=True)
    subprocess.check_call(cmd)
    all.append(json.load(open(out)))
summary={"label":LABEL,"base":BASE,"runs":N,"files":[os.path.join(OUTDIR,f"{LABEL}-r{i}.json") for i in range(1,N+1)],"results":all}
json.dump(summary, open(os.path.join(OUTDIR,f"{LABEL}-summary.json"),"w"), ensure_ascii=False, indent=1)
print("saved summary", flush=True)

