#!/bin/bash
# Record the environment of a run: host, GPU, driver, CUDA, CPU, RAM, disk,
# repository commit and the model pack. Paste the output into a report.
set -u

echo "== date =="
date -Is
echo "== host =="
hostnamectl 2>/dev/null | sed -n '1,8p' || uname -a
echo "== kernel =="
uname -a
echo "== gpu =="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv
nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.corrected.volatile.total --format=csv 2>/dev/null || true
echo "== cuda =="
nvcc --version 2>/dev/null | tail -2 || echo "nvcc not found"
echo "== cpu =="
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|Flags' | cut -c1-160
echo "== memory =="
free -h
echo "== disk =="
df -h / /home 2>/dev/null | tail -3
echo "== repo =="
if git rev-parse --show-toplevel >/dev/null 2>&1; then
  git rev-parse --short HEAD
  git status --short | head -5
else
  echo "not a git checkout"
fi
echo "== pack =="
for f in "$@"; do
  [ -f "$f" ] && ls -l "$f" && sha256sum "$f"
done
