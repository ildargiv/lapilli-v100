# V100 16GB (Linux) - Lapilli notes

Tested on **Tesla V100-SXM2-16GB** (linux1, Ubuntu 22.04, Ryzen 5 5600G AVX2/no AVX-512, 96 GiB RAM, NVMe, CUDA 12.9). Upstream Lapilli README marks V100 16GB as not tested.

## Model / pack

- Pack: `Qwen3.8-FN-UD-IQ3_XXS.basalt` from `jesdga/Qwen3.8-Flash-Next-Basalt` (87,602,085,467 bytes).
- Not in Lapilli's officially tested pack list (README tests `UD-Q4_K_XL` and `IQ3_S`).
- API alias kept Strata-compatible: `qwen3.8-flash-next-iq3_s`.

## Build

```bash
git clone https://github.com/ildargiv/lapilli-v100.git
cd lapilli-v100
cmake -S . -B build70 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DBASALT_MODELS=qwen4exp -DBUILD_TESTING=OFF
cmake --build build70 -j
# bin/basalt lands in bin/
```

## Run (16GB-stable config, 2026-11-22)

```bash
bin/basalt serve /path/to/Qwen3.8-FN-UD-IQ3_XXS.basalt \
  --alias qwen3.8-flash-next-iq3_s \
  --gpu 0 --kv int8 --kv-resident 32768 --max-context 262144 \
  --mmap-experts --resident-experts \
  --ram-reserve-mib 10240 --vram-reserve-mib 700 \
  --port 8080
```

Observed at load: `weights 13.5/13.7 GiB on GPU + ~37.7 GiB system RAM`, `kv 1.0GiB cuda0 solo=262144` with `kv-resident=32768`, `ram-experts=50.5GB pool=5`, `expert cache 4602 slots / 8.82 GiB`, `hit ~59-74%`, `pcie=0.0%`.

### OOM finding (2026-11-22)

With `--vram-reserve-mib 256` (and no `kv-resident` tuning), `auto expert cache` grew to ~4831-5002 slots (9.25-9.58 GiB). On 8k/32k prompt prefill the engine hit `mmq: out of memory: cudaFuncSetAttribute/cudaMalloc` and restart-looped. Stable fix: `--kv-resident 32768 --vram-reserve-mib 700`. `--kv k8v4` is **incompatible** with `--kv-resident` in the tested build (`--kv k8v4 does not support --kv-resident streaming (yet)`).

Warnings seen on V100 sm_70:
- `mid_chain: device not covered (80 SMs, 98304 B optin smem; need 16, 101376): unfused kernels run`
- `fallback: GDN qkv + z without dense_tc_pair (types 14/14, 8/8)`
- `CPU has no AVX-512: expert kernels run on AVX-2`

## Benchmark protocol

- Harness: `tools/ab_bench.py` (client-class, streaming TTFT/tok/s).
- Cases: warmup + `short-1k(886)/256`, `medium-8k(6214)/256`, `long-32k(24359)/256`.
- Repeats: 4 per experiment, idle, one change at a time.
- Raw JSON: `results/v100-16gb-linux/`.

## Results (medians, n=4, client)

| Experiment | short decode | medium decode | long decode | short TTFT | medium TTFT | long TTFT | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline 21.11 (int8, res256) | 57.8 | 75.4 | 68.0 | 1.75 | 6.07 | 20.88 | pre-OOM-fix config |
| A vram256 | 52.4 | 76.45 | 67.9 | 1.75 | 6.07 | 20.89 | neutral |
| B k8v4 | **67.75** | **76.5** | **74.15** | 1.75 | 6.07 | 20.92 | KV quality tradeoff; **incompatible with kv-resident** in tested build |
| C1 spec6 | 56.85 | 76.6 | 67.65 | 1.75 | 6.06 | 20.96 | no gain |
| C2 spec0 | 68.15 | 75.75 | 67.55 | 1.75 | 6.09 | 21.08 | no stable gain |
| D route-prior1 | 55.5 | 76.45 | 67.0 | 1.76 | 6.35 | 21.98 | lossy, worse TTFT/prefill |
| E tail-skip9 | 55.75 | 76.8 | 68.4 | 1.75 | 6.35 | 22.37 | lossy, worse TTFT/prefill |
| **F stable (int8, kv-resident 32768, res700)** | **67.75** | **75.95** | **70.35** | **1.78** | **6.02** | **20.75** | **stable 16GB config, no OOM** |
| G stable + adapt-every1/swaps96 | 67.7 | 75.15 | 68.25 | 1.79 | 6.03 | 20.77 | no gain vs F |

## Strata reference (not apples-to-apples)

Strata 0.1.60-cuda12, GSQ-RCO IQ3_S GGUF, kv int8, kv-resident 32768, spec 6, dflash2 q4_0-selq4, expert cache 4497/8.58 GiB, hit 97.7-99.1%, n=1:

| Case | Strata decode | Lapilli F decode | Strata prefill | Lapilli F prefill | Lapilli hit | Strata hit |
|---|---:|---:|---:|---:|---:|---:|
| short-1k | 91.4 | 67.8 | 410.3 | 497.3 | ~59-71% | 97.7% |
| medium-8k | 89.3 | 76.0 | 869.7 | 1032.5 | ~73% | 98.x% |
| long-32k | 94.6 | 70.4 | 1212.5 | 1174.0 | ~73-74% | 99.1% |

## Takeaways for 16GB fork

1. **16GB needs explicit VRAM budgeting**: `kv-resident 32768` + `vram-reserve-mib 700` avoided the auto-cache OOM. `res256` was not enough on this setup.
2. **Main decode limit is expert hit**: ~73% hit vs Strata 97-99% at similar cache size (8.82 vs 8.58 GiB). Explicit adaptive flags did not improve hit on tested workload.
3. **k8v4 was the only decode win** in old tests, but it is a KV quality tradeoff and incompatible with `kv-resident` in the tested build; not a drop-in default for this stable config.
4. **Lossy routing flags did not help**; route-prior/tail-skip regressed TTFT/prefill.
5. **sm_70 work remains**: mid_chain smem (101376 > Volta 98304) and dense_tc_pair GDN fallback (types 14/8).
