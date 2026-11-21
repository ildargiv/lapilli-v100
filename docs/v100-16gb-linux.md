# V100 16GB (Linux) ??? Lapilli notes

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

## Run (16GB-safe baseline used in tests)

```bash
bin/basalt serve /path/to/Qwen3.8-FN-UD-IQ3_XXS.basalt \
  --alias qwen3.8-flash-next-iq3_s \
  --gpu 0 --kv int8 --max-context 262144 \
  --mmap-experts --resident-experts \
  --ram-reserve-mib 10240 --vram-reserve-mib 256 \
  --port 8080
```

Observed at load: `weights 13.5/13.7 GiB on GPU + ~37.7 GiB system RAM`, `kv 0.7/4.0GiB cuda0 elastic`, `ram-experts=50.5GB pool=5`, `hit ~61-74%`, `pcie=0.0%`.

Warnings seen on V100 sm_70:
- `mid_chain: device not covered (80 SMs, 98304 B optin smem; need 16, 101376): unfused kernels run`
- `fallback: GDN qkv + z without dense_tc_pair (types 14/14, 8/8)`
- `CPU has no AVX-512: expert kernels run on AVX-2`

## Benchmark protocol

- Harness: `ai-flight-log-v100-16gb/tools/ab_bench.py` (client-class, streaming TTFT/tok/s).
- Cases: warmup + `short-1k(886)/256`, `medium-8k(6214)/256`, `long-32k(24359)/256`.
- Repeats: 4 per experiment, idle, one change at a time.
- Raw JSON: `results/v100-16gb-linux/`.

## Results (medians, n=4)

| Experiment | short decode | medium decode | long decode | short TTFT | medium TTFT | long TTFT | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline (int8, res700, spec8) | 57.8 | 75.4 | 68.0 | 1.75 | 6.07 | 20.88 |  |
| A vram256 | 52.4 | 76.45 | 67.9 | 1.75 | 6.07 | 20.89 | neutral |
| B k8v4 | **67.75** | **76.5** | **74.15** | 1.75 | 6.07 | 20.92 | **KV quality tradeoff** |
| C1 spec6 | 56.85 | 76.6 | 67.65 | 1.75 | 6.06 | 20.96 | no gain |
| C2 spec0 | 68.15 | 75.75 | 67.55 | 1.75 | 6.09 | 21.08 | no stable gain |
| D route-prior1 | 55.5 | 76.45 | 67.0 | 1.76 | 6.35 | 21.98 | lossy, worse TTFT/prefill |
| E tail-skip9 | 55.75 | 76.8 | 68.4 | 1.75 | 6.35 | 22.37 | lossy, worse TTFT/prefill |

## Strata reference (not apples-to-apples)

Strata 0.1.60-cuda12, GSQ-RCO IQ3_S GGUF, kv int8, kv-resident 32768, spec 6, dflash2 q4_0-selq4, expert cache 4497/8.58 GiB, hit 97.7???99.1%, n=1 (ai-flight-log-v100-16gb):

| Case | Strata decode | Lapilli base decode | Lapilli B k8v4 decode | Strata prefill | Lapilli base prefill |
|---|---:|---:|---:|---:|---:|
| short-1k | 91.4 | 57.8 | 67.75 | 410.3 | 507.0 |
| medium-8k | 89.3 | 75.4 | 76.5 | 869.7 | 1025.0 |
| long-32k | 94.6 | 68.0 | 74.15 | 1212.5 | 1167.0 |

## Takeaways for 16GB fork

1. **Main limit is VRAM expert cache**: ~13.7 GiB dense weights on GPU leave little room for expert cache; hit stays ~61-74% vs Strata's 97-99%.
2. **k8v4 is the only meaningful flag win** (+9-17% decode) but it is a KV quality tradeoff; treat as opt-in speed profile, not silent default.
3. **Lossy routing flags (route-prior, route-tail-skip) did not help here**; they regressed prefill/TTFT.
4. **sm_70 work needed**: mid_chain smem (101376 > Volta 98304) and dense_tc_pair GDN fallback (types 14/8).
5. **Fork should expose VRAM budget/cache visibility** in startup/status and provide a 16GB preset with explicit quality warnings.
