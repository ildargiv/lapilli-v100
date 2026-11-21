# Lapilli: Basalt on NVIDIA V100

Lapilli is a port of [Basalt](https://github.com/jesdga95/basalt), the local inference engine for **Qwen3.8-Flash-Next**,
to **NVIDIA V100** GPUs (Volta, compute capability 7.0). Upstream Basalt targets Blackwell (sm_120) only and does not take
other architectures; this fork adds sm_70 and keeps the Blackwell build as it was.

It also brings a chat page, served at `/`: conversations, reasoning levels, image input, tool calls shown as cards, and a live
monitor of the GPUs. Basalt's own dashboard moves to `/dashboard`. The server keeps Basalt's OpenAI and Anthropic APIs.

*Lapilli are the small fragments a volcano throws out: small pieces of basalt.*

<p align="center"><img src="docs/img/lapilli-chat.png" alt="The Lapilli chat page: a question, the collapsed reasoning and the formatted answer" width="820"></p>
<p align="center"><img src="docs/img/lapilli-monitor.png" alt="The live monitor while an answer is written: speed, the two V100s, CPU, RAM and disk" width="600">&nbsp;&nbsp;<img src="docs/img/lapilli-phone.png" alt="The chat page on a phone" width="178"></p>
<p align="center"><sub>The chat page, the live monitor while an answer is being written, and the same page on a phone. Screenshots of this build on two V100s.</sub></p>

> Based on Basalt commit `6809edfa` (November 15, 2026). Basalt moves fast; this fork is a snapshot of it and does not follow
> upstream. Provided as is, without support (see [Support](#support)).

## What was measured

One machine: 2 x V100-SXM2 32 GB (on PCIe 3.0 x16 and x4 links, peer-to-peer off: see below), Intel Core i9-14900K, 96 GB of
RAM, Windows 10, CUDA 12.6, NVIDIA driver 561.17, the `Qwen3.8-FN-UD-Q4_K_XL` pack (115 GB) on an NVMe SSD, experts
memory-mapped. Measured on November 20, 2026, with the build from this repository; each speed is the median of three runs.
Strata v0.2.13, the V100-capable engine Basalt descends from, ran the same measurements on the same machine earlier.

| | Lapilli | Strata v0.2.13, same machine |
|---|---|---|
| Decode, short answer (256 tokens, temperature 0) | 91 tokens/s | 58 to 72 tokens/s |
| Decode, long answer with reasoning (1,500 tokens) | 140 tokens/s (133 to 145) | 68 to 94 tokens/s |
| Prompt processing, 3,000 tokens | 1,295 tokens/s | |
| Prompt processing, 4,200 tokens | 1,170 tokens/s | 250 to 490 tokens/s |
| Prompt processing, 8,000 / 16,000 tokens | 1,504 / 1,448 tokens/s | |
| Time to load the model | 34 s | |

Quality, with Basalt's own gate (`basalt-bench --quality 100`, 100 chunks through the decode path, against the BF16 reference):

| | Lapilli on 2 x V100 | Basalt's published figures for this pack (RTX 5090) |
|---|---|---|
| Perplexity | 3.3206 | 3.321 |
| Mean KLD | 0.04932 | 0.0496 |
| Same top-1 token | 92.75% | 92.63% |

The port keeps the pack's quality. 108 of Basalt's 122 GPU parity tests pass on a V100; the others test paths a V100 does not
take with this pack: Blackwell-only kernels (FP8, FP4, TMA), thread-block clusters (the engine uses its cluster-free fallback),
more than 96 KB of shared memory per block, and two bitwise comparisons between paths that add in a different order on Volta.

## Tested by users

**One V100 PCIe 32 GB on Linux**, reported by u/nebod99 (i9-10940X, 96 GB of RAM, CUDA 12.6.3 in Docker, the `IQ3_S` pack,
thinking on, a 7,242-token prompt and 1,024 generated tokens per run), against Strata on the same machine:

| | Strata, 128K context | Lapilli, 128K context |
|---|---|---|
| Decode, long prompt | 55.6 tokens/s | **100.9 tokens/s** |
| First token, long prompt | 22.6 s | **8.7 s** |
| Launch to ready | 70 s | 90 s |

The first run outside the machine it was developed on, and the first on Linux, on a single card and on PCIe.

Quality, on the same machine: on more than 2,500 questions selected from public benchmarks (thinking on, temperature 0),
Lapilli scores the same as Strata with the same model, tied with the best results nebod99 has ever had from any model or
engine.

Full report, with the 16K runs, the quality results and the Linux build:
[docs/community/one-v100-pcie-linux.md](docs/community/one-v100-pcie-linux.md).

A big thank you to nebod99 for the time and care put into these tests, and for sharing them so that every V100 owner can
benefit from them.

### V100-SXM2 16GB on Linux (community, 2026-11-21)

**One V100-SXM2 16 GB on Linux**, tested by [@ildargiv](https://github.com/ildargiv) (Ubuntu 22.04, Ryzen 5 5600G, AVX2/no AVX-512, 96 GB RAM, NVMe, CUDA 12.9). Model/pack: `Qwen3.8-FN-UD-IQ3_XXS.basalt` from [jesdga/Qwen3.8-Flash-Next-Basalt](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt) (87.6 GB), **not in Lapilli's officially tested pack list**. API alias kept Strata-compatible: `qwen3.8-flash-next-iq3_s`.

Build:

```bash
cmake -S . -B build70 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DBASALT_MODELS=qwen4exp -DBUILD_TESTING=OFF
cmake --build build70 -j
```

Run (16GB-safe baseline used in tests):

```bash
bin/basalt serve /path/to/Qwen3.8-FN-UD-IQ3_XXS.basalt \
  --alias qwen3.8-flash-next-iq3_s \
  --gpu 0 --kv int8 --max-context 262144 \
  --mmap-experts --resident-experts \
  --ram-reserve-mib 10240 --vram-reserve-mib 256 \
  --port 8080
```

Client-class benchmark (`tools/ab_bench.py` / ai-flight-log harness, streaming TTFT/tok/s, `temperature 0`, `max_tokens 256`, **n=4 medians**, idle, one change at a time). Cases: `short-1k(886)`, `medium-8k(6214)`, `long-32k(24359)` prompt tokens.

| Experiment | short decode | medium decode | long decode | short TTFT | medium TTFT | long TTFT | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| Baseline (`int8`, res700, spec8) | 57.8 | 75.4 | 68.0 | 1.75s | 6.07s | 20.88s | |
| A `vram-reserve-mib 256` | 52.4 | 76.45 | 67.9 | 1.75s | 6.07s | 20.89s | neutral |
| B `--kv k8v4` | **67.75** | **76.5** | **74.15** | 1.75s | 6.07s | 20.92s | **KV quality tradeoff** |
| C1 `--spec 6` | 56.85 | 76.6 | 67.65 | 1.75s | 6.06s | 20.96s | no gain |
| C2 `--spec 0` | 68.15 | 75.75 | 67.55 | 1.75s | 6.09s | 21.08s | no stable gain |
| D `--route-prior 1.0` | 55.5 | 76.45 | 67.0 | 1.76s | 6.35s | 21.98s | lossy, worse TTFT/prefill |
| E `--route-tail-skip 9` | 55.75 | 76.8 | 68.4 | 1.75s | 6.35s | 22.37s | lossy, worse TTFT/prefill |

Strata reference (not apples-to-apples): Strata 0.1.60-cuda12, GSQ-RCO IQ3_S GGUF, kv int8, kv-resident 32768, spec 6, dflash2 q4_0-selq4, expert cache 4497/8.58 GiB, hit 97.7???99.1%, **n=1** client: short 410.3/91.4, medium 869.7/89.3, long 1212.5/94.6 (prefill/decode tok/s). Lapilli base is faster on prefill/TTFT for short/medium, but behind on decode; B (k8v4) partially closes decode at a KV quality tradeoff.

Observed bottlenecks on 16GB: dense ~13.5/13.7 GiB on GPU, `kv 0.7/4.0GiB cuda0 elastic`, `hit ~61-74%`, `pcie 0.0%`. sm_70 warnings: `mid_chain: device not covered (80 SMs, 98304 B optin smem; need 16, 101376): unfused kernels`, `GDN qkv+z without dense_tc_pair (types 14/14, 8/8)`, no AVX-512.

Raw data: [`results/v100-16gb-linux/`](results/v100-16gb-linux/). Detailed notes: [docs/v100-16gb-linux.md](docs/v100-16gb-linux.md). Flight log entry: [ai-flight-log-v100-16gb #006](https://github.com/ildargiv/ai-flight-log-v100-16gb/blob/master/journal/entries/006.md).

## Requirements

- One or more NVIDIA V100 GPUs. Tested: two V100-SXM2 32 GB; one V100 PCIe 32 GB by a user; one V100-SXM2 16 GB on Linux (community test, see [Tested by users](#tested-by-users)).
- Windows 10 or 11 x64, or Linux (run by a user with CUDA 12.6.3 in Docker; the build steps below are for Windows).
- CUDA Toolkit 12.6 to 12.9 (CUDA 13 dropped Volta), the Visual Studio 2022 Build Tools (C++), CMake 3.24+, Python 3.10+.
- System RAM and a fast SSD for the experts that do not fit in VRAM: the pack is 115 GB. Tested with 96 GB of RAM.
- A K-quant pack. NVFP4 packs need FP4 hardware and do not run on a V100. Tested: `Qwen3.8-FN-UD-Q4_K_XL`.

## Prebuilt archive (Windows)

Each release carries `lapilli-v100-<tag>-windows-x64.zip`: `bin\` (`basalt`, `basalt-bench`, `basalt-probe`,
`basalt-vision`), the server, the chat page, `data\`, `docs\`, `requirements.txt` and the licences. Like Basalt's archives, it
holds no CUDA libraries and no Python environment. It needs:

- An NVIDIA driver that supports CUDA 12.6 (tested: 561.17).
- The CUDA 12 runtime libraries on the `PATH`: `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll`. The CUDA Toolkit 12.6
  to 12.9 installs them.
- The Microsoft Visual C++ 2015-2022 Redistributable (x64).
- Python 3.10+ with the requirements in a `.venv` at the archive's top directory, which `bin\basalt.exe serve` uses when it is
  there: `py -3 -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`.

Then follow [Run](#run).

## Build (Windows)

From an *x64 Native Tools Command Prompt for VS 2022*:

```bat
git clone https://github.com/<this fork> lapilli && cd lapilli
cmake -S . -B build70 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 -DBASALT_MODELS=qwen4exp -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
cmake --build build70
py -3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The executables land in `bin\`. A full build, tests included, took 49 minutes on a 24-core i9-14900K with 6 parallel jobs. `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` avoids a duplicate `cudart` at link time on
Windows. `-DBUILD_TESTING=ON` adds the parity tests (`ctest --test-dir build70 -L gpu`).

## Run

```bat
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-UD-Q4_K_XL.basalt --local-dir models
bin\basalt.exe serve models\Qwen3.8-FN-UD-Q4_K_XL.basalt --gpu 0,1 --kv int8 --mmap-experts --resident-experts --peer-exclusive --vram-reserve-mib 700 --ram-reserve-mib 16384 --max-context 262144 --vision dynamic --port 8080
```

Then open `http://127.0.0.1:8080/`. Every flag works as in Basalt: see its [running](docs/running.md) and
[configuration](docs/configuration/index.md) pages (this repository's `docs/` is Basalt's documentation, written for Blackwell).

## What changed from Basalt

- **Build**: compute capability 7.0 is accepted next to 12.0, with CUDA 12.6 or newer.
- **A compatibility layer** (`include/basalt/kernels/volta_compat.cuh`): every newer PTX instruction the engine uses (`cp.async`,
  `ldmatrix`, the `mma` shapes of Turing and Ampere, tf32 and bf16 conversions, `griddepcontrol`, clusters) gets a correct sm_70
  equivalent, so every kernel compiles and runs right on a V100.
- **V100 paths for the hot kernels**: llama.cpp's dp4a path for the quantized prompt products; int8 prompt attention on the
  V100 tensor cores (`mma.m8n8k4`, from Strata); hyper-connection reads, the router, the decode experts and the GDN recurrence
  rewritten for Volta; the expert exchange between two cards tuned for V100.
- **Peer-to-peer access is off by default on Volta**: on the test machine, direct copies between the two V100s returned wrong
  data. The copies go through the host instead. `BASALT_PEER_NO_P2P=0` turns peer access back on, at your own risk.
- **Server**: the exact resume of a follow-up turn (the previous answer's tokens are reused as they were sampled, so the
  engine always finds its cache).
- **The Lapilli chat page** at `/`; Basalt's dashboard at `/dashboard`.

The `BASALT_*` environment variables added by the port are the A/B switches used while measuring it; their defaults are the
best measured on V100, and they are not meant as settings.

## Not tested

NVLink with peer access on; packs other than `UD-Q4_K_XL`, `IQ3_S` and `UD-IQ3_XXS` (community-tested on V100 16GB: `UD-IQ3_XXS`, see [community test](#v100-sxm2-16gb-on-linux-community-2026-11-21)); Basalt's other models (DeepSeek-V4.1-Flash and
Qwen3.8-27B are not built: `-DBASALT_MODELS=qwen4exp`); Blackwell builds of this fork.

## Support

This fork is shared as is, in the hope that it helps other V100 owners. There is no promise of support or updates. Issues
may be read, but may not get an answer.

## Credits and license

MIT, like Basalt ([LICENSE](LICENSE)). Lapilli stands on the work of others:

- [Basalt](https://github.com/jesdga95/basalt) by Jesús Gulfo (MIT): the engine.
- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors (MIT): Basalt's origin, and the V100
  prompt-attention kernel used here.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): the dp4a prompt products.
- [marked](https://github.com/markedjs/marked) (MIT): the chat page's text formatting. Geist font (SIL Open Font License).

Every notice is in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES). Basalt's own README, with its models and documentation, is at
[github.com/jesdga95/basalt](https://github.com/jesdga95/basalt).

French: [README.fr.md](README.fr.md).
