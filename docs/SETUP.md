# Setup: reproduce the ladder on your own dual-GPU box

Paths below are placeholders. Swap in yours. I ran this on Ubuntu with no Docker.

I did not rebuild everything from scratch to write this guide. The commands come from what I ran on the box. Where I'm filling a gap, it says so.

## 0. Check the fit first

The model files are 96.5 GB. The experts that don't fit in VRAM live in RAM, so you need free RAM for the remainder plus page cache.

| Check | Command | What you need |
|---|---|---|
| VRAM | `nvidia-smi --query-gpu=name,memory.total --format=csv` | 2 cards. I had 2 x 20 GB. With 2 x 24 GB you should get a lower `--n-cpu-moe` than 28. |
| RAM | `grep MemTotal /proc/meminfo` | At least 128 GB is comfortable. I had 262 GB and did not test lower. |
| Disk | `df -h` | About 100 GB for the GGUFs, plus about 1 GB for Maya's pack. |
| CUDA | `nvcc --version` | 12.x. Maya's installer wants 12.0 or newer. I used 12.6. |

## 1. Get the model

```bash
pip install -U "huggingface_hub[cli]"
hf download peasantsmith/GLM-5.3-Flash-Maya-GGUF --include "Maya-S-v2-IQ2_XXS/*" --local-dir /path/to/models
```

You should end up with three shards (`-00001-of-00003.gguf` and so on). Point everything at shard 1. Expected sizes on disk: shard 1 is 47,835,993,056 bytes, shard 2 is 47,797,478,080, shard 3 is 835,121,120.

## 2. Path A: ik_llama.cpp ladder

Build ik_llama.cpp with CUDA. I ran commit `009dce1`. I did not record the exact configure line. This is the shape:

```bash
git clone https://github.com/ikawrakow/ik_llama.cpp && cd ik_llama.cpp
git checkout 009dce1
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j --target llama-server
```

`86` is Ampere (3080, 3090). Change it for other cards. If your CUDA libs are not on the default path, export `LD_LIBRARY_PATH` first.

Run the sweep:

```bash
export IK_BIN=$PWD/build/bin/llama-server
export MODEL_GGUF=/path/to/models/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
cd /path/to/glm-5.3-flash-dual-gpu
REPS=3 ROWS_FILE=rows/batch1.psv scripts/decode-matrix.sh '.+'
REPS=3 ROWS_FILE=rows/batch3.psv scripts/decode-matrix.sh 'b3-m28-c20k-ub128'
python3 scripts/decode-summary.py runs/decode-results.csv
```

To run one serving command by hand, this is the best row:

```bash
GGML_CUDA_NO_PINNED=1 $IK_BIN -m $MODEL_GGUF \
  -ngl 999 -fa on -sm layer --numa distribute -t 40 \
  -c 20480 -ub 128 --n-cpu-moe 28 \
  --spec-type mtp:n_max=1,p_min=0.0 \
  --host 127.0.0.1 --port 8181
```

Change `-t` to your physical core count. `--numa distribute` only matters on a multi-socket box.

### Finding your own `--n-cpu-moe`

This is the one setting that moved the number. `--n-cpu-moe N` keeps the experts of the first N layers on CPU. Lower N puts more on GPU and is faster.

1. Start at 30 and confirm it loads.
2. Drop N by 1 and re-run. Stop at the first value that fails to load.
3. If the load error names a buffer bigger than the free VRAM on one card, try a `-ts a,b` split that gives that card less. On my box no split rescued N below 28.
4. Smaller context and a smaller `-ub` use less VRAM. N=28 loaded at `-c 20480 -ub 128` and at `-c 8192` with `-ub 256` and with the default `-ub`. I did not isolate which of these mattered, and I never tried N=28 at 20k context with the default `-ub`.

Watch `nvidia-smi` after load. Leave a gigabyte or two free per card for the KV cache and compute buffers.

## 3. Path B: Project Maya

```bash
git clone https://github.com/mw00/project-maya && cd project-maya
git checkout 6c9a27b
./maya.py --help
```

Follow Maya's README for the first-run install. The command I used for this model:

```bash
./.venv/bin/python maya.py \
  --gguf-dir /path/to/models/Maya-S-v2-IQ2_XXS \
  --draft mtp --context 16384 --gpus 0,1 \
  --vram-reserve-mib 2000 --port 8180 \
  --nvcc /path/to/cuda-12.6/bin/nvcc \
  --yes --plain --no-start
```

That builds its CUDA engine (276 build steps for me), packs the GGUF (a roughly 1 GB pack folder) and writes a config JSON plus a start script. It does not start the server. See `maya/config.example.json` for what mine looked like.

Notes:

- `--vram-reserve-mib 2000` is what I used. I also ran 1000 and could not tell the two apart.
- **To change context, edit `--max-context` in the config JSON.** `--context` on the command line did not take effect for later runs.
- For 32k context I edited that value to `32768` and left everything else alone.

Benchmark it with the same client:

```bash
export MAYA_DIR=/path/to/project-maya
export CFG=/path/to/config.json
scripts/maya-bench.sh my-run-1
python3 scripts/decode-summary.py runs/decode-results.csv
```

`maya-bench.sh` starts Maya on port 8180, waits for health, runs three reps of the client, then kills the process group and waits for VRAM to clear. Extra arguments pass straight to `maya.py` (for example `--drafting off`).

Long context ladder:

```bash
MAYA_DIR=$MAYA_DIR BASE_CFG=$CFG scripts/deep-ladder.sh
```

The 1M run took about 17 minutes to first token on my box. Plan for that.

## 4. Reading your numbers

- Take the median of the reps, and look at the range. Under about 0.8 tok/s on ik_llama.cpp, or 3 to 4 on Maya, is noise.
- Rep 1 is usually slowest. Maya's expert cache warms across requests.
- Reps 2 and 3 of the long prompt are served from the prompt cache. Only trust rep 1 for prefill.
- Run each candidate at least twice on separate loads before you believe a gain.

## 5. Check quality before you use it

I did not. At minimum, before you rely on this setup:

1. Run 20 or so of your own real prompts at temperature 0 on both stacks and diff or eyeball the outputs.
2. Run your own eval, or at least a perplexity pass, on this quant against a bigger one you trust.
3. Re-run with MTP on and off and confirm outputs match at temperature 0.

## 6. Cleanup

Both runners kill their server by process group. If you interrupt a sweep, check `nvidia-smi` and `ss -ltnp` for a leftover server before the next run. Both runners refuse to start if the GPUs hold more than 2 GB or the port is taken.
