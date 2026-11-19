# Results (raw CSV: `decode-results.csv`)

Decode = tok/s on a 256-token generation after a short prompt, temperature 0, streamed, one request at a time (`scripts/smoke-single.py`, `single` leg). Prefill = tok/s on a 16.7k-token prompt (`long` leg, first rep only, because reps 2 and 3 hit the server's prompt cache and finish in about half a second). Median of the reps shown. Noise between identical configs was about 0.5 tok/s on ik_llama.cpp, so treat gaps under about 0.8 tok/s as no difference.

All runs: GLM-5.3-Flash Maya-S v2 IQ2_XXS (3 GGUF shards, 96.5 GB), 2x RTX 3080 20 GB, dual Xeon Gold 6138, 262 GB DDR4. See the README for versions.

## ik_llama.cpp ladder (layer split, experts partly on CPU)

| Row | What changed | Decode tok/s (median) | Range | Reps | Prefill tok/s |
|---|---|---|---|---|---|
| `glm-base` | `--n-cpu-moe 30`, 20k ctx (baseline) | **15.7** | 15.7-15.7 | 2 | 94 |
| `glm-numa` | + `--numa distribute -t 40` | **16.0** | 15.8-16.2 | 2 | 100 |
| `glm-numa-mtp1` | + MTP draft depth 1 | **16.9** | 16.3-17.4 | 2 | 97 |
| `glm-numa-mtp2` | MTP draft depth 2 | **16.2** | 16.1-16.4 | 2 | 94 |
| `b2-ref` | re-run of the MTP-1 row (noise check) | **16.5** | 16.4-17.4 | 3 | 93 |
| `b2-moe29` | `--n-cpu-moe 29`, 20k ctx (long leg crashed) | **17.4** | 17.4-17.4 | 1 | n/a |
| `b2-moe28-c8k` | `--n-cpu-moe 28`, 8k ctx (long leg crashed) | **18.0** | 18.0-18.0 | 1 | n/a |
| `b3-m28-c8k-ub256` | `--n-cpu-moe 28`, 8k ctx, `-ub 256` (long leg crashed) | **17.4** | 17.3-18.3 | 3 | n/a |
| `b3-m28-c20k-ub128` | **`--n-cpu-moe 28`, 20k ctx, `-ub 128` (best ik)** | **18.4** | 18.2-18.4 | 3 | 80 |

### ik_llama.cpp flags that did nothing (inside noise)

| Row | What changed | Decode tok/s (median) | Range | Reps | Prefill tok/s |
|---|---|---|---|---|---|
| `b2-rtr` | + `-rtr` (runtime repack) | **15.8** | 15.6-16.9 | 3 | 95 |
| `b2-muge` | + `-muge` (merge up/gate experts) | **16.4** | 15.3-16.5 | 3 | 95 |
| `b2-mqkv` | + `-mqkv` (merge QKV) | **16.4** | 16.3-17.0 | 3 | 97 |
| `b2-graph` | `-sm graph --max-gpu 2` | **16.4** | 16.4-17.2 | 3 | 97 |
| `b2-t36` | `-t 36` instead of 40 | **15.8** | 15.8-16.4 | 3 | 91 |

### ik_llama.cpp rows that did not load (VRAM wall)

Every `--n-cpu-moe` value below 28 failed to load, at every `-ts` split tried. The one failure I read in full (`--n-cpu-moe 26`, 20k ctx) was VRAM: the model wanted about 89.9 GB of device buffers against 37.3 GB free, and GPU 1 had 19.8 GB free for a 22.6 GB buffer. The other rows end in the same `unable to load model` line. I did not read each log, so the cause for those is assumed to be the same.

| Row | Flags | Result |
|---|---|---|
| `glm-numa-moe26` | `--n-cpu-moe 26`, 20k ctx | load failed |
| `b3-m27-c8k-ub256` | `--n-cpu-moe 27`, 8k ctx, `-ub 256` | load failed |
| `b3-m26-c8k-ub256` | `--n-cpu-moe 26`, 8k ctx, `-ub 256` | load failed |
| `b3-nomtp-m27-c8k-ub256` | `--n-cpu-moe 27`, no MTP | load failed |
| `b3-nomtp-m26-c8k-ub256` | `--n-cpu-moe 26`, no MTP | load failed |
| `b4-m27-ts55_45` | m27, -ts 55/45, 20k ctx | load failed |
| `b4-m27-ts60_40` | m27, -ts 60/40, 20k ctx | load failed |
| `b4-m27-ts70_30` | m27, -ts 70/30, 20k ctx | load failed |
| `b4-m26-ts60_40` | m26, -ts 60/40, 20k ctx | load failed |
| `b4-m26-ts70_30` | m26, -ts 70/30, 20k ctx | load failed |
| `b4-m26-ts80_20` | m26, -ts 80/20, 20k ctx | load failed |
| `b4-m25-ts70_30` | m25, -ts 70/30, 20k ctx | load failed |
| `b4-m25-ts80_20` | m25, -ts 80/20, 20k ctx | load failed |

## Project Maya (VRAM expert cache, same GGUF files)

| Row | What changed | Decode tok/s (median) | Range | Reps | Prefill tok/s |
|---|---|---|---|---|---|
| `maya-s-res2000` | 16k ctx, 2000 MiB VRAM reserve, MTP drafting | **37.1** | 32.9-37.3 | 3 | n/a |
| `maya-s-c32k-res2000` | same as the row above (ran at 16k, see note) | **33.3** | 31.2-34.3 | 3 | n/a |
| `maya-s-c32k-res1000` | reserve 1000 MiB (16k, flag from row label) | **33.7** | 31.7-34.4 | 3 | n/a |
| `maya-s-c32k-res2000-nommap` | `--mmap-experts` off (16k, flag from row label) | **32.9** | 29.8-33.0 | 3 | n/a |
| `maya-ctx32k` | true 32k ctx | **31.4** | 31.0-35.2 | 3 | 945 |
| `maya-ctx32k-rep2` | true 32k ctx, repeat | **33.2** | 30.9-34.6 | 3 | 941 |
| `maya-ctx32k-pf32k` | true 32k ctx, `--prefill 32768` | **32.4** | 30.0-35.2 | 3 | 743 |
| `maya-ctx32k-nodraft` | true 32k ctx, `--drafting off` | **32.1** | 31.9-35.0 | 3 | 951 |

Note: the four `maya-s-*` rows ran at 16k context. Three of them were meant to be 32k: passing `--context 32768` on the `maya.py` command line did not override the config, and every one of those logs says `context 16384 tokens`. Editing `--max-context` in the config JSON did work, and the `maya-ctx32k*` rows (log says `context 32768 tokens`) are the real 32k runs. `maya-s-res2000` and `maya-s-c32k-res2000` are the same effective config and landed at 37.1 and 33.3, so run-to-run spread on Maya is at least that wide. The long-prompt leg returned `HTTP Error 400: Bad Request` on every 16k row (the 16.7k-token prompt does not fit in 16k), so those rows have no prefill number. `maya-s-c32k-res2000-pf32k` failed to load (cause not investigated) and was redone as `maya-ctx32k-pf32k`.

## Context depth ladder on Maya (one run per depth)

| Context set | Prompt tokens | Cold time to first token | Prefill tok/s | Decode tok/s | Needle found |
|---|---|---|---|---|---|
| 128k | 116,523 | 100 s | 1,165 | 27.9 | yes |
| 256k | 232,943 | 193 s | 1,205 | 25.3 | yes |
| 512k | 465,632 | 423 s | 1,100 | 27.5 | yes |
| 1M | 931,009 | 1,021 s | 911 | 20.7 | yes |

Needle = a passphrase placed about 5% into non-repeating filler text, asked for at the end (`needle=True` in the CSV). It tests lookup only. One run per depth, no repeats.
