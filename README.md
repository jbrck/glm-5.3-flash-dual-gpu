# GLM-5.3-Flash on two consumer GPUs: a measured decode-speed ladder

I wanted the fastest way to serve GLM-5.3-Flash (321B MoE, about 18B active) on a dual-GPU box with a lot of RAM and not much VRAM. This repo has every row I ran, the harness that ran them, and the settings that moved the number. It also has the ones that didn't.

**Speed only.** I did not test output quality. Read [What I did not test](#what-i-did-not-test) before you trust any of this.

## Headline

| Stack | Decode tok/s | vs baseline |
|---|---|---|
| ik_llama.cpp, plain layer split (baseline) | 15.7 | 1.00x |
| ik_llama.cpp, best settings found | 18.4 (17.4 on a fresh-clone re-run) | 1.11x to 1.17x |
| Project Maya, 16k context | 33 to 37 (best row 37.1) | 2.1x to 2.4x |
| Project Maya, 32k context | 31 to 33 | about 2.0x |

Prefill moved more than decode. ik_llama.cpp prefilled a 16.7k-token prompt at 80 to 100 tok/s. Maya did the same prompt at about 940 tok/s (roughly 10x).

Same GGUF files for every row. Single request, temperature 0, one fixed prompt. Medians of 2 to 3 reps (a few rows have 1). Maya's spread between identical configs was wider than ik_llama.cpp's, so I give it as a range. Full tables with ranges and rep counts: [results/RESULTS.md](results/RESULTS.md). Raw CSV: [results/decode-results.csv](results/decode-results.csv).

## Rig

| Part | Spec |
|---|---|
| GPUs | 2x RTX 3080, 20 GB each (modded cards, 40 GB total), driver 595.99.02 |
| CPU | 2x Xeon Gold 6138 (40 cores, 80 threads, 2 NUMA nodes, AVX2) |
| RAM | 262 GB DDR4 (all channels populated) |
| Model | GLM-5.3-Flash Maya-S v2 IQ2_XXS, 3 GGUF shards, 96.5 GB, from [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) (`Maya-S-v2-IQ2_XXS/`) |
| ik_llama.cpp | commit `009dce1` (2026-11-04), CUDA on, Release |
| Project Maya | v1.1.52, commit `6c9a27b` ([mw00/project-maya](https://github.com/mw00/project-maya)) |
| CUDA | 12.6 toolkit (nvcc), sm_86 |

The model is 96.5 GB and you have 40 GB of VRAM. That is the whole game. Every speedup below is a way of deciding which expert weights sit on the GPUs and which stay in RAM.

## The ladder (ik_llama.cpp)

Each row adds one change. Rep counts are small, so read the Range column.

| Step | Change | Decode tok/s | Range | Call |
|---|---|---|---|---|
| 0 | Baseline: `-sm layer --n-cpu-moe 30` | 15.7 | 15.7-15.7 (2 reps) | start here |
| 1 | `--numa distribute -t 40` | 16.0 | 15.8-16.2 | tiny gain, keep |
| 2 | MTP draft depth 1 (`--spec-type mtp:n_max=1,p_min=0.0`) | 16.9 | 16.3-17.4 | keep, draft accept rate 83 to 86% in the log |
| 3 | `--n-cpu-moe 28 -c 20480 -ub 128` | **18.4** | 18.2-18.4 (3 reps) | best ik row |

Step 3 is the only gain bigger than the noise, and it is just "put more expert layers on the GPU until it stops fitting". 28 was the lowest that loaded.

### What lost or did nothing

Noise between identical rows was about 0.5 tok/s (the MTP-1 row re-ran at 16.5 against 16.9). Anything inside that band I count as no change.

| Flag | Decode tok/s | Verdict |
|---|---|---|
| MTP draft depth 2 | 16.2 | worse than depth 1 |
| `-rtr` | 15.8 | no change (no repack target for this quant, my guess, not checked) |
| `-muge` | 16.4 | no change |
| `-mqkv` | 16.4 | no change |
| `-sm graph --max-gpu 2` | 16.4 | no change |
| `-t 36` | 15.8 | no worse or better than 40 |
| `--n-cpu-moe 27, 26, 25`, any `-ts` split, with or without MTP | n/a | would not load |

I skipped `-ger` and `-ser` on purpose. Both change which experts run, which is a quality trade I wasn't going to make blind.

## Project Maya

Maya keeps the hottest experts in VRAM, the next tier in RAM, and the rest on disk, and it ships its own CUDA engine. Same GGUF files. On this box it about doubled ik_llama.cpp's decode and was about 10x on prefill, with no tuning beyond the reserve and context settings in the table.

| Row | Context | Decode tok/s (median) | Range |
|---|---|---|---|
| `maya-s-res2000`, 2000 MiB VRAM reserve, MTP drafting | 16k | 37.1 | 32.9-37.3 |
| `maya-s-c32k-res2000`, identical config, re-run | 16k | 33.3 | 31.2-34.3 |
| `maya-s-c32k-res1000`, 1000 MiB reserve | 16k | 33.7 | 31.7-34.4 |
| `maya-s-c32k-res2000-nommap`, mmap experts off | 16k | 32.9 | 29.8-33.0 |
| `maya-ctx32k`, same config as the first row | 32k | 31.4 | 31.0-35.2 |
| `maya-ctx32k-rep2`, repeat of the row above | 32k | 33.2 | 30.9-34.6 |
| `maya-ctx32k-pf32k`, `--prefill 32768` | 32k | 32.4 | 30.0-35.2 |
| `maya-ctx32k-nodraft`, `--drafting off` | 32k | 32.1 | 31.9-35.0 |

Maya's own log reported a 59 to 67% expert-cache hit rate on the 32k runs.

What this table does and doesn't say:

- **Two runs of the identical 16k config gave 37.1 and 33.3.** That is the noise floor. Rep 1 of each run was usually the slowest (cold expert cache), which is why the Range column starts low.
- **The effect of 16k versus 32k context is smaller than that spread.** Do not read the 37.1 as a 16k advantage.
- **Drafting on or off, `--prefill 32768`, reserve 1000 versus 2000, and mmap off all fall inside the noise.** I can't tell any of them apart with 3 reps.
- **Prefill is the clean win.** About 940 tok/s cold on the 16.7k prompt at 32k context, in both runs that measured it (945 and 941), versus 80 to 100 on ik_llama.cpp. The 16k rows have no prefill number because that prompt is longer than a 16k context and the server returned HTTP 400.

### Long context (one run per size, no repeats)

| Context filled | Prompt tokens | Time to first token (cold) | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|
| 128k | 116,523 | 100 s | 1,165 | 27.9 |
| 256k | 232,943 | 193 s | 1,205 | 25.3 |
| 512k | 465,632 | 423 s | 1,100 | 27.5 |
| 1M | 931,009 | 1,021 s | 911 | 20.7 |

The model found a passphrase buried 5% into the text at every size. That is a lookup test on non-repeating filler. It says the context window works. It does not say the model reasons well at 1M tokens.

## What I did not test

I measured tokens per second and nothing else. No perplexity, no benchmark scores, no side-by-side reading of outputs. So:

- **The quant itself is the biggest quality hit and I did not measure it.** IQ2_XXS is roughly 2-bit on the routed experts. The [model card](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) claims 97.9% of the full FP8 model on four zero-shot tasks for Maya-S. That is the publisher's number, not mine, and zero-shot multiple choice is a weak proxy for long, agentic or code-heavy use.
- **MTP drafting should be output-preserving at temperature 0.** The target model verifies every drafted token. I believe that but did not diff outputs with and without it. The 83 to 86% accept rate is a speed stat. It is not a quality stat.
- **`--n-cpu-moe`, `-ub`, `-ngl`, NUMA and thread settings are placement and scheduling.** They should not change outputs. Same caveat: I did not diff.
- **I don't know if Maya's expert cache and prefetch are bit-exact against llama.cpp.** I did not compare Maya's output to ik_llama.cpp's on the same prompt. If you care, do that before you switch.
- **The 1M context run is a single-needle lookup.** One run per size.

Speed caveats:

- **One prompt.** Decode is measured on one fixed 256-token explanation prompt. Draft acceptance and Maya's expert-cache hit rate depend on content. Your workload will land somewhere else, likely lower on code and structured output.
- **Repetitive long prompt.** The 16.7k prefill prompt is a repeated word list. Cheap to prefill and easy to draft. Real documents will prefill slower.
- **Reps are low.** 2 to 3 per row, a few with 1. The `--n-cpu-moe 29` and `28 -c 8192` rows have one decode rep and crashed on the long-prompt leg. I report them but don't lean on them.
- **Single request only.** No concurrency tests.
- **Warm cache.** Maya's reps run back to back, so later reps see a warmer expert cache. Rep 1 is often the slowest. On the 16.7k prompt, reps 2 and 3 hit the server's prompt cache, so I only report prefill from rep 1.
- **One machine.** Different RAM speed, PCIe layout, or 24 GB cards will move the numbers. The ceiling here was VRAM, not bandwidth to the cards.
- **Host toggles untested.** I had no sudo on the box. NUMA balancing, CPU governor and THP are unexplored.

## Replication check

I ran `b3-m28-c20k-ub128` once from a fresh clone of this repo, on the same box, with `REPS=1`. It loaded (227 s, cold page cache) and finished clean. Decode came in at **17.4 tok/s**, against 18.2 on rep 1 of the original run (18.4 median over 3). Prefill matched: 79.9 against 80.4. One rep is not enough to say whether the 0.8 gap is noise or a real difference between runs, so treat the best ik row as roughly 17.4 to 18.4, and the gain over baseline as roughly 1.1x to 1.17x.

## Replicate it

Short version below. Full walk-through: [docs/SETUP.md](docs/SETUP.md). There is also an agent skill at [skill/glm-dual-gpu-decode/SKILL.md](skill/glm-dual-gpu-decode/SKILL.md) that encodes the procedure and the pitfalls.

```bash
git clone https://github.com/jbrck/glm-5.3-flash-dual-gpu && cd glm-5.3-flash-dual-gpu

export IK_BIN=/path/to/ik_llama.cpp/build/bin/llama-server
export MODEL_GGUF=/path/to/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf

REPS=3 ROWS_FILE=rows/batch1.psv scripts/decode-matrix.sh '.+'
python3 scripts/decode-summary.py runs/decode-results.csv
```

The runner refuses to start if the port is taken or the GPUs already hold more than 2 GB. It loads one row at a time, waits for health, runs warmup, single and long legs, appends to a CSV, then tears the server down and waits for VRAM to clear.

Rows live in `rows/*.psv`. Format: `name|ENV VARS|binary|model|alias|server flags`. `@IK@` and `@MODEL@` expand from the env vars above. Edit `--n-cpu-moe` for your VRAM. Rows that fail to load log `loadfail` and the runner moves on.

## Layout

| Path | What |
|---|---|
| `scripts/decode-matrix.sh` | ik_llama.cpp sweep runner |
| `scripts/maya-bench.sh` | same harness for Maya |
| `scripts/deep-ladder.sh`, `scripts/deep-bench.py` | long-context ladder and needle check |
| `scripts/smoke-single.py` | the client: warmup, single and long legs, streams, reads usage |
| `scripts/decode-summary.py` | median, min and max per row from the CSV |
| `rows/batch1-4.psv` | the exact flag sets for every ik row, in the order I ran them |
| `maya/config.example.json` | the Maya config I ended up with, paths replaced by placeholders |
| `results/` | raw CSV and generated tables |

## Gotchas I hit

- **`GGML_CUDA_NO_PINNED=1`** is set on every ik row. Loading 90 GB+ without it caused trouble on this box. I did not isolate how much it matters for speed.
- **Maya's `--context` flag did not override the config.** The server still said 16384. Edit `--max-context` in the config JSON.
- **Maya needs nvcc to set up.** If it says the CUDA toolkit is missing and you have one in a non-default path, pass `--nvcc`.
- **systemd-oomd can kill a session mid-load** when the model sits in RAM. Maya's README flags this (issue 497). I didn't hit it.
- **A mid-sweep kill leaves VRAM held.** The runner waits for VRAM to drop and kills by process group, never by name.

## Credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) by ikawrakow. [Project Maya](https://github.com/mw00/project-maya) by mw00. Quants by [peasantsmith](https://huggingface.co/peasantsmith). GLM-5.3-Flash by Z.ai. I wrote the harness and ran the sweeps.

## License

MIT. See [LICENSE](LICENSE). The model and engines carry their own licenses.
