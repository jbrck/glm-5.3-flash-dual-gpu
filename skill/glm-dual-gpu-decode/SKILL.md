---
name: glm-dual-gpu-decode
description: Use when tuning decode speed for a big MoE on 2 GPUs plus RAM. Sweep, noise-gate, keep winners.
---

# Tune MoE decode speed on a dual-GPU box

A procedure for finding the fastest way to serve a model that is bigger than your VRAM. Built from a GLM-5.3-Flash run (96.5 GB IQ2_XXS on 2 x 20 GB). Numbers and rows are in the repo README. This skill is the method.

Speed only. This procedure does not check output quality. Run that check yourself before you ship the winner.

## Rules

1. **Look at the box first.** Free VRAM, free RAM, running servers, and whether you have sudo. If you have no sudo, host toggles (CPU governor, NUMA balancing, THP) are out. Say so and skip them.
2. **One change per row.** Each row adds or swaps one flag against a named reference row. Otherwise you can't tell what moved the number.
3. **Re-run the reference.** Run the current best row again in the same batch. The gap between its two runs is your noise floor. On ik_llama.cpp it was about 0.5 tok/s. On Maya it was 3 to 4. A gain smaller than the floor is no gain.
4. **3 reps, separate loads.** Report median and range. Two runs of one load share a warm cache.
5. **Same prompt, temperature 0, streamed, one request at a time.** Decode tok/s is completion tokens over the span from first to last token, so prefill doesn't leak in.
6. **Rep 1 only for prefill.** Later reps hit the prompt cache and look absurdly fast.
7. **Start small, widen.** Limit a batch to the rows you can interpret. I ran 5, 8, 6, then 8 rows. Write each batch's result before you plan the next.
8. **Don't trust a row that crashed partway.** A decode number from a row whose long leg died is one rep of evidence. Say so in the table.
9. **No quality-changing flags without asking.** Expert-skipping options (`-ger`, `-ser` in ik_llama.cpp) change which experts run. Skip them or get an explicit yes.

## Procedure

1. Pick the harness. `scripts/decode-matrix.sh` takes a rows file and runs warmup, single and long legs per row, then tears down. It refuses to start if the port is busy or the GPUs hold over 2 GB.
2. Row 0 is the plain baseline: layer split, enough experts on CPU to load (`--n-cpu-moe 30` here).
3. Climb in this order. Cheap and safe first.
   1. NUMA and thread count (`--numa distribute -t <physical cores>`). Multi-socket only.
   2. Speculative decoding. MTP depth 1 first. Depth 2 was worse. Look at the draft accept rate in the log.
   3. Move experts onto the GPUs: lower `--n-cpu-moe` one step at a time until the load fails.
   4. Free VRAM for step 3: smaller `-c`, smaller `-ub`.
   5. Merge and repack flags (`-muge`, `-mqkv`, `-rtr`), `-sm graph`. These did nothing here. Try them, but don't expect much.
4. When load fails at the lowest `--n-cpu-moe`, read one failure log. Look for "tried to put X GB buffer on GPU N (Y GB free)". That is a VRAM wall. Try `-ts` splits once, then stop. More rows won't help.
5. Try the other engine. A VRAM-resident expert cache (Project Maya) doubled decode and gave about 10x prefill on this box with the same GGUF files. If one engine hits a VRAM wall, a different memory strategy may beat any flag.
6. Write down what lost, with numbers. Rejected rows are half the value for the next person.

## Pitfalls

- **Pinned memory on 90 GB+ loads.** `GGML_CUDA_NO_PINNED=1` was set on every ik row. I did not measure its speed effect.
- **Killing by name.** Don't `pkill -f` a server. It can match the wrong process. Start the server in its own process group and kill the group, then wait for VRAM to drop.
- **A leftover server fouls the next row.** The runner checks used VRAM and the port first.
- **Config key versus CLI flag.** On Maya, `--context` on the command line didn't override the config. The ready line in the log (`context N tokens`) is the truth. Check it.
- **A long prompt can exceed a small context.** A 16.7k prompt on a 16k context returns HTTP 400. The row has no prefill number. Don't treat it as a failure of the config.
- **systemd-oomd** can kill a session mid-load when the model sits in RAM. Run the load detached and read the log, not the terminal.
- **Long runs.** Cold loads take minutes (about 4 min for the first load after boot, much less once the files are in page cache). A 1M-token context fill took about 17 minutes to first token. Run these as background jobs and get notified on exit. Don't poll in a loop.

## Report format

For every row: name, the one change, median, min-max, rep count. A line for rows that failed to load, with the buffer error. A caveats block that says what you did not test. Mine said: no quality tests, one prompt, repetitive long prompt, low rep counts, one machine, no host toggles, no concurrency.

## Files

All in the repo root.

| File | Use |
|---|---|
| `scripts/decode-matrix.sh` | ik_llama.cpp sweep runner |
| `scripts/maya-bench.sh` | Maya runner, same legs |
| `scripts/deep-ladder.sh`, `scripts/deep-bench.py` | long-context ladder and needle check |
| `scripts/smoke-single.py` | client |
| `scripts/decode-summary.py` | median and range per row |
| `rows/batch1-4.psv` | the real rows, in the order I ran them |
| `docs/SETUP.md` | full setup walk-through |
