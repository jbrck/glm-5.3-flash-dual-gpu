#!/usr/bin/env python3
"""Smoke bench for OpenAI-compatible LLM servers.
Usage: smoke-bench.py --url URL --model ALIAS --label label --out results.csv
"""
import argparse, concurrent.futures, csv, json, time, urllib.request

def chat(url, model, prompt, max_tokens, timeout=1200):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None; last = None; usage = None; n_deltas = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    n_deltas += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    last = time.perf_counter() - t0
            if chunk.get("usage"):
                usage = chunk["usage"]
    if usage is None:  # server without stream_options support: estimate from chunks
        usage = {"prompt_tokens": 0, "completion_tokens": n_deltas}
    if ttft is None:  # server streamed usage but no visible delta
        ttft = time.perf_counter() - t0
    return ttft, (last or ttft), usage

LONG_PROMPT = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
               "mike november oscar papa quebec romeo sierra tango uniform victor ") * 450

LEGS = [
    ("warmup", 1, "Say OK.", 8),
    ("single", 1, "Explain how a diesel engine works, covering compression ignition, "
                  "the four-stroke cycle, turbocharging, and common failure modes.", 256),
    ("long",   1, LONG_PROMPT + "\n\nSummarize the text above in one sentence.", 128),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    header = ["engine/model", "leg", "users", "ttft_s", "prefill_tps",
              "gen_tps_single", "total_tps_conc", "prompt_tokens",
              "completion_tokens", "notes"]
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(header)
        for leg, users, prompt, mt in LEGS:
            try:
                rows = run_leg(args, leg, users, prompt, mt)
            except Exception as e:
                rows = [[args.label, leg, users, "", "", "", "", "", "",
                         "ERROR: %s" % e]]
            for r in rows:
                w.writerow(r)
                print(",".join(str(x) for x in r), flush=True)
            f.flush()

def run_leg(args, leg, users, prompt, mt):
    rows = []
    if users == 1:
        t0 = time.perf_counter()
        ttft, last, usage = chat(args.url, args.model, prompt, mt)
        wall = time.perf_counter() - t0
        gen_span = max(last - ttft, 0.01)
        gen_tps = usage["completion_tokens"] / gen_span if leg != "warmup" else 0.0
        prefill_tps = usage["prompt_tokens"] / max(ttft, 0.001)
        note = "" if usage["prompt_tokens"] else "no_usage_chunk_est"
        rows.append([args.label, leg, users, round(ttft, 2), round(prefill_tps, 1),
                     round(gen_tps, 1), round(gen_tps, 1), usage["prompt_tokens"],
                     usage["completion_tokens"], note])
    else:
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(users) as ex:
            results = list(ex.map(lambda _: chat(args.url, args.model, prompt, mt),
                                  range(users)))
        wall = time.perf_counter() - t0
        ttfts = [r[0] for r in results]
        total_out = sum(r[2]["completion_tokens"] for r in results)
        rows.append([args.label, leg, users, round(max(ttfts), 2), 0.0, 0.0,
                     round(total_out / wall, 1), results[0][2]["prompt_tokens"],
                     total_out, "worst_ttft=%.1fs" % max(ttfts)])
    return rows

if __name__ == "__main__":
    main()
