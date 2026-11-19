#!/usr/bin/env python3
import csv, statistics, collections, sys
path = sys.argv[1] if len(sys.argv) > 1 else "runs/decode-results.csv"
g = collections.defaultdict(list)
pf = collections.defaultdict(list)
fails = {}
H=["engine/model","leg","users","ttft_s","prefill_tps","gen_tps_single","total_tps_conc","prompt_tokens","completion_tokens","notes"]
for r in csv.DictReader(open(path, newline=""), fieldnames=H):
    if r["leg"] == "leg": continue
    lab = r["engine/model"]
    if r["leg"] == "loadfail":
        fails[lab] = r["notes"]
    elif r["leg"] == "single" and "#r" in lab and r["gen_tps_single"]:
        g[lab.split("#r")[0]].append(float(r["gen_tps_single"]))
    elif r["leg"] == "long" and lab.endswith("#r1") and r["prefill_tps"]:
        pf[lab.split("#r")[0]].append(float(r["prefill_tps"]))
rows = sorted(g.items(), key=lambda kv: -statistics.median(kv[1]))
print(f"{'row':36} {'decode':>7} {'min':>6} {'max':>6} {'n':>2} {'prefill':>8}")
for k, v in rows:
    p = statistics.median(pf[k]) if pf[k] else 0.0
    print(f"{k:36} {statistics.median(v):7.1f} {min(v):6.1f} {max(v):6.1f} {len(v):2d} {p:8.1f}")
for k, n in fails.items():
    print(f"LOADFAIL {k}: {n[:120]}")
