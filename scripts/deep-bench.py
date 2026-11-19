#!/usr/bin/env python3
"""deep-bench.py: fill a prompt to N tokens, hide a needle early, measure TTFT + decode + retrieval.
usage: deep-bench.py --url http://127.0.0.1:8180 --model glm-5.3-flash --tokens 120000 --label X --out CSV
"""
import argparse, json, random, time, urllib.request, sys, subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-new", type=int, default=160)
ap.add_argument("--timeout", type=int, default=7200)
a = ap.parse_args()

NEEDLE = "amber-falcon-7731"
rnd = random.Random(42)
words = ("river stone harbor lantern copper meadow signal winter orchard timber ledger "
         "compass violet anchor pillar market thread bridge engine garden window canyon "
         "pocket rocket silver cabin forest ribbon tunnel basket island carpet mirror").split()

def sentence():
    n = rnd.randint(9, 18)
    s = " ".join(rnd.choice(words) for _ in range(n))
    return f"{s.capitalize()} {rnd.randint(100, 99999)}."

def post(body, stream=False, timeout=600):
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def build(chars):
    out, n = [], 0
    needle_at = int(chars * 0.05)
    placed = False
    while n < chars:
        s = sentence()
        out.append(s)
        n += len(s) + 1
        if not placed and n >= needle_at:
            out.append(f"NOTE: the vault passphrase is {NEEDLE}. Remember it.")
            placed = True
    return " ".join(out)

# calibrate chars/token with a probe
probe = build(8000)
r = json.load(post({"model": a.model, "messages": [{"role": "user", "content": probe + "\nSay ok."}],
                    "max_tokens": 4, "temperature": 0}))
ptoks = r["usage"]["prompt_tokens"]
cpt = len(probe) / max(1, ptoks)
print(f"calibrate: {len(probe)} chars = {ptoks} tokens ({cpt:.2f} chars/token)", flush=True)

text = build(int(a.tokens * cpt))
q = "\n\nQuestion: what is the vault passphrase mentioned in the notes above? Answer with the passphrase only."
body = {"model": a.model, "messages": [{"role": "user", "content": text + q}],
        "max_tokens": a.max_new, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True}}
t0 = time.time()
tfirst = None
chunks = 0
content = ""
usage = {}
try:
    resp = post(body, timeout=a.timeout)
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        j = json.loads(d)
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices", []):
            dl = ch.get("delta", {})
            piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
            if piece:
                if tfirst is None:
                    tfirst = time.time()
                chunks += 1
                content += dl.get("content") or ""
    t1 = time.time()
    ttft = (tfirst or t1) - t0
    ctoks = usage.get("completion_tokens", chunks)
    ptoks = usage.get("prompt_tokens", int(a.tokens))
    dec = (ctoks - 1) / max(0.001, t1 - (tfirst or t0)) if ctoks > 1 else 0.0
    found = NEEDLE in content
    mem = subprocess.run("free -g | sed -n 2p; nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\\n' ' '",
                         shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")
    print(f"{a.label}: prompt={ptoks} ttft={ttft:.1f}s prefill={ptoks/max(0.001,ttft):.0f} tok/s "
          f"decode={dec:.1f} tok/s completion={ctoks} needle_in_answer={found}", flush=True)
    print(f"mem: {mem}", flush=True)
    print(f"answer: {content.strip()[:120]!r}", flush=True)
    with open(a.out, "a") as f:
        f.write(f"{a.label},deep,1,{ttft:.2f},{ptoks/max(0.001,ttft):.1f},{dec:.1f},{dec:.1f},{ptoks},{ctoks},needle={found}\n")
except Exception as e:
    print(f"{a.label}: FAIL {type(e).__name__}: {e}", flush=True)
    with open(a.out, "a") as f:
        f.write(f"{a.label},deepfail,1,,,,,,,{type(e).__name__}: {str(e)[:100]}\n")
    sys.exit(2)
