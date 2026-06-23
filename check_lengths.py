"""
check_lengths.py — audit token lengths of SFT + DPO data against max_seq_len.

Uses the SAME tokenizer + prompt formatting as training, so the numbers match
exactly what the trainer will see. Reports the distribution, counts over-length
samples, and (importantly) flags cases where truncation would eat into the
COMPLETION (think+code) rather than just the prompt.

Usage:
    python check_lengths.py \
        --model JetBrains/Mellum2-12B-A2.5B-Thinking \
        --sft out/sft_clean.jsonl \
        --dpo out/dolci_dpo_pairs.jsonl \
        --max-seq-len 24000
    # add --no-chat-template if your prompts are already templated
"""

import argparse
import json

from transformers import AutoTokenizer

from data_utils import build_prompt_text


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def lens(tok, prompt_text, completion_text, use_ct):
    p = len(tok(build_prompt_text(tok, prompt_text, use_ct),
                add_special_tokens=False)["input_ids"])
    c = len(tok(completion_text, add_special_tokens=False)["input_ids"])
    c += 1  # eos appended in training
    return p, c, p + c


def summarize(name, totals, comps, prompts, max_len):
    n = len(totals)
    over = [i for i, t in enumerate(totals) if t > max_len]
    # truncation eats the completion when prompt alone already >= max_len,
    # OR completion alone > max_len (completion gets hard-capped).
    comp_damaged = [i for i in range(n)
                    if prompts[i] >= max_len or comps[i] > max_len]
    s = sorted(totals)
    pct = lambda q: s[min(n - 1, int(q * n))]
    print(f"\n=== {name}  (n={n}, max_seq_len={max_len}) ===")
    print(f"  total tokens  min/median/p90/p99/max: "
          f"{s[0]} / {pct(.5)} / {pct(.9)} / {pct(.99)} / {s[-1]}")
    print(f"  over max_seq_len:        {len(over)}  ({100*len(over)/n:.1f}%)")
    print(f"  completion damaged:      {len(comp_damaged)}  "
          f"<-- these lose think/code, consider DROPPING")
    if over:
        worst = sorted(over, key=lambda i: -totals[i])[:5]
        print(f"  worst offenders (idx:total): "
              + ", ".join(f"{i}:{totals[i]}" for i in worst))
    return over, comp_damaged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sft")
    ap.add_argument("--dpo")
    ap.add_argument("--max-seq-len", type=int, default=24000)
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--dump-over", help="optional path to write over-length ids")
    args = ap.parse_args()
    use_ct = not args.no_chat_template

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    over_ids = []

    if args.sft:
        rows = load_jsonl(args.sft)
        P, C, T = [], [], []
        for r in rows:
            p, c, t = lens(tok, r["question"], r["reasoning"], use_ct)
            P.append(p); C.append(c); T.append(t)
        over, dmg = summarize("SFT  " + args.sft, T, C, P, args.max_seq_len)
        over_ids += [(args.sft, rows[i].get("id", i)) for i in over]

    if args.dpo:
        rows = load_jsonl(args.dpo)
        # DPO: check BOTH branches; the longer branch governs.
        P, C, T = [], [], []
        for r in rows:
            pg, cg, tg = lens(tok, r["prompt"], r["good"], use_ct)
            pb, cb, tb = lens(tok, r["prompt"], r["bad"], use_ct)
            # worst branch per sample
            if tg >= tb:
                P.append(pg); C.append(cg); T.append(tg)
            else:
                P.append(pb); C.append(cb); T.append(tb)
        over, dmg = summarize("DPO  " + args.dpo, T, C, P, args.max_seq_len)
        over_ids += [(args.dpo, rows[i].get("id", i)) for i in over]

    if args.dump_over and over_ids:
        with open(args.dump_over, "w") as f:
            for src, _id in over_ids:
                f.write(json.dumps({"source": src, "id": _id}) + "\n")
        print(f"\nWrote {len(over_ids)} over-length ids -> {args.dump_over}")

    print("\nTip: 'completion damaged' counts are the ones that actually hurt "
          "training. If small, drop them; if large, raise max_seq_len or "
          "shorten prompts.")


if __name__ == "__main__":
    main()