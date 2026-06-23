# %%
"""
Downloads and decontaminates the following datasets:
  - SFT data:  OpenCodeReasoning (medium-hard)         -> out/sft_clean.jsonl
  - DPO data:  Dolci-Think-RL-7B-Completions-DPO (code) -> out/dolci_clean.jsonl

Both are decontaminated against LiveCodeBench v6 (exact id/url, BM25 token
containment, MiniLM cosine).
LCB itself must be present in data/lcb_v6.jsonl to perform decontamination.
"""
import json
import re
from pathlib import Path

from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# %%
DATA = Path("data")
OUT = Path("out")

SEED = 42

# --- OCR (SFT) ---
OCR_TARGET = 3000
OCR_SHUFFLE_BUFFER = 50000
MEDIUM_HARD = {"medium", "hard", "very_hard", "medium_hard", "hardest",
               "competition", "interview"}

# --- Dolci (DPO) ---
DOLCI_TARGET = 5000
DOLCI_REPO = "allenai/Dolci-Think-RL-7B-Completions-DPO"
DOLCI_SPLIT = "coding"


# --- Decontamination thresholds ---
CONTAIN_THRESH = 0.60
COSINE_THRESH = 0.85
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Output paths (used by the skip-if-exists guards) ---
SFT_OUT = OUT / "sft_clean.jsonl"
DOLCI_OUT = OUT / "dolci_clean.jsonl"

LCB_REPO = "livecodebench/code_generation_lite"
LCB_SHARDS = ["test.jsonl", "test2.jsonl", "test3.jsonl",
              "test4.jsonl", "test5.jsonl", "test6.jsonl"]

_CODE_FENCE = re.compile(r"```")
_word = re.compile(r"[a-z0-9]+")


def toks(s):
    return _word.findall((s or "").lower())


def norm_id(s):
    return re.sub(r"\s+", "", (s or "").strip().lower())


def device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def count_lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def add_think_tags(completion: str) -> str:
    """Wrap the reasoning prose that appears BEFORE the LAST code fence in
    <think>...</think>, leaving the final code block (and anything after)
    untouched.
 
    The LAST fence is used (not the first) because a reasoning model can emit
    ```python blocks INSIDE its reasoning trace. The final answer's code is the
    last fenced block, so everything before it -- including any intermediate
    code blocks in the reasoning -- is the think section.
 
    A fenced code block opens and closes with ```, so "the last code block"
    starts at the second-to-last ``` marker. We wrap everything before that.
 
    - If there is no code fence, the whole thing is treated as reasoning -> wrapped.
    - If the last code block starts at position 0 (no prose before it), nothing
      is wrapped (no think section).
    - If a <think> tag is already present, return unchanged (idempotent).
    """
    if not completion:
        return completion
    if "<think>" in completion:
        return completion  # already tagged, don't double-wrap
 
    # Find all fence positions.
    fences = [m.start() for m in _CODE_FENCE.finditer(completion)]
    if not fences:
        # No code block at all -> whole completion is reasoning.
        body = completion.strip()
        return f"<think>\n{body}\n</think>" if body else completion
 
    # The final code block is delimited by the last pair of fences. Its opening
    # fence is the second-to-last ``` (or the last one if fences are unbalanced).
    if len(fences) >= 2:
        last_block_start = fences[-2]
    else:
        # Only one fence in the whole string -> treat it as the opening fence
        # of the final (unclosed) code block.
        last_block_start = fences[-1]
 
    pre = completion[:last_block_start]
    rest = completion[last_block_start:]
    pre_stripped = pre.strip()
    if not pre_stripped:
        # Completion begins with the final code block -> no reasoning to wrap.
        return completion
    return f"<think>\n{pre_stripped}\n</think>\n{rest}"

# %%
class Decontaminator:
    """Flags candidates that overlap LiveCodeBench v6 via exact id/url, token
    containment, or embedding cosine."""

    def __init__(self, lcb):
        self.ids = {norm_id(x["question_id"]) for x in lcb if x.get("question_id")}
        self.urls = {norm_id(x["url"]) for x in lcb if x.get("url")}
        texts = [x["question"] for x in lcb]
        self.tok_sets = [set(toks(t)) for t in texts]
        self.bm25 = BM25Okapi([toks(t) for t in texts])
        self.model = SentenceTransformer(EMB_MODEL, device=device())
        self.emb = self.model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
        )

    def check(self, text, ext_id="", url=""):
        if ext_id and norm_id(ext_id) in self.ids:
            return "exact_id"
        if url and norm_id(url) in self.urls:
            return "exact_url"
        q = toks(text)
        if q:
            j = int(self.bm25.get_scores(q).argmax())
            contain = len(set(q) & self.tok_sets[j]) / len(set(q))
            if contain >= CONTAIN_THRESH:
                return f"contain={contain:.3f}->lcb#{j}"
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        cos = float((self.emb @ vec).max())
        if cos >= COSINE_THRESH:
            return f"cosine={cos:.3f}"
        return None


def pull_lcb():
    out = DATA / "lcb_v6.jsonl"
    if out.exists():
        return [json.loads(l) for l in out.open()]

    seen, rows = set(), []
    for shard in LCB_SHARDS:
        path = hf_hub_download(LCB_REPO, shard, repo_type="dataset")
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            qid = str(ex.get("question_id", ""))
            if qid in seen:
                continue
            seen.add(qid)
            rows.append({
                "question_id": qid,
                "title": ex.get("question_title", ""),
                "url": ex.get("platform_url") or ex.get("url") or "",
                "platform": ex.get("platform", ""),
                "question": ex.get("question_content", "") or "",
                "contest_date": str(ex.get("contest_date", "")),
            })

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


# %% ----------------------------- OCR (SFT) -----------------------------
def keep_difficulty(d):
    d = str(d).strip().lower() if d is not None else ""
    if d in MEDIUM_HARD:
        return True
    return bool(re.fullmatch(r"\d+", d)) and int(d) >= 7


def pull_ocr(decon):
    ds = load_dataset(
        "nvidia/OpenCodeReasoning", "split_0", split="split_0", streaming=True
    ).shuffle(seed=SEED, buffer_size=OCR_SHUFFLE_BUFFER)
    print(f"downloaded OCR (streaming, buffer {OCR_SHUFFLE_BUFFER})")
    kept, removed = [], []
    seen = 0
    for ex in ds:
        seen += 1
        if not keep_difficulty(ex.get("difficulty")):
            continue
        question = ex.get("input", "") or ""
        if question.strip() in ("", "-"):
            continue
        ext_id = str(ex.get("id", ""))
        reason = decon.check(question, ext_id=ext_id)
        if reason:
            removed.append({"id": ext_id, "reason": reason})
            continue
        kept.append({
            "id": ext_id or f"ocr_{len(kept)}",
            "question": question,
            "solution": ex.get("solution", "") or "",
            "reasoning": ex.get("output", "") or "",
            "difficulty": ex.get("difficulty"),
            "source": ex.get("source"),
            "dataset": ex.get("dataset"),
        })
        if len(kept) >= OCR_TARGET:
            break

    print(f"OCR: streamed {seen}, dropped {len(removed)} contaminated (kept {len(kept)})")
    with SFT_OUT.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    return kept, removed


# %% ----------------------------- Dolci (DPO) -----------------------------
def extract_question(messages):
    """Dolci stores the prompt in messages[0]['content']."""
    if not messages:
        return ""
    first = messages[0]
    if isinstance(first, dict):
        return first.get("content", "") or ""
    return str(first)


def parse_ground_truth(gt):
    """ground_truth is a JSON-encoded string of a list of assert strings.
    Defensive: may already be a list, or occasionally non-JSON."""
    if isinstance(gt, list):
        return gt
    if not isinstance(gt, str):
        return []
    try:
        parsed = json.loads(gt)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def pull_dolci(decon):
    # Load the FULL coding split into memory (~2GB), then shuffle and take 20k.
    ds = load_dataset(DOLCI_REPO, split=DOLCI_SPLIT)  # non-streaming
    print(f"loaded full Dolci '{DOLCI_SPLIT}' split: {len(ds)} rows")
    ds = ds.shuffle(seed=SEED)
    print("shuffled")

    kept, removed = [], []
    seen = 0
    for ex in ds:
        seen += 1
        question = extract_question(ex.get("messages"))
        tests = parse_ground_truth(ex.get("ground_truth"))
        raw_outputs = ex.get("outputs") or []
        
        # EXTRACT STATS EARLY:
        total_rollouts = ex.get("total_rollouts")
        total_correct = ex.get("total_correct_rollouts")
        
        # DIVERSITY FILTER FOR DPO:
        # We need at least 1 pass and at least 1 fail to create a chosen/rejected pair.
        # Skip if all rollouts passed (no rejected) OR if all failed (no chosen).
        if total_rollouts is not None and total_correct is not None:
            if total_correct == total_rollouts or total_correct == 0:
                continue

        # Need a prompt, >=1 test, and >=1 completion to score later.
        if not question.strip() or not tests or not raw_outputs:
            continue
        
        ext_id = str(ex.get("custom_id", ""))
        reason = decon.check(question, ext_id=ext_id)
        if reason:
            removed.append({"id": ext_id, "reason": reason})
            continue
        
 
        # Add <think> tags to the pre-code reasoning of each completion so that
        # SFT and DPO share the same <think>...</think> + code structure.
        outputs = [add_think_tags(o) for o in raw_outputs]
 
        kept.append({
            "id": ext_id or f"dolci_{len(kept)}",
            "question": question,
            "tests": tests,                       # list of assert strings
            "outputs": outputs,                   # think-tagged completions
            "passrate": ex.get("passrate"),
            "total_rollouts": ex.get("total_rollouts"),
            "total_correct_rollouts": ex.get("total_correct_rollouts"),
            "original_dataset": ex.get("original_dataset"),
        })
        if len(kept) >= DOLCI_TARGET:
            break
        if len(kept) % 1000 ==0:
            print(len(kept),' dolci samples done')
 
    print(f"Dolci: scanned {seen}, dropped {len(removed)} contaminated (kept {len(kept)})")
    with DOLCI_OUT.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    # save 10 random samples for manual inspection.
    import random
    rng = random.Random(SEED)
    sample_n = min(10, len(kept))
    samples = rng.sample(kept, sample_n)
    with (OUT / "random_samples_dolci.jsonl").open("w") as f:
        for r in samples:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {sample_n} random samples to {OUT / 'random_samples_dolci.jsonl'}")

    return kept, removed

# %% --------------------------------------------------------
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# Decide upfront what work remains so we only build LCB if needed.
need_ocr = not SFT_OUT.exists()
need_dolci = not DOLCI_OUT.exists()

if SFT_OUT.exists():
    print(f"SKIP OCR: {SFT_OUT} already exists ({count_lines(SFT_OUT)} rows)")
if DOLCI_OUT.exists():
    print(f"SKIP Dolci: {DOLCI_OUT} already exists ({count_lines(DOLCI_OUT)} rows)")

decon = None
lcb = None
if need_ocr or need_dolci:
    print("downloading LCB")
    lcb = pull_lcb()
    decon = Decontaminator(lcb)
else:
    print("Nothing to do -- both outputs already present.")

# %%
ocr_kept, ocr_removed = ([], [])
if need_ocr:
    ocr_kept, ocr_removed = pull_ocr(decon)

# %%
dolci_kept, dolci_removed = ([], [])
if need_dolci:
    dolci_kept, dolci_removed = pull_dolci(decon)

# %%
# Only (re)write the report for the parts we actually ran this time.
report = {
    "thresholds": {"containment": CONTAIN_THRESH, "cosine": COSINE_THRESH},
    "summary": {
        "lcb": (len(lcb) if lcb is not None else "skipped"),
        "ocr_kept": (len(ocr_kept) if need_ocr else "skipped(exists)"),
        "ocr_removed": (len(ocr_removed) if need_ocr else "skipped(exists)"),
        "dolci_kept": (len(dolci_kept) if need_dolci else "skipped(exists)"),
        "dolci_removed": (len(dolci_removed) if need_dolci else "skipped(exists)"),
    },
    "removed_ocr": ocr_removed if need_ocr else [],
    "removed_dolci": dolci_removed if need_dolci else [],
}
with (OUT / "decontamination_report.json").open("w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report["summary"], indent=2))
