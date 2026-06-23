"""
Shared data utilities for Mellum2 SFT + DPO.

Used by BOTH the raw-FSDP scripts and the TRL scripts so that tokenization,
prompt formatting, and loss masking are byte-for-byte identical across backends.
That identity is what makes the raw-FSDP vs TRL GPU-usage comparison fair.

Masking rule (SFT): prompt tokens -> -100, completion (think+code) -> trained.
Masking rule (DPO): prompt tokens -> -100 in both chosen and rejected branches;
                    per-token logprobs are summed over completion tokens only.
"""

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt_text(tokenizer, question: str, use_chat_template: bool) -> str:
    """Format the user turn exactly as the model expects at inference.

    Mellum2-Thinking is a chat/reasoning model, so by default we run the
    tokenizer chat template with add_generation_prompt=True. The completion
    (which already contains <think>...</think> + ```python fence) is appended
    raw afterwards. If your `prompt`/`question` field is ALREADY fully
    templated, pass use_chat_template=False to avoid double-wrapping.
    """
    if use_chat_template and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return question


def _encode_pair(tokenizer, prompt_text: str, completion_text: str,
                 max_seq_len: int) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) with the prompt masked to IGNORE_INDEX.

    Truncates from the LEFT of the prompt if the pair exceeds max_seq_len, so
    the completion (the thing we train on) is preserved. Appends EOS.
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    # Keep completion intact; trim prompt from the left if needed.
    overflow = len(prompt_ids) + len(completion_ids) - max_seq_len
    if overflow > 0:
        prompt_ids = prompt_ids[overflow:]

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids[:]
    # Hard cap (in case completion alone exceeds max_seq_len).
    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    return input_ids, labels


class SFTDataset(Dataset):
    """Yields {input_ids, labels} for causal LM SFT with prompt masked.

    Expects rows with keys: 'question' (prompt) and 'reasoning' (completion,
    i.e. the full <think>...</think> + ```python ...``` string).
    """

    def __init__(self, path, tokenizer, max_seq_len=8192, use_chat_template=True):
        self.rows = load_jsonl(path)
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.use_chat_template = use_chat_template

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = build_prompt_text(self.tok, r["question"], self.use_chat_template)
        completion = r["reasoning"]
        input_ids, labels = _encode_pair(self.tok, prompt, completion, self.max_seq_len)
        return {"input_ids": input_ids, "labels": labels}


class DPODataset(Dataset):
    """Yields tokenized chosen/rejected branches for raw-FSDP DPO.

    Expects rows with keys: 'prompt', 'good' (chosen), 'bad' (rejected).
    Each branch is (prompt + completion) with the prompt masked.
    """

    def __init__(self, path, tokenizer, max_seq_len=8192, use_chat_template=True):
        self.rows = load_jsonl(path)
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.use_chat_template = use_chat_template

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt = build_prompt_text(self.tok, r["prompt"], self.use_chat_template)
        ch_ids, ch_lab = _encode_pair(self.tok, prompt, r["good"], self.max_seq_len)
        rj_ids, rj_lab = _encode_pair(self.tok, prompt, r["bad"], self.max_seq_len)
        return {
            "chosen_input_ids": ch_ids, "chosen_labels": ch_lab,
            "rejected_input_ids": rj_ids, "rejected_labels": rj_lab,
        }


def _pad(seqs: list[list[int]], pad_value: int) -> torch.Tensor:
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


@dataclass
class SFTCollator:
    pad_token_id: int

    def __call__(self, batch):
        input_ids = _pad([b["input_ids"] for b in batch], self.pad_token_id)
        labels = _pad([b["labels"] for b in batch], IGNORE_INDEX)
        attention_mask = (input_ids != self.pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": attention_mask}


@dataclass
class DPOCollator:
    pad_token_id: int

    def __call__(self, batch):
        out = {}
        for branch in ("chosen", "rejected"):
            ids = _pad([b[f"{branch}_input_ids"] for b in batch], self.pad_token_id)
            lab = _pad([b[f"{branch}_labels"] for b in batch], IGNORE_INDEX)
            out[f"{branch}_input_ids"] = ids
            out[f"{branch}_labels"] = lab
            out[f"{branch}_attention_mask"] = (ids != self.pad_token_id).long()
        return out


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Sum log p(token) over positions where labels != IGNORE_INDEX.

    logits: [B, T, V]  labels: [B, T]  ->  [B] summed completion logprob.
    Standard next-token shift.
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != IGNORE_INDEX
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = torch.gather(logp, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask).sum(dim=-1)