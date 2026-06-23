"""
train_fsdp_sft.py — Supervised fine-tuning of Mellum2-12B (MoE) with raw PyTorch FSDP.

Launch (4xA100 single node):
    torchrun --nproc_per_node=4 train_fsdp_sft.py \
        --model JetBrains/Mellum2-12B-A2.5B-Thinking \
        --data out/sft_clean.jsonl \
        --out ckpts/sft \
        --epochs 2 --max-seq-len 8192 --micro-bsz 1 --grad-accum 8 --lr 1e-5

Design choices (see README_training.md):
  - FULL_SHARD (ZeRO-3), transformer-block auto-wrap (decoder layer class
    auto-detected at runtime, so it works regardless of Mellum2's MoE block name).
  - Mixed precision: bf16 params/compute, fp32 gradient reduction + buffers
    (keeps the MoE router numerically stable).
  - Activation checkpointing on every decoder block (needed for 12B at 8k on 80GB).
  - output_router_logits=False so the MoE load-balancing aux loss does not
    interfere with the masked LM loss.
  - Saves a CONSOLIDATED HF checkpoint (config + safetensors + tokenizer) so it
    drops straight into the LCB harness via --local_model_path.
"""

import torch.cuda.nvtx as nvtx
from train_obs import StepLogger, nvtx_range, reduce_sum, MeanAccumulator

import argparse
import functools
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import SFTDataset, SFTCollator


def is_main():
    return dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def detect_decoder_layer_cls(model):
    """Return the transformer decoder block class for the auto-wrap policy.

    Robust to model internals: grabs the class of the first element in the
    decoder layer ModuleList rather than importing a hardcoded name.
    """
    # Common HF layout: model.model.layers is the decoder stack.
    layers = model.model.layers
    return type(layers[0])


def setup():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--micro-bsz", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="stop after N optimizer steps (profiling/smoke). -1 = full run")
    p.add_argument("--jsonl", default=None,
                   help="per-step JSONL log path (default: <out>/metrics.jsonl)")
    args = p.parse_args()
    PROFILE = os.environ.get("PROFILE_ENABLE", "0") == "1"
    P_START = int(os.environ.get("PROFILE_START_STEP", "10"))
    P_STOP  = int(os.environ.get("PROFILE_STOP_STEP", "16"))
    _profiling_active = False

    setup()
    rank = dist.get_rank()
    world = dist.get_world_size()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Load on CPU first; FSDP shards as it wraps (avoids 4x full-model spikes).
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False

    layer_cls = detect_decoder_layer_cls(model)
    log(f"[wrap] decoder layer class = {layer_cls.__name__}")

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,   # fp32 grad reduction -> stable MoE router
        buffer_dtype=torch.float32,
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        limit_all_gathers=True,
    )

    jsonl_path = args.jsonl or os.path.join(args.out, "metrics.jsonl")
    logger = StepLogger(jsonl_path,
                        run_name=("sft" if "sft" in args.out else "dpo"),
                        enabled=is_main())

    # Activation checkpointing on each decoder block.
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda m: isinstance(m, layer_cls),
    )

    ds = SFTDataset(args.data, tok, args.max_seq_len, not args.no_chat_template)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(
        ds, batch_size=args.micro_bsz, sampler=sampler,
        collate_fn=SFTCollator(tok.pad_token_id), num_workers=2, pin_memory=True,
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), fused=True,
    )
    steps_per_epoch = len(dl) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup))
        * (1.0 if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    )

    model.train()
    global_step = 0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        running_tok = 0
        loss_acc = MeanAccumulator()
        for it, batch in enumerate(dl):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            with nvtx_range("forward"):
                out = model(**batch)
                loss = out.loss / args.grad_accum
            with nvtx_range("backward"):
                loss.backward()

            loss_acc.add(out.loss.item())                 # un-divided, per micro-batch
            running_tok += (batch["labels"] != -100).sum().item()

            if (it + 1) % args.grad_accum == 0:
                with nvtx_range("optim"):
                    gnorm = model.clip_grad_norm_(1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                global_step += 1

                # ---- scoped nsys capture window ----
                if PROFILE and not _profiling_active and global_step == P_START:
                    torch.cuda.synchronize(); nvtx.range_push("CAPTURE")
                    torch.cuda.cudart().cudaProfilerStart(); _profiling_active = True
                if PROFILE and _profiling_active and global_step == P_STOP:
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop(); nvtx.range_pop()
                    _profiling_active = False

                if global_step % args.log_every == 0:
                    dt = time.time() - t0
                    step_loss = loss_acc.global_mean()    # mean over microbatches + ranks
                    tps = reduce_sum(float(running_tok)) / dt
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    logger.log(epoch=epoch, step=global_step, total=total_steps,
                               loss=float(step_loss),
                               grad_norm=float(gnorm),
                               lr=float(sched.get_last_lr()[0]),
                               tok_per_s=float(tps), peak_mem_gb=float(peak))
                    t0 = time.time(); running_tok = 0
                    torch.cuda.reset_peak_memory_stats()

                loss_acc.reset()                          # reset for next opt step

                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop = True
                    break

    logger.close()

    # ---- consolidated HF save (rank0) so LCB can load it directly ----
    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg):
        state = model.state_dict()
    if is_main():
        os.makedirs(args.out, exist_ok=True)
        # Re-instantiate a clean HF model to attach the consolidated weights.
        cfg_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16
        )
        cfg_model.load_state_dict(state)
        cfg_model.save_pretrained(args.out, safe_serialization=True)
        tok.save_pretrained(args.out)
        log(f"[save] consolidated HF checkpoint -> {args.out}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()