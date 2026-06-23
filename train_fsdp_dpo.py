"""
train_fsdp_dpo.py — Direct Preference Optimization of Mellum2-12B (MoE) with raw FSDP.

Launch (4xA100 single node):
    torchrun --nproc_per_node=4 train_fsdp_dpo.py \
        --policy ckpts/sft \
        --data out/dolci_dpo_pairs.jsonl \
        --out ckpts/dpo \
        --epochs 1 --max-seq-len 8192 --micro-bsz 1 --grad-accum 8 \
        --lr 5e-7 --beta 0.1

Notes:
  - Policy model = your SFT checkpoint (trainable). Reference model = the SAME
    SFT checkpoint, frozen (requires_grad=False, eval mode). Both are FSDP-wrapped
    with identical config; on 4xA100 80GB (320GB) holding both sharded is fine.
  - DPO loss: -log_sigmoid(beta * ((lp_pol_ch - lp_ref_ch) - (lp_pol_rej - lp_ref_rej)))
  - Memory-saving alt (not used here, kept simple): precompute & cache the frozen
    ref logprobs in a first pass, then train the policy alone.
  - Same FSDP / mixed-precision / activation-checkpointing setup as the SFT script.
"""

import torch.cuda.nvtx as nvtx
from train_obs import StepLogger, nvtx_range, reduce_sum, MeanAccumulator

import argparse
import functools
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import DPODataset, DPOCollator, sequence_logprobs


def is_main():
    return dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


def detect_decoder_layer_cls(model):
    return type(model.model.layers[0])


def build_fsdp(model, layer_cls, trainable: bool):
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    m = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True,
        limit_all_gathers=True,
        cpu_offload=CPUOffload(offload_params=True) if not trainable else None,
    )
    if trainable:
        apply_activation_checkpointing(
            m,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda mod: isinstance(mod, layer_cls),
        )
    return m


def load_base(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        attn_implementation="flash_attention_2",
    )
    if hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = False
    return model


def branch_logps(model, batch, branch):
    out = model(
        input_ids=batch[f"{branch}_input_ids"],
        attention_mask=batch[f"{branch}_attention_mask"],
    )
    return sequence_logprobs(out.logits, batch[f"{branch}_labels"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="SFT checkpoint (HF format)")
    p.add_argument("--ref", default=None, help="defaults to --policy")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--micro-bsz", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
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
    
    ref_path = args.ref or args.policy

    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    rank, world = dist.get_rank(), dist.get_world_size()

    tok = AutoTokenizer.from_pretrained(args.policy)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    policy_raw = load_base(args.policy)
    ref_raw = load_base(ref_path)
    layer_cls = detect_decoder_layer_cls(policy_raw)
    log(f"[wrap] decoder layer class = {layer_cls.__name__}")

    policy = build_fsdp(policy_raw, layer_cls, trainable=True)
    ref = build_fsdp(ref_raw, layer_cls, trainable=False)
    ref.eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)

    ds = DPODataset(args.data, tok, args.max_seq_len, not args.no_chat_template)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True)
    dl = DataLoader(
        ds, batch_size=args.micro_bsz, sampler=sampler,
        collate_fn=DPOCollator(tok.pad_token_id), num_workers=2, pin_memory=True,
    )

    opt = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, betas=(0.9, 0.95), fused=True
    )

    jsonl_path = args.jsonl or os.path.join(args.out, "metrics.jsonl")
    logger = StepLogger(jsonl_path,
                        run_name=("sft" if "sft" in args.out else "dpo"),
                        enabled=is_main())
    
    steps_per_epoch = len(dl) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup))
    )

    policy.train()
    gstep = 0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        t0 = time.time(); seen = 0
        loss_acc = MeanAccumulator()
        acc_acc  = MeanAccumulator()   # reward accuracy
        kl_acc   = MeanAccumulator()   # KL-to-ref proxy
        rm_acc   = MeanAccumulator()   # reward margin
        for it, batch in enumerate(dl):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            with nvtx_range("policy_forward"):
                pol_ch  = branch_logps(policy, batch, "chosen")
                pol_rej = branch_logps(policy, batch, "rejected")
            with nvtx_range("ref_forward"):
                with torch.no_grad():
                    ref_ch  = branch_logps(ref, batch, "chosen")
                    ref_rej = branch_logps(ref, batch, "rejected")

            pol_margin = pol_ch - pol_rej
            ref_margin = ref_ch - ref_rej
            logits = args.beta * (pol_margin - ref_margin)
            with nvtx_range("backward"):
                loss = -F.logsigmoid(logits).mean() / args.grad_accum
                loss.backward()

            # ---- per micro-batch metrics (un-divided) ----
            loss_acc.add((-F.logsigmoid(logits)).mean().item())
            acc_acc.add((logits > 0).float().mean().item())
            chosen_reward   = (args.beta * (pol_ch  - ref_ch )).mean().item()
            rejected_reward = (args.beta * (pol_rej - ref_rej)).mean().item()
            rm_acc.add(chosen_reward - rejected_reward)
            kl_acc.add(0.5 * ((pol_ch - ref_ch).mean().item()
                              + (pol_rej - ref_rej).mean().item()))
            seen += 1

            if (it + 1) % args.grad_accum == 0:
                with nvtx_range("optim"):
                    gnorm = policy.clip_grad_norm_(1.0)
                    opt.step(); sched.step()
                    opt.zero_grad(set_to_none=True)
                gstep += 1

                if PROFILE and not _profiling_active and gstep == P_START:
                    torch.cuda.synchronize(); nvtx.range_push("CAPTURE")
                    torch.cuda.cudart().cudaProfilerStart(); _profiling_active = True
                if PROFILE and _profiling_active and gstep == P_STOP:
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop(); nvtx.range_pop()
                    _profiling_active = False

                if gstep % args.log_every == 0:
                    dt = time.time() - t0
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    logger.log(epoch=epoch, step=gstep, total=total_steps,
                               loss=float(loss_acc.global_mean()),
                               grad_norm=float(gnorm),
                               reward_acc=float(acc_acc.global_mean()),
                               reward_margin=float(rm_acc.global_mean()),
                               kl_to_ref=float(kl_acc.global_mean()),
                               lr=float(sched.get_last_lr()[0]),
                               ex_per_s=float(reduce_sum(float(seen)) / dt),
                               peak_mem_gb=float(peak))
                    t0 = time.time(); seen = 0
                    torch.cuda.reset_peak_memory_stats()

                loss_acc.reset(); acc_acc.reset(); kl_acc.reset(); rm_acc.reset()

                if args.max_steps > 0 and gstep >= args.max_steps:
                    stop = True
                    break

    logger.close()

    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(policy, StateDictType.FULL_STATE_DICT, save_cfg):
        state = policy.state_dict()
    if is_main():
        os.makedirs(args.out, exist_ok=True)
        clean = AutoModelForCausalLM.from_pretrained(args.policy, torch_dtype=torch.bfloat16)
        clean.load_state_dict(state)
        clean.save_pretrained(args.out, safe_serialization=True)
        tok.save_pretrained(args.out)
        log(f"[save] consolidated HF checkpoint -> {args.out}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()