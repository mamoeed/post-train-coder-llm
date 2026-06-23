# Experimental Post-Training Loop on Mellum 2 (12B)

This repository contains the data cleaning scripts, local DPO scoring setup, and FSDP training configs used to fine-tune the JetBrains Mellum 2 12B model.


## Datasets
* **SFT:** 3,000 samples filtered from `nvidia/OpenCodeReasoning` (OCR), focusing on medium and hard problems that include thinking traces.
* **DPO:** 2,439 clean pairs prepared from `allenai/Dolci-Think-RL-7B-Completions-DPO`. 
* **Processing:** All samples were decontaminated against the LiveCodeBench v6 dataset using exact matches and text similarity filters. The DPO pairs were verified locally using a Docker sandbox setup to filter for syntactically valid outputs where the chosen response met the logic requirements and the rejected response failed.

## Infrastructure & Distributed Training
* **Hardware:** A rented cloud cluster of 4x A100 (80GB) and a local RTX 3090 (used for dataset decontamination, testing vLLM inference, testing training, and Docker scoring) and .
* **FSDP Setup:** Trained using PyTorch FSDP (`FULL_SHARD`) across the 4 GPUs. Every decoder layer is sharded across the cluster. While the model fits easily within this VRAM pool, this strategy incurs a communication overhead since all 64 experts must be synchronized across the GPUs during the passes, even though only 8 experts are active per token. 

## Training Configurations

### SFT
* Epochs: 2
* Micro-batch size: 1
* Gradient accumulation steps: 8
* Learning rate: 1e-5 (linear warmup and decay)
* Optimizer: AdamW

### DPO
* Epochs: 1
* Micro-batch size: 1
* Gradient accumulation steps: 8
* Learning rate: 5e-7
* Beta: 0.1

The total compute time on the cluster was roughly 2.5 hours excluding testing and setup.

## Evaluation & Results
Evaluation was performed on LiveCodeBench v6 using an identical configuration and a generation length constraint for both models.

* **Original Mellum 2 Base:** 0.30 Pass@1
* **Post-Trained Checkpoint:** 0.19 Pass@1

Hugging Face: https://huggingface.co/mamoeed/Mellum2-12B-post-post-train

### Analysis
The drop in benchmark performance indicates style overfitting. Forcing the model to adopt the specific reasoning and formatting structure found in the training sets likely disrupted its raw code syntax generation or biased it away from the exact structure expected by the LiveCodeBench evaluation harness. Under a limited compute budget we could only use a small train set and could not experiment with different training strategies. To fix these, future runs should blend the training samples with standard base-formatting examples and experiment with different training recipes. 
