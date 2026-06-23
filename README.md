# Description
Code for supervised finetuning (SFT) and direct preference optimization (DPO) of Mellum 2, a 12B param mixture-of-experts architecture (2.5B active) specialized for coding tasks. We evaluate the Pass@1 score of the model on LiveCodeBench v6. 

# Datasets
For SFT we download nvidia/OpenCodeReasoning (OCR) dataset, shortlist 3000 decontaminated medium and hard problems with think traces. 
For DPO we download allenai/Dolci-Think-RL-7B-Completions-DPO (Dolci), decontaminate, score the "good" and "bad" locally through a Docker setup and shortlist pairs which have syntactially valid by logically good (or bad) scores. 

# Hardware
For training and inference we use a 4xA100 (80GB each) machine. 

# Distributed training setup
The model is sharded using torch FSDP (FULL_SHARD) across 4 GPUs. Each decoder layer is sharded across the GPUs. While the model can train comfortably inside the machine, this strategy introduces high network cost because all experts must be loaded for each GPU although only 8 out of 64 experts are actually used.

# Training configs
## SFT: 
Epochs:2, Micro-Batch-size:1, Gradient accumulation: 8, LR: 1e-5 with a linear warmup and linear decay, AdamW.
## DPO:
Epochs: 1, Micro-Batch-size:1, Gradient accumulation: 8, LR: 5e-7, Beta: 0.1.
Both runs took a combined of 2-3 hours excluding data preparation and environment setup.

# Evaluation
We evaluated the model results (before and after) on the LiveCodeBench v6. The Pass@1 for the original model is 0.3 while for the post-trained it is 0.19. This decrease in result is most likely because of the overfitting on small dataset and forcing the model to adopt a strict reasoning style. 