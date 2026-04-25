# DK-Root: A Joint Data-and-Knowledge-Driven Framework for Root Cause Analysis of QoE Degradations in Mobile Networks[[Paper](https://arxiv.org/abs/2511.11737)]


## What Is Included

- **Core classifier**: `models/model.py` implements the encoder and classification head used for root-cause prediction.
- **Training entry point**: `main.py` supports supervised training, contrastive pretraining modes and fine-tuning hooks.
- **Pipeline**: `Paper_Diffusion_semi_supervised_pipeline.py` orchestrates three stages: diffusion pretraining, contrastive representation pretraining, and expert-label fine-tuning.
- **Data loader**: `dataloader/dataloader.py` expects PyTorch `.pt` dictionaries with metadata.
- **Augmentation**: `Diffusion_aug_main.py` and `dataloader/augmentations.py` provide the conditional diffusion augmentation path described in the paper.


## Quick Start


```bash
python Paper_Diffusion_semi_supervised_pipeline.py \
  --diffusion_num_epochs 1 \
  --stage2_num_epoch 1 \
  --stage3_num_epoch 1 \
  --device cpu
```

## Methodology Alignment

The code mirrors the paper pipeline at the implementation level:

1. **Stage I**: `Diffusion_aug_main.py --training_mode diffusion_train_labeled` trains a class-conditional diffusion model on expert-labeled KPI sequences.
2. **Stage II**: `main.py --training_mode self_supervised --aug_method diffusion` pretrains the 1D-CNN encoder and Temporal-Contrast head on rule-labeled data with augmented views.
3. **Stage III**: `main.py --training_mode ft_2p` loads the Stage-II encoder and fine-tunes the full classifier on expert-labeled data.

## Citation

If you use this code, please cite the corresponding DK-Root paper when it becomes publicly available.
