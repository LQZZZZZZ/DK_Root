# DK-Root: A Joint Data-and-Knowledge-Driven Framework for Root Cause Analysis of QoE Degradations in Mobile Networks[[Paper](https://arxiv.org/abs/2511.11737)]


## What Is Included

- **Core classifier**: `models/model.py` implements the simple CNN encoder and classification head used for root-cause prediction.
- **Training entry point**: `main.py` supports supervised training, contrastive pretraining modes, fine-tuning hooks, and CPU/GPU selection.
- **Paper pipeline**: `Paper_Diffusion_semi_supervised_pipeline.py` orchestrates the paper's three stages: diffusion pretraining, contrastive representation pretraining, and expert-label fine-tuning.
- **Data loader**: `dataloader/dataloader.py` expects PyTorch `.pt` dictionaries with `samples`, `labels`, and optional metadata.
- **Example data only**: `dataloader/data_example/` contains one anonymized 16-sample batch for smoke testing. The full dataset is not included.
- **Optional augmentation**: `Diffusion_aug_main.py` and `dataloader/augmentations.py` provide the conditional diffusion augmentation path described in the paper.

## Repository Layout

```text
DK_Root/
  main.py
  Diffusion_aug_main.py
  Paper_Diffusion_semi_supervised_pipeline.py
  config_files/dk_root_Configs.py
  dataloader/
    augmentations.py
    dataloader.py
    data_example/
      train_2p_labeled.pt
      val.pt
  models/
    model.py
    TC.py
    attention.py
    loss.py
  trainer/trainer.py
  requirements.txt
```

## Data Format

Each dataset file is a PyTorch dictionary:

```python
{
    "samples": torch.Tensor,      # shape: [num_samples, 33, 40]
    "labels": torch.LongTensor,   # shape: [num_samples]
    "window_num": torch.Tensor,   # optional sample ids
    "fields": list[str],          # optional KPI names
    "label_classes": list[str],   # optional class names
}
```

The public example keeps only one batch (`16` samples) and is intended only to verify that the code runs. It is not suitable for reproducing paper-level metrics.

The bundled `fields` metadata contains the 33 model-input KPI names used by the public example, and every field name matches the KPI definitions in the paper appendix. The appendix also lists additional KPIs used by the rule-based labeler; those rule-labeling-only KPIs are not required as model input features in the released example.

## Environment

The code was checked with Python from the existing `0102_env` environment and the package versions in `requirements.txt`.

If you need to install dependencies in another environment:

```bash
pip install -r requirements.txt
```

## Quick Start

Run a one-epoch smoke test on the bundled single-batch example:

```bash
cd DK_Root
python main.py --training_mode supervised --num_epoch 1 --device cpu
```

Outputs are written under `experiments_logs/`, which is ignored by Git.

To smoke-test the full three-stage paper pipeline on the bundled example, run:

```bash
python Paper_Diffusion_semi_supervised_pipeline.py \
  --diffusion_num_epochs 1 \
  --stage2_num_epoch 1 \
  --stage3_num_epoch 1 \
  --device cpu
```

When `--rule_data_path` is not provided, the pipeline creates a temporary demo `train.pt` copy under `experiments_logs/demo_rule_data/` so that Stage II can run on the public example. This fallback is only for smoke testing; paper-level experiments require a separate rule-labeled dataset.

## Running With Your Own Data

Place your full dataset outside the repository or in an ignored directory. For supervised CNN training, the directory should contain:

```text
train_2p_labeled.pt
val.pt
```

Then run:

```bash
python main.py \
  --training_mode supervised \
  --data_path /path/to/your/data/root \
  --selected_dataset . \
  --num_epoch 150 \
  --device cuda
```

For modes that use rule-labeled data, provide a directory containing `train.pt` via `--rule_data_path`. The rule-labeled file should follow the same dictionary format and use the same 33 model-input KPI ordering.

## Methodology Alignment

The code mirrors the paper pipeline at the implementation level:

1. **Stage I**: `Diffusion_aug_main.py --training_mode diffusion_train_labeled` trains a class-conditional diffusion model on expert-labeled KPI sequences.
2. **Stage II**: `main.py --training_mode self_supervised --aug_method diffusion` pretrains the 1D-CNN encoder and Temporal-Contrast head on rule-labeled data with augmented views.
3. **Stage III**: `main.py --training_mode ft_2p` loads the Stage-II encoder and fine-tunes the full classifier on expert-labeled data.

`main.py --training_mode supervised` is retained as a supervised baseline and quick smoke test. The standalone quick-start uses the default `--aug_method normal` through `config_files/dk_root_Configs.py` so it does not require a pretrained diffusion checkpoint.

## Privacy Notes

- The full original dataset is not included.
- The bundled `.pt` files contain only a single smoke-test batch.
- Metadata in the example data uses English KPI/class names and synthetic sample ids.
- Generated logs, checkpoints, and additional `.pt` files are ignored by `.gitignore`.

## Citation

If you use this code, please cite the corresponding DK-Root paper when it becomes publicly available.
