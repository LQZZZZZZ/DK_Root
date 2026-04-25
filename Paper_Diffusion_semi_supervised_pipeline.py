"""
DK-Root: Three-Stage Semi-Supervised Pipeline
==============================================
This script reproduces the full three-stage training pipeline described in the paper:

  Stage I  – Conditional Diffusion Model Pre-training (Sec. IV-A)
             Train a class-conditional DDPM on expert-labeled data to learn
             the conditional distribution p(x | class).  The trained model is
             used in Stage II as a semantic-preserving data augmentor.

  Stage II – Contrastive Representation Pre-training (Sec. IV-B)
             Pre-train the 1D-CNN encoder + Temporal-Contrast (TC) head via a
             joint NTXent + SupCon objective on rule-labeled data.
             Weak/strong augmentations produced by Stage I are used here.
             Training mode: ``self_supervised``

  Stage III – Fine-tuning with Expert Labels (Sec. IV-C)
              Load the Stage-II encoder weights and fine-tune the full model
              (encoder + classifier) on the small expert-labeled split.
              Training mode: ``ft_2p``

Usage (quick-start on bundled example data, CPU):
-------------------------------------------------
    python Paper_Diffusion_semi_supervised_pipeline.py

The script creates experiment logs under ./experiments_logs/ .
Pass ``--device cuda`` or ``--device cuda:1`` to use a GPU.

For multi-seed sweeps, wrap this script in a shell loop or refer to the
paper's experimental setup (Appendix, Table I) for full hyperparameters.
"""

import argparse
import os
import shutil
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DK-Root full three-stage pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(here, "dataloader", "data_example")

    parser.add_argument("--seed", default=0, type=int,
                        help="Global random seed (controls all three stages)")
    parser.add_argument("--device", default="cpu", type=str,
                        help="Compute device: 'cpu', 'cuda', or 'cuda:N'")
    parser.add_argument("--data_path", default=default_data, type=str,
                        help="Root directory containing train_2p_labeled.pt and val.pt")
    parser.add_argument("--rule_data_path", default=None, type=str,
                        help="Directory with rule-labeled train.pt used in Stage II. "
                             "If None, --data_path is reused (demo mode).")
    parser.add_argument("--logs_save_dir",
                        default=os.path.join(here, "experiments_logs"),
                        type=str, help="Root directory for logs and checkpoints")
    parser.add_argument("--experiment_description", default="dk_root", type=str,
                        help="Top-level experiment group name")
    parser.add_argument("--run_description", default="full_pipeline", type=str,
                        help="Run identifier appended to log paths")

    # Per-stage epoch overrides (useful for smoke-tests or ablations).
    parser.add_argument("--diffusion_num_epochs", default=None, type=int,
                        help="Override Stage-I diffusion training epochs "
                             "(paper default: 1000)")
    parser.add_argument("--stage2_num_epoch", default=None, type=int,
                        help="Override Stage-II contrastive pre-training epochs "
                             "(paper default: 60)")
    parser.add_argument("--stage3_num_epoch", default=None, type=int,
                        help="Override Stage-III fine-tuning epochs "
                             "(paper default: 150)")

    return parser.parse_args()


def run_stage(description: str, cmd: list) -> None:
    """Print a stage banner and execute *cmd* as a subprocess."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {description}")
    print(f"{sep}")
    print("Command:", " ".join(str(c) for c in cmd))
    print()
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"Stage failed with exit code {result.returncode}. "
            "Check the log files under --logs_save_dir for details."
        )


def prepare_rule_data(args: argparse.Namespace) -> str:
    """Return a rule-labeled directory, creating a demo train.pt if needed."""
    if args.rule_data_path:
        return args.rule_data_path
    existing_train = os.path.join(args.data_path, "train.pt")
    if os.path.exists(existing_train):
        return args.data_path
    expert_train = os.path.join(args.data_path, "train_2p_labeled.pt")
    if not os.path.exists(expert_train):
        raise FileNotFoundError(
            "Demo fallback requires train_2p_labeled.pt when --rule_data_path is not provided."
        )
    demo_rule_dir = os.path.join(args.logs_save_dir, "demo_rule_data", f"seed_{args.seed}")
    os.makedirs(demo_rule_dir, exist_ok=True)
    shutil.copy2(expert_train, os.path.join(demo_rule_dir, "train.pt"))
    print(
        "No --rule_data_path was provided; using a demo rule-labeled copy at "
        f"{demo_rule_dir}. This is only for smoke testing."
    )
    return demo_rule_dir


def main() -> None:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable

    # In demo mode, create train.pt from the bundled expert batch under ignored logs.
    rule_data_path = prepare_rule_data(args)

    # Arguments shared across Stage II and Stage III.
    common = [
        "--seed", str(args.seed),
        "--device", args.device,
        "--data_path", args.data_path,
        "--logs_save_dir", args.logs_save_dir,
        "--experiment_description", args.experiment_description,
        "--run_description", args.run_description,
    ]

    # ------------------------------------------------------------------
    # Stage I – Conditional Diffusion Model Pre-training
    # ------------------------------------------------------------------
    # The UNet1D-based diffusion model learns the class-conditional
    # distribution p(x | class) from expert-labeled KPI sequences.
    # The saved checkpoint is subsequently consumed by DataTransform_diffusion
    # (dataloader/augmentations.py) to synthesise weak augmentations
    # (small reverse-diffusion time steps) and strong augmentations
    # (large time steps) for Stage II contrastive learning.
    # Paper reference: Section IV-A; Appendix Table I, Stage I row.
    # ------------------------------------------------------------------
    stage1_cmd = [
        python,
        os.path.join(here, "Diffusion_aug_main.py"),
        "--seed", str(args.seed),
        "--device", args.device,
        "--data_path", args.data_path,
        "--training_mode", "diffusion_train_labeled",
    ]
    if args.diffusion_num_epochs is not None:
        stage1_cmd += ["--diffusion_num_epochs", str(args.diffusion_num_epochs)]

    run_stage("Stage I  – Conditional Diffusion Model Pre-training", stage1_cmd)

    # ------------------------------------------------------------------
    # Stage II – Contrastive Representation Pre-training
    # ------------------------------------------------------------------
    # The 1D-CNN encoder and Temporal-Contrast (TC) head are jointly
    # pre-trained with a combined NTXent + SupCon objective on the
    # (potentially noisy) rule-labeled dataset.
    # Augmented positive pairs are generated by the Stage-I checkpoint.
    # Paper reference: Section IV-B, Eq. (4)–(8);
    #                  Appendix Table I, Stage II row (lr=3e-4, epochs=60).
    # ------------------------------------------------------------------
    stage2_cmd = [
        python,
        os.path.join(here, "main.py"),
        "--training_mode", "self_supervised",
        "--rule_data_path", rule_data_path,
        "--aug_method", "diffusion",
    ] + common
    if args.stage2_num_epoch is not None:
        stage2_cmd += ["--num_epoch", str(args.stage2_num_epoch)]

    run_stage("Stage II – Contrastive Representation Pre-training (self_supervised)", stage2_cmd)

    # ------------------------------------------------------------------
    # Stage III – Expert-Guided Fine-tuning
    # ------------------------------------------------------------------
    # The encoder weights from Stage II are loaded; the full model
    # (encoder + linear classifier) is fine-tuned end-to-end on the
    # small expert-verified split with standard cross-entropy loss.
    # Paper reference: Section IV-C;
    #                  Appendix Table I, Stage III row (lr=3e-4, epochs=150).
    # ------------------------------------------------------------------
    stage3_cmd = [
        python,
        os.path.join(here, "main.py"),
        "--training_mode", "ft_2p",
    ] + common
    if args.stage3_num_epoch is not None:
        stage3_cmd += ["--num_epoch", str(args.stage3_num_epoch)]

    run_stage("Stage III – Expert-Guided Fine-tuning (ft_2p)", stage3_cmd)

    print("\n" + "=" * 60)
    print("  DK-Root pipeline completed.")
    print("  Logs and checkpoints stored in:")
    print(f"    {args.logs_save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
