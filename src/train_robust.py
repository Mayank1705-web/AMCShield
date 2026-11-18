"""
train_robust.py

AMCShield - Robust CNN Training Runner

Purpose:
    Train RobustCNN using PGD adversarial training.

Dataset:
    RadioML 2018.01A

Input:
    (B, 2, 1024)

Classes:
    24

Training:
    PGD ONLY

Output:
    checkpoints/robust_best.pt
"""

import torch

from src.data_loader import create_dataloaders
from src.model_robust import RobustCNN
from src.train import train_model, load_config
from src.training_dashboard_hooks import should_stop_training
import argparse
from src.attacks import pgd_attack


def main():

    print("=" * 70)
    print("AMCShield Robust CNN Training")
    print("=" * 70)

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    config = load_config()

    # ------------------------------------------------------------
    # Optional command-line overrides. Every option defaults to
    # "no change", and they only modify THIS run's in-memory config,
    # never config.yaml, so the evaluation scripts keep using the
    # shared values (e.g. pgd_steps: 10 for attacking).
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="PGD adversarial training of RobustCNN"
    )
    parser.add_argument(
        "--init-from", default=None,
        help="start from this checkpoint (adversarial fine-tuning), "
             "e.g. checkpoints/baseline_best.pt",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--lr", type=float, default=None,
        help="learning rate (use about 1e-4 when fine-tuning)",
    )
    parser.add_argument(
        "--pgd-steps", type=int, default=None,
        help="PGD steps used DURING TRAINING only",
    )
    parser.add_argument(
        "--max-train-batches", type=int, default=None,
        help="cap the number of training batches per epoch",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the resolved settings and exit without training",
    )
    args = parser.parse_args()

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["learning_rate"] = args.lr

    batch_size = config.get("batch_size", 128)
    num_classes = config["dataset"].get("num_classes", 24)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dataset      : {config['dataset']['name']}")
    print(f"Classes      : {num_classes}")
    print(f"Batch size   : {batch_size}")
    print(f"Epochs       : {config.get('epochs', 30)}")
    print(f"Learning rate: {config.get('learning_rate', 0.001)}")
    print(f"Device       : {device}")

    # ------------------------------------------------------------
    # PGD configuration
    # ------------------------------------------------------------

    attack_config = config.get("attack", {})

    epsilon_config = attack_config.get(
        "epsilon",
        config.get("epsilon", 0.02)
    )

    # ------------------------------------------------------------
    # IMPORTANT:
    # The configuration may contain:
    #
    # epsilon:
    #   fgsm: 0.02
    #   pgd: 0.02
    #   mim: 0.02
    #
    # For robust training we MUST use PGD epsilon only.
    # ------------------------------------------------------------

    if isinstance(epsilon_config, dict):

        pgd_epsilon = float(
            epsilon_config.get("pgd", 0.02)
        )

    else:

        pgd_epsilon = float(
            epsilon_config
        )

    pgd_steps = attack_config.get(
        "pgd_steps",
        config.get("pgd_steps", 10)
    )

    pgd_steps = int(pgd_steps)
    if args.pgd_steps is not None:
        pgd_steps = args.pgd_steps

    print(f"PGD epsilon  : {pgd_epsilon}")
    print(f"PGD steps    : {pgd_steps}")
    print(
        f"Init from    : "
        f"{args.init_from or 'random initialisation (from scratch)'}"
    )
    print(f"Batches/epoch: {args.max_train_batches or 'all'}")

    if args.dry_run:
        print("\nDry run: nothing was trained. These are the settings "
              "that would be used.")
        return

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------

    print("\nCreating DataLoaders...")

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=0,
    )

    print(f"Train batches : {len(train_loader):,}")
    print(f"Val batches   : {len(val_loader):,}")
    print(f"Test batches  : {len(test_loader):,}")

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    print("\nCreating RobustCNN...")

    model = RobustCNN(
        num_classes=num_classes
    )

    if args.init_from:
        init_ckpt = torch.load(
            args.init_from, map_location="cpu", weights_only=False
        )
        init_state = (
            init_ckpt.get("model_state_dict")
            or init_ckpt.get("state_dict")
            or init_ckpt
        )
        model.load_state_dict(init_state)
        print(f"Initialised from: {args.init_from}")

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_params:,}"
    )

    # ------------------------------------------------------------
    # PGD adversarial training hook
    # ------------------------------------------------------------

    def adversarial_fn(model, signals, labels):

        return pgd_attack(
            model=model,
            x=signals,
            y=labels,
            epsilon=pgd_epsilon,
            steps=pgd_steps,
        )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    checkpoint_path = (
        "checkpoints/robust_best.pt"
    )

    print("\nStarting PGD adversarial training...")
    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print("=" * 70)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        adversarial_fn=adversarial_fn,
        checkpoint_path=checkpoint_path,
        max_train_batches=args.max_train_batches,
        should_stop=should_stop_training,
    )

    # ------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ROBUST TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best checkpoint: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()