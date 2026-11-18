"""
quick_check.py -- two fast sanity checks after robust training.

Run from the project root (virtual environment active):

    python quick_check.py

CHECK 1  Was BatchNorm really updated during robust training?
         The robust model started from the baseline, so its BatchNorm
         counters start at the baseline's values. They must END HIGHER.
         If they are equal, the model ran in eval mode during training
         (the old bug) and the run must be redone.

CHECK 2  Is the robust model actually harder to attack than the baseline?
         Compares accuracy under FGSM and PGD (epsilon from config) on a
         fixed random sample of test signals, overall and by SNR group.

Takes a few minutes on a GPU. Nothing is written to disk.
"""

import argparse

import torch

from src.attacks import fgsm_attack, pgd_attack
from src.data_loader import create_dataloaders
from src.model_baseline import BaselineCNN
from src.model_robust import RobustCNN

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=3072, help="random test signals to attack")
parser.add_argument("--epsilon", type=float, default=0.02)
parser.add_argument("--pgd-steps", type=int, default=10, help="PGD steps for ATTACKING (not training)")
parser.add_argument("--baseline", default="checkpoints/baseline_best.pt")
parser.add_argument("--robust", default="checkpoints/robust_best.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(model_cls, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = model_cls(num_classes=24)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


baseline, base_ckpt = load(BaselineCNN, args.baseline)
robust, rob_ckpt = load(RobustCNN, args.robust)

# ----------------------------------------------------------------------
# CHECK 1: BatchNorm counters must have grown beyond the baseline's
# ----------------------------------------------------------------------
print("=" * 70)
print("CHECK 1: BatchNorm statistics were updated during robust training")
print("=" * 70)
base_sd = base_ckpt["model_state_dict"]
rob_sd = rob_ckpt["model_state_dict"]
keys = [k for k in rob_sd if k.endswith("num_batches_tracked")]
deltas = [int(rob_sd[k]) - int(base_sd[k]) for k in keys]
mean_shift = max(
    float((rob_sd[k] - base_sd[k]).abs().max())
    for k in rob_sd if k.endswith("running_mean")
)
print(f"Robust checkpoint is from epoch {rob_ckpt.get('epoch')} "
      f"(val acc {100 * rob_ckpt.get('val_accuracy', float('nan')):.2f}%)")
print(f"BatchNorm layers checked          : {len(keys)}")
print(f"Counter growth vs baseline (min)  : {min(deltas):,}")
print(f"Counter growth vs baseline (max)  : {max(deltas):,}")
print(f"Largest running_mean change       : {mean_shift:.6f}")
check1 = len(keys) > 0 and min(deltas) > 0
print("CHECK 1:", "PASS (BatchNorm was training)" if check1 else
      "FAIL (counters did not grow -> model was in eval mode during training)")

# ----------------------------------------------------------------------
# CHECK 2: robustness comparison on a fixed random test sample
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"CHECK 2: accuracy under attack (eps={args.epsilon}, "
      f"PGD-{args.pgd_steps}, {args.samples} random test signals)")
print("=" * 70)

_, _, test_loader = create_dataloaders(batch_size=128, num_workers=0)
dataset = test_loader.dataset
gen = torch.Generator().manual_seed(0)
picks = torch.randperm(len(dataset), generator=gen)[: args.samples].tolist()

signals, labels, snrs = [], [], []
for i in picks:
    s, l, r = dataset[i]
    signals.append(s)
    labels.append(l)
    snrs.append(r)
X = torch.stack(signals).float()
Y = torch.stack(labels).long().view(-1)
S = torch.stack(snrs).view(-1)

groups = {
    "all SNRs": torch.ones_like(S, dtype=torch.bool),
    "low  (< 0 dB)": S < 0,
    "mid  (0..10 dB)": (S >= 0) & (S <= 10),
    "high (> 10 dB)": S > 10,
}


def accuracy_flags(model, attack=None):
    """Per-signal correctness (bool tensor on CPU) under an optional attack."""
    flags = []
    for start in range(0, len(X), 128):
        x = X[start:start + 128].to(device)
        y = Y[start:start + 128].to(device)
        if attack is not None:
            x = attack(model, x, y)
        with torch.no_grad():
            flags.append((model(x).argmax(1) == y).cpu())
    return torch.cat(flags)


attacks = {
    "clean": None,
    "FGSM": lambda m, x, y: fgsm_attack(m, x, y, args.epsilon),
    f"PGD-{args.pgd_steps}": lambda m, x, y: pgd_attack(m, x, y, args.epsilon, steps=args.pgd_steps),
}

table = {}
for model_name, model in (("baseline", baseline), ("robust", robust)):
    for attack_name, attack in attacks.items():
        print(f"  running {model_name:8s} / {attack_name} ...", flush=True)
        table[(model_name, attack_name)] = accuracy_flags(model, attack)

print()
header = f"{'group':18s}{'attack':10s}{'baseline':>10s}{'robust':>10s}{'gain':>9s}"
print(header)
print("-" * len(header))
for group_name, mask in groups.items():
    n = int(mask.sum())
    for attack_name in attacks:
        b = 100 * table[("baseline", attack_name)][mask].float().mean().item()
        r = 100 * table[("robust", attack_name)][mask].float().mean().item()
        print(f"{group_name:18s}{attack_name:10s}{b:9.1f}%{r:9.1f}%{r - b:+8.1f}  (n={n})")
    print()

pgd_name = f"PGD-{args.pgd_steps}"
all_mask = groups["all SNRs"]
b_pgd = 100 * table[("baseline", pgd_name)][all_mask].float().mean().item()
r_pgd = 100 * table[("robust", pgd_name)][all_mask].float().mean().item()
check2 = (r_pgd - b_pgd) > 5.0
print(f"CHECK 2: robust {r_pgd:.1f}% vs baseline {b_pgd:.1f}% under {pgd_name} "
      f"-> {'PASS' if check2 else 'FAIL'} (rule of thumb: robust should beat baseline by more than 5 points)")

print("\n" + "=" * 70)
print("RESULT:", "both checks passed - safe to re-run the attack scripts."
      if (check1 and check2) else "at least one check failed - send me this output before re-running anything.")
print("=" * 70)
