"""
check_attack_strength.py -- is the attack ladder FGSM >= PGD, MIM restored?

Run from the project root (virtual environment active):

    python check_attack_strength.py

Background: on this project's baseline, FGSM (1 step) left ~5% accuracy while
PGD-10 left ~20% and MIM-10 ~35%. An iterative attack should be at least as
strong as FGSM. The suspected cause is the step size (epsilon / steps) combined
with a random start. This script attacks the SAME random test signals with the
old and the new settings and prints the accuracy under each.

Lower accuracy = stronger attack. Takes a few minutes on a GPU; writes nothing.
"""

import argparse

import torch

from src.attacks import fgsm_attack, mim_attack, pgd_attack
from src.data_loader import create_dataloaders
from src.model_baseline import BaselineCNN

parser = argparse.ArgumentParser()
parser.add_argument("--samples", type=int, default=3072)
parser.add_argument("--epsilon", type=float, default=0.02)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--checkpoint", default="checkpoints/baseline_best.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps, steps = args.epsilon, args.steps

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model = BaselineCNN(num_classes=24)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device).eval()

# Fixed random sample of test signals (same on every run).
_, _, test_loader = create_dataloaders(batch_size=128, num_workers=0)
dataset = test_loader.dataset
picks = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))[: args.samples].tolist()
items = [dataset[i] for i in picks]
X = torch.stack([it[0] for it in items]).float()
Y = torch.stack([it[1] for it in items]).long().view(-1)
S = torch.stack([it[2] for it in items]).view(-1)

variants = {
    "clean":                       None,
    "FGSM":                        lambda m, x, y: fgsm_attack(m, x, y, eps),
    "PGD  OLD (alpha=eps/steps)":  lambda m, x, y: pgd_attack(m, x, y, eps, steps=steps, alpha=eps / steps),
    "PGD  NEW (alpha=2.5eps/steps)": lambda m, x, y: pgd_attack(m, x, y, eps, steps=steps),
    "MIM  OLD (rand start, eps/steps)": lambda m, x, y: mim_attack(m, x, y, eps, steps=steps, alpha=eps / steps, random_start=True),
    "MIM  canonical (eps/steps)":  lambda m, x, y: mim_attack(m, x, y, eps, steps=steps, alpha=eps / steps),
    "MIM  NEW (alpha=2.5eps/steps)": lambda m, x, y: mim_attack(m, x, y, eps, steps=steps),
}

correct, l2_mean, at_edge = {}, {}, {}
for name, attack in variants.items():
    print(f"  running {name} ...", flush=True)
    flags, l2s, edges = [], [], []
    for start in range(0, len(X), 128):
        x0 = X[start:start + 128].to(device)
        y = Y[start:start + 128].to(device)
        x = x0
        if attack is not None:
            x = attack(model, x0, y)
            d = (x - x0).flatten(1)
            l2s.append(d.norm(dim=1).cpu())
            edges.append((d.abs() >= 0.99 * eps).float().mean(dim=1).cpu())
        with torch.no_grad():
            flags.append((model(x).argmax(1) == y).cpu())
    correct[name] = torch.cat(flags)
    if l2s:
        l2_mean[name] = torch.cat(l2s).mean().item()
        at_edge[name] = 100 * torch.cat(edges).mean().item()

groups = {
    "all SNRs": torch.ones_like(S, dtype=torch.bool),
    "mid  (0..10 dB)": (S >= 0) & (S <= 10),
    "high (> 10 dB)": S > 10,
}
print(f"\nAccuracy of the BASELINE under attack (eps={eps}, {steps} steps, n={len(X)}). Lower = stronger attack.\n")
header = f"{'attack':32s}" + "".join(f"{g:>18s}" for g in groups)
print(header)
print("-" * len(header))
acc = {}
for name in variants:
    row = f"{name:32s}"
    for g, mask in groups.items():
        a = 100 * correct[name][mask].float().mean().item()
        acc[(name, g)] = a
        row += f"{a:17.1f}%"
    print(row)

print("\nHow big is each attack's perturbation? (FGSM puts every coordinate at +/- eps)\n")
print(f"{'attack':34s}{'mean L2':>10s}{'% coords at eps edge':>24s}")
print("-" * 68)
for name in variants:
    if name in l2_mean:
        print(f"{name:34s}{l2_mean[name]:10.3f}{at_edge[name]:23.1f}%")

g = "all SNRs"
fgsm = acc[("FGSM", g)]
pgd_old, pgd_new = acc[("PGD  OLD (alpha=eps/steps)", g)], acc[("PGD  NEW (alpha=2.5eps/steps)", g)]
mim_old, mim_new = acc[("MIM  canonical (eps/steps)", g)], acc[("MIM  NEW (alpha=2.5eps/steps)", g)]
print()
print(f"Cause confirmed (new PGD stronger than old PGD): {'YES' if pgd_new < pgd_old else 'NO'}  ({pgd_old:.1f}% -> {pgd_new:.1f}%)")
print(f"Cause confirmed (MIM with 2.5eps/steps stronger than MIM with eps/steps): {'YES' if mim_new < mim_old else 'NO'}  ({mim_old:.1f}% -> {mim_new:.1f}%)")
ladder = pgd_new <= fgsm + 1.0 and mim_new <= fgsm + 1.0
print(f"Ladder restored (PGD-new and MIM-new no weaker than FGSM {fgsm:.1f}%, 1-point tolerance): {'YES' if ladder else 'NO'}")
if not ladder:
    print("-> Send me this output; the attacks need another look before anything is re-run.")
