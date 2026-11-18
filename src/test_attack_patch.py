"""
test_attack_patch.py  --  regression tests for the attacks.py fixes.

Run from the project root:   python -m src.test_attack_patch
(or:  python test_attack_patch.py  from a folder where `src.attacks` imports)

Uses tiny toy models (BatchNorm + Dropout, and BatchNorm + BiLSTM like the
real surrogate), so it needs no dataset or checkpoints and runs on CPU in
seconds. Exit code is 0 only if every check passes.
"""
import sys
import torch
import torch.nn as nn

from src.attacks import (
    fgsm_attack, pgd_attack, mim_attack, cw_attack,
    black_box_attack, blackbox_transfer_attack,
)

torch.manual_seed(0)
B, L, C = 4, 1024, 24
EPS = 0.02
results = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    results.append(bool(ok))


class ToyTarget(nn.Module):                      # BatchNorm + Dropout, like the baseline
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 8, 7, padding=3)
        self.bn = nn.BatchNorm1d(8)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(8, C)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h = torch.relu(self.bn(self.conv(x))).mean(-1)
        return self.fc(self.drop(h))


class ToySurrogate(nn.Module):                   # BatchNorm + BiLSTM, like the surrogate
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 8, 5, padding=2)
        self.bn = nn.BatchNorm1d(8)
        self.pool = nn.MaxPool1d(16)
        self.lstm = nn.LSTM(8, 16, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(32, C)

    def forward(self, x):
        h = self.pool(torch.relu(self.bn(self.conv(x)))).permute(0, 2, 1)
        out, _ = self.lstm(h)
        return self.fc(out[:, -1])


x = torch.randn(B, 2, L)                         # signed, unit-variance IQ-like batch
y = torch.randint(0, C, (B,))

attacks = {
    "fgsm": lambda m: fgsm_attack(m, x, y, EPS),
    "pgd":  lambda m: pgd_attack(m, x, y, EPS, steps=3),
    "mim":  lambda m: mim_attack(m, x, y, EPS, steps=3),
    "cw":   lambda m: cw_attack(m, x, y),
}

print("\n--- 1. attacks must not change the caller's train/eval mode ---")
for name, fn in attacks.items():
    m = ToyTarget(); m.train()
    mean0, nbt0 = m.bn.running_mean.clone(), m.bn.num_batches_tracked.clone()
    fn(m)
    check(f"{name}: model still in train mode afterwards",
          m.training and m.bn.training and m.drop.training)
    check(f"{name}: BatchNorm buffers untouched",
          torch.equal(m.bn.running_mean, mean0) and torch.equal(m.bn.num_batches_tracked, nbt0))
    m.eval(); fn(m)
    check(f"{name}: an eval-mode model stays in eval mode", not m.training)

print("\n--- 2. L-infinity bound for FGSM / PGD / MIM ---")
for name in ("fgsm", "pgd", "mim"):
    m = ToyTarget().eval()
    linf = (attacks[name](m) - x).abs().max().item()
    check(f"{name}: max |x_adv - x| = {linf:.4f} <= {EPS}", linf <= EPS + 1e-6)

m = ToyTarget().eval()
for fn_name, fn in (("pgd", pgd_attack), ("mim", mim_attack)):
    adv = fn(m, x, y, EPS, steps=3, alpha=0.01, random_start=False)
    check(f"{fn_name}: accepts alpha / random_start and stays within epsilon",
          (adv - x).abs().max().item() <= EPS + 1e-6)

print("\n--- 3. C&W must work in signal space (no [0,1] clamping) ---")
m = ToyTarget().eval()
with torch.no_grad():
    clean_pred = m(x).argmax(1)
try:
    adv = cw_attack(m, x, clean_pred, c=10.0, steps=50, lr=0.05)
except TypeError:                                  # original signature has no c/steps/lr
    adv = cw_attack(m, x, clean_pred)
with torch.no_grad():
    flipped = m(adv).argmax(1) != clean_pred
delta_l2 = (adv - x).flatten(1).norm(dim=1)
x_l2 = x.flatten(1).norm(dim=1)
changed = delta_l2 > 1e-6
check("cw: output keeps negative values (signal not clamped to [0,1])", bool((adv < 0).any()))
check("cw: flips the prediction of at least one sample", bool(flipped.any()))
check("cw: perturbation is smaller than the signal it perturbs (L2 < ||x||)",
      bool(changed.any()) and bool((delta_l2[changed] < x_l2[changed]).all()))

print("\n--- 4. black-box path: surrogate left exactly as found ---")
for start_mode in ("eval", "train"):
    target, sur = ToyTarget().eval(), ToySurrogate()
    sur.train(start_mode == "train")
    mean0 = sur.bn.running_mean.clone()
    nbt0 = sur.bn.num_batches_tracked.clone()
    lstm_p0 = sur.lstm.dropout
    black_box_attack(target, x, y, sur, EPS, steps=3)
    check(f"[{start_mode}] surrogate BatchNorm running stats unchanged",
          torch.equal(sur.bn.running_mean, mean0) and torch.equal(sur.bn.num_batches_tracked, nbt0))
    check(f"[{start_mode}] surrogate training flag restored",
          sur.training == (start_mode == "train") and sur.bn.training == (start_mode == "train"))
    check(f"[{start_mode}] surrogate parameters still trainable (requires_grad restored)",
          all(p.requires_grad for p in sur.parameters()))
    check(f"[{start_mode}] LSTM dropout value restored", sur.lstm.dropout == lstm_p0)

target, sur = ToyTarget().eval(), ToySurrogate().eval()
torch.manual_seed(1); a = black_box_attack(target, x, y, sur, EPS, steps=3)
torch.manual_seed(1); b = black_box_attack(target, x, (y + 1) % C, sur, EPS, steps=3)
check("ground-truth labels are NOT used (different y, same seed -> same output)", torch.equal(a, b))
check("target model never differentiated (no .grad on its parameters)",
      all(p.grad is None for p in target.parameters()))
torch.manual_seed(1); c = blackbox_transfer_attack(sur, target, x, y, EPS, steps=3)
check("blackbox_transfer_attack matches black_box_attack (single implementation)", torch.equal(a, c))
check("black-box perturbation within epsilon", (a - x).abs().max().item() <= EPS + 1e-6)

print(f"\n{sum(results)}/{len(results)} checks passed")

def test_all_attack_checks_pass():
    assert len(results) == 32
    assert all(results)

if __name__ == "__main__":
    sys.exit(0 if all(results) else 1)