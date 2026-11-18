"""
test_metrics_core.py -- checks metrics_core.py against scikit-learn.

Run from the project root:   python test_metrics_core.py
Needs scikit-learn (your evaluate.py already imports it). Exit code 0 = all passed.
"""
import sys
import tempfile
import os

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
                             precision_recall_fscore_support)

from metrics_core import (attack_summary, build_report, confusion_matrix, metrics_from_cm,
                          top_confusions, write_report)

results = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    results.append(bool(ok))


def close(a, b):
    return bool(np.allclose(a, b, atol=1e-9))


def compare(tag, y_true, y_pred, n):
    m = metrics_from_cm(confusion_matrix(y_true, y_pred, n))
    labels = list(range(n))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for avg in ("macro", "weighted"):
        ap, ar, af, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        check(f"{tag}: {avg} precision / recall / F1 match sklearn",
              close([m[f"{avg}_precision"], m[f"{avg}_recall"], m[f"{avg}_f1"]], [ap, ar, af]))
    check(f"{tag}: per-class precision / recall / F1 match sklearn",
          close(m["per_class"]["precision"], p) and close(m["per_class"]["recall"], r) and close(m["per_class"]["f1"], f))
    check(f"{tag}: accuracy matches", close(m["accuracy"], np.mean(np.asarray(y_true) == np.asarray(y_pred))))
    check(f"{tag}: balanced accuracy matches sklearn", close(m["balanced_accuracy"], balanced_accuracy_score(y_true, y_pred)))
    check(f"{tag}: MCC matches sklearn", close(m["mcc"], matthews_corrcoef(y_true, y_pred)))
    check(f"{tag}: Cohen's kappa matches sklearn", close(m["cohen_kappa"], cohen_kappa_score(y_true, y_pred)))
    pc = m["per_class"]
    total = m["n_samples"]
    check(f"{tag}: TP+FP+FN+TN = N for every class",
          all(pc["tp"][i] + pc["fp"][i] + pc["fn"][i] + pc["tn"][i] == total for i in labels))


rng = np.random.default_rng(0)
n = 24

# 1. balanced classes, ~60% correct, errors spread randomly
y_true = rng.integers(0, n, 6000)
y_pred = np.where(rng.random(6000) < 0.6, y_true, rng.integers(0, n, 6000))
compare("balanced", y_true, y_pred, n)

# 2. imbalanced classes, errors concentrated on neighbouring classes
y_true = rng.choice(n, 6000, p=np.r_[np.full(4, 0.15), np.full(20, 0.4 / 20)])
y_pred = np.where(rng.random(6000) < 0.7, y_true, (y_true + 1) % n)
compare("imbalanced", y_true, y_pred, n)

# 3. a class that never appears in the truth but IS predicted, and one that is never predicted
y_true = rng.integers(0, n - 2, 3000)
y_pred = np.where(rng.random(3000) < 0.5, y_true, rng.integers(1, n - 1, 3000))
compare("missing classes", y_true, y_pred, n)

# 4. perfect predictions
y_true = rng.integers(0, n, 1000)
m = metrics_from_cm(confusion_matrix(y_true, y_true, n))
check("perfect predictions: accuracy = F1 = MCC = kappa = 1",
      close([m["accuracy"], m["macro_f1"], m["mcc"], m["cohen_kappa"]], [1, 1, 1, 1]))

# 5. attack summary on a hand-made example
yt = np.array([0, 1, 2, 3, 4, 5])
clean = np.array([0, 1, 2, 0, 0, 0])          # first 3 correct
adv = np.array([0, 0, 2, 3, 4, 5])            # attack flips sample 1; samples 3,4,5 happen to become right
a = attack_summary(yt, clean, adv)
check("attack summary: ASR = flipped / originally-correct = 1/3", close(a["attack_success_rate"], 1 / 3))
check("attack summary: accuracy under attack = 5/6", close(a["accuracy_under_attack"], 5 / 6))

# 6. top confusions
cm = np.zeros((4, 4), dtype=int)
cm[0, 0] = 10; cm[1, 2] = 7; cm[3, 1] = 3
tc = top_confusions(cm, ["a", "b", "c", "d"], k=5)
check("top confusions: largest wrong pair first, zeros excluded",
      tc[0] == {"true": "b", "predicted": "c", "count": 7} and len(tc) == 2)

# 7. full report + files
names = [f"c{i}" for i in range(n)]
y_true = rng.integers(0, n, 4000)
y_pred = np.where(rng.random(4000) < 0.6, y_true, rng.integers(0, n, 4000))
snr = rng.choice(np.arange(-20, 32, 2), 4000)
rep = build_report(y_true, y_pred, snr, names)
check("report: 26 SNR rows and overall N = 4000", len(rep["per_snr"]) == 26 and rep["overall"]["n_samples"] == 4000)
sel = snr == 10
check("report: per-SNR accuracy matches a direct calculation",
      close([r for r in rep["per_snr"] if r["snr"] == 10][0]["accuracy"], np.mean(y_true[sel] == y_pred[sel])))
with tempfile.TemporaryDirectory() as d:
    files = write_report(rep, names, d, "unit")
    check("report: JSON and three CSV files written and non-empty",
          len(files) == 4 and all(os.path.getsize(f) > 0 for f in files))

print(f"\n{sum(results)}/{len(results)} checks passed")

def test_all_metrics_checks_pass():
    assert len(results) == 31
    assert all(results)

if __name__ == "__main__":
    sys.exit(0 if all(results) else 1)
