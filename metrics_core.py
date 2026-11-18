"""
metrics_core.py -- classification metrics computed from a confusion matrix.

Pure numpy (no torch, no sklearn), so it can be unit-tested anywhere.
Convention: cm[i, j] = number of samples whose TRUE class is i and whose
PREDICTED class is j.  Every metric below is derived from that one matrix.
"""

import csv
import json
import os

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    idx = y_true * num_classes + y_pred
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def _safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.zeros(np.broadcast(a, b).shape, dtype=np.float64)
    np.divide(a, b, out=out, where=(b != 0))
    return out


def metrics_from_cm(cm):
    """All per-class and aggregate metrics from one confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    tp = np.diag(cm)
    support = cm.sum(axis=1)          # true samples per class
    pred_count = cm.sum(axis=0)       # predicted samples per class
    fp = pred_count - tp
    fn = support - tp
    tn = total - tp - fp - fn

    precision = _safe_div(tp, pred_count)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)

    present = (support > 0) | (pred_count > 0)      # classes that occur at all
    has_truth = support > 0

    def macro(v):
        return float(v[present].mean()) if present.any() else 0.0

    def weighted(v):
        return float((v * support).sum() / support.sum()) if support.sum() else 0.0

    # Multiclass Matthews correlation coefficient (Gorodkin) and Cohen's kappa
    correct = tp.sum()
    cov_pred_true = (pred_count * support).sum()
    denom = np.sqrt((total ** 2 - (pred_count ** 2).sum()) * (total ** 2 - (support ** 2).sum()))
    mcc = float((correct * total - cov_pred_true) / denom) if denom > 0 else 0.0
    p_obs = correct / total if total else 0.0
    p_exp = cov_pred_true / total ** 2 if total else 0.0
    kappa = float((p_obs - p_exp) / (1 - p_exp)) if p_exp < 1 else 0.0

    return {
        "n_samples": int(total),
        "accuracy": float(p_obs),
        "balanced_accuracy": float(recall[has_truth].mean()) if has_truth.any() else 0.0,
        "macro_precision": macro(precision),
        "macro_recall": macro(recall),
        "macro_f1": macro(f1),
        "macro_specificity": macro(specificity),
        "weighted_precision": weighted(precision),
        "weighted_recall": weighted(recall),
        "weighted_f1": weighted(f1),
        "mcc": mcc,
        "cohen_kappa": kappa,
        "per_class": {
            "support": support.astype(int).tolist(),
            "tp": tp.astype(int).tolist(),
            "fp": fp.astype(int).tolist(),
            "fn": fn.astype(int).tolist(),
            "tn": tn.astype(int).tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "specificity": specificity.tolist(),
        },
    }


def top_confusions(cm, class_names, k=8):
    """The k most frequent wrong (true -> predicted) pairs."""
    cm = np.asarray(cm)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:k]
    rows = []
    for flat in order:
        i, j = divmod(int(flat), cm.shape[1])
        if off[i, j] > 0:
            rows.append({"true": class_names[i], "predicted": class_names[j], "count": int(off[i, j])})
    return rows


def attack_summary(y_true, clean_pred, adv_pred):
    """
    accuracy under attack, and attack success rate (ASR) = the share of
    originally-correct samples that the attack turned into errors.
    """
    y_true = np.asarray(y_true).reshape(-1)
    clean_ok = np.asarray(clean_pred).reshape(-1) == y_true
    adv_ok = np.asarray(adv_pred).reshape(-1) == y_true
    n_clean_ok = int(clean_ok.sum())
    return {
        "n_samples": int(len(y_true)),
        "clean_accuracy": float(clean_ok.mean()) if len(y_true) else 0.0,
        "accuracy_under_attack": float(adv_ok.mean()) if len(y_true) else 0.0,
        "n_originally_correct": n_clean_ok,
        "n_flipped_by_attack": int((clean_ok & ~adv_ok).sum()),
        "attack_success_rate": float((clean_ok & ~adv_ok).sum() / n_clean_ok) if n_clean_ok else 0.0,
    }


def build_report(y_true, y_pred, snr, class_names):
    """Overall + per-class + per-SNR report from raw predictions."""
    n = len(class_names)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    snr = np.rint(np.asarray(snr, dtype=np.float64).reshape(-1)).astype(int)

    cm = confusion_matrix(y_true, y_pred, n)
    overall = metrics_from_cm(cm)

    per_snr = []
    for s in sorted(set(snr.tolist())):
        mask = snr == s
        m = metrics_from_cm(confusion_matrix(y_true[mask], y_pred[mask], n))
        per_snr.append({
            "snr": int(s),
            "n_samples": m["n_samples"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "mcc": m["mcc"],
        })
    return {
        "overall": overall,
        "per_snr": per_snr,
        "top_confusions": top_confusions(cm, class_names),
        "confusion_matrix": cm,
    }


def write_report(report, class_names, out_dir, name):
    """Write JSON + three CSVs. Returns the list of files written."""
    os.makedirs(out_dir, exist_ok=True)
    written = []

    pc = report["overall"]["per_class"]
    path = os.path.join(out_dir, f"{name}_per_class.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "precision", "recall", "f1", "specificity", "tp", "fp", "fn", "tn"])
        for i, c in enumerate(class_names):
            w.writerow([c, pc["support"][i], f"{pc['precision'][i]:.4f}", f"{pc['recall'][i]:.4f}",
                        f"{pc['f1'][i]:.4f}", f"{pc['specificity'][i]:.4f}",
                        pc["tp"][i], pc["fp"][i], pc["fn"][i], pc["tn"][i]])
    written.append(path)

    path = os.path.join(out_dir, f"{name}_per_snr.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "n_samples", "accuracy", "macro_f1", "mcc"])
        for r in report["per_snr"]:
            w.writerow([r["snr"], r["n_samples"], f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}", f"{r['mcc']:.4f}"])
    written.append(path)

    path = os.path.join(out_dir, f"{name}_confusion_matrix.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true \\ predicted"] + list(class_names))
        for i, c in enumerate(class_names):
            w.writerow([c] + [int(v) for v in report["confusion_matrix"][i]])
    written.append(path)

    path = os.path.join(out_dir, f"{name}_metrics.json")
    payload = {k: v for k, v in report.items() if k != "confusion_matrix"}
    payload["class_names"] = list(class_names)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    written.append(path)
    return written
