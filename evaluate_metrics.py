"""
evaluate_metrics.py -- full classification report for a trained model.

Run from the project root (virtual environment active):

    python evaluate_metrics.py --model baseline
    python evaluate_metrics.py --model robust
    python evaluate_metrics.py --model robust --attack pgd

Clean evaluation: accuracy, balanced accuracy, precision / recall / F1
(macro and weighted), specificity, MCC, Cohen's kappa, a per-class table,
a per-SNR table and the confusion matrix, on the test set.

With --attack: the same model is also attacked (FGSM, PGD or MIM) and the
script reports accuracy under attack, attack success rate (ASR) and the
full classification report of the attacked predictions.

The test loader uses a fixed pseudo-random order (data_loader.py), so
--max-samples and --attack-samples take the SAME random samples every time:
baseline and robust results are directly comparable.

Files are written to --out-dir (default results/metrics).
"""

import argparse
import csv
import os

import numpy as np

from metrics_core import attack_summary, build_report, write_report

try:                                    # the class names your project already stores
    from src.evaluate import DEFAULT_CLASS_NAMES as CLASS_NAMES
except Exception:                       # pragma: no cover
    CLASS_NAMES = [f"class_{i:02d}" for i in range(24)]


# ----------------------------------------------------------------------
# printing / plotting helpers (no torch needed)
# ----------------------------------------------------------------------
def print_overall(title, overall):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    rows = [
        ("samples", f"{overall['n_samples']:,}"),
        ("accuracy", f"{100 * overall['accuracy']:.2f}%"),
        ("balanced accuracy", f"{100 * overall['balanced_accuracy']:.2f}%"),
        ("macro precision", f"{100 * overall['macro_precision']:.2f}%"),
        ("macro recall", f"{100 * overall['macro_recall']:.2f}%"),
        ("macro F1", f"{100 * overall['macro_f1']:.2f}%"),
        ("weighted F1", f"{100 * overall['weighted_f1']:.2f}%"),
        ("macro specificity", f"{100 * overall['macro_specificity']:.2f}%"),
        ("MCC", f"{overall['mcc']:.4f}"),
        ("Cohen's kappa", f"{overall['cohen_kappa']:.4f}"),
    ]
    for k, v in rows:
        print(f"  {k:20s}{v:>12s}")


def print_worst_classes(overall, class_names, k=5):
    f1 = overall["per_class"]["f1"]
    order = sorted(range(len(class_names)), key=lambda i: f1[i])[:k]
    print(f"\n  {k} hardest classes (lowest F1):")
    for i in order:
        pc = overall["per_class"]
        print(f"    {class_names[i]:12s} F1 {100 * f1[i]:5.1f}%   precision {100 * pc['precision'][i]:5.1f}%   recall {100 * pc['recall'][i]:5.1f}%")


def print_snr_table(per_snr, every=4):
    print("\n  SNR (dB)   samples   accuracy   macro-F1")
    for r in per_snr:
        if r["snr"] % every == 0 or r["snr"] in (-20, 30):
            print(f"  {r['snr']:>7}   {r['n_samples']:>7,}   {100 * r['accuracy']:7.2f}%   {100 * r['macro_f1']:7.2f}%")


def print_confusions(top):
    if top:
        print("\n  most frequent mistakes (true -> predicted):")
        for t in top[:5]:
            print(f"    {t['true']:10s} -> {t['predicted']:10s} {t['count']:>7,}")


def plot_confusion(cm, class_names, path, title):
    """Row-normalised confusion matrix as a PNG. Silently skipped if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="share of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_attack_per_snr(path, snr, y_true, clean_pred, adv_pred):
    snr = np.rint(np.asarray(snr, dtype=float)).astype(int)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "n_samples", "n_originally_correct", "clean_accuracy", "accuracy_under_attack", "attack_success_rate"])
        for s in sorted(set(snr.tolist())):
            m = snr == s
            a = attack_summary(y_true[m], clean_pred[m], adv_pred[m])
            w.writerow([s, a["n_samples"], a["n_originally_correct"], f"{a['clean_accuracy']:.4f}",
                        f"{a['accuracy_under_attack']:.4f}", f"{a['attack_success_rate']:.4f}"])


# ----------------------------------------------------------------------
# model / data glue
# ----------------------------------------------------------------------
def iterate(loader, limit):
    """Yield (x, y, snr) batches; stop after exactly `limit` samples if given."""
    seen = 0
    for x, y, snr in loader:
        if limit is not None and seen + len(y) > limit:
            keep = limit - seen
            x, y, snr = x[:keep], y[:keep], snr[:keep]
        yield x, y.view(-1).long(), snr.view(-1)
        seen += len(y)
        if limit is not None and seen >= limit:
            break


def main():
    import torch

    from src.attacks import fgsm_attack, mim_attack, pgd_attack
    from src.data_loader import create_dataloaders
    from src.model_baseline import BaselineCNN
    from src.model_robust import RobustCNN

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["baseline", "robust"], required=True)
    parser.add_argument("--checkpoint", default=None, help="default: checkpoints/<model>_best.pt")
    parser.add_argument("--attack", choices=["none", "fgsm", "pgd", "mim"], default="none")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--attack-steps", type=int, default=10, help="iterations for PGD / MIM")
    parser.add_argument("--max-samples", type=int, default=None, help="limit the clean evaluation (default: whole test set)")
    parser.add_argument("--attack-samples", type=int, default=12288)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir", default="results/metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or f"checkpoints/{args.model}_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = (BaselineCNN if args.model == "baseline" else RobustCNN)(num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"Model      : {args.model}   ({ckpt_path})")
    print(f"Checkpoint : epoch {ckpt.get('epoch')}, validation accuracy {100 * ckpt.get('val_accuracy', float('nan')):.2f}%")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device     : {device}")

    _, _, test_loader = create_dataloaders(batch_size=args.batch_size, num_workers=0)

    # ---------------- clean evaluation ----------------
    print("\nEvaluating on the test set ...")
    preds, labels, snrs = [], [], []
    done = 0
    with torch.no_grad():
        for i, (x, y, s) in enumerate(iterate(test_loader, args.max_samples)):
            preds.append(model(x.to(device).float()).argmax(1).cpu())
            labels.append(y)
            snrs.append(s)
            done += len(y)
            if (i + 1) % 100 == 0:
                print(f"  ... {done:,} samples", flush=True)
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(preds).numpy()
    snr = torch.cat(snrs).numpy()

    report = build_report(y_true, y_pred, snr, CLASS_NAMES)
    print_overall(f"CLEAN RESULTS - {args.model}", report["overall"])
    print_worst_classes(report["overall"], CLASS_NAMES)
    print_snr_table(report["per_snr"])
    print_confusions(report["top_confusions"])
    files = write_report(report, CLASS_NAMES, args.out_dir, args.model)
    png = os.path.join(args.out_dir, f"{args.model}_confusion_matrix.png")
    if plot_confusion(report["confusion_matrix"], CLASS_NAMES, png, f"{args.model}: confusion matrix (clean)"):
        files.append(png)

    # ---------------- optional attacked evaluation ----------------
    if args.attack != "none":
        attack_fn = {
            "fgsm": lambda m, x, y: fgsm_attack(m, x, y, args.epsilon),
            "pgd": lambda m, x, y: pgd_attack(m, x, y, args.epsilon, steps=args.attack_steps),
            "mim": lambda m, x, y: mim_attack(m, x, y, args.epsilon, steps=args.attack_steps),
        }[args.attack]
        print(f"\nAttacking {args.attack_samples:,} test signals with {args.attack.upper()} (epsilon={args.epsilon}) ...")
        a_true, a_clean, a_adv, a_snr = [], [], [], []
        done = 0
        for i, (x, y, s) in enumerate(iterate(test_loader, args.attack_samples)):
            x, y = x.to(device).float(), y.to(device)
            with torch.no_grad():
                clean_pred = model(x).argmax(1)
            x_adv = attack_fn(model, x, y)
            with torch.no_grad():
                adv_pred = model(x_adv).argmax(1)
            a_true.append(y.cpu()); a_clean.append(clean_pred.cpu()); a_adv.append(adv_pred.cpu()); a_snr.append(s)
            done += len(y)
            if (i + 1) % 10 == 0:
                print(f"  ... {done:,} samples", flush=True)
        a_true = torch.cat(a_true).numpy(); a_clean = torch.cat(a_clean).numpy()
        a_adv = torch.cat(a_adv).numpy(); a_snr = torch.cat(a_snr).numpy()

        summary = attack_summary(a_true, a_clean, a_adv)
        print("\n" + "=" * 70)
        print(f"UNDER {args.attack.upper()} ATTACK - {args.model}  (epsilon={args.epsilon})")
        print("=" * 70)
        print(f"  signals attacked        {summary['n_samples']:>10,}")
        print(f"  clean accuracy          {100 * summary['clean_accuracy']:>9.2f}%")
        print(f"  accuracy under attack   {100 * summary['accuracy_under_attack']:>9.2f}%")
        print(f"  attack success rate     {100 * summary['attack_success_rate']:>9.2f}%   "
              f"({summary['n_flipped_by_attack']:,} of {summary['n_originally_correct']:,} correct signals flipped)")

        adv_report = build_report(a_true, a_adv, a_snr, CLASS_NAMES)
        print_overall(f"ATTACKED PREDICTIONS - {args.model} / {args.attack.upper()}", adv_report["overall"])
        name = f"{args.model}_{args.attack}"
        files += write_report(adv_report, CLASS_NAMES, args.out_dir, name)
        per_snr_path = os.path.join(args.out_dir, f"{name}_attack_per_snr.csv")
        write_attack_per_snr(per_snr_path, a_snr, a_true, a_clean, a_adv)
        files.append(per_snr_path)

    print("\nFiles written:")
    for f in files:
        print("  " + f)


if __name__ == "__main__":
    main()
