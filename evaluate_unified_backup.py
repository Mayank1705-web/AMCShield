"""
AMCShield unified, paired, class-balanced evaluation.

Run from the project root:
    python evaluate_unified.py

This script does NOT train models or modify checkpoints.
It evaluates baseline + robust on the exact same fixed test samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

from src.attacks import (
    fgsm_attack, pgd_attack, mim_attack, cw_attack, black_box_attack
)
from src.data_loader import create_dataloaders
from src.evaluate import DEFAULT_CLASS_NAMES
from src.model_baseline import BaselineCNN
from src.model_robust import RobustCNN
from src.model_surrogate import SurrogateModel


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "config.yaml"
CHECKPOINTS = ROOT / "checkpoints"
OUT = ROOT / "results" / "unified_evaluation"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ckpt(model, path: Path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return {
        "path": str(path.relative_to(ROOT)),
        "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "val_accuracy": payload.get("val_accuracy") if isinstance(payload, dict) else None,
        "sha256": sha256(path),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config():
    with CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_paired_balanced(loader, snrs, class_count, per_class):
    """
    Fixed-order test loader + deterministic first-K selection per (SNR,class).
    The same tensor collection is reused for both target models and all attacks.
    """
    buckets = {(s, c): [] for s in snrs for c in range(class_count)}
    for xb, yb, sb in loader:
        xb = xb.cpu().float()
        yb = yb.view(-1).cpu().long()
        sb = sb.view(-1).cpu().long()
        for i in range(len(yb)):
            key = (int(sb[i]), int(yb[i]))
            if key in buckets and len(buckets[key]) < per_class:
                buckets[key].append(xb[i].clone())

    missing = {str(k): len(v) for k, v in buckets.items() if len(v) < per_class}
    if missing:
        raise RuntimeError(
            "Could not construct the requested class-balanced sample set. "
            f"Missing bucket counts (first 20): {dict(list(missing.items())[:20])}"
        )

    xs, ys, ss = [], [], []
    for snr in snrs:
        for cls in range(class_count):
            for sample in buckets[(snr, cls)]:
                xs.append(sample)
                ys.append(cls)
                ss.append(snr)
    return torch.stack(xs), torch.tensor(ys), torch.tensor(ss)


def batches(x, y, batch_size):
    for start in range(0, len(y), batch_size):
        yield x[start:start + batch_size], y[start:start + batch_size]


@torch.no_grad()
def predict(model, x, device, batch_size):
    out = []
    for start in range(0, len(x), batch_size):
        logits = model(x[start:start + batch_size].to(device))
        out.append(logits.argmax(1).cpu())
    return torch.cat(out).numpy()


def attack_batch(name, model, surrogate, x, y, args):
    if name == "fgsm":
        return fgsm_attack(model, x, y, args.epsilon)
    if name == "pgd":
        return pgd_attack(model, x, y, args.epsilon, steps=args.steps)
    if name == "mim":
        return mim_attack(model, x, y, args.epsilon, steps=args.steps)
    if name == "cw":
        return cw_attack(model, x, y, c=args.cw_c, kappa=0.0,
                         steps=args.cw_steps, lr=args.cw_lr)
    if name == "blackbox":
        return black_box_attack(model, x, y, surrogate, args.epsilon,
                                steps=args.steps)
    raise ValueError(name)


def evaluate_one(model_name, model, surrogate, x, y, snr, clean_pred,
                 attack_name, args, device):
    rows = []
    per_snr = []
    all_adv_pred = []
    all_linf = []
    all_l2 = []
    for s in sorted(set(snr.tolist())):
        idx = np.flatnonzero(snr == s)
        xs = x[idx]
        ys_t = torch.as_tensor(y[idx], dtype=torch.long)
        ys = ys_t.numpy()
        clean = clean_pred[idx]
        adv_preds, linfs, l2s = [], [], []
        for xb, yb in batches(xs, ys_t, args.batch_size):
            xb = xb.to(device)
            yb = yb.to(device)
            adv = attack_batch(attack_name, model, surrogate, xb, yb, args)
            with torch.no_grad():
                pred = model(adv).argmax(1).cpu().numpy()
            adv_preds.extend(pred.tolist())
            delta = (adv.detach() - xb).flatten(1)
            linfs.extend(delta.abs().max(1).values.cpu().numpy().tolist())
            l2s.extend(torch.linalg.vector_norm(delta, ord=2, dim=1).cpu().numpy().tolist())

        adv_pred = np.asarray(adv_preds, dtype=np.int64)
        clean_ok = clean == ys
        adv_ok = adv_pred == ys
        n_clean = int(clean_ok.sum())
        flipped = int((clean_ok & ~adv_ok).sum())
        asr = flipped / n_clean if n_clean else float("nan")
        row = {
            "model": model_name, "attack": attack_name, "snr_db": int(s),
            "n_samples": int(len(idx)), "n_originally_correct": n_clean,
            "n_flipped": flipped, "clean_accuracy": float(clean_ok.mean()),
            "accuracy_under_attack": float(adv_ok.mean()),
            "attack_success_rate": float(asr),
            "mean_linf": float(np.mean(linfs)) if linfs else 0.0,
            "max_linf": float(np.max(linfs)) if linfs else 0.0,
            "mean_l2": float(np.mean(l2s)) if l2s else 0.0,
            "max_l2": float(np.max(l2s)) if l2s else 0.0,
        }
        rows.append(row)
        all_adv_pred.extend(adv_pred.tolist())
        all_linf.extend(linfs)
        all_l2.extend(l2s)
        print(f"{model_name:8s} {attack_name:9s} SNR {s:>3} | "
              f"acc {100*row['accuracy_under_attack']:6.2f}% | "
              f"ASR {100*asr:6.2f}% | n_correct {n_clean:4d}", flush=True)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=10,
                        help="samples per modulation class per SNR (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cw-steps", type=int, default=50)
    parser.add_argument("--cw-lr", type=float, default=0.01)
    parser.add_argument("--cw-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-cw", action="store_true")
    parser.add_argument("--skip-blackbox", action="store_true")
    args = parser.parse_args()

    if args.per_class < 1 or args.batch_size < 1 or args.steps < 1:
        parser.error("--per-class, --batch-size and --steps must be positive")

    if OUT.exists() and any(OUT.iterdir()):
        parser.error(
            f"Output directory already contains files: {OUT}. "
            "Move/backup that directory before running again; this script will not overwrite it."
        )

    set_seed(args.seed)
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = cfg["dataset"]
    snrs = list(range(ds["snr_range"][0], ds["snr_range"][1] + 1, ds["snr_step"]))
    nclasses = int(ds["num_classes"])

    print("=" * 76)
    print("AMCShield unified paired evaluation")
    print(f"Device: {device} | SNR levels: {len(snrs)} | classes: {nclasses}")
    print(f"Balanced sample count: {args.per_class} x {nclasses} x {len(snrs)} "
          f"= {args.per_class*nclasses*len(snrs):,} per model")
    print("No training; checkpoints are read-only.")
    print("=" * 76)

    baseline = BaselineCNN(num_classes=nclasses)
    robust = RobustCNN(num_classes=nclasses)
    surrogate = SurrogateModel(num_classes=nclasses)
    ckpt_info = {
        "baseline": load_ckpt(baseline, CHECKPOINTS / "baseline_best.pt", device),
        "robust": load_ckpt(robust, CHECKPOINTS / "robust_best.pt", device),
        "surrogate": load_ckpt(surrogate, CHECKPOINTS / "surrogate_best.pt", device),
    }

    _, _, test_loader = create_dataloaders(
        hdf5_path=ds["path"], batch_size=256, num_workers=0
    )
    print("Collecting fixed, class-balanced paired test sample set ...", flush=True)
    x, y, snr_t = collect_paired_balanced(
        test_loader, snrs, nclasses, args.per_class
    )
    snr = snr_t.numpy()
    y_np = y.numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x, "y": y, "snr": snr_t},
               OUT / f"paired_samples_{args.per_class}_per_class.pt")

    models = {"baseline": baseline, "robust": robust}
    attack_names = ["fgsm", "pgd", "mim"]
    if not args.skip_cw:
        attack_names.append("cw")
    if not args.skip_blackbox:
        attack_names.append("blackbox")

    all_rows = []
    clean_rows = []
    clean_predictions = {}
    for model_name, model in models.items():
        print(f"\nCLEAN evaluation: {model_name}", flush=True)
        clean_pred = predict(model, x, device, args.batch_size)
        clean_predictions[model_name] = clean_pred
        for s in snrs:
            m = snr == s
            clean_rows.append({
                "model": model_name, "snr_db": s, "n_samples": int(m.sum()),
                "n_correct": int((clean_pred[m] == y_np[m]).sum()),
                "accuracy": float((clean_pred[m] == y_np[m]).mean()),
            })

        for attack_name in attack_names:
            print(f"\nATTACK: {model_name} / {attack_name}", flush=True)
            rows = evaluate_one(
                model_name, model, surrogate, x, y_np, snr,
                clean_pred, attack_name, args, device
            )
            all_rows.extend(rows)
            write_csv(OUT / f"{model_name}_{attack_name}_per_snr.csv", rows)

    write_csv(OUT / "clean_per_snr.csv", clean_rows)
    write_csv(OUT / "attack_results_per_snr.csv", all_rows)

    # Long-form paired generalization gap: robust ASR minus baseline ASR.
    gap_rows = []
    index = {(r["model"], r["attack"], r["snr_db"]): r for r in all_rows}
    for attack in attack_names:
        for s in snrs:
            b = index[("baseline", attack, s)]
            r = index[("robust", attack, s)]
            gap_rows.append({
                "attack": attack, "snr_db": s,
                "baseline_asr": b["attack_success_rate"],
                "robust_asr": r["attack_success_rate"],
                "gap_robust_minus_baseline": (
                    r["attack_success_rate"] - b["attack_success_rate"]
                    if np.isfinite(b["attack_success_rate"]) and np.isfinite(r["attack_success_rate"])
                    else float("nan")
                ),
                "baseline_n_originally_correct": b["n_originally_correct"],
                "robust_n_originally_correct": r["n_originally_correct"],
            })
    write_csv(OUT / "generalization_gap.csv", gap_rows)

    metadata = {
        "status": "evaluation_completed",
        "seed": args.seed,
        "dataset": ds["name"],
        "snr_levels": snrs,
        "num_classes": nclasses,
        "samples_per_class_per_snr": args.per_class,
        "total_samples": int(len(y)),
        "asr_definition": "n_flipped / n_originally_correct; originally-correct samples only",
        "epsilon": args.epsilon,
        "iterative_steps": args.steps,
        "cw": {"steps": args.cw_steps, "lr": args.cw_lr, "c": args.cw_c,
               "note": "not epsilon bounded"},
        "attacks": attack_names,
        "checkpoints": ckpt_info,
        "limitations": [
            "One deterministic class-balanced test subset; not a multi-seed estimate.",
            "C&W uses the project's fixed-c implementation and may return unchanged samples when no adversarial example is found.",
            "Attack randomness is seeded, but CUDA kernels may not be fully deterministic.",
        ],
    }
    with (OUT / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 76)
    print("UNIFIED EVALUATION FINISHED")
    print(f"Output directory: {OUT}")
    print(f"Clean rows: {len(clean_rows)} | attack rows: {len(all_rows)} | gap rows: {len(gap_rows)}")
    print("Review metadata.json and all counts before plotting.")
    print("=" * 76)


if __name__ == "__main__":
    main()
