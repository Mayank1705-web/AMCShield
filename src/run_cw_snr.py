import torch
import numpy as np

from src.model_robust import RobustCNN
from src.data_loader import create_dataloaders
from src.attacks import cw_attack


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = r"checkpoints\robust_best.pt"

BATCH_SIZE = 32

# Number of representative samples initially collected per SNR.
SAMPLES_PER_SNR = 64

# Maximum number of CLEAN-CORRECT samples that will actually
# be attacked by C&W for each SNR.
CW_SUBSET_SIZE = 32

# Pilot SNR.
PILOT_SNR = 10

# C&W configuration
CW_STEPS = 50
CW_C = 1.0
CW_KAPPA = 0.0
CW_LR = 0.01

RANDOM_SEED = 42


# ============================================================
# Utility functions
# ============================================================

def load_model(device):
    """Load the trained RobustCNN checkpoint."""

    model = RobustCNN(num_classes=24)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("\n✓ Robust checkpoint loaded")
    print(
        f"  Checkpoint epoch      : "
        f"{checkpoint['epoch']}"
    )
    print(
        f"  Validation accuracy   : "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    return model


def collect_snr_indices(dataset):
    """
    Collect all dataset indices grouped by SNR.
    """

    print("\nCollecting representative samples by SNR...")

    rng = np.random.default_rng(RANDOM_SEED)

    all_snr_indices = {}

    for index in range(len(dataset)):

        _, _, snr = dataset[index]

        snr_value = int(snr.item())

        if snr_value not in all_snr_indices:
            all_snr_indices[snr_value] = []

        all_snr_indices[snr_value].append(index)

    snr_indices = {}

    for snr_value in sorted(all_snr_indices):

        available = np.array(
            all_snr_indices[snr_value],
            dtype=np.int64
        )

        sample_count = min(
            SAMPLES_PER_SNR,
            len(available)
        )

        selected = rng.choice(
            available,
            size=sample_count,
            replace=False
        )

        snr_indices[snr_value] = selected.tolist()

    print(
        f"✓ SNR buckets collected: "
        f"{len(snr_indices)}"
    )

    return snr_indices


def load_batch(dataset, indices, device):
    """
    Load a list of dataset indices into tensors.
    """

    signals = []
    labels = []

    for idx in indices:

        x, y, _ = dataset[idx]

        signals.append(x)
        labels.append(y)

    x = torch.stack(signals).float().to(device)
    y = torch.stack(labels).long().to(device)

    return x, y


def compute_distortion(x, x_adv):
    """
    Compute L-infinity and L2 perturbation magnitudes.

    Returns:
        linf_per_sample
        l2_per_sample
    """

    delta = x_adv - x

    # L-infinity for each sample
    linf = (
        delta
        .abs()
        .flatten(start_dim=1)
        .max(dim=1)
        .values
    )

    # L2 for each sample
    l2 = (
        delta
        .flatten(start_dim=1)
        .norm(p=2, dim=1)
    )

    return linf, l2


# ============================================================
# Single SNR evaluation
# ============================================================

def evaluate_snr(
    model,
    dataset,
    indices,
    snr_value,
    device,
):
    """
    Evaluate C&W at one SNR.

    Important:
        Only samples that are correctly classified on the
        clean signal are eligible for the C&W attack.

    CW_SUBSET_SIZE limits the number of clean-correct samples
    actually attacked.
    """

    # --------------------------------------------------------
    # Load samples
    # --------------------------------------------------------

    x, y = load_batch(
        dataset,
        indices,
        device
    )

    total_samples = x.shape[0]

    # --------------------------------------------------------
    # Clean prediction
    # --------------------------------------------------------

    with torch.no_grad():

        clean_logits = model(x)

        clean_pred = torch.argmax(
            clean_logits,
            dim=1
        )

    clean_correct_mask = (
        clean_pred == y
    )

    clean_correct_count = (
        clean_correct_mask
        .sum()
        .item()
    )

    clean_accuracy = (
        clean_correct_count /
        total_samples
    )

    # --------------------------------------------------------
    # Select ONLY clean-correct samples
    # --------------------------------------------------------

    correct_indices = torch.nonzero(
        clean_correct_mask,
        as_tuple=False
    ).flatten()

    if correct_indices.numel() == 0:

        print(
            f"\nSNR {snr_value:>3} dB:"
            f" No clean-correct samples."
        )

        return {
            "snr": snr_value,
            "samples": total_samples,
            "clean_correct": 0,
            "attacked": 0,
            "clean_accuracy": clean_accuracy,
            "cw_accuracy": 0.0,
            "attack_success_rate": 0.0,
            "mean_linf": 0.0,
            "max_linf": 0.0,
            "mean_l2": 0.0,
            "max_l2": 0.0,
        }

    # --------------------------------------------------------
    # Apply subset limit HERE
    # --------------------------------------------------------

    attack_count = min(
        CW_SUBSET_SIZE,
        correct_indices.numel()
    )

    correct_indices = correct_indices[:attack_count]

    x_attack = x[correct_indices]
    y_attack = y[correct_indices]

    # --------------------------------------------------------
    # Run C&W
    # --------------------------------------------------------

    x_adv = cw_attack(
        model=model,
        x=x_attack,
        y=y_attack,
    )

    # --------------------------------------------------------
    # Verify tensor shape
    # --------------------------------------------------------

    if x_adv.shape != x_attack.shape:

        raise RuntimeError(
            "C&W returned an unexpected shape:\n"
            f"Original      : {tuple(x_attack.shape)}\n"
            f"Adversarial   : {tuple(x_adv.shape)}"
        )

    # --------------------------------------------------------
    # Adversarial prediction
    # --------------------------------------------------------

    with torch.no_grad():

        adv_logits = model(x_adv)

        adv_pred = torch.argmax(
            adv_logits,
            dim=1
        )

    adv_correct = (
        adv_pred == y_attack
    )

    cw_correct_count = (
        adv_correct
        .sum()
        .item()
    )

    # --------------------------------------------------------
    # C&W attack success rate
    #
    # Since every attacked sample was clean-correct:
    #
    # ASR =
    # initially-correct samples changed to incorrect
    # --------------------------------------------------------

    attack_success_count = (
        (~adv_correct)
        .sum()
        .item()
    )

    attack_success_rate = (
        attack_success_count /
        attack_count
        if attack_count > 0
        else 0.0
    )

    cw_accuracy = (
        cw_correct_count /
        attack_count
        if attack_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Distortion
    # --------------------------------------------------------

    linf, l2 = compute_distortion(
        x_attack,
        x_adv
    )

    mean_linf = linf.mean().item()
    max_linf = linf.max().item()

    mean_l2 = l2.mean().item()
    max_l2 = l2.max().item()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {
        "snr": snr_value,
        "samples": total_samples,
        "clean_correct": clean_correct_count,
        "attacked": attack_count,
        "clean_accuracy": clean_accuracy,
        "cw_accuracy": cw_accuracy,
        "attack_success_rate": attack_success_rate,
        "mean_linf": mean_linf,
        "max_linf": max_linf,
        "mean_l2": mean_l2,
        "max_l2": max_l2,
    }

    return result


# ============================================================
# Pretty printer
# ============================================================

def print_result(result, title=None):

    if title:
        print()
        print("=" * 100)
        print(title)
        print("=" * 100)

    print(
        f"SNR                         : "
        f"{result['snr']} dB"
    )

    print(
        f"Samples                     : "
        f"{result['samples']}"
    )

    print(
        f"Clean-correct samples      : "
        f"{result['clean_correct']}"
    )

    print(
        f"C&W attacked samples       : "
        f"{result['attacked']}"
    )

    print(
        f"Clean accuracy             : "
        f"{result['clean_accuracy'] * 100:.2f}%"
    )

    print(
        f"C&W accuracy               : "
        f"{result['cw_accuracy'] * 100:.2f}%"
    )

    print(
        f"C&W attack success rate    : "
        f"{result['attack_success_rate'] * 100:.2f}%"
    )

    print(
        f"Mean L_inf distortion      : "
        f"{result['mean_linf']:.6f}"
    )

    print(
        f"Max L_inf distortion       : "
        f"{result['max_linf']:.6f}"
    )

    print(
        f"Mean L2 distortion         : "
        f"{result['mean_l2']:.6f}"
    )

    print(
        f"Max L2 distortion          : "
        f"{result['max_l2']:.6f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)
    print("AMCShield - C&W Robustness by SNR")
    print("=" * 100)

    print("\nConfiguration")
    print("-" * 100)

    print(
        f"Batch size                  : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Samples / SNR               : "
        f"{SAMPLES_PER_SNR}"
    )

    print(
        f"C&W subset limit            : "
        f"{CW_SUBSET_SIZE}"
    )

    print(
        f"C&W steps                   : "
        f"{CW_STEPS}"
    )

    print(
        f"C&W c                       : "
        f"{CW_C}"
    )

    print(
        f"C&W kappa                   : "
        f"{CW_KAPPA}"
    )

    print(
        f"C&W learning rate           : "
        f"{CW_LR}"
    )

    print(
        f"Pilot SNR                   : "
        f"{PILOT_SNR} dB"
    )

    print(
        "\nImportant: C&W is NOT constrained "
        "to epsilon = 0.02."
    )

    print(
        "Actual L_inf and L2 distortion "
        "will be reported."
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice                      : "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU                         : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(device)

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    _, _, test_loader = create_dataloaders(
        batch_size=1024,
        num_workers=0
    )

    dataset = test_loader.dataset

    print(
        f"✓ Test dataset loaded: "
        f"{len(dataset):,} samples"
    )

    # --------------------------------------------------------
    # Collect SNR samples
    # --------------------------------------------------------

    snr_indices = collect_snr_indices(
        dataset
    )

    if PILOT_SNR not in snr_indices:

        raise RuntimeError(
            f"Pilot SNR {PILOT_SNR} dB "
            "was not found in the dataset."
        )

    # ========================================================
    # STEP 1: PILOT TEST
    # ========================================================

    print()
    print("=" * 100)
    print(
        f"STEP 1 - C&W PILOT TEST "
        f"({PILOT_SNR} dB)"
    )
    print("=" * 100)

    print(
        f"\nRunning C&W on up to "
        f"{CW_SUBSET_SIZE} clean-correct samples..."
    )

    pilot_result = evaluate_snr(
        model=model,
        dataset=dataset,
        indices=snr_indices[PILOT_SNR],
        snr_value=PILOT_SNR,
        device=device,
    )

    print_result(
        pilot_result,
        title="C&W PILOT RESULT"
    )

    # --------------------------------------------------------
    # Pilot sanity checks
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("PILOT SANITY CHECK")
    print("=" * 100)

    pilot_ok = True

    if pilot_result["attacked"] == 0:

        print(
            "✗ No clean-correct samples were "
            "available for the pilot."
        )

        pilot_ok = False

    else:

        print(
            "✓ Clean-correct samples selected."
        )

    if pilot_result["mean_linf"] == 0.0:

        print(
            "⚠ Mean L_inf distortion is zero."
        )

        print(
            "  Check the C&W implementation/output."
        )

    else:

        print(
            "✓ Non-zero C&W perturbation detected."
        )

    if pilot_result["mean_l2"] == 0.0:

        print(
            "⚠ Mean L2 distortion is zero."
        )

        print(
            "  Check the C&W implementation/output."
        )

    else:

        print(
            "✓ Non-zero L2 perturbation detected."
        )

    if pilot_result["cw_accuracy"] < 1.0:

        print(
            "✓ C&W changed at least one "
            "initially-correct prediction."
        )

    else:

        print(
            "⚠ C&W did not fool any pilot sample."
        )

    # --------------------------------------------------------
    # Shape sanity check
    # --------------------------------------------------------

    print(
        "\n✓ Expected RF tensor shape: "
        "(B, 2, 1024)"
    )

    # --------------------------------------------------------
    # Stop if pilot failed
    # --------------------------------------------------------

    if not pilot_ok:

        print(
            "\n✗ Pilot failed. "
            "Full sweep will NOT be started."
        )

        return

    # ========================================================
    # STEP 2: FULL SNR SWEEP
    # ========================================================

    print()
    print("=" * 100)
    print("STEP 2 - FULL C&W SNR SWEEP")
    print("=" * 100)

    print(
        f"\nRunning C&W from "
        f"-20 dB to +30 dB..."
    )

    print(
        f"Maximum attacked samples per SNR: "
        f"{CW_SUBSET_SIZE}"
    )

    print()

    results = {}

    for snr_value in sorted(snr_indices):

        print(
            f"\nRunning SNR {snr_value:>3} dB..."
        )

        result = evaluate_snr(
            model=model,
            dataset=dataset,
            indices=snr_indices[snr_value],
            snr_value=snr_value,
            device=device,
        )

        results[snr_value] = result

        print(
            f"  Clean: "
            f"{result['clean_accuracy'] * 100:.2f}% | "
            f"C&W: "
            f"{result['cw_accuracy'] * 100:.2f}% | "
            f"ASR: "
            f"{result['attack_success_rate'] * 100:.2f}% | "
            f"Mean L_inf: "
            f"{result['mean_linf']:.6f} | "
            f"Mean L2: "
            f"{result['mean_l2']:.4f}"
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print("=" * 120)
    print("FINAL C&W SNR RESULTS")
    print("=" * 120)

    print(
        f"{'SNR':>6} | "
        f"{'Clean':>8} | "
        f"{'Correct':>8} | "
        f"{'Attack':>8} | "
        f"{'C&W':>8} | "
        f"{'ASR':>8} | "
        f"{'Mean L∞':>10} | "
        f"{'Max L∞':>10} | "
        f"{'Mean L2':>10} | "
        f"{'Max L2':>10}"
    )

    print("-" * 120)

    for snr_value in sorted(results):

        r = results[snr_value]

        print(
            f"{snr_value:>6} | "
            f"{r['clean_accuracy'] * 100:>7.2f}% | "
            f"{r['clean_correct']:>8} | "
            f"{r['attacked']:>8} | "
            f"{r['cw_accuracy'] * 100:>7.2f}% | "
            f"{r['attack_success_rate'] * 100:>7.2f}% | "
            f"{r['mean_linf']:>10.6f} | "
            f"{r['max_linf']:>10.6f} | "
            f"{r['mean_l2']:>10.4f} | "
            f"{r['max_l2']:>10.4f}"
        )

    print("=" * 120)

    print("\n✓ C&W evaluation completed.")

    print(
        "\nNote:"
        "\n  C&W results are NOT directly comparable "
        "to FGSM/PGD/MIM using ε = 0.02."
        "\n  C&W is reported with its actual "
        "optimization distortion."
    )


if __name__ == "__main__":
    main()