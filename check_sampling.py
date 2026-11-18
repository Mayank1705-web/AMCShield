"""
check_sampling.py -- is the per-SNR evaluation sample representative?

Run from the project root (virtual environment active):

    python check_sampling.py

run_baseline_attacks.py (and run_cw_snr.py, run_robust_blackbox.py) take
the FIRST 256 test samples they meet at each SNR, walking an unshuffled
test loader. If the test indices are stored in dataset order, that can
mean every SNR bucket is filled from only one or two modulation classes.

This script reproduces that selection and reports how many of the 24
classes each SNR bucket actually contains. Nothing is written to disk.
"""

from collections import Counter

import numpy as np

from src.data_loader import create_dataloaders

NEED = 256  # samples per SNR, same as run_baseline_attacks.py

idx = np.load("data/processed/test_indices.npy")
print(f"test_indices.npy: {len(idx):,} entries")
print(f"sorted ascending: {bool((np.diff(idx) >= 0).all())}")
print(f"first 5: {idx[:5]}   last 5: {idx[-5:]}")

_, _, test_loader = create_dataloaders(batch_size=128, num_workers=0)

per_snr = {}  # snr -> labels of the first NEED samples seen, in loader order
batches = 0
for _x, y, snr in test_loader:
    batches += 1
    for label, s in zip(y.view(-1).tolist(), snr.view(-1).tolist()):
        bucket = per_snr.setdefault(int(round(s)), [])
        if len(bucket) < NEED:
            bucket.append(int(label))
    if len(per_snr) >= 26 and all(len(v) >= NEED for v in per_snr.values()):
        break

print(f"\nread {batches} batches to fill every SNR bucket\n")
print(f"{'SNR (dB)':>8} | {'classes present':>15} | three most common (class, count)")
print("-" * 70)
distinct = []
for s in sorted(per_snr):
    counts = Counter(per_snr[s])
    distinct.append(len(counts))
    print(f"{s:>8} | {len(counts):>9} of 24 | {counts.most_common(3)}")

print()
print(f"classes per SNR bucket: min {min(distinct)}, max {max(distinct)} (of 24)")
if max(distinct) <= 6:
    print("RESULT: BIASED - the per-SNR samples cover only a few modulation "
          "classes, so the reported accuracy/ASR is NOT representative.")
elif min(distinct) >= 20:
    print("RESULT: OK - every SNR bucket covers (nearly) all classes.")
else:
    print("RESULT: PARTIAL - some buckets miss many classes; send me this output.")
