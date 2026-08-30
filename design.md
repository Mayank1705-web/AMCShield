# Design Decisions — Attack-Agnostic Generalization Robustness

## 1. Dataset
- **Source:** RadioML 2016.10a (DeepSig), CC BY-NC-SA 4.0 — academic use only.
- **Format:** pickle dict keyed by `(modulation, SNR)` -> 1000 samples of shape `2x128`.
- **Classes:** 11 modulation types.
- **SNR range:** -20dB to +18dB in 2dB steps.
- **Why 2016.10a over 2018.01a:** smaller, faster to iterate on within the timeline;
  2018.01a (24 classes) is a reasonable future extension, not the MVP target.

## 2. Preprocessing
- **Normalization:** per-sample amplitude normalization (unit energy).
- **Split strategy:** stratified by `(modulation, SNR)`, 70/15/15 train/val/test. This
  matters more than usual here because every downstream result (generalization-gap
  heatmap) is SNR-bucketed — an unstratified split will produce a heatmap with noisy or
  misleading cells at underrepresented SNR levels.

## 3. Baseline Model
- **Architecture:** CNN (Conv1d layers over the 2-channel I/Q input) -> pooling -> dense ->
  softmax.
- **Why not a Transformer or foundation model:** the literature-reviewed "foundation model
  + prompt tuning" direction requires pretrained wireless foundation models that are not
  readily reproducible solo/small-team — explicitly deferred, not pursued here.
- **Loss:** cross-entropy. **Optimizer:** Adam, lr from `config.yaml`.

## 4. Robust Model — Single-Attack Adversarial Training
- **Defense:** adversarial training against **PGD only**. This is a deliberate choice: the
  project's contribution is testing whether this single-attack hardening generalizes, so
  the robust model must be trained against exactly one attack type to make that question
  answerable.
- **Training procedure:** during training, a fraction of each batch (`adv_train_ratio` in
  `config.yaml`) is replaced with PGD-perturbed examples generated on-the-fly.
- **Why not multi-attack adversarial training:** training against multiple attack types
  simultaneously would answer a different (also interesting, but different) question —
  "does multi-attack training generalize better than single-attack training" — which is a
  reasonable stretch goal for Phase 5 if time allows, but the core deliverable is measuring
  the gap under single-attack training first.

## 5. Surrogate Model — Black-Box Attack Path
- **Architecture:** deliberately different from the baseline (e.g., if baseline is a plain
  CNN, surrogate is a CNN-LSTM, or a CNN with different depth/kernel sizes) — using an
  identical architecture as the surrogate would artificially inflate transfer success and
  misrepresent a realistic black-box threat model.
- **Training procedure (based on Usama et al. 2019's query-based methodology):**
  1. Query the baseline model with a subset of training data to collect
     input -> predicted-label pairs (the attacker never sees ground-truth labels or
     baseline weights/gradients).
  2. Train the surrogate purely on these query-response pairs.
  3. Craft adversarial examples on the surrogate using PGD.
  4. Transfer these examples to attack the real baseline and robust models; measure attack
     success rate there — this is the "black-box" number.

## 6. Attack Suite
| Attack | Type | Role in this project |
|---|---|---|
| FGSM | White-box, single-step | Fast baseline vulnerability check (MVP) |
| PGD | White-box, iterative | What the robust model is trained against |
| MIM | White-box, iterative + momentum | Unseen-attack generalization test |
| C&W | White-box, optimization-based | Unseen-attack generalization test (evaluate on a subset only — expensive) |
| Black-box surrogate-transfer PGD | Black-box | Unseen-attack + unseen-threat-model generalization test |

All attacks implemented via `torchattacks` — no custom attack math needs to be written,
keeping this inside the existing stack.

## 7. Evaluation Metrics
- **Accuracy-vs-SNR curve** — baseline and robust model, clean condition (sanity check).
- **Attack success rate per attack type, per SNR bucket** — computed for every attack in
  the suite above, against both baseline and robust model.
- **Generalization-gap heatmap (core contribution):** for the robust model, compare attack
  success rate under PGD (trained-against) vs. FGSM/MIM/C&W/black-box (unseen), across the
  full SNR range. Rows = attack type, columns = SNR bucket, cell = attack success rate (or
  the delta vs. PGD's own success rate). This is the project's headline result.
- **Confusion matrix** — per model, at a fixed representative SNR (e.g., 0dB).

## 8. Config Schema (`config.yaml`)
```yaml
batch_size: 128
epochs: 30
learning_rate: 0.001
epsilon:
  fgsm: 0.02
  pgd: 0.02
  mim: 0.02
cw_eval_subset_size: 300   # samples per SNR bucket for C&W (expensive attack)
pgd_steps: 10
mim_steps: 10
snr_range: [-20, 18]
adv_train_ratio: 0.5       # fraction of each batch replaced with PGD examples during robust training
surrogate_query_budget: 5000   # number of baseline queries used to train the surrogate
```

## 9. Explicit Non-Goals
- No real captured RF hardware/SDR integration (synthetic data only, stated as a
  limitation, not hidden).
- No channel-aware / over-the-air attack modeling (Rayleigh/Rician fading) — documented as
  future work in the report, not implemented.
- No foundation-model-based defenses, no frequency-masked-autoencoder purification, no
  dual-domain contrastive learning — each is independently thesis-scale.
- No deployment/serving layer — this is a research pipeline, not a product.

## 10. Optional Stretch (only if Phases 1-6 finish ahead of schedule)
- Minimal channel simulation: apply a simple additive Rayleigh fading transform to
  adversarial examples before evaluation, to see whether channel noise degrades attack
  success rate — a lightweight, non-committal touch on Gap #3 from the literature review,
  without building full channel-aware attack optimization.

## 11. Frontend (Demo + Dashboard) — Presentation Layer, Not Research Infrastructure
- **Purpose:** thesis-defense presentation aid only. It validates nothing about the
  research question and is explicitly not part of the MVP (see phases.md Phase 7).
- **Stack:** Streamlit (Python-native, no new language, reads PyTorch/NumPy/matplotlib
  outputs directly) + Plotly, only if the interactive upgrade is reached.
- **Default behavior — static charts first:** the Dashboard tab renders **pre-generated,
  static charts** (accuracy-vs-SNR, confusion matrices, generalization-gap heatmap) loaded
  from `results/metrics.json` and `results/plots/`. This is the committed deliverable.
- **Interactive upgrade — explicitly optional:** live Plotly interactivity (zoom, hover,
  filtering) on the generalization-gap heatmap is attempted only in Week 9, and only if
  Weeks 7-8 finished on schedule with no carryover debugging work. It is never a blocking
  dependency for any other phase.
- **Why this split matters:** the interactive version was originally scoped as the default,
  which created a hidden dependency — Person C's frontend work would stall if Person B's
  Week 7 black-box attack work ran late, consuming the project's only real buffer week.
  Defaulting to static charts removes that coupling: static charts can be generated the
  moment `metrics.json` exists, with no additional frontend engineering required to call
  the deliverable "done."
