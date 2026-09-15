# Execution Flow

This document traces how data and control move through the RF-AMC pipeline.
See `architecture.md` for the component map this flow is built on.

## 1. Training Pipeline Flow (Baseline & Robust Models)

```
RadioML 2016.10a (.pkl)
  |
  v
data_loader.py
  - parses pickle dict keyed by (modulation, SNR)
  - outputs: X (I/Q arrays), y (labels), snr (per-sample metadata)
  |
  v
preprocess.py
  - per-sample amplitude normalization
  - stratified split by (modulation, SNR) -> train/val/test
  - SNR metadata carried through every step (never dropped)
  |
  v
train.py  (shared loop, used by both baseline and robust training runs)
  |
  +-- baseline run: standard supervised training
  |     model_baseline.py -> checkpoints/baseline_model.pt
  |
  +-- robust run: adversarial_fn hook active
        - each batch: fraction (adv_train_ratio) replaced with
          PGD-perturbed examples generated via attacks.py
        model_robust.py -> checkpoints/robust_model.pt
```

## 2. Surrogate / Black-Box Attack Flow

```
checkpoints/baseline_model.pt
  |
  v  (query-only access — no weights/gradients shared)
Query baseline with subset of training data
  -> collect (input, predicted-label) pairs
  |
  v
model_surrogate.py  (architecturally different from baseline)
  -> trained purely on query-response pairs
  -> checkpoints/surrogate_model.pt
  |
  v
attacks.py: craft PGD adversarial examples ON THE SURROGATE
  |
  v
Transfer these examples to attack the REAL baseline/robust models
  -> this is the black-box attack success measurement
```

**Critical distinction:** gradients for white-box attacks (FGSM, PGD, MIM, C&W)
flow through the real target model. Gradients for the black-box attack flow
only through the surrogate; only the resulting adversarial examples ever touch
the real baseline/robust models. Mixing these up silently invalidates the
black-box claim — see `architecture.md` Section 4, integration point #3.

## 3. Evaluation Flow

```
checkpoints/{baseline,robust,surrogate}_model.pt
  |
  v
attacks.py
  - white-box: FGSM, PGD, MIM, C&W directly on baseline/robust models
  - black-box: surrogate-crafted PGD examples transferred to baseline/robust
  |
  v
evaluate.py
  - accuracy_by_snr(model, test_data) -> per-SNR-bucket accuracy
  - attack_success_rate(model, test_data, attack_fn) -> per attack, per SNR
  - generalization_gap(robust_model_results) ->
        compares robust model's PGD performance (trained-against)
        vs. FGSM/MIM/C&W/black-box performance (unseen)
        across every SNR bucket
  - confusion_matrix_plot(model, test_data) at a fixed representative SNR
  |
  v
results/metrics.json  +  results/plots/*.png
```

## 4. Frontend Flow

```
Demo tab (frontend/app.py)
  - user selects a signal + attack type + epsilon
  - LIVE: loads checkpoints/*.pt directly, calls into attacks.py
  - renders clean vs. attacked prediction + I/Q waveform
  - NOTE: uses the SAME config.yaml epsilon/step values as the offline
    evaluation pipeline. If config.yaml changes after metrics.json was
    generated, Demo tab (live) and Dashboard tab (precomputed) will disagree.
    Regenerate metrics.json whenever attack parameters change.

Dashboard tab (frontend/app.py)
  - READ-ONLY: loads results/metrics.json and results/plots/
  - never re-runs training or attacks
  - default rendering: static charts
  - Week 9 optional upgrade: interactive Plotly (zoom/hover/filter)
```

## 5. Error Propagation Notes
- If `preprocess.py`'s stratified split silently drops SNR metadata, every
  downstream heatmap cell becomes invalid without an obvious error — this is
  the highest-priority failure mode to guard against with an explicit check
  (e.g., assert SNR array length matches X/y length at every stage boundary).
- If a Colab GPU session disconnects mid-training, `train.py` must have
  checkpointed recently enough that the training run can resume rather than
  restart from scratch — checkpoint every few epochs, not only at the end.
- If C&W is run on the full test set instead of the configured subset, expect
  the evaluation step to become a practical bottleneck rather than a hard
  error — this fails silently as "still running," not as an exception.

## 6. Update Policy
Update this document whenever:
- A new file is added that changes the data/control flow above
- The distinction between white-box and black-box gradient flow changes
- The frontend's live-vs-precomputed data sourcing changes
