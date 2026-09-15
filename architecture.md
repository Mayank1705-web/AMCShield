# Architecture — Attack-Agnostic Generalization Robustness Pipeline

## 1. High-Level Flow

```
[RadioML .pkl]
      |
      v
data_loader.py  --> raw X (I/Q signals), y (labels), snr (per-sample)
      |
      v
preprocess.py   --> normalized, stratified train/val/test splits (saved to data/processed/)
      |
      +-----------------------------+-------------------------------+
      v                             v                               v
model_baseline.py            model_robust.py                model_surrogate.py
(standard CNN)          (PGD-adversarially trained)     (different architecture,
      |                             |                     trained on query-response
      v                             v                     pairs from baseline)
   train.py  <-- shared training loop, used by baseline + robust
      |                             |
      v                             v
checkpoints/baseline_model.pt   checkpoints/robust_model.pt   checkpoints/surrogate_model.pt
      |                             |                               |
      +-----------------------------+-------------------------------+
                                     v
                          attacks.py (FGSM, PGD, MIM, C&W,
                             black-box surrogate-transfer)
                                     v
                              evaluate.py
              (accuracy-vs-SNR, confusion matrix,
               generalization-gap heatmap: attack type x SNR)
                                     v
                     results/plots/, results/metrics.json
                                     v
                              report/thesis.md
```

## 2. Components

| Component | Responsibility | Depends on |
|---|---|---|
| `data_loader.py` | Parse RadioML pickle -> raw arrays | RadioML dataset file |
| `preprocess.py` | Normalize, split, optional STFT spectrogram conversion | `data_loader.py` |
| `model_baseline.py` | Standard CNN/CNN-LSTM architecture | -- |
| `model_robust.py` | Same base architecture, trained with PGD adversarial training | `model_baseline.py` |
| `model_surrogate.py` | Architecturally different model (e.g., shallower CNN or CNN-LSTM if baseline is CNN), trained on query-response pairs from the baseline to simulate black-box attacker knowledge | `model_baseline.py` (queried, not copied) |
| `train.py` | Generic training loop (epochs, optimizer, checkpointing, optional adversarial-example injection) | `preprocess.py`, model files |
| `attacks.py` | FGSM, PGD, MIM, C&W implementations (wraps `torchattacks`) + black-box surrogate-transfer logic | Trained model checkpoints |
| `evaluate.py` | Accuracy-vs-SNR, confusion matrix, generalization-gap heatmap (attack type x SNR) | Trained models + `attacks.py` |
| `utils.py` | Seed control, plotting helpers, checkpoint I/O | Used by all of the above |
| `frontend/app.py` | Streamlit demo (clean vs. attacked prediction) + results dashboard | `results/metrics.json`, model checkpoints |
| `config.yaml` | Central hyperparameters (batch size, epochs, lr, epsilon per attack, PGD/MIM steps, C&W eval subset size) | Read by `train.py`, `attacks.py` |

## 3. Data Flow Details

- **Input shape:** each RadioML sample is `2 x 128` (I and Q channels, 128 time steps).
- **Labels:** modulation class (11 classes in 2016.10a).
- **SNR metadata:** must be carried alongside `X`/`y` through preprocessing and through
  every attack/evaluation step — required for the generalization-gap heatmap, which is
  the project's core deliverable. Do not discard it at any stage.
- **Splits:** stratify by `(modulation_class, SNR)` jointly, not just by class.

## 4. Tricky Integration Points

1. **Shared `train.py` for two different training regimes.** Baseline training is standard
   supervised training; robust training injects PGD-generated adversarial examples mid-loop.
   `train.py` must accept a hook (`adversarial_fn=None`) rather than being duplicated.
2. **Surrogate model training requires query-only access to the baseline**, not direct
   weight access — this simulates a realistic black-box attacker. Implement this as:
   run baseline inference on a subset of training data to generate (input, predicted-label)
   pairs, then train the surrogate purely on those pairs. Do not let `model_surrogate.py`
   import or share weights with `model_baseline.py`.
3. **Attack generation needs gradients through the model.** For white-box attacks
   (FGSM/PGD/MIM/C&W), gradients flow through the real target model. For the black-box
   attack, gradients flow through the **surrogate**, and only the resulting adversarial
   examples are transferred to attack the real baseline/robust models — verify this
   distinction with a unit test before scaling up, it is the easiest place to introduce a
   silent bug that invalidates the "black-box" claim.
4. **C&W attack cost.** C&W is an optimization loop per sample and is far slower than
   FGSM/PGD/MIM. Evaluate it on a representative subset of the test set (e.g., a few
   hundred samples per SNR bucket), not the full test set — document this subsampling
   explicitly in the report so results remain defensible.
5. **Generalization-gap heatmap construction.** For each (attack type, SNR bucket) cell,
   compute attack success rate separately for the baseline and the robust model. The
   "generalization gap" itself is the difference between the robust model's performance
   against PGD (what it was trained on) and its performance against each unseen attack —
   this derived metric is the project's headline result, keep its computation in one
   clearly isolated function in `evaluate.py` for traceability.

## 5. Storage / Artifacts

- `data/raw/` -- original dataset (gitignored, large)
- `data/processed/` -- split arrays (gitignored, regenerable from raw)
- `checkpoints/` -- model weights, including `surrogate_model.pt` (gitignored, large)
- `results/` -- plots + `metrics.json`, including the generalization-gap heatmap
  (tracked in git -- this is your core evidence)

## 6. Deferred / Future-Work Components (not built, but documented for the thesis)
- `channel_sim.py` (not built) — would simulate Rayleigh/Rician fading between
  perturbation generation and classification, addressing the channel-aware attack gap.
  Documented in the report as a natural extension, not implemented in this timeline.

## 7. Frontend Architecture (`frontend/app.py`)

**Role:** thesis-defense presentation layer only — it validates nothing about the research
question and is not part of the MVP (see phases.md Phase 7). It is a thin read-only client
over artifacts the core pipeline already produces; it does not run training or attacks
itself.

```
results/metrics.json ---\
results/plots/       ----+---> frontend/app.py (Streamlit) ---> two tabs
checkpoints/*.pt      ---/                                      |
                                                                  +-- Demo tab: pick a
                                                                  |   signal + attack type,
                                                                  |   show clean vs.
                                                                  |   attacked prediction
                                                                  +-- Dashboard tab:
                                                                      render SNR curves,
                                                                      confusion matrices,
                                                                      generalization-gap
                                                                      heatmap
```

- **Demo tab dependencies:** loads `checkpoints/baseline_model.pt` and
  `checkpoints/robust_model.pt` directly, runs a forward pass plus an on-the-fly attack
  call into `attacks.py` — this is the one place the frontend touches model-inference code
  directly rather than just reading precomputed results.
- **Dashboard tab dependencies:** reads only `results/metrics.json` and pre-rendered files
  in `results/plots/` — it does not re-run any computation. This keeps it decoupled from
  the training/attack pipeline's runtime.
- **Default rendering mode: static charts.** The Dashboard tab renders pre-generated
  matplotlib/Plotly figures loaded from disk, not live interactive charts, by default (see
  design.md Section 11 for the full rationale). Interactive Plotly (zoom/hover/filter) is
  an optional Week 9 upgrade attempted only if earlier phases finished on schedule.
- **Tricky integration point:** the Demo tab's live attack call means it needs the same
  `config.yaml` epsilon/step values as the offline evaluation pipeline — if these drift out
  of sync (e.g., someone changes `epsilon.fgsm` in `config.yaml` after `metrics.json` was
  generated), the Demo tab's live results and the Dashboard tab's precomputed results will
  disagree. Regenerate `metrics.json` any time `config.yaml` attack parameters change.
