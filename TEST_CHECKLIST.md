# Test Checklist — RF-AMC Project

This is the project's verification contract. Document actual commands and
expected results. "Done" means verified — if a check cannot be run, state
NOT VERIFIED and explain why. Never fabricate results.

## Environment Setup

Command:
```
pip install -r requirements.txt
```

Expected:
All dependencies install without error. Verify `torch`, `torchattacks`,
`numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `streamlit`, `plotly`
are importable.

## Data Pipeline

Command:
```
python -m src.data_loader
```

Expected:
Loads the RadioML pickle without error, prints dataset shape (samples, 2, 128)
and class/SNR counts. Fails loudly if `data/raw/RML2016.10a_dict.pkl` is
missing — should not fail silently.

Command:
```
python -m src.preprocess
```

Expected:
Produces `data/processed/train.npy`, `val.npy`, `test.npy` (or `.pt`
equivalents), each with matching X/y/SNR array lengths. Verify the split is
stratified — SNR distribution in test set should roughly match the full
dataset's SNR distribution, not be skewed toward any one bucket.

## Model Training

Command:
```
python -m src.train --model baseline --config configs/config.yaml
```

Expected:
Training loss decreases over epochs; final validation accuracy is in a
plausible range for RadioML 2016.10a baselines (consult literature — a
sanity-check range, not an exact target). Produces
`checkpoints/baseline_model.pt`.

Command:
```
python -m src.train --model robust --config configs/config.yaml
```

Expected:
Training completes without crashing; the adversarial-example injection
(`adv_train_ratio`) is visibly active in logs. Produces
`checkpoints/robust_model.pt`. Sanity check: robust model's accuracy against
PGD attack must exceed baseline's accuracy against PGD attack — if not,
adversarial training did not work and must be debugged before proceeding.

## Attacks

Command:
```
python -m src.attacks --attack fgsm --model checkpoints/baseline_model.pt
python -m src.attacks --attack pgd --model checkpoints/baseline_model.pt
python -m src.attacks --attack mim --model checkpoints/baseline_model.pt
python -m src.attacks --attack cw --model checkpoints/baseline_model.pt
```

Expected:
Each attack runs without error and produces a non-trivial drop in accuracy
vs. clean baseline accuracy. C&W should only run on the configured subset
size (`cw_eval_subset_size`), not the full test set — verify sample count in
logs matches the config value.

## Surrogate / Black-Box Attack

Command:
```
python -m src.model_surrogate --query-budget 5000
```

Expected:
Surrogate model trains on query-response pairs only (verify no direct import
of `model_baseline` weights in the diff/code). Produces
`checkpoints/surrogate_model.pt`.

Command:
```
python -m src.attacks --attack blackbox --surrogate checkpoints/surrogate_model.pt --target checkpoints/baseline_model.pt
```

Expected:
Adversarial examples crafted on the surrogate measurably reduce the target
model's accuracy when transferred — confirms the black-box path is wired
correctly, not accidentally using white-box gradients on the real target.

## Evaluation

Command:
```
python -m src.evaluate --generate-heatmap
```

Expected:
Produces `results/metrics.json` and `results/plots/generalization_gap_heatmap.png`
(or equivalent). Heatmap must have a cell for every (attack type, SNR bucket)
combination — no missing cells, no NaN values without explanation.

## Frontend

Command:
```
streamlit run frontend/app.py
```

Expected:
App launches without error. Demo tab: selecting a signal + attack produces a
visibly different (attacked) prediction vs. clean prediction for at least
some samples. Dashboard tab: renders static charts from `results/metrics.json`
without attempting to re-run training or attacks (verify no training/attack
function calls in the Dashboard tab's code path).

## Lint / Type Checking

Command:
```
<add actual project lint/typecheck command once tooling is chosen, e.g. ruff/mypy>
```

Expected:
NOT VERIFIED — tooling not yet selected as of this document's creation.
Update this section once the team picks a linter/type-checker.

## Application Smoke Test

Describe:
- Start: `streamlit run frontend/app.py`
- Action: load a clean signal in the Demo tab, apply FGSM at a low SNR bucket
  (e.g., -10dB), observe the predicted label change
- Expected: prediction changes from correct to incorrect (or shows a visibly
  different confidence distribution), confirming the attack pipeline and
  frontend are correctly wired end-to-end

## Full Pipeline Sanity Check (run before any phase merge)

Expected sequence, each step must PASS before merging to `main`:
1. Data pipeline — PASS/FAIL/NOT RUN
2. Baseline training — PASS/FAIL/NOT RUN
3. Robust training (PGD sanity check: robust > baseline against PGD) — PASS/FAIL/NOT RUN
4. Attack suite (all 4 white-box attacks) — PASS/FAIL/NOT RUN
5. Surrogate + black-box attack — PASS/FAIL/NOT RUN
6. Evaluation (heatmap generated, no missing cells) — PASS/FAIL/NOT RUN
7. Frontend smoke test — PASS/FAIL/NOT RUN
