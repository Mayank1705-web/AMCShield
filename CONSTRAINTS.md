# AI Development Constraints — RF-AMC Project

This file defines what an AI agent (or teammate) is and is not allowed to do
when working on this repository. Check this file before every implementation.
If a requested change conflicts with a constraint below: STOP, explain the
conflict, and ask for explicit permission before proceeding.

## Protected Areas
- Do not modify the core research design (PGD-only adversarial training,
  the fixed attack suite of FGSM/PGD/MIM/C&W/black-box, or the requirement
  that every result be SNR-stratified) without explicit team discussion —
  these define the thesis's actual research question. See `DECISIONS.md`.
- Do not modify `model_surrogate.py` to import weights, gradients, or
  architecture from `model_baseline.py` — this invalidates the black-box
  threat model that the project's novelty claim depends on.
- Do not modify `config.yaml`'s attack parameters (epsilon, PGD/MIM steps,
  `cw_eval_subset_size`) without regenerating `results/metrics.json`
  afterward — stale precomputed results silently desynchronize from the
  live Demo tab.

## Dependencies
- Do not introduce a new dependency without asking first.
- Approved stack only: Python, PyTorch, torchattacks, NumPy, SciPy, pandas,
  matplotlib, seaborn, Jupyter, Streamlit, Plotly.
- Do not suggest or add TensorFlow, JAX, a web framework (Flask/Django/etc.),
  a database, or Docker — explicitly out of scope per `PRD.md` Section 2.2.
- Do not add foundation-model dependencies (pretrained wireless backbones,
  RFPrompt-style architectures) — deferred to future work.

## Architecture
- Follow the file/module boundaries defined in `architecture.md` and
  `.cursorrules`. Each branch owner (`person-a-data`, `person-b-attacks`,
  `person-c-eval`) owns a specific file set — do not edit another owner's
  files directly on your own branch; wait for the next phase merge into
  `main`, then pull `main` into your branch.
- `train.py` must remain a single shared generic training loop (supporting
  both plain supervised training and PGD-adversarial training via a hook),
  not duplicated per model type.
- Do not bypass the SNR-stratification requirement anywhere in the
  pipeline — every split, attack evaluation, and heatmap cell must remain
  SNR-bucketed.

## Dataset
- Do not use RadioML data for any commercial purpose — CC BY-NC-SA 4.0,
  academic use only, attribution to DeepSig required.
- Do not discard per-sample SNR metadata at any stage of `data_loader.py` or
  `preprocess.py` — this is required for every downstream evaluation.

## Scope
- Do not expand into channel-aware/over-the-air attack modeling,
  foundation-model-based defenses, or dual-domain contrastive learning as
  core deliverables — all explicitly deferred to future work
  (`PRD.md` Section 2.2). These are common temptations that resurface during
  literature review; treat them as out of scope unless the team explicitly
  re-scopes the project.
- Do not run C&W attack evaluation on the full test set by default — respect
  `cw_eval_subset_size` in `config.yaml`; it is computationally expensive.
- Do not default the frontend Dashboard tab to interactive Plotly rendering —
  static charts are the committed deliverable (see `DECISIONS.md` frontend
  entry). Interactive is an opt-in Week 9 stretch only, never a blocking
  dependency for any other phase.
- Do not let the Dashboard tab re-run training or attacks — it must only read
  `results/metrics.json` and `results/plots/`. Live computation belongs
  exclusively in the Demo tab.
- Do not modify unrelated files or perform opportunistic refactoring during a
  focused task — one logical change per request.

## Testing
- Do not remove or weaken a test simply to make it pass.
- Do not claim a training run or evaluation "worked" without checking the
  actual output numbers/plots — see `TEST_CHECKLIST.md`.

## Code Style
- Type hints on all function signatures.
- All hyperparameters live in `config.yaml` — never hardcode batch size,
  learning rate, epochs, epsilon values, or PGD/MIM steps inline in scripts.
- Always set and log a random seed (`utils.set_seed`) at the start of any
  training or attack script — reproducibility matters for the thesis.

## Git Workflow
- Follow the branch structure in `phases.md`: three branches merging into
  `main` at defined phase boundaries only, via Pull Request (not direct push
  to `main` — branch protection is enabled).
- Use `--no-ff` merges (or PR merges, which behave equivalently) so `main`'s
  history shows each phase as a distinct, traceable merge point.
