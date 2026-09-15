# Current Project State

## Completed
- Literature-grounded problem definition and locked-in research angle: whether
  PGD-only adversarial training generalizes to unseen attacks (FGSM, MIM, C&W,
  black-box surrogate-transfer), stratified across the full SNR range.
- Planning documentation complete: `PRD.md`, `architecture.md`, `design.md`,
  `phases.md`, `.cursorrules` — all cross-checked for consistency (frontend scope
  was initially missing from `architecture.md`/`.cursorrules`, now fixed).
- Tech stack finalized: Python, PyTorch, torchattacks, NumPy/SciPy,
  pandas/matplotlib/seaborn, Streamlit + Plotly (frontend, static-charts default).
- Dataset selected and download links confirmed: RadioML 2016.10a (DeepSig),
  CC BY-NC-SA 4.0.
- Team structure defined: 3 people, branch-per-person (`person-a-data`,
  `person-b-attacks`, `person-c-eval`), 9-week timeline with mandatory
  phase-based merge points.
- Git workflow and GitHub repo setup steps documented (CMD-specific commands
  provided for the team leader).

## In Progress
- Repository has not yet been created on GitHub as of this document's writing.
- No code has been written yet — project is still in the planning/setup phase.

## Known Problems
- None yet — no code exists to have bugs. First real risk area once code starts:
  the shared `train.py` interface must be agreed on by Person A and Person B
  before both start building against it (see `architecture.md` Section 4,
  integration point #1).

## Next Steps
1. Team leader creates the GitHub repo and pushes initial structure (folders,
   `.gitignore`, the five planning docs) to `main`.
2. Create the three branches and add teammates as collaborators.
3. Person A begins Phase 1: `data_loader.py` + `preprocess.py` (Week 1).
4. Person C begins `train.py` skeleton + `frontend/app.py` scaffold (Week 1).
5. Confirm `train.py`'s interface (inputs/outputs, the `adversarial_fn` hook) as
   a team before Week 2, per the Week 1 sync point in `phases.md`.

## Important Context
- The project's core deliverable is the **generalization-gap heatmap**
  (attack type x SNR) — this must never be cut from scope, per the cut list in
  `phases.md`.
- The frontend defaults to **static charts**, not live interactive dashboards —
  this was a deliberate fix after a mid-project sanity check found the
  interactive-by-default plan created a hidden dependency risk (see
  `design.md` Section 11 and `DECISIONS.md` entry on frontend scope).
- RadioML 2016.10a is CC BY-NC-SA 4.0 — academic use only, not for commercial use.
- Every dataset split, attack evaluation, and heatmap cell must be
  SNR-stratified — this is load-bearing for the entire research contribution,
  not an optional nicety.

## Files Recently Changed
- `PRD.md`, `architecture.md`, `design.md`, `phases.md`, `.cursorrules` —
  rewritten to reflect the locked-in generalization-gap angle and the frontend's
  static-charts-default fallback rule. No code files exist yet.

## Things to Avoid
- Do not default the frontend Dashboard tab to interactive Plotly — this
  previously created a hidden dependency between Person B's Week 7 workload and
  Person C's buffer week (Week 9). Static charts are the committed default.
- Do not let `model_surrogate.py` import or share weights with
  `model_baseline.py` — this would invalidate the black-box threat model.
- Do not expand scope into channel-aware/over-the-air attacks, foundation-model
  defenses, or dual-domain contrastive learning — all explicitly deferred to
  future work per `PRD.md` Section 2.2. These come up repeatedly in literature
  review and are a recurring scope-creep temptation.
- Do not run C&W on the full test set by default — it is computationally
  expensive; respect `cw_eval_subset_size` in `config.yaml`.
