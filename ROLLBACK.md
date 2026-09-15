# Rollback Procedures — RF-AMC Project

Document how changes can be safely reversed. Before making a risky change:
identify the rollback point, explain the rollback strategy, and make sure the
change is reversible where practical. Never make a risky change without
knowing how to undo it.

## Standard Rollback (typical code change)

1. Identify the commit that introduced the problem:
   ```
   git log --oneline -- <affected file>
   ```
2. Revert the specific commit (preserves history, safer than reset):
   ```
   git revert <commit-hash>
   ```
3. If the change was already merged into `main` via Pull Request, open a new
   PR with the revert commit rather than force-pushing over `main` — branch
   protection prevents direct pushes to `main` anyway.
4. Restore affected files if a partial revert is needed:
   ```
   git checkout <good-commit-hash> -- <path/to/file>
   ```
5. Run the relevant section of `TEST_CHECKLIST.md` to confirm the rollback
   restored working behavior.

## Large/Risky Changes (e.g., changing config.yaml attack parameters,
## restructuring train.py's shared loop, modifying the data split logic)

Before making the change:
- **Safe rollback point:** note the current commit hash on `main` before
  starting (`git rev-parse HEAD`).
- **Rollback strategy:** since this project has no database or production
  config, rollback is git-only — revert the relevant commit(s) and re-run the
  affected pipeline stage from `TEST_CHECKLIST.md`.

### Config changes (`config.yaml`)
- If attack epsilon/step values change, `results/metrics.json` and all
  `results/plots/` become stale relative to the new config. Rollback means:
  revert the config change, then regenerate `metrics.json` from the reverted
  config to resync the Demo tab (live) and Dashboard tab (precomputed).

### Data split changes (`preprocess.py`)
- Changing the stratification logic invalidates every downstream trained
  model and result. Rollback means: revert `preprocess.py`, delete
  `data/processed/*` (gitignored, regenerable), re-run preprocessing, and
  **retrain both baseline and robust models from scratch** — checkpoints
  trained on the old split are not valid for the reverted split.

### Shared training loop changes (`train.py`)
- Since both baseline and robust models depend on `train.py`, a broken change
  here affects both. Rollback means: revert `train.py`, confirm both
  `checkpoints/baseline_model.pt` and `checkpoints/robust_model.pt` still
  load and evaluate correctly against the reverted loop (they should, since
  checkpoints are independent of the training code that produced them) —
  only retrain if the checkpoint files themselves were also regenerated
  during the broken period.

### Surrogate model changes (`model_surrogate.py`)
- If a change accidentally introduces weight-sharing with
  `model_baseline.py` (a constraint violation, see `CONSTRAINTS.md`), revert
  immediately and retrain the surrogate from a clean, query-only state —
  any black-box attack results generated during the violation period must be
  discarded and regenerated, since they may not represent a genuine black-box
  threat model.

## Post-Rollback Verification
After any rollback:
1. Run the relevant checks from `TEST_CHECKLIST.md`.
2. Confirm `git diff` between the current state and the last known-good
   commit shows no unexpected residual changes.
3. Update `HANDOVER.md`'s "Known Problems" section if the rollback was due to
   a bug that needs further investigation, or clear it if fully resolved.
4. If the rollback affects a merged phase (past a sync point in
   `phases.md`), notify the team — other branches may have already pulled the
   now-reverted change into their own branch via `git merge main`.

## What Is NOT Easily Reversible
- **Retraining time.** Reverting `preprocess.py` after models were already
  trained on the old split means retraining from scratch — budget hours, not
  minutes, for this rollback path. This is the most expensive rollback
  scenario in the project.
- **C&W evaluation runs.** Given their computational cost, a bad C&W
  evaluation run (e.g., wrong `cw_eval_subset_size`) is expensive to redo —
  double-check `config.yaml` before triggering a full C&W run, not after.
