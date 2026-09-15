# Phases — MVP to Full Version

## Guiding Principle
Validate the core premise before investing in the full attack suite: does PGD-adversarial
training actually improve robustness against PGD itself? If that basic sanity check fails,
the generalization-gap question downstream has nothing valid to stand on.

---

## Phase 0 — MVP (proves the concept is worth pursuing)
**Goal:** does adversarial training against PGD actually work, and does an unseen attack
(FGSM) reveal any generalization gap at all?

- Download + parse RadioML 2016.10a
- Train baseline CNN (no tuning needed yet)
- Train robust model with PGD adversarial training
- Run FGSM (unseen attack) against both models
- Produce: accuracy-vs-SNR chart (clean) + FGSM attack-success comparison, baseline vs robust

**Stop-and-check point:** if the robust model isn't more resistant to PGD than baseline,
something is wrong in the adversarial training loop — fix before proceeding. If FGSM
success rate against the robust model is *identical* to PGD success rate, that itself is
an interesting early signal worth flagging, not a bug.

---

## Phase 1 — Data Pipeline (Week 1)
- `data_loader.py`, `preprocess.py`
- Owner: Mayank Ingole [EN24CS3T10017]
- Output: clean, stratified, saved train/val/test splits

## Phase 2 — Baseline Model (Weeks 2-3)
- `model_baseline.py`, `train.py` (shared loop, built by Person C)
- Owner: Mayank Ingole [EN24CS3T10017] (model), Person C (training infra)
- Output: trained baseline checkpoint + accuracy-vs-SNR curve
- **Merge to main at end of Week 3**

## Phase 3 — Attack Suite: PGD + FGSM (Week 4)
- `attacks.py` — implement PGD (for training) and FGSM (first unseen-attack test)
- Owner: Mayank Ingole [EN24CS3T10017]
- Output: FGSM attack success rate against baseline, documented vulnerability

## Phase 4 — Robust Model: PGD Adversarial Training (Weeks 5-6)
- `model_robust.py`, adversarial training integrated into `train.py`
- Owner: Mayank Ingole [EN24CS3T10017]
- Output: trained robust checkpoint, validated against PGD (sanity check from Phase 0)
- **Merge to main at end of Week 6**

## Phase 5 — Generalization Suite: MIM, C&W, Black-Box Surrogate (Week 7, extends into Week 8 if needed)
- `attacks.py` extended with MIM and C&W
- `model_surrogate.py` built and trained on baseline query-response pairs
- Black-box surrogate-transfer attack implemented
- Owner: Mayank Ingole [EN24CS3T10017] (attacks/surrogate), Person C (evaluation integration)
- Output: full generalization-gap heatmap (attack type x SNR) for baseline and robust model
- **This phase produces the project's core contribution — do not compress this to add
  stretch goals elsewhere.**

## Phase 6 — Documentation & Report (Week 8)
- Methodology + results writeup, code cleanup, README, final report
- Owner: Mayank Ingole [EN24CS3T10017] drives, A and B contribute their sections
- Output: final thesis/report document, clean GitHub repo, explicit gap-analysis framing

## Phase 7 — Frontend Demo & Dashboard (rides alongside Phases 1-7, polished in Week 9)
**Not part of the MVP.** The frontend validates nothing about the research question — it is
presentation polish for the thesis defense, added after the core MVP was already confirmed
worthwhile. Build it in thin slices riding on top of artifacts that already exist each week,
not as a parallel full-time track:

| Week | Frontend task | Depends on |
|---|---|---|
| 1 | Skeleton `frontend/app.py`, empty tabs | Nothing |
| 3 | Demo tab wired to baseline model (clean predictions only) | Person A's baseline checkpoint |
| 4 | Add FGSM attack visualization to Demo tab | Person B's `attacks.py` |
| 6 | Add robust-model toggle to Demo tab | Person B's robust checkpoint |
| 7 | Dashboard tab — **default to static charts** (see fallback rule below) | Person C's `evaluate.py` output |
| 9 | Polish, only upgrade to interactive if Week 7-8 finished on schedule | — |

**Fallback rule (locked in after sanity-check review):** the Dashboard tab defaults to
**static, pre-rendered charts** (matplotlib/Plotly figures exported as images or a single
non-interactive render), not a live interactive dashboard. Interactive
zoom/hover/filtering on the generalization-gap heatmap is an upgrade attempted **only in
Week 9, only if Weeks 7-8 finished on schedule with no carryover work.** This removes the
dependency risk where Person B running late on the black-box attack (Week 7, the heaviest
technical week) would otherwise block Person C's buffer week too — static charts can be
generated from `metrics.json` the moment it exists, with no further frontend engineering
required to be "done."

**Rationale:** thesis committees grade the research contribution (the generalization-gap
result), not dashboard interactivity. This fallback protects the one buffer week the
project actually has, rather than letting frontend polish consume it.

---

## MVP vs Full Version Summary

| | MVP (Phase 0) | Full Version (Phase 1-6) |
|---|---|---|
| Attacks | FGSM only (unseen-attack sanity check) | FGSM + PGD + MIM + C&W + black-box surrogate |
| Models | Baseline + PGD-robust | Baseline + PGD-robust + surrogate |
| Evaluation | Single attack-success comparison | Full generalization-gap heatmap (attack x SNR) |
| Defense | PGD adversarial training | Same (single-attack training is the point, not a limitation) |
| Output | Sanity-check plots | Full report, reproducible repo, benchmarked generalization-gap result |

## Explicit Cut List (if time runs short, cut in this order)
1. **Interactive Dashboard upgrade** (Week 9 stretch) — cut first; static charts (the
   Week 7 default) already satisfy the frontend requirement on their own.
2. **Optional Rayleigh-fading stretch goal** (design.md Section 10) — cut next, it was
   never core scope.
3. **C&W attack** — most expensive attack to run; drop it before dropping MIM or the
   black-box surrogate attack, since MIM is cheap and the black-box result is more central
   to the novelty claim than C&W specifically.
4. **Reduce C&W/black-box evaluation to a smaller SNR subset** (e.g., every 4dB instead of
   every 2dB) rather than dropping them entirely — a sparser heatmap is still a valid
   result; no heatmap at all is not.
5. **Do not cut:** the PGD vs. FGSM generalization comparison — this is the minimum viable
   version of the core contribution and must survive any scope reduction.

## Sanity-Check Note (added after mid-project review)
The frontend was added after the original 8-week plan without removing anything else,
leaving effectively zero real buffer once prep time and frontend work are counted. The
static-charts fallback above exists specifically to protect against this — treat any
further scope additions with the same scrutiny before adding them on top of an already
full plan.
