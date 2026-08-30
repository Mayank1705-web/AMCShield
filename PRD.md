# PRD — Attack-Agnostic Robustness Generalization in AMC Under Low-SNR Conditions

## 1. Problem Statement
RF signal classifiers used in SIGINT/EW to identify modulation type perform well on clean
signals but degrade sharply in two realistic conditions:
1. **Low-SNR noise** — weak/degraded signals common in real interception scenarios.
2. **Adversarial interference** — deliberately crafted perturbations (spoofing/jamming)
   designed to fool the classifier.

A deeper literature-grounded problem sits underneath this: published defenses (primarily
adversarial training) are almost always trained and evaluated against a **single attack
type** — usually FGSM or PGD. It is not well established whether a model hardened this way
actually generalizes to attack types it never saw during training (MIM, C&W, black-box
surrogate-transfer attacks), or whether this apparent robustness is an illusion specific to
the attack it was trained against. This "attack-agnostic generalization gap" is confirmed
as under-studied across the reviewed literature, and is the specific problem this project
targets.

## 2. Objective
Determine whether adversarial training against one attack (PGD) generalizes to unseen
attack types, and how that generalization gap changes across the SNR range — producing
benchmarked, SNR-stratified evidence rather than a single aggregate number.

### 2.1 Literature Gap Analysis & Locked-In Unique Angle
A review of AMC adversarial-robustness literature (Sadeghi & Larsson 2018; Liu et al. 2020;
Usama et al. 2019; Zhang et al. 2021; Sahay et al. 2020/2021; Bamdad et al. 2024;
Restuccia et al. 2025, plus the extended gap analysis reviewed for this revision) shows five
recurring gaps:

1. **Synthetic dataset over-reliance** (RadioML lacks real channel effects) — real but out
   of scope; requires SDR hardware.
2. **Adversarial training overfits to a single attack type, generalizing poorly to unseen
   attack modalities** — confirmed, under-studied, and directly buildable with the current
   stack. **This is the locked-in project focus.**
3. **Digital attacks are decoupled from physical channel effects** (channel-aware /
   over-the-air attacks) — real gap, too much channel-modeling infrastructure for this
   timeline; documented as future work.
4. **Iterative defenses are computationally incompatible with real-time edge SDRs** —
   real but orthogonal to a classification-accuracy-focused thesis; noted as future work.
5. **Single-domain (time-only or frequency-only) feature processing** — real gap, but the
   proposed fix (foundation models, contrastive dual-domain learning) is multi-year-scale
   research infrastructure, not reproducible solo/small-team in 8 weeks.

**Locked-in unique angle:** combine the project's existing SNR-stratified evaluation
approach with Gap #2 —
- Adversarially train the robust model against **PGD only**.
- Evaluate it against attacks it never saw during training: **FGSM, MIM, C&W, and a
  black-box surrogate-transfer PGD attack.**
- Stratify every result across the full SNR range (-20dB to +18dB), producing a
  **generalization-gap heatmap** (attack type × SNR) for both the baseline and the
  PGD-adversarially-trained model.
- Report whether the generalization gap widens or narrows at low SNR — this specific
  interaction is not addressed cleanly anywhere in the reviewed literature.

### 2.2 Explicitly Not Pursued (and why)
- **Over-the-air / channel-realistic attacks** — genuine gap, deferred to future work;
  would require Rayleigh/Rician channel simulation infrastructure beyond current scope.
  A minimal version (simple additive Rayleigh fading before attack) is listed as an
  optional stretch goal only, not a core deliverable.
- **Foundation-model-based defenses (RFPrompt, frozen MoE backbones)** — requires
  pretrained wireless foundation models not readily available or reproducible solo.
- **Energy-Adaptive Frequency Masked Autoencoder purification, dual-domain contrastive
  learning** — each is independently thesis-scale; not attempted here.
- **Edge/real-time deployment optimization** — out of scope; this is a research/evaluation
  pipeline, not a deployed system.

## 3. Target Beneficiaries
| Beneficiary | Value |
|---|---|
| Defense/EW researchers | Evidence on whether single-attack adversarial training is a false sense of security |
| Spectrum monitoring / cognitive radio | Improved understanding of robustness limits under realistic noise |
| Academic community | SNR-stratified, multi-attack generalization benchmark — a genuine, citable gap |
| SIGINT/ELINT (conceptual) | Methodology transferable to real pipelines; honest limitations documented |

## 4. Scope

### In Scope (MVP)
- Baseline CNN trained on RadioML 2016.10a
- PGD-adversarial-trained robust model
- FGSM attack evaluated against both models (proves basic vulnerability + basic defense)
- Accuracy-vs-SNR curve for both models, clean and under FGSM

### In Scope (Full Version)
- MIM and C&W attacks added (both from `torchattacks`, no new attack infrastructure)
- Black-box surrogate-transfer attack (`model_surrogate.py`)
- Full generalization-gap heatmap: attack type (FGSM, PGD, MIM, C&W, black-box) × SNR,
  for both baseline and robust model
- Thesis writeup with explicit discussion of the generalization gap finding

### Out of Scope
- Real-world RF signal capture/hardware (SDR integration)
- Channel-aware / over-the-air attack modeling (documented as future work)
- Foundation-model-based defenses
- Deployment as a live/production classification service
- Any commercial use of RadioML (license is CC BY-NC-SA — academic use only)

## 5. Success Criteria
- Baseline model reproduces accuracy in the range reported by prior RadioML literature
- PGD-adversarial training measurably improves robustness against PGD itself (sanity check
  that adversarial training worked at all)
- **Generalization-gap heatmap produced**: robustness against FGSM/MIM/C&W/black-box
  compared to robustness against PGD, across the full SNR range — the project's core
  contribution
- A clear, evidence-backed statement of whether the generalization gap widens, narrows, or
  stays constant at low SNR
- Reproducible codebase + report suitable for academic submission

## 6. Known Risks / Limitations (state explicitly in report, don't hide)
- RadioML is **synthetic** data — results may not generalize to real captured RF signals
- C&W attack is computationally expensive (optimization loop per sample) — budget
  evaluation-set size accordingly, don't run it on the full test set if time is short
- Class/SNR imbalance can silently skew results if splits aren't stratified
- Free-tier GPU (Colab) can disconnect mid-run — checkpointing is mandatory, not optional
- Black-box surrogate transfer success depends on surrogate architecture choice — document
  this choice explicitly as a limitation, not a guaranteed worst-case black-box attack

## 7. Deliverables
- GitHub repo (code, README, requirements.txt)
- Trained baseline + PGD-robust model checkpoints, plus one surrogate model checkpoint
- Evaluation plots: accuracy-vs-SNR, confusion matrices, generalization-gap heatmap
- Final report/thesis document with explicit gap-analysis framing
