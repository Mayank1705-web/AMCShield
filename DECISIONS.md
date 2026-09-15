# Decisions Log

## [Planning Phase] — Locked-In Research Angle: Attack-Agnostic Generalization Gap

### Context
Initial project concept (RadioML + FGSM/PGD + adversarial training) is
well-trodden in published literature. A literature review was needed to find a
genuinely under-studied, but realistically buildable, angle for an 8-9 week
student thesis project.

### Options Considered
- Channel-aware / over-the-air attack modeling (Gap #3 from literature review)
- Foundation-model-based defenses with prompt tuning (research vector from
  extended literature review)
- Dual-domain contrastive learning with energy-adaptive masking
- Attack-agnostic generalization: train against one attack (PGD), test against
  unseen attacks (FGSM, MIM, C&W, black-box), stratified by SNR

### Decision
Attack-agnostic generalization gap, combined with SNR-stratified evaluation.

### Reason
Channel-aware attacks require channel-modeling infrastructure beyond the
timeline. Foundation-model and dual-domain contrastive approaches are
independently thesis-scale research programs, not reproducible solo/small-team
in 8-9 weeks. The generalization-gap angle is genuinely under-studied in the
reviewed literature, requires no new infrastructure beyond the existing
PyTorch/torchattacks stack, and produces a single clear, citable, defensible
result (the generalization-gap heatmap).

### Tradeoffs
Gained: a realistic, finishable, and citable research contribution.
Sacrificed: the more ambitious (and more novel) channel-aware and
foundation-model directions — documented as future work rather than abandoned.

### Impact
Defines the entire evaluation pipeline (`attacks.py`, `evaluate.py`), the
robust model's training procedure (PGD-only, not multi-attack), and the need
for a separate `model_surrogate.py` for the black-box attack path.

### AI Model
Claude (conversation-based planning session)

---

## [Planning Phase] — Robust Model Trained Against PGD Only (Not Multi-Attack)

### Context
Needed to decide whether the robust model should be adversarially trained
against a single attack type or multiple attack types simultaneously.

### Options Considered
- Train against PGD only
- Train against a mix of FGSM + PGD + MIM simultaneously

### Decision
Train against PGD only.

### Reason
The project's research question is specifically whether single-attack training
generalizes. Training against multiple attacks would answer a different
(also interesting) question and would muddy the core result. Multi-attack
training is documented as a reasonable stretch goal only if time allows.

### Tradeoffs
Gained: a clean, answerable research question with an unambiguous experimental
design.
Sacrificed: a "better" defense in absolute terms — the robust model is
expected to underperform against unseen attacks by design; that gap is the
point, not a flaw.

### Impact
`train.py`'s adversarial training hook, `model_robust.py`, and the entire
framing of `evaluate.py`'s generalization-gap computation.

### AI Model
Claude (conversation-based planning session)

---

## [Planning Phase] — Surrogate Model Must Be Architecturally Different from Baseline

### Context
Designing the black-box surrogate-transfer attack path required deciding
whether the surrogate model should share the baseline's architecture.

### Options Considered
- Same architecture as baseline (simpler to implement)
- Deliberately different architecture (e.g., CNN-LSTM if baseline is CNN)

### Decision
Deliberately different architecture, trained only on query-response pairs from
the baseline (no weight sharing, no direct architecture copy).

### Reason
Using an identical architecture as the surrogate would artificially inflate
transfer attack success and misrepresent a realistic black-box threat model,
where an attacker does not know the target's exact architecture.

### Tradeoffs
Gained: a more realistic and defensible black-box threat model.
Sacrificed: some attack strength — a matched-architecture surrogate would
likely produce a higher (but less honest) attack success rate.

### Impact
`model_surrogate.py` is a standalone file, explicitly forbidden from importing
or sharing weights with `model_baseline.py` (enforced in `.cursorrules`).

### AI Model
Claude (conversation-based planning session)

---

## [Planning Phase] — Frontend Defaults to Static Charts, Not Interactive Dashboard

### Context
A frontend (Streamlit demo + results dashboard) was added to the project scope
after the original 8-week plan was already finalized, without removing other
work to compensate. A mid-project sanity-check review flagged this as a risk:
the interactive Dashboard tab depended on Person B's Week 7 black-box attack
work, meaning a delay there would consume Person C's only buffer week (Week 9)
as well.

### Options Considered
- Interactive Plotly dashboard by default (original plan)
- Static, pre-rendered charts by default, with interactive as an opt-in
  Week 9 stretch only if earlier phases finish on schedule

### Decision
Static charts by default; interactive Plotly is an optional, non-blocking
Week 9 upgrade.

### Reason
Thesis committees grade the research contribution (the generalization-gap
result), not dashboard interactivity. Static charts can be generated the
moment `metrics.json` exists, removing the hidden cross-person dependency and
protecting the project's only real schedule buffer.

### Tradeoffs
Gained: removed a scheduling dependency risk; protected the buffer week.
Sacrificed: a more polished, interactive thesis-defense demo — accepted as a
reasonable cost, since the demo is presentation polish, not research
infrastructure.

### Impact
`frontend/app.py` Dashboard tab implementation, `phases.md` Phase 7,
`design.md` Section 11.

### AI Model
Claude (conversation-based planning session, added during mid-project sanity
check)

---

## [Planning Phase] — Dataset: RadioML 2016.10a Over 2018.01a

### Context
DeepSig publishes multiple RadioML dataset versions; needed to pick one for
the MVP.

### Options Considered
- RadioML 2016.10a (11 modulation classes, smaller)
- RadioML 2018.01a (24 modulation classes, larger, more recent)

### Decision
RadioML 2016.10a.

### Reason
Smaller dataset size means faster iteration within the 8-9 week timeline.
2018.01a is documented as a reasonable future extension, not the MVP target.

### Tradeoffs
Gained: faster training/evaluation cycles, more time for the attack suite and
generalization analysis.
Sacrificed: results are on an older, smaller benchmark — less comprehensive
than what 2018.01a would offer.

### Impact
`data_loader.py`, `preprocess.py`, and all reported results are scoped to
2016.10a's 11 classes and SNR range (-20dB to +18dB).

### AI Model
Claude (conversation-based planning session)
