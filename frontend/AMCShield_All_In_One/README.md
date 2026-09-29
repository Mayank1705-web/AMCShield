# AMCShield — Unified Frontend

This package merges the complete AMCShield frontend into a single folder while preserving the individual page functionality and the common dark military-tech visual system.

## Pages

- `index.html` — Secure Login
- `dashboard.html` — Dashboard
- `dataset.html` — Dataset
- `model-training.html` — Model Training
- `adversarial-attacks.html` — Adversarial Attacks
- `evaluation-results.html` — Evaluation Results
- `visualization.html` — Visualization
- `reports.html` — Reports
- `about.html` — About Project

## Structure

```text
AMCShield/
├── index.html
├── dashboard.html
├── dataset.html
├── model-training.html
├── adversarial-attacks.html
├── evaluation-results.html
├── visualization.html
├── reports.html
├── about.html
├── assets/
│   ├── amcshield-logo.png
│   └── amcshield-hero.png
├── login.css / login.js
├── dashboard.css / dashboard.js
├── dataset.css / dataset.js
├── model-training.css / model-training.js
├── adversarial-attacks.css / adversarial-attacks.js
├── evaluation-results.css / evaluation-results.js
├── visualization.css / visualization.js
├── reports.css / reports.js
└── about.css / about.js
```

## Backend integration

The JavaScript files keep the original `/api/...` endpoints. Run the AMCShield FastAPI backend from the same origin (or configure your server/proxy for `/api`) so the dynamic metrics, charts, training status, attack results, evaluation results, visualization data, and reports continue to load from the real project data.

No experimental metrics were added to the frontend during this merge.

## Run locally

For a simple static preview:

```bash
python -m http.server 5500
```

Then open:

`http://localhost:5500/`

For the full application, serve this frontend through the AMCShield backend so `/api/...` requests resolve correctly.
