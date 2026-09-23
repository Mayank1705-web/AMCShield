# AMCShield — Robust Automatic Modulation Classification

AMCShield is a deep-learning project for evaluating and improving the robustness of Automatic Modulation Classification (AMC) systems against adversarial attacks.

The project compares a pretrained baseline AMC model with an adversarially trained robust model using the **RadioML2018.01A** dataset and evaluates both white-box and black-box attack scenarios.

> **Project status:** Training and attack evaluation are complete. The notebooks and result files document the final experimental results. No retraining or attack reruns are required for the final analysis.

---

## 1. Project Objectives

The main objectives of AMCShield are to:

- Build an AMC baseline classifier for IQ radio signals.
- Train a robust AMC model using adversarial training.
- Evaluate model behavior under multiple adversarial attacks.
- Compare baseline and robust models across different SNR levels.
- Evaluate transfer-based black-box robustness using a surrogate model.
- Quantify the generalization gap between baseline and robust models.

---

## 2. Dataset

AMCShield uses **RadioML2018.01A**.

| Property | Value |
|---|---|
| Dataset | RadioML2018.01A |
| Number of modulation classes | 24 |
| Signal representation | IQ |
| Input shape | `(2, 1024)` |
| SNR range | -20 dB to 30 dB |
| SNR step | 2 dB |
| Train split | 70% |
| Validation split | 15% |
| Test split | 15% |

The dataset is loaded lazily from HDF5 to avoid loading the complete dataset into memory.

---

## 3. Models

### Baseline Model

The baseline AMC classifier is trained without adversarial training and provides the reference point for robustness evaluation.

### Robust Model

The robust AMC model is trained using adversarial examples generated with PGD during training.

Training configuration includes:

- Batch size: 128
- Learning rate: 0.001
- Maximum epochs: 20
- PGD epsilon: 0.02
- PGD steps: 10
- Optimizer: Adam
- Device: CUDA when available

The saved checkpoints are:

```text
checkpoints/baseline_best.pt
checkpoints/robust_best.pt
```

The checkpoints are already trained and are used by the evaluation pipeline.

---

## 4. Adversarial Attacks

AMCShield evaluates five attack settings.

| Attack | Category | Configuration |
|---|---|---|
| FGSM | White-box | ε = 0.02 |
| PGD | White-box | ε = 0.02, 10 steps |
| MIM | White-box | ε = 0.02, 10 steps |
| C&W | Optimization-based | 50 optimization steps |
| Black-box | Transfer-based | ε = 0.02, 10 steps |

### Black-box Threat Model

The black-box evaluation uses a separate surrogate model.

- Query budget: 5,000
- Surrogate architecture: CNN + BiLSTM hybrid
- Surrogate validation accuracy: approximately 41.90%
- Target-model gradients are not used to generate the transferred attack.
- Surrogate labels are obtained from model predictions rather than ground-truth labels.

---

## 5. Evaluation Metrics

The project records:

- Clean accuracy
- Attack accuracy
- Attack Success Rate (ASR)
- Accuracy drop
- Mean L∞ perturbation
- Maximum L∞ perturbation
- Mean L2 perturbation
- Maximum L2 perturbation
- Generalization gap across SNR

The generalization-gap analysis uses:

```text
Generalization Gap = Baseline ASR - Robust ASR
```

The gap is reported in percentage points.

---

## 6. Experimental Results

The final evaluation results are stored in `results/`.

### Main result files

```text
results/
├── baseline_clean_results.csv
├── baseline_fgsm_results.csv
├── baseline_pgd_results.csv
├── baseline_mim_results.csv
├── baseline_cw_results.csv
├── baseline_blackbox_results.csv
├── clean_results.csv
├── fgsm_results.csv
├── pgd_results.csv
├── mim_results.csv
├── cw_results.csv
├── blackbox_results.csv
├── generalization_gap.csv
├── generalization_gap.json
├── master_results.csv
└── master_results.json
```

Generated plots are stored under:

```text
results/plots/
```

The master result table contains the consolidated SNR-wise comparison of baseline and robust evaluation results.

---

## 7. Notebooks

The analysis is organized into five notebooks.

```text
notebooks/
├── 01_data_exploration.ipynb
├── 02_baseline_training.ipynb
├── 03_attacks_demo.ipynb
├── 04_robust_vs_baseline.ipynb
└── 05_final_results.ipynb
```

### Notebook 01 — Data Exploration

Documents:

- Dataset structure
- Signal representation
- Classes
- SNR distribution
- Basic dataset exploration

### Notebook 02 — Baseline Training

Documents the baseline model and its training/checkpoint information.

### Notebook 03 — Attacks Demo

Documents the implemented adversarial attacks and their evaluation results.

### Notebook 04 — Robust vs Baseline

Provides detailed SNR-wise comparisons between the baseline and robust models.

### Notebook 05 — Final Results

Provides the final consolidated analysis using the existing result files and `master_results.csv`, including:

- Summary tables
- Clean accuracy comparison
- Attack success-rate plots
- C&W distortion analysis
- Black-box comparison
- Generalization-gap visualization
- Key observations
- Limitations
- Conclusion

No model training or attack execution is performed in this notebook.

---

## 8. Repository Structure

```text
AMCShield/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── attacks.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model_baseline.py
│   ├── model_robust.py
│   ├── model_surrogate.py
│   ├── train_surrogate.py
│   └── ...
│
├── checkpoints/
│   ├── baseline_best.pt
│   ├── robust_best.pt
│   └── surrogate_best.pt
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_attacks_demo.ipynb
│   ├── 04_robust_vs_baseline.ipynb
│   └── 05_final_results.ipynb
│
├── results/
│   ├── *.csv
│   ├── *.json
│   └── plots/
│
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 9. Reproducing the Analysis

The final notebooks are designed to analyze already-generated results.

For example:

```bash
python src/run_baseline_attacks.py
python src/run_generalization_heatmap.py
python src/master_results.py
```

These scripts are part of the experimental pipeline. The final-results notebook itself does not retrain models or rerun attacks.

> Existing checkpoints and generated CSV/JSON result files should be preserved when reproducing the final report.

---

## 10. Important Experimental Limitations

The results should be interpreted within the scope of the implemented evaluation.

1. Attack experiments use sampled evaluation subsets rather than the complete test set for every attack.
2. The number of samples used for different attack evaluations is not identical in every existing experiment.
3. The existing robust C&W evaluation was performed using its recorded exploratory evaluation subset and should not be described as a 300-sample-per-SNR experiment.
4. C&W is not constrained by the same ε = 0.02 L∞ projection used by FGSM, PGD, and MIM.
5. Black-box transfer effectiveness depends on the surrogate model and its agreement with the target model.
6. The experiments cover the attacks implemented in this repository and do not represent every possible adversarial threat.
7. Conclusions are specific to RadioML2018.01A, the evaluated SNR range, model architectures, and attack configurations.

---

## 11. Key Project Takeaway

AMCShield provides an end-to-end experimental pipeline for studying adversarial robustness in automatic modulation classification.

The project combines:

```text
RadioML2018.01A
        ↓
Baseline AMC
        ↓
Adversarial Attacks
        ↓
PGD-Based Robust Training
        ↓
Robust AMC
        ↓
White-box + Black-box Evaluation
        ↓
SNR-wise Robustness Analysis
        ↓
Generalization Gap
```

The final results allow the behavior of the baseline and adversarially trained models to be examined across signal-to-noise conditions and multiple attack types.

---

## 12. Future Work

Potential extensions include:

- Multi-attack adversarial training
- Stronger or adaptive adversarial attacks
- Improved surrogate models for black-box evaluation
- Larger query-budget studies
- Additional RF datasets
- Transformer-based AMC architectures
- More extensive cross-SNR and cross-dataset generalization studies
- Deployment-oriented robustness evaluation

---

## 13. Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- h5py
- YAML
- CUDA
- RadioML2018.01A

---

## 14. Author

**Mayank Ingole**  
B.Tech — Computer Science & Engineering  
Medicaps University

---

## 15. License

This repository is intended primarily for academic/project work. Add an explicit open-source license if the repository is later distributed under one.
