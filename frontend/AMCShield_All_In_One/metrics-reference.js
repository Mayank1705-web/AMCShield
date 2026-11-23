/**
 * metrics-reference.js
 * AMCShield — ML Metrics Reference
 *
 * Data + render logic for the comprehensive ML metrics reference page.
 * No external dependencies. Pure vanilla JS.
 */

/* ══════════════════════════════════════════════════════════════════
   DATA — 53 metrics across 6 categories
   Each metric:
   {
     id:        unique slug
     cat:       'validation' | 'classification' | 'regression'
               | 'clustering' | 'ranking' | 'system'
     name:      display name
     alias:     short aliases / other names (string)
     formula:   ASCII formula shown in a monospace box (optional)
     desc:       explanation (1-3 sentences)
     range:     { min, max, good, label }  —  good = fraction [0-1] where the marker sits
     interp:    array of { text, type }   type = 'good' | 'bad' | 'info'
     usage:     AMCShield project usage note (optional string)
   }
══════════════════════════════════════════════════════════════════ */

const CATEGORIES = {
  validation:     { label: "① Testing & Validation Methods",          color: "#f4c542" },
  classification: { label: "② Classification Parameters",             color: "#20c9d8" },
  regression:     { label: "③ Regression Parameters",                 color: "#52d273" },
  clustering:     { label: "④ Clustering Parameters (Unsupervised)",  color: "#9d7fe8" },
  ranking:        { label: "⑤ Ranking & Recommendation Parameters",   color: "#f4834a" },
  system:         { label: "⑥ System & Production Parameters",        color: "#e87fd0" },
};

const METRICS = [
  /* ─────────────────────────────────────────────
     1. VALIDATION METHODS
  ───────────────────────────────────────────── */
  {
    id: "holdout",
    cat: "validation",
    name: "Holdout (Train-Val-Test Split)",
    alias: "Train-Validation-Test Split",
    formula: "Train ≈ 60-70%  |  Val ≈ 10-20%  |  Test ≈ 10-20%",
    desc: "The dataset is split once into non-overlapping train, validation, and test partitions. The model is trained on train, hyper-parameters are tuned on val, and the final number is reported on test — which must never be touched until the very end.",
    interp: [
      { text: "Fast to run", type: "good" },
      { text: "Single random split = high variance on small data", type: "bad" },
      { text: "AMCShield uses fixed index files", type: "info" },
    ],
    usage: "AMCShield uses fixed index files (train_indices.npy / val_indices.npy / test_indices.npy) for 383 k test samples. The test loader uses a fixed pseudo-random seed so results are reproducible.",
  },
  {
    id: "kfold",
    cat: "validation",
    name: "K-Fold Cross-Validation",
    alias: "CV, K-Fold CV",
    formula: "Score = (1/K) · Σ_k Score_k\n\nTypical K = 5 or 10",
    desc: "Data is divided into K equal folds. The model trains K times; each fold serves as the test set once. The final score is the average over all K runs. Reduces variance compared with a single holdout split.",
    interp: [
      { text: "K=5 is a good default", type: "good" },
      { text: "K times the training cost", type: "bad" },
      { text: "Not used in AMCShield (RadioML is large enough)", type: "info" },
    ],
    usage: "Not used: with 2.5 M samples the single holdout is stable. K-fold would be prohibitively expensive on the RTX 3050.",
  },
  {
    id: "stratified-kfold",
    cat: "validation",
    name: "Stratified K-Fold",
    alias: "Stratified CV",
    formula: "Each fold preserves the class distribution\nof the full dataset.",
    desc: "Like K-Fold, but each fold is built so every class appears in proportion. Essential when classes are imbalanced; prevents a fold from having too few samples of a rare class.",
    interp: [
      { text: "Recommended for imbalanced classes", type: "good" },
      { text: "Cannot be used directly for multi-label", type: "bad" },
      { text: "RadioML 2018.01A is roughly balanced across classes and SNRs", type: "info" },
    ],
    usage: "RadioML is balanced, so plain k-fold would work too. Per-SNR evaluation in AMCShield manually ensures balanced sampling (256 samples per class per SNR level).",
  },
  {
    id: "loocv",
    cat: "validation",
    name: "Leave-One-Out Cross-Validation",
    alias: "LOOCV, LOO-CV",
    formula: "K = N  (each sample is the test set once)\nScore = (1/N) · Σ_i Score_i",
    desc: "Extreme case of K-fold where K equals the number of samples. Each individual sample is held out once. Gives an almost unbiased estimate but is extremely expensive for large datasets.",
    interp: [
      { text: "Minimum bias", type: "good" },
      { text: "N training runs — infeasible for large datasets", type: "bad" },
      { text: "Only practical for datasets with < 1000 samples", type: "info" },
    ],
    usage: "Not applicable for AMCShield — 2.5 M samples makes LOOCV computationally impossible.",
  },
  {
    id: "timeseries-split",
    cat: "validation",
    name: "Time-Series Split",
    alias: "Rolling Window, Expanding Window, Walk-Forward CV",
    formula: "Train on [0..t], Test on [t..t+h]\nt advances → h = forecast horizon",
    desc: "Respects temporal ordering: training always precedes the test window. Used when data has time dependencies (stock prices, sensor streams). Prevents 'look-ahead bias' that would occur in random splits.",
    interp: [
      { text: "Prevents data leakage in temporal data", type: "good" },
      { text: "Earlier windows have smaller train sets → higher variance", type: "bad" },
      { text: "Not applicable to I/Q signal classification (no time ordering between samples)", type: "info" },
    ],
  },
  {
    id: "bootstrapping",
    cat: "validation",
    name: "Bootstrapping",
    alias: "Bootstrap Resampling, .632 Bootstrap",
    formula: "Draw N samples WITH replacement.\nOut-of-bag (OOB) set ≈ 36.8% of data.\nP(not selected) = (1 - 1/N)^N → 1/e ≈ 0.368",
    desc: "Creates many datasets by sampling with replacement. Useful for estimating confidence intervals of any statistic. The 'out-of-bag' samples (not drawn) serve as a natural test set.",
    interp: [
      { text: "Excellent for confidence interval estimation", type: "good" },
      { text: "Some samples appear multiple times — not truly independent", type: "bad" },
      { text: "Used in ensemble methods like Random Forests", type: "info" },
    ],
    usage: "Could be used in a future work to report confidence intervals on the AMCShield accuracy numbers (currently single-point estimates only).",
  },
  {
    id: "baseline-comparison",
    cat: "validation",
    name: "Baseline Model Comparison",
    alias: "Null Model, Majority Classifier, Random Baseline",
    formula: "Baseline accuracy = max class frequency\n\nGain = Model_acc − Baseline_acc",
    desc: "Before claiming a model is good, compare it to the simplest possible predictor: always predicting the majority class. If your model barely beats the baseline, it has not learned much.",
    interp: [
      { text: "Sanity-check every evaluation with a baseline", type: "good" },
      { text: "Misleading if classes are severely imbalanced", type: "bad" },
      { text: "AMCShield BaselineCNN 62.75% vs random 1/24 ≈ 4.2%", type: "info" },
    ],
    usage: "In AMCShield, 'baseline' means the undefended CNN (BaselineCNN), not a random predictor. The comparison is BaselineCNN vs RobustCNN under various attacks.",
  },
  {
    id: "ab-testing",
    cat: "validation",
    name: "A/B Testing",
    alias: "Online Controlled Experiment, Split Testing",
    formula: "Traffic split 50/50 between Model A and Model B.\nStatistical significance test on outcome metric.",
    desc: "Live traffic is randomly routed to two model versions; real user interactions determine which performs better. The gold standard for production ML deployment decisions.",
    interp: [
      { text: "Real-world feedback, no simulation assumptions", type: "good" },
      { text: "Requires production traffic — not available offline", type: "bad" },
      { text: "Risk of exposing users to a weaker model during the test", type: "bad" },
    ],
    usage: "Not applicable in the academic setting. Mentioned as future work for a real SDR (Software-Defined Radio) deployment scenario.",
  },
  {
    id: "shadow-deployment",
    cat: "validation",
    name: "Shadow Deployment",
    alias: "Shadow Mode, Mirror Deployment, Canary (passive)",
    formula: "Real traffic → Model A (live)\n               ↘ Model B (shadow, logged only)",
    desc: "The shadow model receives a copy of real traffic and makes predictions, but its output is never shown to users. Logs are compared offline. Zero risk while testing a new model in production conditions.",
    interp: [
      { text: "Safe: users never see shadow model output", type: "good" },
      { text: "Double compute cost", type: "bad" },
      { text: "Requires a production system to mirror traffic from", type: "info" },
    ],
    usage: "Relevant for a future AMCShield API deployment where the robust model runs in shadow mode alongside the baseline to compare real-time classification quality.",
  },

  /* ─────────────────────────────────────────────
     2. CLASSIFICATION
  ───────────────────────────────────────────── */
  {
    id: "confusion-matrix",
    cat: "classification",
    name: "Confusion Matrix",
    alias: "Error Matrix, TP / TN / FP / FN",
    formula: "          Predicted+   Predicted−\nActual+     TP            FN\nActual−     FP            TN",
    desc: "An N×N table showing how often each true class was predicted as each other class. The diagonal = correct predictions. Off-diagonal cells reveal which classes the model confuses. The foundation from which all other classification metrics derive.",
    interp: [
      { text: "High diagonal values → good model", type: "good" },
      { text: "Off-diagonal clusters reveal systematic confusion", type: "bad" },
      { text: "24×24 matrix for AMCShield (24 modulation classes)", type: "info" },
    ],
    usage: "evaluate_metrics.py computes the full 24×24 confusion matrix. Stored to results/. Useful for seeing which modulation classes (e.g. QAM64 vs QAM256) are confused.",
  },
  {
    id: "accuracy",
    cat: "classification",
    name: "Accuracy",
    alias: "Top-1 Accuracy, Overall Accuracy",
    formula: "Accuracy = (TP + TN) / (TP + TN + FP + FN)\n         = Correct / Total",
    desc: "The fraction of all predictions that are correct. The most intuitive metric but misleading when classes are imbalanced — a model that always predicts the majority class can have high accuracy while being useless.",
    range: { min: 0, max: 1, good: 0.628, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "High value → more correct predictions", type: "good" },
      { text: "Misleading on imbalanced classes", type: "bad" },
      { text: "AMCShield baseline clean val accuracy: 62.75%", type: "info" },
    ],
    usage: "Primary metric for AMCShield model selection (best checkpoint by clean val accuracy). Per-SNR accuracy plotted in the generalization gap analysis.",
  },
  {
    id: "precision",
    cat: "classification",
    name: "Precision",
    alias: "Positive Predictive Value, PPV",
    formula: "Precision = TP / (TP + FP)",
    desc: "Of all the samples the model PREDICTED as positive, what fraction were truly positive? High precision = few false alarms. Important when false positives are costly.",
    range: { min: 0, max: 1, good: 0.75, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "High value → few false alarms", type: "good" },
      { text: "Can be artificially high by being very conservative", type: "bad" },
    ],
    usage: "Computed per-class (macro/weighted) in evaluate_metrics.py. Low-SNR classes have poor precision because the model guesses randomly.",
  },
  {
    id: "recall",
    cat: "classification",
    name: "Recall",
    alias: "Sensitivity, True Positive Rate, TPR, Hit Rate",
    formula: "Recall = TP / (TP + FN)",
    desc: "Of all truly positive samples, what fraction did the model correctly detect? High recall = few missed detections. Important when false negatives are costly.",
    range: { min: 0, max: 1, good: 0.75, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "High value → few missed detections", type: "good" },
      { text: "Trivially maximized by predicting everything as positive", type: "bad" },
    ],
    usage: "In adversarial settings, recall drops dramatically under attacks — the model 'misses' the correct class. Used alongside precision in the per-class analysis.",
  },
  {
    id: "specificity",
    cat: "classification",
    name: "Specificity",
    alias: "True Negative Rate, TNR, Selectivity",
    formula: "Specificity = TN / (TN + FP)",
    desc: "Of all truly negative samples, what fraction did the model correctly label as negative? The complement of the false positive rate. High specificity = few false alarms on negative examples.",
    range: { min: 0, max: 1, good: 0.85, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "High value → model correctly rejects negatives", type: "good" },
      { text: "Must be balanced with recall (sensitivity)", type: "info" },
    ],
  },
  {
    id: "f1",
    cat: "classification",
    name: "F1-Score",
    alias: "F-Measure, F1",
    formula: "F1 = 2 · (Precision · Recall) / (Precision + Recall)\n   = 2·TP / (2·TP + FP + FN)",
    desc: "The harmonic mean of precision and recall. A single balanced metric that penalizes extreme imbalance between precision and recall. Preferred over accuracy when classes are imbalanced.",
    range: { min: 0, max: 1, good: 0.65, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "Balances precision and recall", type: "good" },
      { text: "Still undefined if TP+FP+FN = 0 (zero-shot class)", type: "bad" },
      { text: "Macro-F1 weights all classes equally; Weighted-F1 weights by support", type: "info" },
    ],
    usage: "Computed by evaluate_metrics.py for AMCShield. Expected to be significantly higher at high SNR (≥ +8 dB) than at low SNR (≤ -10 dB).",
  },
  {
    id: "fbeta",
    cat: "classification",
    name: "F-Beta Score",
    alias: "Fβ, F-Measure (general)",
    formula: "Fβ = (1 + β²) · (Precision · Recall) / (β² · Precision + Recall)\n\nβ > 1 → Recall favored\nβ < 1 → Precision favored",
    desc: "Generalizes the F1-score with a β parameter that lets you weight precision and recall differently. F2 (β=2) values recall twice as much; F0.5 (β=0.5) values precision twice as much.",
    range: { min: 0, max: 1, good: 0.65, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "Flexible weighting of precision vs recall", type: "good" },
      { text: "β must be chosen by domain experts", type: "info" },
    ],
    usage: "In a spectrum-defense context (false alarms costly), β < 1 would be preferred. AMCShield currently uses plain F1 (β=1).",
  },
  {
    id: "auc-roc",
    cat: "classification",
    name: "AUC-ROC",
    alias: "AUROC, Area Under the ROC Curve",
    formula: "ROC plots TPR vs FPR at every decision threshold.\nAUC = ∫ TPR d(FPR)\n\nAUC = 0.5 → random  |  1.0 → perfect",
    desc: "Measures the model's ability to discriminate between positive and negative classes across all possible thresholds. Threshold-independent. 0.5 = random classifier; 1.0 = perfect.",
    range: { min: 0.5, max: 1, good: 0.65, label: "0.5 → 1   (higher is better)" },
    interp: [
      { text: "AUC > 0.9 → excellent discrimination", type: "good" },
      { text: "AUC ≈ 0.5 → useless (random)", type: "bad" },
      { text: "One-vs-Rest AUC needed for multi-class", type: "info" },
    ],
    usage: "Not currently computed in AMCShield's evaluate_metrics.py but could be added with sklearn.metrics.roc_auc_score(multiclass='ovr') for a future report section.",
  },
  {
    id: "auc-pr",
    cat: "classification",
    name: "AUC-PR",
    alias: "AUPRC, Average Precision",
    formula: "PR curve plots Precision vs Recall.\nAUC-PR = area under that curve.\n\nBaseline = class prevalence (random classifier).",
    desc: "Like AUC-ROC but plots precision against recall. More informative than AUC-ROC when classes are heavily imbalanced because it focuses on the minority (positive) class. The random baseline is the class prevalence, not 0.5.",
    range: { min: 0, max: 1, good: 0.7, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "Better than AUC-ROC for imbalanced datasets", type: "good" },
      { text: "Sensitive to class imbalance — baseline shifts", type: "info" },
    ],
  },
  {
    id: "log-loss",
    cat: "classification",
    name: "Log Loss",
    alias: "Logarithmic Loss, Cross-Entropy Loss",
    formula: "LogLoss = -(1/N) · Σ Σ y_ic · log(p_ic)\n\ny_ic = 1 if sample i is class c\np_ic = predicted probability",
    desc: "Measures the quality of the probability estimates, not just the hard predictions. Heavily penalizes confident wrong answers. Lower is better. This is what PyTorch's CrossEntropyLoss minimizes during training.",
    range: { min: 0, max: null, good: 0.3, label: "0 → ∞   (lower is better)" },
    interp: [
      { text: "0 → perfect (never happens)", type: "good" },
      { text: "Very sensitive to overconfident wrong predictions", type: "bad" },
      { text: "AMCShield uses CE loss as training objective", type: "info" },
    ],
    usage: "The training and validation loss curves in AMCShield are cross-entropy. Early stopping uses validation loss as a signal (patience 7 epochs).",
  },
  {
    id: "mcc",
    cat: "classification",
    name: "Matthews Correlation Coefficient",
    alias: "MCC, Phi Coefficient",
    formula: "MCC = (TP·TN − FP·FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))\n\nRange: −1 (worst) → 0 (random) → +1 (perfect)",
    desc: "A correlation coefficient between the actual and predicted binary labels. MCC is considered the most informative single metric for binary classification: it only gives a high score if the model performs well on all four confusion-matrix quadrants, even when classes are imbalanced.",
    range: { min: -1, max: 1, good: 0.75, label: "−1 → +1   (higher is better)" },
    interp: [
      { text: "MCC > 0.7 → strong model", type: "good" },
      { text: "MCC = 0 → random baseline", type: "bad" },
      { text: "Multi-class generalization available in scikit-learn", type: "info" },
    ],
    usage: "Computed by evaluate_metrics.py using sklearn. MCC will be low at low SNR (model at chance level) and high at high SNR. Adversarial attacks collapse MCC toward 0.",
  },
  {
    id: "cohens-kappa",
    cat: "classification",
    name: "Cohen's Kappa",
    alias: "κ (Kappa), Inter-Rater Reliability",
    formula: "κ = (p_o − p_e) / (1 − p_e)\n\np_o = observed agreement\np_e = expected agreement by chance",
    desc: "Measures how much better the classifier does compared to a random classifier that respects the class distribution. Accounts for the possibility of agreement occurring by chance, unlike plain accuracy.",
    range: { min: -1, max: 1, good: 0.65, label: "−1 → +1   (higher is better)" },
    interp: [
      { text: "κ > 0.8 → near-perfect", type: "good" },
      { text: "κ 0.4-0.6 → moderate", type: "info" },
      { text: "κ < 0.2 → slight agreement (barely better than chance)", type: "bad" },
    ],
    usage: "Computed alongside MCC in evaluate_metrics.py. Kappa is especially meaningful in AMCShield's balanced-class setting where accuracy is not misleading, but kappa provides a chance-corrected view.",
  },

  /* ─────────────────────────────────────────────
     3. REGRESSION
  ───────────────────────────────────────────── */
  {
    id: "mae",
    cat: "regression",
    name: "Mean Absolute Error",
    alias: "MAE, L1 Loss",
    formula: "MAE = (1/N) · Σ |y_i − ŷ_i|",
    desc: "The average of the absolute differences between predicted and actual values. Easy to interpret (same units as the target). Less sensitive to large outlier errors than MSE.",
    range: { min: 0, max: null, good: 0.1, label: "0 → ∞   (lower is better)" },
    interp: [
      { text: "Robust to outliers compared to MSE", type: "good" },
      { text: "Gradient undefined at zero (non-differentiable there)", type: "bad" },
    ],
  },
  {
    id: "mse",
    cat: "regression",
    name: "Mean Squared Error",
    alias: "MSE, L2 Loss (regression)",
    formula: "MSE = (1/N) · Σ (y_i − ŷ_i)²",
    desc: "The average of the squared errors. Penalizes large errors heavily due to the squaring. The most common loss function for regression tasks. Units are squared — harder to interpret directly.",
    range: { min: 0, max: null, good: 0.1, label: "0 → ∞   (lower is better)" },
    interp: [
      { text: "Differentiable everywhere — easy to optimize", type: "good" },
      { text: "Very sensitive to outliers", type: "bad" },
    ],
  },
  {
    id: "rmse",
    cat: "regression",
    name: "Root Mean Squared Error",
    alias: "RMSE",
    formula: "RMSE = √(MSE) = √((1/N) · Σ (y_i − ŷ_i)²)",
    desc: "Square root of MSE, bringing the error back to the same units as the target. The most widely reported regression metric. Penalizes large errors more than MAE.",
    range: { min: 0, max: null, good: 0.1, label: "0 → ∞   (lower is better)" },
    interp: [
      { text: "Same units as target — easier to interpret than MSE", type: "good" },
      { text: "Still sensitive to outliers", type: "bad" },
    ],
  },
  {
    id: "rmsle",
    cat: "regression",
    name: "Root Mean Squared Logarithmic Error",
    alias: "RMSLE",
    formula: "RMSLE = √((1/N) · Σ (log(ŷ_i + 1) − log(y_i + 1))²)",
    desc: "Applies RMSE in log-transformed space. Treats relative errors equally regardless of the magnitude of the target. Useful when the target spans many orders of magnitude (e.g. sales, prices).",
    range: { min: 0, max: null, good: 0.1, label: "0 → ∞   (lower is better)" },
    interp: [
      { text: "Penalizes under-predictions more than over-predictions", type: "info" },
      { text: "Only valid for non-negative targets", type: "bad" },
    ],
  },
  {
    id: "r2",
    cat: "regression",
    name: "R-Squared",
    alias: "R², Coefficient of Determination",
    formula: "R² = 1 − SS_res / SS_tot\n\nSS_res = Σ (y_i − ŷ_i)²\nSS_tot = Σ (y_i − ȳ)²",
    desc: "The proportion of variance in the target explained by the model. 1.0 = perfect; 0.0 = no better than the mean baseline; negative = worse than predicting the mean.",
    range: { min: -Infinity, max: 1, good: 0.85, label: "−∞ → 1   (higher is better)" },
    interp: [
      { text: "R² > 0.9 → strong fit", type: "good" },
      { text: "R² < 0 → model is worse than the mean baseline", type: "bad" },
      { text: "Can be misleading for non-linear relationships", type: "info" },
    ],
  },
  {
    id: "adj-r2",
    cat: "regression",
    name: "Adjusted R-Squared",
    alias: "Adj. R², R̄²",
    formula: "Adj. R² = 1 − (1−R²)·(N−1)/(N−p−1)\n\nN = samples, p = number of features",
    desc: "Penalizes R² for adding extra features that do not genuinely improve the model. R² always increases with more features; Adjusted R² can decrease if irrelevant features are added.",
    range: { min: -Infinity, max: 1, good: 0.82, label: "−∞ → 1   (higher is better)" },
    interp: [
      { text: "Better than R² for model selection with many features", type: "good" },
      { text: "Still assumes a linear relationship", type: "bad" },
    ],
  },
  {
    id: "mape",
    cat: "regression",
    name: "MAPE",
    alias: "Mean Absolute Percentage Error",
    formula: "MAPE = (100/N) · Σ |y_i − ŷ_i| / |y_i|   [%]",
    desc: "Expresses errors as a percentage of the actual values. Easy to communicate to non-technical stakeholders. Undefined when any true value is zero; inflated when true values are small.",
    range: { min: 0, max: null, good: 0.1, label: "0% → ∞   (lower is better)" },
    interp: [
      { text: "< 5% → excellent", type: "good" },
      { text: "Undefined if any y_i = 0", type: "bad" },
    ],
  },
  {
    id: "smape",
    cat: "regression",
    name: "sMAPE",
    alias: "Symmetric MAPE",
    formula: "sMAPE = (100/N) · Σ |y_i − ŷ_i| / ((|y_i| + |ŷ_i|) / 2)   [%]",
    desc: "Symmetric version of MAPE that treats over- and under-predictions equally. Still problematic when both y and ŷ are near zero, but avoids the asymmetry of plain MAPE.",
    range: { min: 0, max: 200, good: 0.1, label: "0% → 200%   (lower is better)" },
    interp: [
      { text: "Symmetric: over- and under-predictions penalized equally", type: "good" },
      { text: "Still problematic near zero values", type: "bad" },
    ],
  },
  {
    id: "huber-loss",
    cat: "regression",
    name: "Huber Loss",
    alias: "Smooth L1 Loss",
    formula: "L_δ = { ½(y−ŷ)²           if |y−ŷ| ≤ δ\n       { δ·(|y−ŷ|−δ/2)   otherwise",
    desc: "Combines the best of MAE and MSE: quadratic for small errors (smooth, easy to optimize), linear for large errors (robust to outliers). The transition threshold δ is a hyperparameter.",
    range: { min: 0, max: null, good: 0.1, label: "0 → ∞   (lower is better)" },
    interp: [
      { text: "Robust to outliers while remaining differentiable", type: "good" },
      { text: "Requires choosing δ", type: "info" },
    ],
  },

  /* ─────────────────────────────────────────────
     4. CLUSTERING
  ───────────────────────────────────────────── */
  {
    id: "silhouette",
    cat: "clustering",
    name: "Silhouette Score",
    alias: "Silhouette Coefficient",
    formula: "s(i) = (b(i) − a(i)) / max(a(i), b(i))\n\na(i) = mean intra-cluster distance\nb(i) = mean nearest-cluster distance\nScore = mean s(i)",
    desc: "Measures how well each sample fits in its own cluster vs the nearest other cluster. +1 = perfect separation; 0 = on boundary; −1 = misclassified.",
    range: { min: -1, max: 1, good: 0.75, label: "−1 → +1   (higher is better)" },
    interp: [
      { text: "Score > 0.5 → reasonable cluster structure", type: "good" },
      { text: "Score < 0 → samples likely in wrong clusters", type: "bad" },
    ],
  },
  {
    id: "davies-bouldin",
    cat: "clustering",
    name: "Davies-Bouldin Index",
    alias: "DB Index, DBI",
    formula: "DB = (1/K) · Σ_k max_{j≠k} (σ_k + σ_j) / d(c_k, c_j)\n\nσ = avg dist to centroid  d = centroid dist",
    desc: "Measures cluster compactness and separation. Lower is better. 0 = perfect clustering. Sensitive to cluster shape; favours convex, well-separated clusters.",
    range: { min: 0, max: null, good: 0.3, label: "0 → ∞   (lower is better)" },
    interp: [
      { text: "0 → perfect clustering", type: "good" },
      { text: "No upper bound — depends on data scale", type: "info" },
    ],
  },
  {
    id: "calinski-harabasz",
    cat: "clustering",
    name: "Calinski-Harabasz Index",
    alias: "CHI, Variance Ratio Criterion",
    formula: "CH = [Tr(B) / (K−1)] / [Tr(W) / (N−K)]\n\nB = between-cluster scatter\nW = within-cluster scatter",
    desc: "Ratio of between-cluster variance to within-cluster variance, normalized by the degrees of freedom. Higher is better: more separated and more compact clusters.",
    range: { min: 0, max: null, good: 0.8, label: "0 → ∞   (higher is better)" },
    interp: [
      { text: "Higher → better-defined clusters", type: "good" },
      { text: "No clear threshold — compare across K values", type: "info" },
    ],
  },
  {
    id: "ari",
    cat: "clustering",
    name: "Adjusted Rand Index",
    alias: "ARI, Corrected Rand Index",
    formula: "ARI = (RI − E[RI]) / (max(RI) − E[RI])\n\nRange: −1 → +1 (0 = random, 1 = perfect)",
    desc: "Compares two clusterings (or a clustering to ground-truth labels) by counting pairs of samples. Adjusted for chance: a random labeling gets 0. Used when ground truth labels exist.",
    range: { min: -1, max: 1, good: 0.75, label: "−1 → +1   (higher is better)" },
    interp: [
      { text: "ARI > 0.9 → near-perfect match to ground truth", type: "good" },
      { text: "ARI = 0 → random labels", type: "bad" },
    ],
  },
  {
    id: "nmi",
    cat: "clustering",
    name: "Normalized Mutual Information",
    alias: "NMI, Normalized MI",
    formula: "NMI(U, V) = 2 · I(U;V) / (H(U) + H(V))\n\nI = mutual information  H = entropy\nRange: 0 → 1",
    desc: "Measures the amount of information shared between the predicted clustering and the ground-truth labels, normalized to [0, 1]. 1 = perfect agreement; 0 = independent.",
    range: { min: 0, max: 1, good: 0.75, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "Symmetric and normalized", type: "good" },
      { text: "Not adjusted for chance — use AMI for fair comparison", type: "info" },
    ],
  },
  {
    id: "inertia",
    cat: "clustering",
    name: "Inertia",
    alias: "Within-Cluster Sum of Squares, WCSS",
    formula: "Inertia = Σ_k Σ_{x ∈ C_k} ‖x − μ_k‖²\n\nμ_k = centroid of cluster k",
    desc: "The sum of squared distances from each point to its assigned cluster centroid. Used in the 'elbow method' to choose the optimal number of clusters K. Lower is better, but always decreases as K increases.",
    range: { min: 0, max: null, good: 0.2, label: "0 → ∞   (lower is better)" },
    interp: [
      { text: "Decreases monotonically with K — use elbow method", type: "info" },
      { text: "Not meaningful across different datasets (not normalized)", type: "bad" },
    ],
  },

  /* ─────────────────────────────────────────────
     5. RANKING & RECOMMENDATION
  ───────────────────────────────────────────── */
  {
    id: "mrr",
    cat: "ranking",
    name: "Mean Reciprocal Rank",
    alias: "MRR",
    formula: "MRR = (1/|Q|) · Σ_q 1 / rank_q\n\nrank_q = position of first relevant result for query q",
    desc: "Averages the reciprocal of the rank of the first relevant item across all queries. If the first correct answer is always at position 1, MRR = 1. Used in question answering and information retrieval.",
    range: { min: 0, max: 1, good: 0.75, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "High MRR → first relevant result appears early", type: "good" },
      { text: "Only cares about the FIRST relevant result", type: "info" },
    ],
  },
  {
    id: "map",
    cat: "ranking",
    name: "Mean Average Precision",
    alias: "MAP, mAP",
    formula: "AP_q = Σ_k P@k · rel(k) / R_q\nMAP = (1/|Q|) · Σ_q AP_q\n\nrel(k)=1 if item k is relevant, R_q=total relevant",
    desc: "Averages the precision at each position where a relevant item is found, then averages over all queries. Considers the full ranked list rather than just the first hit.",
    range: { min: 0, max: 1, good: 0.75, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "Sensitive to the ordering of all relevant results", type: "good" },
      { text: "Expensive to compute for large result lists", type: "info" },
    ],
    usage: "mAP (mean Average Precision) is also widely used in object detection tasks. Not directly applicable to AMCShield's classification setting.",
  },
  {
    id: "ndcg",
    cat: "ranking",
    name: "NDCG",
    alias: "Normalized Discounted Cumulative Gain, nDCG",
    formula: "DCG_p = Σ_{i=1}^{p} rel_i / log₂(i+1)\nNDCG_p = DCG_p / IDCG_p\n\nIDCG = ideal (perfect) DCG",
    desc: "Measures the quality of a ranked list by discounting the value of relevant items that appear lower in the list (log discount). Normalized to [0,1] by dividing by the ideal ranking. The standard metric for search engine quality.",
    range: { min: 0, max: 1, good: 0.8, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "Handles graded relevance (not just binary)", type: "good" },
      { text: "Requires a relevance-graded gold standard", type: "info" },
    ],
  },
  {
    id: "precision-at-k",
    cat: "ranking",
    name: "Precision@K and Recall@K",
    alias: "P@K, R@K, top-K precision",
    formula: "P@K = (relevant items in top K) / K\nR@K = (relevant items in top K) / (total relevant)",
    desc: "Precision@K measures what fraction of the top-K results are relevant. Recall@K measures what fraction of all relevant items appear in the top K. Used in recommendation systems and search.",
    range: { min: 0, max: 1, good: 0.75, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "Direct measure of top-K recommendation quality", type: "good" },
      { text: "Precision@K decreases as K increases (more noise)", type: "info" },
    ],
    usage: "In AMCShield, top-1 accuracy is essentially P@1. Top-5 accuracy could be computed if the report needs it (model returns a ranked list of modulation classes).",
  },
  {
    id: "hit-rate",
    cat: "ranking",
    name: "Hit Rate",
    alias: "HR, Recall@K (binary relevance)",
    formula: "HR@K = (users with ≥1 relevant item in top K) / (all users)",
    desc: "The fraction of queries where at least one relevant item appears in the top-K results. Binary: either the user finds something relevant or not. Common in recommendation systems.",
    range: { min: 0, max: 1, good: 0.8, label: "0 → 1   (higher is better)" },
    interp: [
      { text: "Easy to interpret: 'did we help the user?'", type: "good" },
      { text: "Does not capture item ranking quality within the top K", type: "bad" },
    ],
  },

  /* ─────────────────────────────────────────────
     6. SYSTEM & PRODUCTION
  ───────────────────────────────────────────── */
  {
    id: "inference-latency",
    cat: "system",
    name: "Inference Latency",
    alias: "Response Time, Prediction Latency, p99 Latency",
    formula: "Latency = time from request received → response sent\n\nCommonly report: mean, p50, p95, p99 (ms)",
    desc: "How long it takes the model to return a prediction for a single input. Critical for real-time applications. Report p99 (worst 1%) not just the mean, since tail latency matters for user experience.",
    interp: [
      { text: "< 10 ms → real-time capable", type: "good" },
      { text: "Report p99, not just mean latency", type: "info" },
      { text: "GPU inference << CPU inference for AMCShield", type: "info" },
    ],
    usage: "The FastAPI backend wraps AMCShield's classifier. Each HTTP request triggers a forward pass. Profiling with time.perf_counter() around the model call will give per-batch latency. Not yet benchmarked.",
  },
  {
    id: "throughput",
    cat: "system",
    name: "Throughput",
    alias: "Predictions per Second, QPS, RPS",
    formula: "Throughput = N_predictions / wall_clock_time  [pred/s]",
    desc: "How many predictions the system can process per second under sustained load. The inverse of latency for batch processing. GPU parallelism makes throughput much higher than 1/latency suggests.",
    interp: [
      { text: "High throughput → efficient use of hardware", type: "good" },
      { text: "Throughput drops under CPU bottlenecks (data loading)", type: "bad" },
    ],
    usage: "The AMCShield test loader (batch=128, 2,996 batches) gives ~383k predictions per evaluation pass. Throughput on RTX 3050 is not yet formally measured.",
  },
  {
    id: "cpu-gpu-util",
    cat: "system",
    name: "CPU / GPU Utilization",
    alias: "GPU Utilization, CPU Load",
    formula: "GPU Util = time GPU cores active / total time  [%]\n\nMonitor: nvidia-smi  or  torch.cuda.memory_summary()",
    desc: "The fraction of time the processor is performing useful computation. Low GPU utilization often means the data pipeline (CPU data loading) is the bottleneck, not the model.",
    interp: [
      { text: "GPU util near 100% → compute-bound (ideal)", type: "good" },
      { text: "GPU util < 50% → data-loading bottleneck", type: "bad" },
      { text: "Use num_workers > 0 in DataLoader to fix I/O bottleneck", type: "info" },
    ],
    usage: "AMCShield DataLoader uses num_workers from configs/config.yaml. On the RTX 3050 laptop, pin_memory=True and num_workers=4 is a good starting point.",
  },
  {
    id: "memory-footprint",
    cat: "system",
    name: "Memory / RAM Footprint",
    alias: "RAM Usage, GPU VRAM, Peak Memory",
    formula: "VRAM = model params (fp32) + activations + gradients + optimizer state\n\nparam bytes = N_params × 4  (fp32)",
    desc: "Total RAM or GPU VRAM consumed during training or inference. Important for deciding whether a model fits on the target hardware. Mixed precision (fp16) roughly halves VRAM.",
    interp: [
      { text: "BaselineCNN: 1.09 M params ≈ 4.2 MB fp32", type: "info" },
      { text: "RTX 3050 laptop: 4 GB VRAM total", type: "info" },
      { text: "During training, batch × 2 × 1024 × 4 bytes = activations per layer", type: "info" },
    ],
    usage: "AMCShield models are small by modern standards. Training a single batch of 128 × 2 × 1024 fp32 = ~1 MB input tensor. No OOM issues reported.",
  },
  {
    id: "model-size",
    cat: "system",
    name: "Model Size on Disk",
    alias: "Checkpoint Size, .pt File Size",
    formula: "File size = params + buffers + optimizer state (if full checkpoint)\n\nInference-only: strip optimizer state → smaller file",
    desc: "The file size of the saved model checkpoint. Affects deployment (edge devices, download time). The full checkpoint includes optimizer state; a 'inference-only' export is smaller.",
    interp: [
      { text: "BaselineCNN full checkpoint ≈ 13-15 MB", type: "info" },
      { text: "Inference-only export (state_dict only) ≈ 4-5 MB", type: "info" },
      { text: "Very small by industry standards", type: "good" },
    ],
    usage: "Checkpoints live in checkpoints/ directory. Use torch.save(model.state_dict(), 'inference.pt') to strip optimizer state for a deployment build.",
  },
  {
    id: "data-drift",
    cat: "system",
    name: "Data Drift & Concept Drift Rate",
    alias: "Dataset Shift, Covariate Shift, Label Shift",
    formula: "Data drift:    P(X) changes over time\nConcept drift: P(Y|X) changes over time\n\nDetect: KS test, PSI, MMD on feature distributions",
    desc: "Data drift occurs when the statistical properties of the input distribution change between training and production. Concept drift is when the relationship between inputs and labels changes. Both cause model performance to degrade silently.",
    interp: [
      { text: "Monitor input distributions continuously in production", type: "good" },
      { text: "Models degrade silently — scheduled re-evaluation is essential", type: "bad" },
      { text: "Real-world RF environments have strong temporal drift (interference, hardware aging)", type: "info" },
    ],
    usage: "RadioML 2018.01A is static — no drift during training. In a real SDR deployment, the RF channel characteristics and interference environment would change, requiring drift monitoring. Currently out of scope (listed as future work in AMCShield).",
  },
];

/* ══════════════════════════════════════════════════════════════════
   RENDER
══════════════════════════════════════════════════════════════════ */

const grid       = document.getElementById("cardsGrid");
const emptyState = document.getElementById("emptyState");
const tabStrip   = document.getElementById("tabStrip");
const searchInput= document.getElementById("metricSearch");
const totalEl    = document.getElementById("totalMetrics");
const activeCatEl= document.getElementById("activeCategory");

let activeCat  = "all";
let searchQuery= "";

function catColor(cat) { return CATEGORIES[cat]?.color ?? "#aaa"; }

function buildRangeBar(metric) {
  if (!metric.range) return "";
  const { min, max, good, label } = metric.range;
  const pct = typeof good === "number" ? Math.min(1, Math.max(0, good)) * 100 : 60;
  return `
    <div class="range-bar-wrap">
      <div class="range-label">
        <span>${label || "Range"}</span>
      </div>
      <div class="range-track">
        <div class="range-fill" data-w="${pct}%"></div>
        <div class="range-marker" data-left="${pct}%"></div>
      </div>
    </div>`;
}

function buildInterpPills(metric) {
  if (!metric.interp?.length) return "";
  const pills = metric.interp.map(p =>
    `<span class="interp-pill ${p.type}">${p.text}</span>`
  ).join("");
  return `<div class="interp-row">${pills}</div>`;
}

function buildUsageNote(metric) {
  if (!metric.usage) return "";
  return `<div class="usage-note"><span class="usage-icon">◈</span><span><strong>AMCShield:</strong> ${metric.usage}</span></div>`;
}

function renderCard(metric, index) {
  const color = catColor(metric.cat);
  const catLabel = CATEGORIES[metric.cat]?.label?.replace(/^[①②③④⑤⑥] /, "") ?? metric.cat;

  const formula = metric.formula
    ? `<div class="formula-box">${escHtml(metric.formula)}</div>`
    : "";

  return `
    <article class="metric-card"
             data-accent="${color}"
             data-cat="${metric.cat}"
             data-name="${metric.name.toLowerCase()}"
             data-alias="${(metric.alias||"").toLowerCase()}"
             data-id="${metric.id}">
      <div class="card-top">
        <span class="card-badge" data-badge="${color}">${catLabel}</span>
        <span class="card-index">#${String(index).padStart(2,"0")}</span>
      </div>
      <div class="card-name">${escHtml(metric.name)}</div>
      ${metric.alias ? `<div class="card-alias">${escHtml(metric.alias)}</div>` : ""}
      <div class="card-desc">${escHtml(metric.desc)}</div>
      ${formula}
      ${buildRangeBar(metric)}
      ${buildInterpPills(metric)}
      ${buildUsageNote(metric)}
    </article>`;
}

function renderSectionDivider(cat) {
  const color = catColor(cat);
  const label = CATEGORIES[cat]?.label ?? cat;
  return `
    <div class="section-divider" data-cat="${cat}">
      <div class="section-divider-line"></div>
      <div class="section-divider-label" data-divider="${color}">
        <span class="cat-dot" data-dot="${color}"></span>
        ${escHtml(label)}
      </div>
      <div class="section-divider-line"></div>
    </div>`;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g,"&amp;")
    .replace(/</g,"&lt;")
    .replace(/>/g,"&gt;");
}

function filterMetrics() {
  const q = searchQuery.trim().toLowerCase();
  return METRICS.filter(m => {
    if (activeCat !== "all" && m.cat !== activeCat) return false;
    if (!q) return true;
    return (
      m.name.toLowerCase().includes(q) ||
      (m.alias||"").toLowerCase().includes(q) ||
      m.desc.toLowerCase().includes(q) ||
      m.id.includes(q) ||
      m.cat.includes(q)
    );
  });
}

/* CSP-safe dynamic styling: markup carries data-* values, CSSOM applies them. */
function applyDynamicStyles(root) {
  root.querySelectorAll("[data-w]").forEach(e => { e.style.width = e.dataset.w; });
  root.querySelectorAll("[data-left]").forEach(e => { e.style.left = e.dataset.left; });
  root.querySelectorAll("[data-accent]").forEach(e => { e.style.setProperty("--card-accent", e.dataset.accent); });
  root.querySelectorAll("[data-badge]").forEach(e => {
    const c = e.dataset.badge;
    e.style.color = c; e.style.borderColor = `${c}33`; e.style.background = `${c}11`;
  });
  root.querySelectorAll("[data-divider]").forEach(e => {
    const c = e.dataset.divider;
    e.style.color = c; e.style.borderColor = `${c}33`;
  });
  root.querySelectorAll("[data-dot]").forEach(e => { e.style.background = e.dataset.dot; });
}

function render() {
  const filtered = filterMetrics();
  grid.innerHTML = "";

  if (!filtered.length) {
    emptyState.classList.remove("hidden");
    totalEl.textContent = "0";
    return;
  }
  emptyState.classList.add("hidden");
  totalEl.textContent = filtered.length;

  // Update active category display
  if (activeCat === "all") {
    activeCatEl.querySelector("strong").textContent = "All";
    activeCatEl.style.setProperty("--cat-active-color", "#20c9d8");
  } else {
    activeCatEl.querySelector("strong").textContent =
      CATEGORIES[activeCat]?.label?.replace(/^[①②③④⑤⑥] /, "").split("(")[0].trim() ?? activeCat;
  }

  // Group by category preserving CATEGORIES order
  const catOrder = Object.keys(CATEGORIES);
  const grouped  = {};
  for (const m of filtered) {
    if (!grouped[m.cat]) grouped[m.cat] = [];
    grouped[m.cat].push(m);
  }

  const showDividers = (activeCat === "all" && !searchQuery.trim());
  let globalIdx = 1;
  let html = "";

  for (const cat of catOrder) {
    const items = grouped[cat];
    if (!items?.length) continue;
    if (showDividers) html += renderSectionDivider(cat);
    for (const m of items) {
      html += renderCard(m, globalIdx++);
    }
  }
  grid.innerHTML = html;
  applyDynamicStyles(grid);
}

/* ── TABS ── */
tabStrip.addEventListener("click", e => {
  const btn = e.target.closest(".tab");
  if (!btn) return;
  document.querySelectorAll(".tab").forEach(t => {
    t.classList.remove("active"); t.setAttribute("aria-selected","false");
  });
  btn.classList.add("active"); btn.setAttribute("aria-selected","true");
  activeCat = btn.dataset.cat;
  render();
});

/* ── SEARCH ── */
let searchDebounce;
searchInput.addEventListener("input", () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => {
    searchQuery = searchInput.value;
    // Switch to 'all' tab when searching so user sees results across categories
    if (searchQuery.trim()) {
      document.querySelectorAll(".tab").forEach(t => {
        const isAll = t.dataset.cat === "all";
        t.classList.toggle("active", isAll);
        t.setAttribute("aria-selected", String(isAll));
      });
      activeCat = "all";
    }
    render();
  }, 180);
});

/* ── SIDEBAR / AUTH (mirror dashboard.js pattern) ── */
function checkApiStatus() {
  const el = document.getElementById("apiStatus");
  const txt = document.getElementById("apiStatusText");
  const det = document.getElementById("apiStatusDetail");
  fetch("/api/health", { credentials: "include" })
    .then(r => {
      if (r.ok) {
        el.className = "api-status online";
        txt.textContent = "API Online";
        det.textContent = "Connected";
      } else { throw new Error("non-200"); }
    })
    .catch(() => {
      el.className = "api-status offline";
      txt.textContent = "API Offline";
      det.textContent = "Cannot connect";
    });
}

function setupSidebar() {
  const btn     = document.getElementById("menuButton");
  const sidebar = document.getElementById("sidebar");
  const overlay = document.getElementById("mobileOverlay");
  if (!btn) return;
  btn.addEventListener("click", () => {
    sidebar.classList.toggle("open");
    overlay.classList.toggle("show");
  });
  overlay.addEventListener("click", () => {
    sidebar.classList.remove("open");
    overlay.classList.remove("show");
  });
}

function setupProfile() {
  try {
    const user = JSON.parse(sessionStorage.getItem("amcshield_user") || "{}");
    if (user.username) {
      document.getElementById("userName").textContent = user.username;
      document.getElementById("userRole").textContent = user.role || "Researcher";
      const av = document.getElementById("profileAvatar");
      if (av) av.textContent = user.username[0].toUpperCase();
    }
  } catch {}
}

document.getElementById("logoutButton")?.addEventListener("click", async () => {
  try {
    await AMCShieldSecurity.fetch("/api/auth/logout", { method: "POST" });
  } finally {
    sessionStorage.clear();
    window.location.href = "/";
  }
});

/* ── BOOT ── */
setupSidebar();
setupProfile();
checkApiStatus();
render(); // initial full render
