# Credit Default Prediction — Homework #2

## Task
Binary classification: predict **Credit Default = 1**.  
Dataset: `train.csv` (7,500 rows × 17 features), `test.csv` (2,500 rows).

## Target Metrics
| Metric | Requirement | Result (best library) |
|---|---|---|
| F1 (positive class) | **> 0.5** | **0.5299 ✓** |
| Gini (2×AUC−1) | — | **0.4691** |

---

## Repository Structure
```
├── HW_DT_GB_final.ipynb          # Main Jupyter notebook (all steps)
├── custom_gbm.py                 # Custom GBM implementation (stand-alone)
├── train.csv / test.csv          # Data
├── predictions.csv               # Test predictions (2500 rows)
├── model_comparison.csv          # All model metrics
├── outlier_summary.csv           # Outlier analysis table
├── README.md                     # This file
└── plots/
    ├── 01_class_imbalance.png
    ├── 02_distributions_by_class.png
    ├── 03_outliers_boxplot.png
    ├── 04_correlation_matrix.png
    ├── 05_feature_importance.png
    ├── 06_custom_gbm_learning_curve.png
    ├── 07_custom_gbm_feature_importance.png
    ├── 08_model_comparison.png
    └── 09_confusion_matrix.png
```

---

## Methodology

### 1. EDA
- Shape, dtypes, missing values, target distribution
- Class imbalance: **71.8% / 28.2%** — addressed with `class_weight='balanced'`
- Distributions split by target class (visual patterns for hypothesis building)

### 2. Outlier Analysis (all numeric features)
| Feature | Issue | Action |
|---|---|---|
| `Current Loan Amount` | Placeholder `99999999` (870 rows, 11.6%) | → NaN → log-transform |
| `Credit Score` | Values > 850 (outside FICO 300–850) | → NaN |
| `Maximum Open Credit` | Max = 1.3B, extreme right tail | → log-transform |
| `Number of Credit Problems` | 13.7% IQR outliers | kept (real events) |
| `Bankruptcies` | 11% IQR + collinear with NCP | removed |
| `Annual Income` | 4.8% IQR + right tail | → log-transform |
Full table: `outlier_summary.csv`

### 3. Feature Engineering
| Feature | Formula | Rationale |
|---|---|---|
| `DTI` | Monthly Debt / (Annual Income/12) | Debt burden — key credit risk indicator |
| `Credit_Utilization` | Current Credit Balance / Max Open Credit | Limit usage (>70% = stress) |
| `Has_Delinquent` | 1 if delinquency exists | Replaces 54%-NaN continuous feature |
| `Log_Income` | log(1 + Annual Income) | Normalises right-skewed income |
| `Log_Loan` | log(1 + Current Loan Amount) | Normalises loan amount |
| `Log_MaxCred` | log(1 + Maximum Open Credit) | Normalises credit limit |

### 4. Correlation Analysis + Feature Selection
**Threshold: |r| > 0.70** (Hair et al., 2019 — standard in credit scoring)
- Rationale: at |r| > 0.70 VIF > 2.04 — multicollinearity distorts LR coefficients and reduces DT interpretability
- Removed 6 collinear features (replaced by engineered equivalents)

**Feature Selection via RF Feature Importance** (threshold = 0.01):
- Features below importance threshold were removed if any
- Prevents noise features from degrading model quality

### 5. Models
| Model | F1 (tuned thr) | Gini | Train time |
|---|---|---|---|
| Logistic Regression | 0.4774 | 0.3006 | 0.06s |
| Decision Tree | 0.4975 | 0.4004 | 0.07s |
| Random Forest | 0.5299 | 0.4691 | 0.69s |
| LightGBM | 0.5198 | 0.4695 | 0.47s |
| **Custom GBM** ⭐ | **0.5290** | **0.4772** | **2.21s** |

### 6. Custom GBM (⭐ hand-written implementation)
Based on Friedman (2001) — «Greedy Function Approximation: A Gradient Boosting Machine»
- **Loss function**: Binary Cross-Entropy
- **Pseudo-residuals**: negative gradient = y - sigmoid(F(x))
- **Base learner**: DecisionTreeRegressor
- **Update rule**: F(x) ← F(x) + η·h(x)
- **Subsampling**: 80% of data per iteration (reduces variance)
- Achieves comparable F1/Gini to LightGBM with transparent, readable code

### 7. Imbalance Handling
- `class_weight='balanced'` for sklearn models
- `scale_pos_weight=2.55` for LightGBM
- **Threshold tuning**: optimal threshold via Precision-Recall curve (maximise F1)

---

## How to Run
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
python3 full_pipeline.py        # runs everything
# OR open HW_DT_GB_final.ipynb in Jupyter
```

## References
- Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*.
- Hair, J.F. et al. (2019). *Multivariate Data Analysis* (8th ed.).
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
