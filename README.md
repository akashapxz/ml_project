# Reliable Early Sepsis Prediction from Sparse and Non-Stationary ICU Time-Series
## Using Temporal Features and Ensemble Learning

**M.Tech Data Science (24DS601 – Machine Learning)**  
**Student:** Akash J P (Register No.: `CB.AI.P2DSC26033`)  
**Institution:** Amrita Vishwa Vidyapeetham  
**Benchmark:** PhysioNet / Computing in Cardiology Challenge 2019

---

### Project Overview
This project investigates whether engineered temporal features (instantaneous $F_1$, delta $F_2$, rolling statistics $F_3$, trends $F_4$) combined with ensemble machine learning (Stacking Classifier) improve early sepsis risk discrimination, probability calibration, and clinical early-warning utility while strictly eliminating temporal and patient data leakage.

### Directory Layout
```
├── data/
│   ├── raw/                # Untouched PhysioNet .psv files (training_setA, training_setB)
│   ├── interim/            # Transformed and validated patient cohorts
│   └── processed/          # Model-ready feature matrices (F1, F2, F3, F4)
├── notebooks/              # Sequentially numbered, fully reproducible research notebooks
├── sepsis_ml/              # Core Python ML research package
│   ├── __init__.py
│   ├── config.py           # Clinical covariates schema, ranges, paths, hyperparameters
│   ├── reproducibility.py  # Global deterministic seed enforcement
│   └── logger.py           # Structured experiment logger
├── experiments/            # Experiment run logs and configuration artifacts
│   ├── baseline/
│   ├── temporal_features/
│   ├── imbalance/
│   ├── horizons/
│   └── reliability/
├── outputs/
│   ├── figures/            # High-resolution publication plots (.png)
│   ├── tables/             # Benchmark results and audit summaries (.csv, .json)
│   ├── models/             # Serialized, frozen scikit-learn/xgboost pipelines (.joblib)
│   └── predictions/        # Out-of-fold and final test prediction sets
├── reports/                # Literature matrices, review presentations, final manuscript
├── tests/                  # Unit tests and leakage audit assertions
├── requirements.txt        # Frozen dependency specification
└── README.md
```

### Reproducibility Policy
- **Global Seed:** 42 across NumPy, Python standard library, Scikit-learn, and XGBoost.
- **Partitioning:** Stratified patient-level splitting (no row-wise leakage).
- **Causality:** At time $t$, only observations at or before $t$ are utilized.
