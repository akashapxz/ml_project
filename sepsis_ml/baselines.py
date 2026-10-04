"""
Benchmark Clinical & Statistical Baselines Module.
Implements:
1. Rule-Based Clinical Scorers: qSOFA, SIRS, Modified Shock Index.
2. Statistical Classifiers: L2-penalized Class-Weighted Logistic Regression,
   k-Nearest Neighbors (k=15), and Support Vector Machine (Linear / RBF).
Evaluates all models on held-out test splits, optimizes utility thresholds on validation data,
and logs AUROC, AUPRC, U_norm, F1, Sensitivity, Specificity, and Brier Score.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sepsis_ml.config import MODELS_DIR, ProjectConfig, TABLES_DIR, TARGET_COL
from sepsis_ml.features import FeatureEngineer
from sepsis_ml.logger import get_logger
from sepsis_ml.metrics import (
    compute_comprehensive_metrics,
    compute_physionet_utility,
    optimize_utility_threshold,
)
from sepsis_ml.preprocessing import ClinicalPreprocessor
from sepsis_ml.reproducibility import seed_everything

logger = get_logger("baselines")


class RuleBasedClinicalScorer:
    """Evaluates standardized bedside clinical rules without training parameters."""

    @staticmethod
    def predict_qsofa(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """qSOFA score: RR >= 22 (+1), SBP <= 100 (+1). Score >= 2 is positive."""
        rr = (df["Resp"] >= 22.0).astype(int)
        sbp = (df["SBP"] <= 100.0).astype(int)
        score = (rr + sbp).values
        # Softmax / Sigmoid pseudo-probability
        prob = 1.0 / (1.0 + np.exp(-(score - 1.0) * 2.0))
        pred = (score >= 2).astype(int)
        return prob, pred

    @staticmethod
    def predict_sirs(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """SIRS criteria (Temp, HR, RR, WBC). Score >= 2 is positive."""
        temp = ((df["Temp"] > 38.0) | (df["Temp"] < 36.0)).astype(int)
        hr = (df["HR"] > 90.0).astype(int)
        resp = (df["Resp"] > 20.0).astype(int)
        wbc = ((df["WBC"] > 12.0) | (df["WBC"] < 4.0)).astype(int)
        score = (temp + hr + resp + wbc).values
        prob = score / 4.0
        pred = (score >= 2).astype(int)
        return prob, pred

    @staticmethod
    def predict_msi(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Modified Shock Index (HR / MAP). MSI >= 1.3 is high risk."""
        msi = (df["HR"] / np.clip(df["MAP"], a_min=20.0, a_max=200.0)).values
        prob = 1.0 / (1.0 + np.exp(-3.0 * (msi - 1.1)))
        pred = (msi >= 1.3).astype(int)
        return prob, pred


def train_and_evaluate_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Optional[ProjectConfig] = None,
    feature_tier: str = "F3",
) -> pd.DataFrame:
    """
    Trains and evaluates all clinical rules and statistical ML baselines:
    1. qSOFA Clinical Rule
    2. SIRS Clinical Rule
    3. Modified Shock Index (MSI)
    4. L2 Class-Weighted Logistic Regression
    5. k-Nearest Neighbors (k=15)
    6. Calibrated Linear Support Vector Machine (loss='log_loss', L2 penalty)
    
    Returns:
        pd.DataFrame summarizing benchmark performance across validation and test sets.
    """
    config = cfg or ProjectConfig()
    seed_everything(config.seed)
    
    # 1. Feature Extraction on Specified Tier
    prep = ClinicalPreprocessor()
    train_prep = prep.fit_transform(train_df)
    val_prep = prep.transform(val_df)
    test_prep = prep.transform(test_df)
    
    fe = FeatureEngineer()
    train_f, feat_cols = fe.build_feature_tier(train_prep, tier=feature_tier)
    val_f, _ = fe.build_feature_tier(val_prep, tier=feature_tier)
    test_f, _ = fe.build_feature_tier(test_prep, tier=feature_tier)
    
    X_train = train_f[feat_cols].values
    y_train = train_f[TARGET_COL].values
    
    X_val = val_f[feat_cols].values
    y_val = val_f[TARGET_COL].values
    
    X_test = test_f[feat_cols].values
    y_test = test_f[TARGET_COL].values
    
    results = []
    
    # ----------------------------------------------------
    # Model 1-3: Rule-Based Clinical Scorers (No Training Required)
    # ----------------------------------------------------
    clinical_rules = {
        "Clinical_qSOFA": RuleBasedClinicalScorer.predict_qsofa,
        "Clinical_SIRS": RuleBasedClinicalScorer.predict_sirs,
        "Clinical_MSI": RuleBasedClinicalScorer.predict_msi,
    }
    
    for rule_name, rule_fn in clinical_rules.items():
        logger.info(f"Evaluating clinical rule: {rule_name}...")
        val_prob, _ = rule_fn(val_f)
        test_prob, _ = rule_fn(test_f)
        
        # Optimize threshold on validation data
        val_eval_df = val_f[["patient_id", "ICULOS", TARGET_COL]].copy()
        val_eval_df["pred_prob"] = val_prob
        best_th, val_u, _ = optimize_utility_threshold(val_eval_df, prob_col="pred_prob", threshold_steps=50)
        
        # Evaluate on Test set using tuned threshold
        test_metrics = compute_comprehensive_metrics(
            y_true=y_test,
            y_prob=test_prob,
            threshold=best_th,
            patient_df=test_f[["patient_id", "ICULOS", TARGET_COL]],
        )
        
        results.append({
            "model": rule_name,
            "feature_set": "Clinical_Rule",
            "val_utility_norm": round(val_u, 4),
            "test_utility_norm": test_metrics.get("utility_norm", 0.0),
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_f1": test_metrics["f1"],
            "test_sensitivity": test_metrics["sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_brier": test_metrics["brier_score"],
            "optimal_threshold": round(best_th, 4),
        })

    # ----------------------------------------------------
    # Model 4-6: Statistical Machine Learning Baselines
    # ----------------------------------------------------
    ml_models: Dict[str, Any] = {
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=1000,
                random_state=config.seed,
            )),
        ]),
        "kNN_k15": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                n_jobs=-1,
            )),
        ]),
        "Linear_SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(
                loss="log_loss",  # Gives calibrated probabilities (logistic loss)
                penalty="l2",
                alpha=0.01,
                class_weight="balanced",
                max_iter=1000,
                random_state=config.seed,
            )),
        ]),
    }
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for model_name, pipeline in ml_models.items():
        logger.info(f"Training statistical baseline: {model_name} on {X_train.shape[1]} features...")
        pipeline.fit(X_train, y_train)
        
        # Save model pipeline
        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, model_path)
        
        # Predict probabilities
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        
        # Optimize decision threshold on Validation partition
        val_eval_df = val_f[["patient_id", "ICULOS", TARGET_COL]].copy()
        val_eval_df["pred_prob"] = val_prob
        best_th, val_u, _ = optimize_utility_threshold(val_eval_df, prob_col="pred_prob", threshold_steps=50)
        
        # Evaluate on held-out Test partition
        test_metrics = compute_comprehensive_metrics(
            y_true=y_test,
            y_prob=test_prob,
            threshold=best_th,
            patient_df=test_f[["patient_id", "ICULOS", TARGET_COL]],
        )
        
        results.append({
            "model": model_name,
            "feature_set": feature_tier,
            "val_utility_norm": round(val_u, 4),
            "test_utility_norm": test_metrics.get("utility_norm", 0.0),
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_f1": test_metrics["f1"],
            "test_sensitivity": test_metrics["sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_brier": test_metrics["brier_score"],
            "optimal_threshold": round(best_th, 4),
        })
        
    benchmark_df = pd.DataFrame(results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(TABLES_DIR / "benchmark_baselines.csv", index=False)
    
    logger.info("Baseline benchmarks completed successfully:")
    for _, row in benchmark_df.iterrows():
        logger.info(
            f"  [{row['model']}] AUROC={row['test_auroc']:.4f} | AUPRC={row['test_auprc']:.4f} | "
            f"U_norm={row['test_utility_norm']:.4f} | Sens={row['test_sensitivity']:.2f} | Spec={row['test_specificity']:.2f} (tau*={row['optimal_threshold']:.3f})"
        )
        
    return benchmark_df
