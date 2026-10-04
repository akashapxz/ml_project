"""
Tree Ensembles & Cross-Validation Ablation Module.
Implements:
1. Patient-grouped StratifiedGroupKFold cross-validation (5 folds).
2. Random Forest, XGBoost, and HistGradientBoosting classifiers with class-imbalance compensation.
3. Full feature ablation experiment across tiers F1, F2, F3, and F4.
4. Out-of-fold and test evaluation logging to outputs/tables/ablation_feature_tiers.csv.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

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

logger = get_logger("tree_ensembles")


def get_tree_models(
    scale_pos_weight: float = 30.0,
    random_state: int = 42
) -> Dict[str, Any]:
    """Factory creating configured tree ensemble classifiers with class weighting."""
    return {
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=120,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "Hist_Gradient_Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def run_cross_validated_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Optional[ProjectConfig] = None,
    n_splits: int = 5,
    tiers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Executes a 5-fold StratifiedGroupKFold cross-validation across models and feature tiers.
    Evaluates out-of-fold and held-out test discrimination, calibration, and utility.
    """
    config = cfg or ProjectConfig()
    seed_everything(config.seed)
    feature_tiers = tiers or ["F1", "F2", "F3", "F4"]
    
    # 1. Preprocess partitions causally
    prep = ClinicalPreprocessor()
    train_prep = prep.fit_transform(train_df)
    val_prep = prep.transform(val_df)
    test_prep = prep.transform(test_df)
    
    # Merge train and val for cross-validation on all development patients
    dev_prep = pd.concat([train_prep, val_prep], ignore_index=True)
    
    # Determine patient-level sepsis label for StratifiedGroupKFold
    patient_status = dev_prep.groupby("patient_id", sort=False)[TARGET_COL].max()
    patient_ids = dev_prep["patient_id"].values
    y_dev = dev_prep[TARGET_COL].values
    
    # Calculate positive scale weight
    n_pos = int(y_dev.sum())
    n_neg = len(y_dev) - n_pos
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    
    fe = FeatureEngineer()
    all_results = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    # Target for splitting: map each row to its patient-level status
    patient_status_map = dev_prep["patient_id"].map(patient_status).values
    
    for tier in feature_tiers:
        logger.info(f"=== Running Feature Tier Ablation: {tier} ===")
        dev_f, feat_cols = fe.build_feature_tier(dev_prep, tier=tier)
        test_f, _ = fe.build_feature_tier(test_prep, tier=tier)
        
        X_dev = dev_f[feat_cols].values
        X_test = test_f[feat_cols].values
        y_test = test_f[TARGET_COL].values
        
        models_dict = get_tree_models(scale_pos_weight=scale_pos_weight, random_state=config.seed)
        
        for model_name, model in models_dict.items():
            logger.info(f"Cross-validating {model_name} on Tier {tier} ({len(feat_cols)} features)...")
            oof_probs = np.zeros(len(dev_f), dtype=float)
            fold_aurocs = []
            fold_auprcs = []
            
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dev, patient_status_map, groups=patient_ids)):
                X_tr_f, y_tr_f = X_dev[train_idx], y_dev[train_idx]
                X_va_f, y_va_f = X_dev[val_idx], y_dev[val_idx]
                
                # Fit clone
                clf = get_tree_models(scale_pos_weight=scale_pos_weight, random_state=config.seed + fold)[model_name]
                clf.fit(X_tr_f, y_tr_f)
                
                val_probs = clf.predict_proba(X_va_f)[:, 1]
                oof_probs[val_idx] = val_probs
                
            # Compute Out-of-Fold (OOF) discrimination
            oof_eval_df = dev_f[["patient_id", "ICULOS", TARGET_COL]].copy()
            oof_eval_df["pred_prob"] = oof_probs
            oof_th, oof_u, _ = optimize_utility_threshold(oof_eval_df, prob_col="pred_prob", threshold_steps=50)
            oof_metrics = compute_comprehensive_metrics(
                y_true=y_dev,
                y_prob=oof_probs,
                threshold=oof_th,
                patient_df=dev_f[["patient_id", "ICULOS", TARGET_COL]],
            )
            
            # Train final model on full development set
            final_clf = get_tree_models(scale_pos_weight=scale_pos_weight, random_state=config.seed)[model_name]
            final_clf.fit(X_dev, y_dev)
            
            # Save best performing models
            model_save_path = MODELS_DIR / f"{model_name}_{tier}.joblib"
            joblib.dump(final_clf, model_save_path)
            
            # Evaluate on held-out Test set
            test_probs = final_clf.predict_proba(X_test)[:, 1]
            test_metrics = compute_comprehensive_metrics(
                y_true=y_test,
                y_prob=test_probs,
                threshold=oof_th,
                patient_df=test_f[["patient_id", "ICULOS", TARGET_COL]],
            )
            
            result_row = {
                "model": model_name,
                "feature_tier": tier,
                "num_features": len(feat_cols),
                "oof_auroc": oof_metrics["auroc"],
                "oof_auprc": oof_metrics["auprc"],
                "oof_utility_norm": oof_metrics.get("utility_norm", 0.0),
                "oof_optimal_threshold": round(oof_th, 4),
                "test_auroc": test_metrics["auroc"],
                "test_auprc": test_metrics["auprc"],
                "test_utility_norm": test_metrics.get("utility_norm", 0.0),
                "test_f1": test_metrics["f1"],
                "test_sensitivity": test_metrics["sensitivity"],
                "test_specificity": test_metrics["specificity"],
                "test_brier": test_metrics["brier_score"],
            }
            all_results.append(result_row)
            logger.info(
                f"  [{model_name} - {tier}] OOF AUROC={oof_metrics['auroc']:.4f} | "
                f"Test AUROC={test_metrics['auroc']:.4f} | Test U_norm={test_metrics.get('utility_norm', 0.0):.4f} "
                f"(tau*={oof_th:.3f})"
            )
            
    ablation_df = pd.DataFrame(all_results)
    ablation_df.to_csv(TABLES_DIR / "ablation_feature_tiers.csv", index=False)
    ablation_df.to_csv(TABLES_DIR / "benchmark_tree_ensembles.csv", index=False)
    logger.info("Ablation experiment completed successfully and saved to outputs/tables/ablation_feature_tiers.csv")
    return ablation_df
