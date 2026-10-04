"""
Official Evaluation Metrics Engine.
Implements the exact PhysioNet / Computing in Cardiology Challenge 2019
Normalized Clinical Utility Metric (U_norm), AUROC, AUPRC, Sensitivity,
Specificity, F1-Score, Brier Score, and Utility-Maximizing Threshold Search.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from sepsis_ml.config import TARGET_COL
from sepsis_ml.logger import get_logger

logger = get_logger("metrics")

# Official PhysioNet 2019 Utility Parameters
DT_EARLY = -12
DT_OPTIMAL_START = -6
DT_OPTIMAL_END = 0
DT_LATE = 3

U_FP = -0.05
U_FN = -2.0
U_TP = 1.0


def compute_patient_utility_matrix(
    n_hours: int,
    is_septic: bool,
    tsepsis_idx: Optional[int]
) -> np.ndarray:
    """
    Computes the official Nx2 utility matrix for an individual patient trajectory.
    u_matrix[t, 0] = utility if predicted negative (pred = 0) at hour t.
    u_matrix[t, 1] = utility if predicted positive (pred = 1) at hour t.
    """
    u_matrix = np.zeros((n_hours, 2), dtype=np.float64)
    
    if not is_septic or tsepsis_idx is None:
        # Non-septic patient: true negative gives 0, false alarm gives -0.05
        u_matrix[:, 0] = 0.0
        u_matrix[:, 1] = U_FP
        return u_matrix
        
    for t in range(n_hours):
        # 1. Prior to warning window
        if t < tsepsis_idx + DT_EARLY:
            u_matrix[t, 0] = 0.0
            u_matrix[t, 1] = U_FP
        # 2. Ramp-up window [-12h, -6h]
        elif tsepsis_idx + DT_EARLY <= t < tsepsis_idx + DT_OPTIMAL_START:
            u_matrix[t, 0] = 0.0
            u_matrix[t, 1] = (t - (tsepsis_idx + DT_EARLY)) / float(DT_OPTIMAL_START - DT_EARLY)
        # 3. Prime early warning window [-6h, 0h]
        elif tsepsis_idx + DT_OPTIMAL_START <= t < tsepsis_idx + DT_OPTIMAL_END:
            u_matrix[t, 0] = 0.0
            u_matrix[t, 1] = U_TP
        # 4. Grace period [0h, +3h]
        elif tsepsis_idx + DT_OPTIMAL_END <= t < tsepsis_idx + DT_LATE:
            slope = (t - tsepsis_idx) / float(DT_LATE)
            u_matrix[t, 0] = U_FN * slope
            u_matrix[t, 1] = U_TP - slope
        # 5. Late post-onset period [> +3h]
        else:
            u_matrix[t, 0] = U_FN
            u_matrix[t, 1] = 0.0
            
    return u_matrix


def compute_physionet_utility(
    df: pd.DataFrame,
    pred_col: str = "pred_binary",
    target_col: str = TARGET_COL,
) -> Dict[str, float]:
    """
    Computes official Normalized Clinical Utility Score (U_norm) across a patient cohort.
    
    Args:
        df: DataFrame with 'patient_id', target_col, and pred_col
        pred_col: Column with binary predictions in {0, 1}
        target_col: Ground truth SepsisLabel column
        
    Returns:
        Dict with keys:
        - utility_norm: Normalized utility in [-inf, 1.0] (0.0 for all zeros, 1.0 for oracle)
        - utility_observed: Raw observed utility sum
        - utility_optimal: Oracle maximum utility sum
        - utility_no_predictions: All-zeros baseline utility sum
    """
    u_observed = 0.0
    u_optimal = 0.0
    u_no_predictions = 0.0
    
    for pid, group in df.groupby("patient_id", sort=False):
        n_hours = len(group)
        labels = group[target_col].values
        preds = group[pred_col].values.astype(int)
        
        is_septic = bool((labels == 1).any())
        if is_septic:
            first_pos = int(np.where(labels == 1)[0][0])
            tsepsis_idx = first_pos + 6  # Challenge ground truth: SepsisLabel starts 6h before onset
        else:
            tsepsis_idx = None
            
        u_matrix = compute_patient_utility_matrix(n_hours, is_septic, tsepsis_idx)
        
        # Observed utility
        u_observed += np.choose(preds, [u_matrix[:, 0], u_matrix[:, 1]]).sum()
        
        # Optimal oracle: picks best action at each hour
        u_optimal += u_matrix.max(axis=1).sum()
        
        # All-negative baseline: pred=0 at every hour
        u_no_predictions += u_matrix[:, 0].sum()
        
    denominator = u_optimal - u_no_predictions
    if denominator == 0.0:
        u_norm = 0.0
    else:
        u_norm = (u_observed - u_no_predictions) / denominator
        
    return {
        "utility_norm": float(u_norm),
        "utility_observed": float(u_observed),
        "utility_optimal": float(u_optimal),
        "utility_no_predictions": float(u_no_predictions),
    }


def optimize_utility_threshold(
    df: pd.DataFrame,
    prob_col: str = "pred_prob",
    target_col: str = TARGET_COL,
    threshold_steps: int = 100
) -> Tuple[float, float, pd.DataFrame]:
    """
    Finds the probability threshold tau* in [0.01, 0.99] that maximizes U_norm.
    
    Returns:
        Tuple of (optimal_threshold: float, max_utility: float, threshold_curve: pd.DataFrame)
    """
    thresholds = np.linspace(0.01, 0.99, threshold_steps)
    records = []
    
    temp_df = df.copy()
    probs = temp_df[prob_col].values
    
    best_thresh = 0.5
    best_u = -float("inf")
    
    for th in thresholds:
        temp_df["temp_pred"] = (probs >= th).astype(np.int8)
        res = compute_physionet_utility(temp_df, pred_col="temp_pred", target_col=target_col)
        u_norm = res["utility_norm"]
        
        records.append({
            "threshold": float(th),
            "utility_norm": u_norm,
            "utility_observed": res["utility_observed"]
        })
        
        if u_norm > best_u:
            best_u = u_norm
            best_thresh = float(th)
            
    curve_df = pd.DataFrame(records)
    logger.info(f"Optimal decision threshold: tau* = {best_thresh:.3f} (Max Utility = {best_u:.4f})")
    return best_thresh, best_u, curve_df


def compute_comprehensive_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
    patient_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Computes clinical classification metrics:
    AUROC, AUPRC, Sensitivity, Specificity, F1-Score, Brier Score, and PhysioNet Utility.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    y_pred_arr = (y_prob_arr >= threshold).astype(int)
    
    # 1. Discrimination (AUROC & AUPRC)
    try:
        auroc = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        auroc = 0.5
        
    try:
        auprc = float(average_precision_score(y_true_arr, y_prob_arr))
    except ValueError:
        auprc = float(np.mean(y_true_arr))
        
    # 2. Operating Point Metrics at threshold
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))
    
    metrics = {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "f1": round(f1, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "brier_score": round(brier, 4),
        "threshold": round(threshold, 4),
        "prevalence": round(float(np.mean(y_true_arr)), 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    
    # 3. PhysioNet Clinical Utility Score (if patient trajectory DataFrame provided)
    if patient_df is not None:
        eval_df = patient_df.copy()
        eval_df["eval_pred"] = y_pred_arr
        u_res = compute_physionet_utility(eval_df, pred_col="eval_pred")
        metrics["utility_norm"] = round(u_res["utility_norm"], 4)
        metrics["utility_observed"] = round(u_res["utility_observed"], 2)
        metrics["utility_optimal"] = round(u_res["utility_optimal"], 2)
        metrics["utility_no_predictions"] = round(u_res["utility_no_predictions"], 2)
        
    return metrics
