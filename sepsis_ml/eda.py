"""
Exploratory Data Analysis and Plot Generation Module.
Produces publication-quality diagnostic plots for Phase 3:
1. Missingness percentage by feature.
2. Missingness heatmap for representative patients.
3. Patient length-of-stay (ICULOS) distribution.
4. Sepsis class distribution (patient-level and hourly-level).
5. Feature distributions before preprocessing (vitals).
6. Example septic vs non-septic patient temporal trajectories.
7. Correlation matrix for continuous vital signs.
8. Per-feature observation frequency by hospital system.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sepsis_ml.config import (
    ALL_COLUMNS,
    DEMOGRAPHIC_VARIABLES,
    FIGURES_DIR,
    LAB_VARIABLES,
    RAW_DATA_DIR,
    TARGET_COL,
    VITAL_SIGNS,
)
from sepsis_ml.data_loader import iter_patient_files, load_patient_psv
from sepsis_ml.logger import get_logger

logger = get_logger("eda")


def generate_all_eda_plots(sample_size: int = 200, output_dir: Optional[Path] = None) -> List[Path]:
    """Generates all 8 diagnostic EDA plots specified in Golden Implementation Guide."""
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(iter_patient_files(RAW_DATA_DIR))[:sample_size]
    logger.info(f"Loading {len(files)} patient files for EDA visualization...")
    
    frames = [load_patient_psv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    
    saved_plots: List[Path] = []
    
    # -------------------------------------------------------------
    # Plot 1: Missingness Percentage by Feature
    # -------------------------------------------------------------
    logger.info("Generating Plot 1: Missingness Percentage by Feature...")
    missing_pct = df[ALL_COLUMNS[:-1]].isna().mean() * 100.0
    missing_pct = missing_pct.sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    colors = ["#2563eb" if col in VITAL_SIGNS else ("#059669" if col in LAB_VARIABLES else "#d97706") for col in missing_pct.index]
    bars = ax.barh(missing_pct.index, missing_pct.values, color=colors, alpha=0.85)
    ax.set_xlabel("Missingness Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Clinical Variables", fontsize=12, fontweight="bold")
    ax.set_title("PhysioNet 2019: Missingness Percentage Across 40 Clinical Covariates", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563eb", label="Vital Signs (8)"),
        Patch(facecolor="#059669", label="Laboratory Analytes (26)"),
        Patch(facecolor="#d97706", label="Demographics / Admin (6)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    p1 = out_dir / "01_missingness_percentage.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    saved_plots.append(p1)

    # -------------------------------------------------------------
    # Plot 2: Missingness Heatmap on Representative Patients
    # -------------------------------------------------------------
    logger.info("Generating Plot 2: Missingness Heatmap...")
    rep_patients = df["patient_id"].unique()[:4]
    sample_df = df[df["patient_id"].isin(rep_patients)].copy()
    
    # Sort by patient and ICULOS
    sample_df = sample_df.sort_values(by=["patient_id", "ICULOS"])
    matrix = sample_df[VITAL_SIGNS + LAB_VARIABLES[:12]].isna().astype(int).T
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(matrix, cmap=["#1e293b", "#f1f5f9"], cbar=False, ax=ax)
    ax.set_title("Clinical Measurement Availability (Dark = Measured, Light = Missing)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sequential Hourly ICU Timestamps across 4 Patients", fontsize=11)
    ax.set_ylabel("Clinical Variables", fontsize=11)
    plt.tight_layout()
    p2 = out_dir / "02_missingness_heatmap.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    saved_plots.append(p2)

    # -------------------------------------------------------------
    # Plot 3: Patient Length of Stay (ICULOS) Distribution
    # -------------------------------------------------------------
    logger.info("Generating Plot 3: Patient Length of Stay Distribution...")
    los_per_patient = df.groupby("patient_id")["ICULOS"].max()
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(los_per_patient, bins=35, kde=True, color="#4f46e5", ax=ax)
    ax.set_xlabel("ICU Length of Stay (Hours)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=11, fontweight="bold")
    ax.set_title(f"Distribution of ICU Stay Durations (Median: {los_per_patient.median():.1f} hrs, Min: {los_per_patient.min():.0f} hrs)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p3 = out_dir / "03_length_of_stay_distribution.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    saved_plots.append(p3)

    # -------------------------------------------------------------
    # Plot 4: Sepsis Class Imbalance Distribution
    # -------------------------------------------------------------
    logger.info("Generating Plot 4: Sepsis Class Imbalance...")
    patient_labels = df.groupby("patient_id")[TARGET_COL].max()
    patient_counts = patient_labels.value_counts()
    hourly_counts = df[TARGET_COL].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(["Non-Septic (0)", "Septic (1)"], [patient_counts.get(0, 0), patient_counts.get(1, 0)], color=["#3b82f6", "#ef4444"])
    ax1.set_title(f"Patient-Level Sepsis Incidence\n(Prevalence: {patient_labels.mean()*100:.1f}%)", fontweight="bold")
    ax1.set_ylabel("Patient Count")
    for i, count in enumerate([patient_counts.get(0, 0), patient_counts.get(1, 0)]):
        ax1.text(i, count + 2, f"{count}", ha="center", fontweight="bold")
        
    ax2.bar(["Non-Sepsis Hour (0)", "Sepsis Hour (1)"], [hourly_counts.get(0, 0), hourly_counts.get(1, 0)], color=["#60a5fa", "#dc2626"])
    ax2.set_title(f"Hourly Row-Level Imbalance\n(Prevalence: {df[TARGET_COL].mean()*100:.2f}%)", fontweight="bold")
    ax2.set_ylabel("Hourly Window Count")
    for i, count in enumerate([hourly_counts.get(0, 0), hourly_counts.get(1, 0)]):
        ax2.text(i, count + 50, f"{count}", ha="center", fontweight="bold")
        
    plt.tight_layout()
    p4 = out_dir / "04_sepsis_class_distribution.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    saved_plots.append(p4)

    # -------------------------------------------------------------
    # Plot 5: Feature Distributions Before Preprocessing
    # -------------------------------------------------------------
    logger.info("Generating Plot 5: Vital Sign Distributions...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, v in enumerate(VITAL_SIGNS):
        valid = df[v].dropna()
        sns.histplot(valid, bins=30, ax=axes[idx], color="#0284c7", kde=True)
        axes[idx].set_title(f"{v} (Observed: {len(valid)})", fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].grid(True, linestyle="--", alpha=0.4)
    plt.suptitle("Raw Vital Sign Distributions (Pre-Imputation)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    p5 = out_dir / "05_vital_sign_distributions.png"
    plt.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(p5)

    # -------------------------------------------------------------
    # Plot 6: Example Patient Temporal Trajectories
    # -------------------------------------------------------------
    logger.info("Generating Plot 6: Patient Temporal Trajectories...")
    septic_pts = df[df[TARGET_COL] == 1]["patient_id"].unique()
    non_septic_pts = df[df[TARGET_COL] == 0]["patient_id"].unique()
    
    if len(septic_pts) > 0 and len(non_septic_pts) > 0:
        sep_id = septic_pts[0]
        non_sep_id = non_septic_pts[0]
        
        sep_df = df[df["patient_id"] == sep_id].sort_values("ICULOS")
        non_sep_df = df[df["patient_id"] == non_sep_id].sort_values("ICULOS")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Heart Rate
        axes[0].plot(sep_df["ICULOS"], sep_df["HR"], "r-o", label=f"Septic ({sep_id})", linewidth=2)
        axes[0].plot(non_sep_df["ICULOS"], non_sep_df["HR"], "b--s", label=f"Non-Septic ({non_sep_id})", alpha=0.7)
        axes[0].set_ylabel("Heart Rate (bpm)", fontweight="bold")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
        # Mean Arterial Pressure
        axes[1].plot(sep_df["ICULOS"], sep_df["MAP"], "r-o", label=f"Septic MAP", linewidth=2)
        axes[1].plot(non_sep_df["ICULOS"], non_sep_df["MAP"], "b--s", label=f"Non-Septic MAP", alpha=0.7)
        axes[1].set_ylabel("MAP (mmHg)", fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        # Sepsis Label
        axes[2].step(sep_df["ICULOS"], sep_df[TARGET_COL], "r-", where="post", label="Sepsis Active Label", linewidth=2.5)
        axes[2].set_ylabel("SepsisLabel", fontweight="bold")
        axes[2].set_xlabel("ICU Length of Stay (Hours)", fontweight="bold")
        axes[2].set_ylim(-0.1, 1.2)
        axes[2].legend(loc="upper left")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        
        plt.suptitle("Comparative Hourly Physiological Trajectories: Septic vs Non-Septic Patient", fontsize=14, fontweight="bold")
        plt.tight_layout()
        p6 = out_dir / "06_patient_temporal_trajectories.png"
        plt.savefig(p6, dpi=300)
        plt.close()
        saved_plots.append(p6)

    # -------------------------------------------------------------
    # Plot 7: Correlation Matrix for Continuous Vitals
    # -------------------------------------------------------------
    logger.info("Generating Plot 7: Correlation Matrix for Continuous Vitals...")
    vitals_corr = df[VITAL_SIGNS].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(vitals_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Pearson Correlation Matrix: Raw Vital Signs", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p7 = out_dir / "07_vital_correlation_matrix.png"
    plt.savefig(p7, dpi=300)
    plt.close()
    saved_plots.append(p7)

    # -------------------------------------------------------------
    # Plot 8: Observation Frequency by Hospital System
    # -------------------------------------------------------------
    logger.info("Generating Plot 8: Observation Frequency by Hospital System...")
    hosp_densities = []
    for h in ["A", "B"]:
        sub = df[df["hospital"] == h]
        d = (1.0 - sub[VITAL_SIGNS].isna().mean()) * 100.0
        for v, val in d.items():
            hosp_densities.append({"Vital": v, "Hospital": f"Hospital {h}", "Density": val})
            
    hosp_df = pd.DataFrame(hosp_densities)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=hosp_df, x="Vital", y="Density", hue="Hospital", palette=["#2563eb", "#ea580c"], ax=ax)
    ax.set_ylabel("Sampling Density (%)", fontweight="bold")
    ax.set_xlabel("Vital Sign", fontweight="bold")
    ax.set_title("Vital Sign Sampling Frequency: Hospital System A vs Hospital System B", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    p8 = out_dir / "08_observation_frequency.png"
    plt.savefig(p8, dpi=300)
    plt.close()
    saved_plots.append(p8)
    
    logger.info(f"All 8 diagnostic plots generated and saved to {out_dir}")
    return saved_plots


if __name__ == "__main__":
    generate_all_eda_plots()
