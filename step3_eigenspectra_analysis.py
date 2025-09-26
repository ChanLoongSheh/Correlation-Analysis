# Program 2 — Step 3: Analysis of Eigensensitivity Spectra
# Author: (Your name)
# Description:
#   Implements Step 3 of the README:
#     - Load the eigensensitivity spectra E(λ) (scores) produced by Step 2.
#     - Plot E_j(λ) overlapped and as separate subplots.
#     - Compute per-component summary metrics (max/min, areas, zero-crossings, centroids, norms).
#     - Save a detailed processing_log.txt explaining what and where the outputs are.
#
# How to run:
#   1) Set INPUT_E_SCORES_CSV to the output CSV from Step 2:
#        outputs_optimal_filter_pca/E_scores_eigensensitivity_spectra.csv
#      Optionally set INPUT_EVR_CSV to:
#        outputs_optimal_filter_pca/pca_explained_variance_ratio.csv
#   2) Run: python step3_eigenspectra_analysis.py
#
# Outputs (saved into a single folder without timestamp: 'outputs_eigenspectra_analysis'):
#   - EigensensitivitySpectra_overlapped.png
#   - EigensensitivitySpectra_separate_grid.png
#   - Eigenspectra_summary_metrics.csv
#   - processing_log.txt
#
# Notes:
#   - E_scores columns (E1..En) are the eigensensitivity spectra E(λ) vs wavelength λ.
#   - All CSV outputs include headers and wavelength λ [µm] as row index where applicable.

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# User-configurable parameters
# ----------------------------

# Input E(λ) (scores) from Step 2
INPUT_E_SCORES_CSV = Path("outputs_optimal_filter_pca") / "E_scores_eigensensitivity_spectra.csv"

# Optional: PCA explained variance ratio from Step 2 (if available)
INPUT_EVR_CSV = Path("outputs_optimal_filter_pca") / "pca_explained_variance_ratio.csv"

# Output directory for Step 3 (no timestamp)
OUTPUT_DIR = Path("outputs_eigenspectra_analysis")

# Plotting defaults
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


# --------------------------------------
# Function 1 — Data Loading
# Title: load_eigenspectra_scores
# --------------------------------------
def load_eigenspectra_scores(
    scores_csv: Path,
    evr_csv: Optional[Path] = None,
) -> Tuple[pd.Index, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load E(λ) eigensensitivity spectra (scores) and optional explained variance ratio.

    Returns
    -------
    wavelength_index : pd.Index
        Wavelength grid λ [µm], sorted ascending.
    E_df : pd.DataFrame
        E(λ) with shape (N_λ, N_comp); columns are E1..En.
    evr_df : Optional[pd.DataFrame]
        Explained variance ratio indexed by component (E1..En) if available; else None.
    """
    if not scores_csv.exists():
        raise FileNotFoundError(f"E_scores file not found: {scores_csv}")

    E_df = pd.read_csv(scores_csv, index_col=0)
    # Drop any accidental 'Unnamed' columns
    to_drop = [c for c in E_df.columns if str(c).startswith("Unnamed:")]
    if to_drop:
        E_df = E_df.drop(columns=to_drop)

    # Ensure wavelength index is numeric and sorted
    try:
        E_df.index = pd.to_numeric(E_df.index, errors="coerce")
    except Exception:
        raise ValueError("Row index of E_scores must be wavelength λ [µm].")
    E_df.index.name = "wavelength_um"
    E_df = E_df.sort_index(axis=0, ascending=True)
    wavelength_index = E_df.index

    # Load optional explained variance ratio
    evr_df = None
    if evr_csv is not None and evr_csv.exists():
        evr_df = pd.read_csv(evr_csv, index_col=0)

    return wavelength_index, E_df, evr_df


# --------------------------------------
# Function 2 — Overlapped Plot
# Title: plot_eigenspectra_overlapped
# --------------------------------------
def plot_eigenspectra_overlapped(
    E_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "EigensensitivitySpectra_overlapped.png",
) -> Path:
    """
    Plot all eigensensitivity spectra E_j(λ) on a single set of axes (overlapped).
    """
    lam = E_df.index.to_numpy(dtype=float)
    cols = list(E_df.columns)
    ncomp = len(cols)

    y_min = float(np.min(E_df.to_numpy()))
    y_max = float(np.max(E_df.to_numpy()))
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ylims = (y_min - pad, y_max + pad)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cols):
        ax.plot(lam, E_df[c].to_numpy(), lw=1.2, color=cmap(i % 10), label=c)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax.set_xlim(lam.min(), lam.max())
    ax.set_ylim(*ylims)
    ax.set_xlabel("Wavelength λ [µm]")
    ax.set_ylabel("Eigensensitivity E_j(λ) [arb. units]")
    ax.set_title("Eigensensitivity Spectra — overlapped")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=min(3, ncomp), fontsize=9)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------
# Function 3 — Separate Subplots
# Title: plot_eigenspectra_separate_grid
# --------------------------------------
def plot_eigenspectra_separate_grid(
    E_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "EigensensitivitySpectra_separate_grid.png",
    ncols: int = 3,
) -> Path:
    """
    Plot each E_j(λ) in a separate subplot (shared axes for fair comparison).
    """
    lam = E_df.index.to_numpy(dtype=float)
    cols = list(E_df.columns)
    ncomp = len(cols)

    # Common y-limits
    y_min = float(np.min(E_df.to_numpy()))
    y_max = float(np.max(E_df.to_numpy()))
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ylims = (y_min - pad, y_max + pad)

    nrows = int(np.ceil(ncomp / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(9.2, 2.8 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    for idx, c in enumerate(cols):
        r = idx // ncols
        k = idx % ncols
        ax = axes[r, k]
        ax.plot(lam, E_df[c].to_numpy(), lw=1.2, color="tab:blue")
        ax.axhline(0.0, color="k", lw=0.7, alpha=0.6)
        ax.set_ylim(*ylims)
        ax.set_title(c)
        ax.grid(True, alpha=0.25)
        if r == nrows - 1:
            ax.set_xlabel("λ [µm]")
        if k == 0:
            ax.set_ylabel("E_j(λ)")

    # Hide unused axes
    for idx in range(ncomp, nrows * ncols):
        r = idx // ncols
        k = idx % ncols
        axes[r, k].axis("off")

    fig.suptitle("Eigensensitivity Spectra — separate panels", y=0.995, fontsize=12)
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------
# Function 4 — Metrics
# Title: compute_eigenspectra_metrics
# --------------------------------------
def compute_eigenspectra_metrics(
    E_df: pd.DataFrame,
    evr_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute per-component diagnostics for E_j(λ):
      - lambda_grid_min_um, lambda_grid_max_um, N_lambda
      - value_max, lambda_at_value_max_um
      - value_min, lambda_at_value_min_um
      - value_absmax, lambda_at_value_absmax_um
      - positive_area (∫ max(E,0) dλ), negative_area (∫ min(E,0) dλ), total_abs_area (∫ |E| dλ)
      - zero_crossings_count
      - centroid_pos_um (centroid of positive lobe), centroid_neg_um (centroid of negative lobe)
      - L2_norm = sqrt(∫ E^2 dλ)
      - explained_variance_ratio (if provided)
    """
    lam = E_df.index.to_numpy(dtype=float)
    if lam.ndim != 1 or len(lam) < 2:
        raise ValueError("Wavelength index must be 1D and length >= 2.")
    dlam = np.diff(lam)
    if np.any(dlam <= 0):
        raise ValueError("Wavelength index must be strictly ascending.")

    records: List[Dict[str, float]] = []
    for c in E_df.columns:
        y = E_df[c].to_numpy(dtype=float)

        # Basic extrema
        idx_max = int(np.argmax(y)); val_max = float(y[idx_max]); lam_max = float(lam[idx_max])
        idx_min = int(np.argmin(y)); val_min = float(y[idx_min]); lam_min = float(lam[idx_min])

        idx_abs = int(np.argmax(np.abs(y))); val_abs = float(y[idx_abs]); lam_abs = float(lam[idx_abs])

        # Areas by trapezoidal integration
        y_pos = np.clip(y, a_min=0.0, a_max=None)
        y_neg = np.clip(y, a_min=None, a_max=0.0)
        A_pos = float(np.trapz(y_pos, lam))
        A_neg = float(np.trapz(y_neg, lam))
        A_abs = float(np.trapz(np.abs(y), lam))

        # Zero crossings: count sign changes (exclude exact zeros from double-counting)
        s = np.sign(y)
        # Replace zeros with nearest nonzero sign to avoid artificial inflation
        for i in range(1, len(s)):
            if s[i] == 0 and s[i - 1] != 0:
                s[i] = s[i - 1]
        for i in range(len(s) - 2, -1, -1):
            if s[i] == 0 and s[i + 1] != 0:
                s[i] = s[i + 1]
        zc_count = int(np.sum(s[1:] * s[:-1] < 0))

        # Centroids (first moment) for positive/negative lobes
        centroid_pos_um = np.nan
        centroid_neg_um = np.nan
        if np.any(y_pos > 0):
            centroid_pos_um = float(np.trapz(lam * y_pos, lam) / A_pos) if A_pos > 0 else np.nan
        if np.any(y_neg < 0):
            wneg = np.abs(y_neg)
            denom = float(np.trapz(wneg, lam))
            centroid_neg_um = float(np.trapz(lam * wneg, lam) / denom) if denom > 0 else np.nan

        # L2 norm (energy-like magnitude)
        L2 = float(np.sqrt(np.trapz(y * y, lam)))

        rec: Dict[str, float] = dict(
            component=c,
            lambda_grid_min_um=float(lam.min()),
            lambda_grid_max_um=float(lam.max()),
            N_lambda=int(len(lam)),
            value_max=val_max,
            lambda_at_value_max_um=lam_max,
            value_min=val_min,
            lambda_at_value_min_um=lam_min,
            value_absmax=val_abs,
            lambda_at_value_absmax_um=lam_abs,
            positive_area=A_pos,
            negative_area=A_neg,
            total_abs_area=A_abs,
            zero_crossings_count=zc_count,
            centroid_pos_um=centroid_pos_um,
            centroid_neg_um=centroid_neg_um,
            L2_norm=L2,
        )

        # If EVR provided, merge by component name (expects index like "E1")
        if evr_df is not None and c in evr_df.index:
            if "explained_variance_ratio" in evr_df.columns:
                rec["explained_variance_ratio"] = float(evr_df.loc[c, "explained_variance_ratio"])
            if "explained_variance" in evr_df.columns:
                rec["explained_variance"] = float(evr_df.loc[c, "explained_variance"])

        records.append(rec)

    out = pd.DataFrame.from_records(records).set_index("component")
    # Order useful columns
    preferred_order = [
        "lambda_grid_min_um", "lambda_grid_max_um", "N_lambda",
        "value_max", "lambda_at_value_max_um",
        "value_min", "lambda_at_value_min_um",
        "value_absmax", "lambda_at_value_absmax_um",
        "positive_area", "negative_area", "total_abs_area",
        "zero_crossings_count",
        "centroid_pos_um", "centroid_neg_um",
        "L2_norm",
        "explained_variance_ratio", "explained_variance",
    ]
    cols = [c for c in preferred_order if c in out.columns] + [c for c in out.columns if c not in preferred_order]
    return out[cols]


# --------------------------------------
# Function 5 — I/O
# Title: save_step3_outputs
# --------------------------------------
def save_step3_outputs(
    output_dir: Path,
    metrics_df: pd.DataFrame,
    fig_paths: List[Path],
) -> List[Path]:
    """
    Save CSVs and figures for Step 3. Returns list of saved file paths.
    """
    saved: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary metrics CSV
    metrics_path = output_dir / "Eigenspectra_summary_metrics.csv"
    metrics_df.to_csv(metrics_path, index=True)
    saved.append(metrics_path)

    # Figures already written; just register paths
    for p in fig_paths:
        saved.append(p)

    return saved


# --------------------------------------
# Function 6 — Logging
# Title: write_detailed_log_step3
# --------------------------------------
def write_detailed_log_step3(
    output_dir: Path,
    input_scores_path: Path,
    input_evr_path: Optional[Path],
    wavelength_index: pd.Index,
    E_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    saved_paths: List[Path],
) -> Path:
    """
    Write a detailed log describing inputs, outputs, and their meanings for Step 3.
    """
    log_lines: List[str] = []

    log_lines.append("Project: Optimal Infrared Filter Selection via Eigenspectra Analysis")
    log_lines.append("Step 3: Analysis of Eigensensitivity Spectra — plotting and diagnostics.")
    log_lines.append("E_scores columns (E1..En) are the eigensensitivity spectra E(λ) vs wavelength λ.")
    log_lines.append("This step reads Step 2 outputs and produces plots and summary metrics.")
    log_lines.append("")

    # Inputs
    log_lines.append("Inputs:")
    log_lines.append(f"  - E_scores (E(λ)): {input_scores_path.resolve()}")
    log_lines.append(f"  - PCA explained_variance_ratio: {input_evr_path.resolve() if input_evr_path is not None and input_evr_path.exists() else 'not provided'}")
    log_lines.append(f"  - Wavelength λ range [µm]: {wavelength_index.min():.6g} to {wavelength_index.max():.6g} ({len(wavelength_index)} points).")
    log_lines.append("")

    # Shapes
    log_lines.append("Data structures (shapes and meanings):")
    log_lines.append(f"  - E(λ): {E_df.shape}; rows = λ [µm]; columns = E1..En (eigensensitivity spectra).")
    log_lines.append(f"  - Summary metrics: {metrics_df.shape}; rows = E1..En; columns = per-component diagnostics.")
    log_lines.append("")

    # Outputs
    log_lines.append("Outputs (what and where):")
    for p in saved_paths:
        fname = p.name
        if fname == "EigensensitivitySpectra_overlapped.png":
            fdesc = "Overlapped plot of E1..En vs wavelength λ."
        elif fname == "EigensensitivitySpectra_separate_grid.png":
            fdesc = "Multi-panel figure with separate subplots for each E_j(λ); shared axes for comparison."
        elif fname == "Eigenspectra_summary_metrics.csv":
            fdesc = ("Per-component diagnostics: λ-grid bounds, N_λ, extrema (values and λ-locations), "
                     "areas (positive/negative/|·|), zero crossings, positive/negative centroids, L2-norm, "
                     "and (if provided) explained_variance_ratio.")
        else:
            fdesc = "Additional output."
        log_lines.append(f"  - {p.resolve()}")
        log_lines.append(f"    Meaning: {fdesc}")

    log_lines.append("")
    log_lines.append("Notes:")
    log_lines.append("- Positive lobes in E_j(λ) indicate wavelengths where radiance increases under the j-th mode;")
    log_lines.append("  negative lobes indicate decreases. The absolute magnitude indicates sensitivity strength.")
    log_lines.append("- Areas and centroids provide quantitative summaries of where sensitivity is concentrated.")
    log_lines.append("- Zero crossings quantify spectral oscillation/structure in E_j(λ).")
    log_lines.append("")

    log_path = output_dir / "processing_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    return log_path


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load E(λ) and optional explained variance ratio
    wavelength_index, E_df, evr_df = load_eigenspectra_scores(
        INPUT_E_SCORES_CSV, evr_csv=INPUT_EVR_CSV
    )

    # Generate figures
    fig_paths: List[Path] = []
    path_over = plot_eigenspectra_overlapped(E_df, OUTPUT_DIR)
    fig_paths.append(path_over)

    path_sep = plot_eigenspectra_separate_grid(E_df, OUTPUT_DIR, filename="EigensensitivitySpectra_separate_grid.png")
    fig_paths.append(path_sep)

    # Compute diagnostics
    metrics_df = compute_eigenspectra_metrics(E_df, evr_df=evr_df)

    # Save outputs (CSV + figs)
    saved_paths = save_step3_outputs(OUTPUT_DIR, metrics_df, fig_paths)

    # Write detailed log
    log_path = write_detailed_log_step3(
        OUTPUT_DIR,
        INPUT_E_SCORES_CSV,
        INPUT_EVR_CSV if INPUT_EVR_CSV.exists() else None,
        wavelength_index,
        E_df,
        metrics_df,
        saved_paths,
    )

    print(f"Step 3 complete. Outputs saved to: {OUTPUT_DIR.resolve()}")
    print(f"Detailed log: {log_path.resolve()}")


if __name__ == "__main__":
    main()