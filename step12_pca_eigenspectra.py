# Program 1 — Step 1 and Step 2 with Eigensensitivity Spectra plots and singular values
# Author: (Your name)
# Description:
#   Implements:
#     - Step 1: Load U(λ), sort λ ascending, visualize with wavelength-accurate heatmap (pcolormesh),
#               de-crowd y-axis ticks, and output diagnostics for PC peak wavelengths.
#     - Step 2: Standardize U(λ) and perform PCA to obtain E(λ) (eigensensitivity spectra).
#   Adds:
#     - Function 8: plot Eigensensitivity Spectra E(λ) — overlapped and separate grid figure.
#     - Function 9: export PCA singular values along with explained variance info.
#   Outputs (all saved into a single folder without timestamp 'outputs_optimal_filter_pca'):
#     - heatmap_U_lambda_vs_PC.png
#     - U_scaled.csv
#     - E_scores_eigensensitivity_spectra.csv
#     - pca_explained_variance_ratio.csv
#     - pca_components_loadings.csv
#     - U_column_peaks.csv
#     - line_PC1_spectrum.png
#     - EigensensitivitySpectra_overlapped.png
#     - EigensensitivitySpectra_separate_grid.png
#     - pca_singular_values.csv
#     - processing_log.txt
#
# How to run:
#   1) Ensure INPUT_CSV_PATH points to PC_U_unit_sensitivity.csv
#   2) Run: python step12_pca_eigenspectra.py
#
# Notes:
#   - E_scores (E1..En) are the eigensensitivity spectra E(λ): each column is the score vs wavelength for that component.
#   - singular_values = sqrt((N_λ - 1) * explained_variance). sqrt(explained_variance) is the std dev of E(λ) along each component.

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ----------------------------
# User-configurable parameters
# ----------------------------

INPUT_CSV_PATH = Path("PC_U_unit_sensitivity.csv")  # input U(λ)
OUTPUT_DIR = Path("outputs_optimal_filter_pca")      # output folder (no timestamp)

ASSUMED_LAMBDA_MIN_UM = 2.5
ASSUMED_LAMBDA_MAX_UM = 20.0

HEATMAP_CMAP = "RdBu_r"  # diverging colormap
N_COMPONENTS: Optional[int] = None  # None => number of columns in U(λ)

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
# Function 1 — Data Loading (with λ sorting)
# Title: load_unit_sensitivity_matrix
# --------------------------------------
def load_unit_sensitivity_matrix(
    csv_path: Path,
    lambda_min_um: float = ASSUMED_LAMBDA_MIN_UM,
    lambda_max_um: float = ASSUMED_LAMBDA_MAX_UM,
) -> Tuple[pd.Index, pd.DataFrame, str]:
    """
    Load the unit-coefficient sensitivity matrix U(λ) from CSV and sort by λ ascending.

    Returns
    -------
    wavelength_index : pd.Index
        Wavelength grid λ [µm] used as row index (sorted ascending).
    U_df : pd.DataFrame
        U(λ) with shape (N_λ, N_PC); columns are original PCs (∂I(λ)/∂cᵢ).
    wavelength_source : str
        'from_file' if a known wavelength column exists; otherwise 'generated'.
    """
    df = pd.read_csv(csv_path)

    # Drop typical unnamed index columns (e.g., 'Unnamed: 0')
    to_drop = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Find wavelength column if present
    candidate_cols = ["wavelength_um", "wavelength", "lambda_um", "lambda", "λ", "Lambda"]
    wl_col = None
    for c in df.columns:
        if str(c).strip().lower() in {cc.lower() for cc in candidate_cols}:
            wl_col = c
            break

    if wl_col is not None:
        wavelength = pd.to_numeric(df[wl_col], errors="coerce").to_numpy()
        U_df = df.drop(columns=[wl_col])
        wavelength_source = "from_file"
    else:
        # No wavelength column; generate uniform λ grid across [2.5, 20] µm
        n_rows = df.shape[0]
        wavelength = np.linspace(lambda_min_um, lambda_max_um, num=n_rows, endpoint=True)
        U_df = df.copy()
        wavelength_source = "generated"

    # Clean/ensure PC column names
    if U_df.columns.isnull().any():
        U_df.columns = [f"PC{i+1}" for i in range(U_df.shape[1])]
    else:
        new_cols: List[str] = []
        for i, c in enumerate(U_df.columns):
            cs = str(c).strip()
            new_cols.append(cs if cs != "" else f"PC{i+1}")
        U_df.columns = new_cols

    # Assemble DataFrame with λ index and sort ascending
    wavelength_index = pd.Index(wavelength, name="wavelength_um")
    U_df.index = wavelength_index

    # Remove any rows with NaN wavelengths (if present)
    if np.isnan(U_df.index.values).any():
        U_df = U_df[~np.isnan(U_df.index.values)]

    U_df = U_df.sort_index(axis=0, ascending=True)
    wavelength_index = U_df.index

    return wavelength_index, U_df, wavelength_source


# -----------------------------------------------------------------
# Function 2 — Heatmap Plotting (pcolormesh with true λ; de-crowded y-axis)
# Title: plot_heatmap_U_pcolormesh
# -----------------------------------------------------------------
def plot_heatmap_U_pcolormesh(
    U_df: pd.DataFrame,
    output_dir: Path,
    cmap_name: str = HEATMAP_CMAP,
    figsize: Tuple[float, float] = (6.8, 7.2),
    y_tick_count: int = 10,         # evenly spaced in λ to avoid crowding (e.g., dense 2.5–4.15 µm region)
    y_tick_format: str = "{:.1f}",  # fewer decimals helps reduce overlap
    y_tick_labelsize: float = 8.0,
) -> Path:
    """
    Create and save a wavelength-accurate heatmap of U(λ) using pcolormesh with true λ edges.
    Y-ticks are placed evenly in λ (not by row index), avoiding label crowding.
    """
    cmap = plt.get_cmap(cmap_name)

    data = U_df.to_numpy()
    lam = U_df.index.to_numpy(dtype=float)
    n_lambda, n_pc = data.shape

    # Compute λ cell edges (safe for non-uniform grids)
    lam_edges = np.empty(n_lambda + 1, dtype=float)
    lam_edges[1:-1] = 0.5 * (lam[:-1] + lam[1:])
    lam_edges[0] = lam[0] - (lam_edges[1] - lam[0])
    lam_edges[-1] = lam[-1] + (lam[-1] - lam_edges[-2])

    x_edges = np.arange(0, n_pc + 1, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(
        x_edges,
        lam_edges,
        data,
        cmap=cmap,
        shading="flat",
    )

    # X-axis ticks at cell centers
    ax.set_xticks(np.arange(n_pc) + 0.5)
    ax.set_xticklabels(list(U_df.columns), rotation=45, ha="right")

    # Y-axis ticks evenly spaced in wavelength
    lam_min = float(lam.min()); lam_max = float(lam.max())
    y_tick_count = max(4, int(y_tick_count))
    yticks = np.linspace(lam_min, lam_max, y_tick_count)
    ytick_labels = [y_tick_format.format(v) for v in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.tick_params(axis="y", labelsize=y_tick_labelsize)

    ax.set_ylabel("Wavelength λ [µm]")
    ax.set_xlabel("Principal Component index (PC)")
    ax.set_title("Unit-Coefficient Sensitivity Matrix U(λ) — wavelength-accurate")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Sensitivity  ∂I(λ)/∂cᵢ  [arbitrary units]")

    fig.tight_layout()
    fig_path = output_dir / "heatmap_U_lambda_vs_PC.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# --------------------------------------
# Function 3 — PCA Pipeline
# Title: scale_and_pca
# --------------------------------------
def scale_and_pca(
    U_df: pd.DataFrame,
    n_components: Optional[int] = None,
):
    """
    Scale U(λ) and perform PCA to obtain E(λ).
    Returns scaler, U_scaled_df, pca, E_scores_df, evr_df, components_df.
    """
    X = U_df.to_numpy()
    if n_components is None:
        n_components = U_df.shape[1]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    U_scaled_df = pd.DataFrame(X_scaled, index=U_df.index.copy(), columns=U_df.columns.copy())

    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    E_scores = pca.fit_transform(X_scaled)  # shape (N_λ, n_components)

    E_cols = [f"E{i+1}" for i in range(n_components)]
    E_scores_df = pd.DataFrame(E_scores, index=U_df.index.copy(), columns=E_cols)

    evr_df = pd.DataFrame(
        {"component": E_cols, "explained_variance_ratio": pca.explained_variance_ratio_}
    ).set_index("component")

    components_df = pd.DataFrame(
        pca.components_, index=[f"EV{i+1}" for i in range(n_components)], columns=U_df.columns.copy()
    )

    return scaler, U_scaled_df, pca, E_scores_df, evr_df, components_df


# --------------------------------------
# Function 4 — I/O
# Title: save_outputs
# --------------------------------------
def save_outputs(
    output_dir: Path,
    U_scaled_df: pd.DataFrame,
    E_scores_df: pd.DataFrame,
    evr_df: pd.DataFrame,
    components_df: pd.DataFrame,
    extra_fig_paths: List[Path],
    peaks_df: pd.DataFrame,
) -> List[Path]:
    """
    Save DataFrames and figures to disk. Returns list of saved file paths.
    """
    saved_paths: List[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    path_U_scaled = output_dir / "U_scaled.csv"
    U_scaled_df.to_csv(path_U_scaled, index=True); saved_paths.append(path_U_scaled)

    path_E_scores = output_dir / "E_scores_eigensensitivity_spectra.csv"
    E_scores_df.to_csv(path_E_scores, index=True); saved_paths.append(path_E_scores)

    path_evr = output_dir / "pca_explained_variance_ratio.csv"
    evr_df.to_csv(path_evr, index=True); saved_paths.append(path_evr)

    path_components = output_dir / "pca_components_loadings.csv"
    components_df.to_csv(path_components, index=True); saved_paths.append(path_components)

    path_peaks = output_dir / "U_column_peaks.csv"
    peaks_df.to_csv(path_peaks, index=True); saved_paths.append(path_peaks)

    for p in extra_fig_paths:
        saved_paths.append(p)

    return saved_paths


# --------------------------------------
# Function 5 — Logging
# Title: write_detailed_log
# --------------------------------------
def write_detailed_log(
    output_dir: Path,
    input_csv_path: Path,
    wavelength_source: str,
    wavelength_index: pd.Index,
    U_df: pd.DataFrame,
    U_scaled_df: pd.DataFrame,
    E_scores_df: pd.DataFrame,
    evr_df: pd.DataFrame,
    components_df: pd.DataFrame,
    fig_paths: List[Path],
    saved_paths: List[Path],
) -> Path:
    """
    Write a detailed log file describing inputs, outputs, and their meanings.
    """
    log_lines: List[str] = []

    log_lines.append("Project: Optimal Infrared Filter Selection via Eigenspectra Analysis")
    log_lines.append("Steps: 1) Data Loading & Visualization; 2) PCA to Eigensensitivity Spectra.")
    log_lines.append("E_scores columns (E1..En) are the eigensensitivity spectra E(λ) vs wavelength λ.")
    log_lines.append("Heatmap uses pcolormesh with true λ cell edges and evenly spaced λ-ticks to avoid crowding.")
    log_lines.append("")

    # Inputs
    log_lines.append("Inputs:")
    log_lines.append(f"  - Input CSV (U(λ)): {input_csv_path.resolve()}")
    log_lines.append(f"  - Wavelength source: {wavelength_source}")
    log_lines.append(f"  - Wavelength λ range [µm]: {wavelength_index.min():.6g} to {wavelength_index.max():.6g} "
                     f"({len(wavelength_index)} points).")
    log_lines.append("")

    # Shapes
    log_lines.append("Data structures (shapes and meanings):")
    log_lines.append(f"  - U(λ): {U_df.shape}; rows = λ [µm], columns = original PCs (∂I(λ)/∂cᵢ).")
    log_lines.append(f"  - U_scaled: {U_scaled_df.shape}; standardized columns (zero-mean, unit-variance).")
    log_lines.append(f"  - E(λ): {E_scores_df.shape}; PCA scores (eigensensitivity spectra) vs λ.")
    log_lines.append(f"  - PCA explained_variance_ratio: length = {len(evr_df)}.")
    log_lines.append(f"  - PCA components (loadings): {components_df.shape} (rows = EV1..EVn; columns = original PCs).")
    log_lines.append("")

    # Outputs
    log_lines.append("Outputs (what and where):")
    for p in saved_paths:
        fname = p.name
        if fname == "U_scaled.csv":
            fdesc = "Standardized U(λ); index = wavelength λ [µm]; columns = original PCs."
        elif fname == "E_scores_eigensensitivity_spectra.csv":
            fdesc = "E(λ): eigensensitivity spectra as PCA scores; index = λ [µm]; columns = E1..En."
        elif fname == "pca_explained_variance_ratio.csv":
            fdesc = "Explained variance ratio for each eigensensitivity spectrum E1..En."
        elif fname == "pca_components_loadings.csv":
            fdesc = "PCA loading matrix (rows = EV1..EVn; columns = original PCs)."
        elif fname == "U_column_peaks.csv":
            fdesc = "Per-PC λ of max, min, and |max| in U(λ); quick check for expected spectral peaks."
        elif fname == "heatmap_U_lambda_vs_PC.png":
            fdesc = "Wavelength-accurate heatmap of U(λ) (true λ grid; evenly spaced y-ticks)."
        elif fname == "line_PC1_spectrum.png":
            fdesc = "Diagnostic line plot of PC1 vs λ with vertical lines at max/min/|max|."
        elif fname == "EigensensitivitySpectra_overlapped.png":
            fdesc = "Overlapped plot of eigensensitivity spectra E1..En vs wavelength λ."
        elif fname == "EigensensitivitySpectra_separate_grid.png":
            fdesc = "Multi-panel figure: separate subplots for each E_j(λ). Shared axes for comparison."
        elif fname == "pca_singular_values.csv":
            fdesc = ("PCA magnitudes: explained_variance, explained_variance_ratio, "
                     "sqrt_explained_variance (std of scores), and singular_values "
                     "with σ_j = sqrt((N_λ − 1) · explained_variance_j).")
        else:
            fdesc = "Additional output."
        log_lines.append(f"  - {p.resolve()}")
        log_lines.append(f"    Meaning: {fdesc}")

    log_lines.append("")
    log_lines.append("Notes:")
    log_lines.append("- E_scores are called 'scores' in PCA because they are projections of samples (here, λ) onto components;")
    log_lines.append("  in this application, they are the eigensensitivity spectra E(λ).")
    log_lines.append("- sqrt(explained_variance) is the standard deviation of E(λ) along each component;")
    log_lines.append("  scikit-learn's singular_values satisfy σ_j = sqrt((N_λ - 1) * explained_variance_j).")
    log_lines.append("- All CSVs include headers and wavelength λ [µm] as the row index where applicable.")
    log_lines.append("")

    log_path = output_dir / "processing_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    return log_path


# --------------------------------------
# Function 6 — Diagnostics
# Title: compute_and_save_column_peaks
# --------------------------------------
def compute_and_save_column_peaks(U_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each PC column, compute λ locations and values of max, min, and |max|.
    Returns a DataFrame indexed by PC column name.
    """
    records = []
    for col in U_df.columns:
        s = U_df[col]
        lam_max = float(s.idxmax()); val_max = float(s.loc[lam_max])
        lam_min = float(s.idxmin()); val_min = float(s.loc[lam_min])
        lam_abs = float(s.abs().idxmax()); val_abs = float(s.loc[lam_abs])
        records.append(
            dict(
                PC=col,
                value_max=val_max, lambda_max_um=lam_max,
                value_min=val_min, lambda_min_um=lam_min,
                value_absmax=val_abs, lambda_absmax_um=lam_abs
            )
        )
    return pd.DataFrame.from_records(records).set_index("PC")


# --------------------------------------
# Function 7 — Diagnostics Plot
# Title: plot_PC1_line
# --------------------------------------
def plot_PC1_line(U_df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Plot PC1 vs λ with vertical lines marking max, min, and |max|.
    """
    fig_path = output_dir / "line_PC1_spectrum.png"
    if "PC1" not in U_df.columns:
        return fig_path

    s = U_df["PC1"]
    lam = s.index.to_numpy(dtype=float)
    val = s.to_numpy(dtype=float)

    lam_max = float(s.idxmax()); val_max = float(s.loc[lam_max])
    lam_min = float(s.idxmin()); val_min = float(s.loc[lam_min])
    lam_abs = float(s.abs().idxmax()); val_abs = float(s.loc[lam_abs])

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(lam, val, color="k", lw=1.2, label="PC1(λ)")
    ax.axvline(lam_max, color="tab:red", ls="--", lw=1.0, label=f"max @ {lam_max:.3f} µm")
    ax.axvline(lam_min, color="tab:blue", ls="--", lw=1.0, label=f"min @ {lam_min:.3f} µm")
    ax.axvline(lam_abs, color="tab:green", ls=":", lw=1.0, label=f"|max| @ {lam_abs:.3f} µm")

    ax.set_xlabel("Wavelength λ [µm]")
    ax.set_ylabel("Sensitivity ∂I(λ)/∂c₁  [arb. units]")
    ax.set_title("PC1 vs λ with peak locations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# --------------------------------------
# Function 8 — Eigensensitivity Plots
# Title: plot_eigensensitivity_spectra
# --------------------------------------
def plot_eigensensitivity_spectra(
    E_scores_df: pd.DataFrame,
    output_dir: Path,
    overlapped_filename: str = "EigensensitivitySpectra_overlapped.png",
    separate_grid_filename: str = "EigensensitivitySpectra_separate_grid.png",
) -> List[Path]:
    """
    Plot Eigensensitivity Spectra E(λ) from E_scores_df.
    Produces:
      - An overlapped plot of all E_j(λ) curves.
      - A multi-panel grid with one subplot per component.
    Returns list of figure paths.
    """
    lam = E_scores_df.index.to_numpy(dtype=float)
    cols = list(E_scores_df.columns)
    ncomp = len(cols)

    # Determine common y-limits for fair visual comparison
    y_min = float(np.min(E_scores_df.to_numpy()))
    y_max = float(np.max(E_scores_df.to_numpy()))
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ylims = (y_min - pad, y_max + pad)

    fig_paths: List[Path] = []

    # Overlapped
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cols):
        ax.plot(lam, E_scores_df[c].to_numpy(), lw=1.2, color=cmap(i % 10), label=c)
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax.set_xlim(lam.min(), lam.max())
    ax.set_ylim(*ylims)
    ax.set_xlabel("Wavelength λ [µm]")
    ax.set_ylabel("Eigensensitivity E_j(λ) [arb. units]")
    ax.set_title("Eigensensitivity Spectra — overlapped")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=min(3, ncomp), fontsize=9)
    fig.tight_layout()
    path_over = output_dir / overlapped_filename
    fig.savefig(path_over, bbox_inches="tight")
    plt.close(fig)
    fig_paths.append(path_over)

    # Separate grid
    ncols = 3
    nrows = int(np.ceil(ncomp / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9.0, 2.6 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for idx, c in enumerate(cols):
        r = idx // ncols; k = idx % ncols
        ax = axes[r, k]
        ax.plot(lam, E_scores_df[c].to_numpy(), lw=1.2, color="tab:blue")
        ax.axhline(0.0, color="k", lw=0.7, alpha=0.6)
        ax.set_ylim(*ylims)
        ax.set_title(c)
        ax.grid(True, alpha=0.25)
        if r == nrows - 1:
            ax.set_xlabel("λ [µm]")
        if k == 0:
            ax.set_ylabel("E_j(λ)")
    # Hide unused subplots if any
    for idx in range(ncomp, nrows * ncols):
        r = idx // ncols; k = idx % ncols
        axes[r, k].axis("off")

    fig.suptitle("Eigensensitivity Spectra — separate panels", y=0.995, fontsize=12)
    fig.tight_layout()
    path_sep = output_dir / separate_grid_filename
    fig.savefig(path_sep, bbox_inches="tight")
    plt.close(fig)
    fig_paths.append(path_sep)

    return fig_paths


# --------------------------------------
# Function 9 — PCA Singular Values
# Title: save_pca_singular_values
# --------------------------------------
def save_pca_singular_values(
    pca: PCA,
    n_samples: int,
    output_dir: Path,
    filename: str = "pca_singular_values.csv",
) -> Path:
    """
    Save PCA magnitude metrics:
      - explained_variance (λ_j),
      - explained_variance_ratio,
      - sqrt_explained_variance = std of scores along component j,
      - singular_values σ_j = sqrt((n_samples - 1) * explained_variance_j).

    Returns
    -------
    path : Path to the saved CSV (index = E1..En).
    """
    comp_labels = [f"E{i+1}" for i in range(len(pca.explained_variance_))]
    ev = pca.explained_variance_.astype(float)
    evr = pca.explained_variance_ratio_.astype(float)
    std_scores = np.sqrt(ev)
    sing = pca.singular_values_.astype(float)  # sklearn provides σ_j directly

    df = pd.DataFrame(
        {
            "explained_variance": ev,
            "explained_variance_ratio": evr,
            "sqrt_explained_variance": std_scores,
            "singular_values": sing,
        },
        index=comp_labels,
    )
    path = output_dir / filename
    df.to_csv(path, index=True)
    return path

# --------------------------------------
# Function 10 — PCA Components Visualization
# Title: plot_pca_components_bargrid
# --------------------------------------
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_pca_components_bargrid(
    components_df: pd.DataFrame,
    output_dir: Path,
    separate_filename: str = "PCA_components_separate_grid.png",
) -> List[Path]:
    """
    Visualize PCA components (loadings) separately as bar charts.

    Parameters
    ----------
    components_df : pd.DataFrame
        PCA loading matrix with shape (n_components, n_features).
        Rows are EV1..EVn (components), columns are original feature names (e.g., PC1..PC9).
        Each row is an eigenvector of the covariance (here, correlation) matrix of U_scaled.
    output_dir : Path
        Directory to save the figure(s).
    separate_filename : str
        Filename for the multi-panel (separate subplots) figure.

    Returns
    -------
    fig_paths : List[Path]
        Paths to saved figure files.

    Notes
    -----
    - Bars show the contribution (loading) of each original feature to a given principal component.
    - Because columns were standardized, these loadings are scale-invariant (correlation PCA).
    - Signs are arbitrary up to a global flip per component.
    """
    fig_paths: List[Path] = []
    comp_names = list(components_df.index)     # e.g., ["EV1", ..., "EV9"]
    feat_names = list(components_df.columns)   # e.g., ["PC1", ..., "PC9"]
    ncomp = len(comp_names)
    nfeat = len(feat_names)

    # Separate grid of bar charts
    ncols = 3
    nrows = int(np.ceil(ncomp / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(10.0, 2.8 * nrows), sharex=False, sharey=True
    )
    axes = np.atleast_2d(axes)

    # Determine symmetric y-limits for comparison across components
    vmax = float(np.abs(components_df.to_numpy()).max())
    ypad = 0.05 * vmax if vmax > 0 else 0.1
    ylims = (-vmax - ypad, vmax + ypad)

    for idx, cname in enumerate(comp_names):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        vals = components_df.loc[cname].to_numpy(dtype=float)
        ax.bar(np.arange(nfeat), vals, color="tab:blue", alpha=0.85)
        ax.set_title(cname)
        ax.set_ylim(*ylims)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(np.arange(nfeat))
        ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=9)
        if c == 0:
            ax.set_ylabel("Loading (eigenvector elements)")
        if r == nrows - 1:
            ax.set_xlabel("Original features")

    # Hide any unused subplots
    for idx in range(ncomp, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    fig.suptitle("PCA Components (Loadings) — separate panels", y=0.995, fontsize=12)
    fig.tight_layout()
    out_path = output_dir / separate_filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    fig_paths.append(out_path)

    return fig_paths

# --------------------------------------
# Function 11 — PCA Components Heatmap
# Title: plot_pca_components_heatmap
# --------------------------------------
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_pca_components_heatmap(
    components_df: pd.DataFrame,
    output_dir: Path,
    filename: str = "PCA_components_heatmap.png",
    cmap_name: str = "RdBu_r",
    figsize: tuple = (8.5, 4.8),
    annotate: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> List[Path]:
    """
    Draw a heatmap of the full PCA loading matrix (components_df).

    Parameters
    ----------
    components_df : pd.DataFrame
        PCA loading matrix with shape (n_components, n_features).
        Rows are EV1..EVn (eigenvectors/components), columns are original features (e.g., PC1..PC9).
    output_dir : Path
        Directory to save the figure.
    filename : str
        Output image filename.
    cmap_name : str
        Diverging colormap name (centered at zero).
    figsize : tuple
        Figure size (width, height).
    annotate : bool
        If True, write numeric loading values in each cell.
    vmin, vmax : Optional[float]
        Color scale limits. If None, symmetrical limits based on max absolute loading.

    Returns
    -------
    fig_paths : List[Path]
        List with the saved figure path.

    Notes
    -----
    - Color scale is symmetric around zero to reflect positive/negative loadings.
    - Axes:
        x-axis = original features (PC1..PC9),
        y-axis = components (EV1..EV9).
    """
    fig_paths: List[Path] = []
    data = components_df.to_numpy(dtype=float)  # shape (n_components, n_features)
    comp_names = list(components_df.index)      # ["EV1", ..., "EV9"]
    feat_names = list(components_df.columns)    # ["PC1", ..., "PC9"]

    ncomp, nfeat = data.shape

    # Symmetric color scale about zero
    if vmin is None or vmax is None:
        amax = float(np.max(np.abs(data))) if data.size else 1.0
        vmin = -amax
        vmax = +amax

    # Use pcolormesh with explicit cell edges to avoid half-pixel shifts
    x_edges = np.arange(0, nfeat + 1, dtype=float)
    y_edges = np.arange(0, ncomp + 1, dtype=float)

    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(
        x_edges, y_edges, data,
        cmap=cmap, shading="flat", vmin=vmin, vmax=vmax
    )

    # Tick labels at cell centers
    ax.set_xticks(np.arange(nfeat) + 0.5)
    ax.set_xticklabels(feat_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(ncomp) + 0.5)
    ax.set_yticklabels(comp_names)

    ax.set_xlabel("Original features (PC)")
    ax.set_ylabel("Components (EV = eigenvectors)")
    ax.set_title("PCA Components (Loadings) — heatmap")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Loading (eigenvector element)")

    # Optional numeric annotations
    if annotate:
        for i in range(ncomp):
            for j in range(nfeat):
                ax.text(
                    j + 0.5, i + 0.5, f"{data[i, j]:.2f}",
                    ha="center", va="center", fontsize=8, color="k"
                )

    fig.tight_layout()
    out_path = output_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    fig_paths.append(out_path)
    return fig_paths

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load U(λ) and sort by λ ascending
    wavelength_index, U_df, wl_source = load_unit_sensitivity_matrix(
        INPUT_CSV_PATH, lambda_min_um=ASSUMED_LAMBDA_MIN_UM, lambda_max_um=ASSUMED_LAMBDA_MAX_UM
    )

    # Step 1: Heatmap of U(λ) — wavelength-accurate with de-crowded y-axis
    fig_paths: List[Path] = []
    heatmap_path = plot_heatmap_U_pcolormesh(
        U_df, OUTPUT_DIR, cmap_name=HEATMAP_CMAP, figsize=(6.8, 7.2),
        y_tick_count=10, y_tick_format="{:.1f}", y_tick_labelsize=8.0
    )
    fig_paths.append(heatmap_path)

    # Step 1 diagnostics
    peaks_df = compute_and_save_column_peaks(U_df)
    pc1_line_path = plot_PC1_line(U_df, OUTPUT_DIR)
    fig_paths.append(pc1_line_path)

    # Step 2: Scaling and PCA to obtain E(λ)
    _, U_scaled_df, pca, E_scores_df, evr_df, components_df = scale_and_pca(
        U_df, n_components=N_COMPONENTS
    )

    # New: Plot eigensensitivity spectra (overlapped + separate grid)
    eig_figs = plot_eigensensitivity_spectra(E_scores_df, OUTPUT_DIR)
    fig_paths.extend(eig_figs)

    # Save outputs (CSVs + figs + peaks)
    saved_paths = save_outputs(
        OUTPUT_DIR, U_scaled_df, E_scores_df, evr_df, components_df, fig_paths, peaks_df
    )

    # New: Save PCA singular values and related quantities
    sv_path = save_pca_singular_values(pca, n_samples=E_scores_df.shape[0], output_dir=OUTPUT_DIR)
    saved_paths.append(sv_path)

    # Main patch — call Function 10 to visualize PCA components (loadings)
    comp_figs = plot_pca_components_bargrid(components_df, OUTPUT_DIR)
    fig_paths.extend(comp_figs)

    # Main patch — call Function 11 to visualize the full PCA components matrix as a heatmap
    comp_heatmap_figs = plot_pca_components_heatmap(
        components_df,
        OUTPUT_DIR,
        filename="PCA_components_heatmap.png",
        cmap_name="RdBu_r",
        figsize=(8.5, 4.8),
        annotate=False  # set True if you want numeric values in cells
    )
    fig_paths.extend(comp_heatmap_figs)

    # Log
    log_path = write_detailed_log(
        OUTPUT_DIR,
        INPUT_CSV_PATH,
        wl_source,
        wavelength_index,
        U_df,
        U_scaled_df,
        E_scores_df,
        evr_df,
        components_df,
        fig_paths + [sv_path],
        saved_paths + [sv_path],
    )

    print(f"Processing complete. Outputs saved to: {OUTPUT_DIR.resolve()}")
    print(f"Detailed log: {log_path.resolve()}")


if __name__ == "__main__":
    main()