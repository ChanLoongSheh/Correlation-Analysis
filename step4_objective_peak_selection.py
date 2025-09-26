# Program 3 — Step 4: Objective Identification of Optimal Wavelengths
# Author: (Your name)
# Description:
#   Implements Step 4 of the README (Project Proposal: Optimal Infrared Filter Selection via
#   Eigenspectra Analysis of PCA-based Weighting Functions).
#
#   Step 4 goal:
#     - Identify objective, spectrally distinct peak wavelengths λ* from the eigensensitivity
#       spectra E_j(λ), emphasizing the most informative components (by explained variance ratio).
#     - Provide a ranked candidate filter list (center wavelength and an initial bandwidth estimate).
#
#   Inputs (produced by Steps 1 and 2):
#     - outputs_optimal_filter_pca/E_scores_eigensensitivity_spectra.csv
#       Meaning: E(λ): eigensensitivity spectra as PCA scores; index = λ [µm]; columns = E1..En.
#     - outputs_optimal_filter_pca/pca_explained_variance_ratio.csv
#       Meaning: Explained variance ratio for each eigensensitivity spectrum E1..En.
#
#   Outputs (saved into a single folder without timestamp: 'outputs_step4_peak_selection'):
#     - Peaks_per_component.csv
#       Meaning: For each component E_j, all peaks on |E_j(λ)| with metrics (center λ, amplitude,
#                sign, prominence, FWHM-like width, left/right bounds, suggested bandwidth).
#     - CandidateFilters_ranked.csv
#       Meaning: A global, deconflicted ranked list of peaks across selected components with a
#                target number of filters. Includes: center λ [µm], parent component, EVR,
#                prominence-based score, suggested FWHM [µm], and selection flags/ranks.
#     - Eigensensitivity_overlapped_with_peaks.png
#       Meaning: All E_j(λ) overlapped with detected peak markers (up/down triangle shows sign).
#     - Eigensensitivity_separate_with_peaks.png
#       Meaning: Separate subplots of E_j(λ) with peak markers (up/down triangles show sign).
#     - CandidateFilters_on_overlapped.png
#       Meaning: Overlapped E_j(λ) with vertical bands at selected filter center λ and suggested
#                FWHM to visualize spectral placement and separation.
#     - processing_log_step4.txt
#       Meaning: A detailed log describing inputs, shapes, parameter settings, and the meaning of
#                each output and its file path.
#
# How to run:
#   1) Make sure Step 2 files exist in 'outputs_optimal_filter_pca'.
#   2) Run: python step4_objective_peak_selection.py
#
# Notes:
#   - E(λ) columns E1..En are the eigensensitivity spectra; index is the wavelength grid λ [µm].
#   - Peaks are located on |E_j(λ)|; the sign of E_j(λ*) at the peak is retained for interpretation.
#   - Widths are computed at rel_height=0.5 (≈ FWHM-like on |E_j|); converted to µm on a possibly
#     non-uniform λ-grid via linear interpolation of fractional indices.
#   - Greedy global selection enforces a minimum inter-filter spacing in µm and prioritizes peaks by
#     a weighted score combining component EVR and normalized prominence.
#
from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, peak_prominences, peak_widths


# ----------------------------
# User-configurable parameters
# ----------------------------

# Inputs from Step 2
INPUT_E_SCORES_CSV = Path("outputs_optimal_filter_pca") / "E_scores_eigensensitivity_spectra.csv"
INPUT_EVR_CSV     = Path("outputs_optimal_filter_pca") / "pca_explained_variance_ratio.csv"

# Output directory for Step 4 (no timestamp)
OUTPUT_DIR = Path("outputs_step4_peak_selection")

# Component selection policy
SELECT_METHOD = "cumulative_evr"  # "cumulative_evr" or "top_n"
CUM_EVR_THRESHOLD = 0.99          # If using cumulative EVR, keep components until >= this threshold
TOP_N_COMPONENTS = 6              # If using "top_n", keep this many components

# Peak detection thresholds (relative to per-component max |E_j|)
MIN_PROMINENCE_FRAC = 0.05        # min prominence as a fraction of max(|E_j|)
MIN_HEIGHT_FRAC     = 0.00        # optional min height of |E_j| (set 0 to disable)
MIN_PEAK_SEP_UM     = 0.10        # minimum spacing between peaks in µm (per-component detection)

# Global filter set synthesis
TARGET_FILTER_COUNT = 12          # desired number of final candidate filters
GLOBAL_MIN_SEP_UM   = 0.50        # enforce minimum separation between selected filters
MIN_BANDWIDTH_UM    = 0.10        # lower bound on suggested FWHM (avoid unrealistically narrow bands)
MAX_BANDWIDTH_UM    = 5.00        # upper bound on suggested FWHM

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
# Title: load_eigenspectra_and_evr
# --------------------------------------
def load_eigenspectra_and_evr(
    scores_csv: Path,
    evr_csv: Optional[Path] = None,
) -> Tuple[pd.Index, pd.DataFrame, Optional[pd.Series]]:
    """
    Load E(λ) eigensensitivity spectra (scores) and optional explained variance ratio.

    Returns
    -------
    wavelength_index : pd.Index
        Wavelength grid λ [µm], sorted ascending.
    E_df : pd.DataFrame
        E(λ) with shape (N_λ, N_comp); columns are E1..En.
    evr : Optional[pd.Series]
        Explained variance ratio indexed by component name (E1..En) if available; else None.

    Variables
    ---------
    λ (lambda_um): wavelength in micrometers [µm].
    E_j(λ): eigensensitivity spectrum for component j (arbitrary units).
    """
    if not scores_csv.exists():
        raise FileNotFoundError(f"E_scores file not found: {scores_csv}")

    E_df = pd.read_csv(scores_csv, index_col=0)
    # Drop accidental 'Unnamed' columns
    to_drop = [c for c in E_df.columns if str(c).startswith("Unnamed:")]
    if to_drop:
        E_df = E_df.drop(columns=to_drop)

    # Ensure wavelength index is numeric and sorted
    E_df.index = pd.to_numeric(E_df.index, errors="coerce")
    E_df.index.name = "wavelength_um"
    E_df = E_df.sort_index(axis=0, ascending=True)
    wavelength_index = E_df.index

    # Load optional EVR
    evr = None
    if evr_csv is not None and evr_csv.exists():
        tmp = pd.read_csv(evr_csv, index_col=0)
        # Standard name 'explained_variance_ratio'
        if "explained_variance_ratio" in tmp.columns:
            evr = tmp["explained_variance_ratio"].astype(float)
        else:
            # Allow single-column CSVs where the first (or only) column is EVR
            evr = tmp.iloc[:, 0].astype(float)
        evr.index = [str(i) for i in evr.index]  # ensure string component names

    return wavelength_index, E_df, evr


# ----------------------------------------------------------
# Function 2 — Helper for λ on fractional indices
# Title: fractional_index_to_lambda
# ----------------------------------------------------------
def fractional_index_to_lambda(frac_idx: float, lam: np.ndarray) -> float:
    """
    Convert a fractional index (e.g., from peak_widths left_ips/right_ips) to λ [µm]
    by linear interpolation on the λ-grid (which may be non-uniform).

    Parameters
    ----------
    frac_idx : float
        Fractional index (e.g., 10.3 means 30% between samples 10 and 11).
    lam : np.ndarray
        Wavelength array [µm] of length N_λ (ascending).

    Returns
    -------
    lam_val : float
        Interpolated wavelength [µm].
    """
    if frac_idx <= 0:
        return float(lam[0])
    n = len(lam)
    if frac_idx >= n - 1:
        return float(lam[-1])

    i0 = int(np.floor(frac_idx))
    i1 = i0 + 1
    t = frac_idx - i0
    return float(lam[i0] + t * (lam[i1] - lam[i0]))


# ------------------------------------------------------------------
# Function 3 — Per-component peak finding on |E_j(λ)|
# Title: find_peaks_per_component
# ------------------------------------------------------------------
def find_peaks_per_component(
    lam: np.ndarray,
    E_df: pd.DataFrame,
    min_prom_frac: float = MIN_PROMINENCE_FRAC,
    min_height_frac: float = MIN_HEIGHT_FRAC,
    min_sep_um: float = MIN_PEAK_SEP_UM,
) -> Dict[str, pd.DataFrame]:
    """
    For each component E_j(λ), find peaks on |E_j(λ)| and compute metrics.

    Returns
    -------
    peaks_by_comp : Dict[str, pd.DataFrame]
        Key = component name (e.g., "E1"); Value = DataFrame with columns:
          - lambda_um: peak center [µm]
          - amplitude: E_j(λ_peak) (signed)
          - abs_amplitude: |E_j(λ_peak)|
          - sign: sign(amplitude) in {-1, 0, +1}
          - prominence: prominence on |E_j|
          - left_um: left bound [µm] at rel_height=0.5 (≈ FWHM-like)
          - right_um: right bound [µm] at rel_height=0.5
          - width_um: FWHM-like width = right_um - left_um
          - rel_height: 0.5 (parameter passed to peak_widths)
          - peak_index: integer sample index of the peak (for reference)
          - peak_rank_by_prom: rank within component by descending prominence (1 = strongest)
          - max_abs_in_component: max(|E_j(λ)|) for normalization
    """
    peaks_by_comp: Dict[str, pd.DataFrame] = {}

    # Approximate samples count per µm for distance conversion (robust to non-uniform grids)
    dlam = np.diff(lam)
    dlam_med = float(np.median(dlam)) if np.all(np.isfinite(dlam)) and len(dlam) > 0 else 0.01
    min_distance_samples = max(1, int(np.floor(min_sep_um / max(dlam_med, 1e-8))))

    for comp in E_df.columns:
        y = E_df[comp].to_numpy(dtype=float)
        y_abs = np.abs(y)

        max_abs = float(np.max(y_abs)) if y_abs.size else 0.0
        if max_abs <= 0:
            # Flat component
            peaks_by_comp[comp] = pd.DataFrame(
                columns=[
                    "lambda_um", "amplitude", "abs_amplitude", "sign",
                    "prominence", "left_um", "right_um", "width_um", "rel_height",
                    "peak_index", "peak_rank_by_prom", "max_abs_in_component"
                ]
            )
            continue

        min_height = float(min_height_frac * max_abs) if min_height_frac > 0 else None
        min_prom = float(min_prom_frac * max_abs)

        # Peak detection on |E_j|
        peaks, props = find_peaks(
            y_abs,
            height=min_height,
            prominence=min_prom,
            distance=min_distance_samples,
        )
        if peaks.size == 0:
            # Relax criteria once (e.g., half the prominence)
            peaks, props = find_peaks(
                y_abs,
                height=min_height if min_height is not None else None,
                prominence=0.5 * min_prom,
                distance=max(1, min_distance_samples // 2),
            )

        if peaks.size == 0:
            peaks_by_comp[comp] = pd.DataFrame(
                columns=[
                    "lambda_um", "amplitude", "abs_amplitude", "sign",
                    "prominence", "left_um", "right_um", "width_um", "rel_height",
                    "peak_index", "peak_rank_by_prom", "max_abs_in_component"
                ]
            )
            continue

        # Prominences and widths at rel_height = 0.5 (≈ FWHM-like)
        prominences = peak_prominences(y_abs, peaks)[0]
        widths, left_ips, right_ips, _ = peak_widths(y_abs, peaks, rel_height=0.5)

        # Convert to physical λ-units
        lam_peaks = lam[peaks].astype(float)
        lam_left = np.array([fractional_index_to_lambda(lp, lam) for lp in left_ips], dtype=float)
        lam_right = np.array([fractional_index_to_lambda(rp, lam) for rp in right_ips], dtype=float)
        width_um = lam_right - lam_left

        # Assemble DataFrame
        records = []
        for i, p in enumerate(peaks):
            amp = float(y[p])
            sign = int(np.sign(amp))
            rec = dict(
                lambda_um=float(lam_peaks[i]),
                amplitude=amp,
                abs_amplitude=float(np.abs(amp)),
                sign=sign,
                prominence=float(prominences[i]),
                left_um=float(lam_left[i]),
                right_um=float(lam_right[i]),
                width_um=float(width_um[i]),
                rel_height=0.5,
                peak_index=int(p),
                max_abs_in_component=max_abs,
            )
            records.append(rec)

        df = pd.DataFrame.from_records(records)
        # Rank by prominence descending (1 = strongest)
        if not df.empty:
            df = df.sort_values(by="prominence", ascending=False).reset_index(drop=True)
            df["peak_rank_by_prom"] = np.arange(1, len(df) + 1, dtype=int)
        peaks_by_comp[comp] = df

    return peaks_by_comp


# -------------------------------------------------------------------------
# Function 4 — Component selection by EVR
# Title: select_components_by_evr
# -------------------------------------------------------------------------
def select_components_by_evr(
    evr: Optional[pd.Series],
    all_components: List[str],
    method: str = SELECT_METHOD,
    cum_threshold: float = CUM_EVR_THRESHOLD,
    top_n: int = TOP_N_COMPONENTS,
) -> List[str]:
    """
    Select components to prioritize (e.g., top by cumulative EVR or a fixed top-N).

    Returns
    -------
    selected : List[str]
        Ordered list of component names (subset of all_components).
    """
    comps = list(all_components)
    if evr is None:
        # No EVR, fall back to first top-N as listed
        return comps[:top_n]

    # EVR indexed by component name (E1..En); sort desc by EVR
    evr_use = evr.reindex(comps).fillna(0.0).astype(float)
    evr_use = evr_use.sort_values(ascending=False)
    if method == "top_n":
        return list(evr_use.index[:top_n])

    # cumulative EVR
    csum = evr_use.cumsum()
    mask = csum <= cum_threshold
    # Ensure at least one (include the one that crosses threshold)
    selected = list(evr_use.index[mask])
    if len(selected) < len(evr_use):
        selected.append(evr_use.index[len(selected)])
    return selected


# ------------------------------------------------------------------------------------
# Function 5 — Aggregate peaks and produce a global, deconflicted filter recommendation
# Title: aggregate_and_rank_candidate_filters
# ------------------------------------------------------------------------------------
def aggregate_and_rank_candidate_filters(
    peaks_by_comp: Dict[str, pd.DataFrame],
    evr: Optional[pd.Series],
    target_count: int = TARGET_FILTER_COUNT,
    global_min_sep_um: float = GLOBAL_MIN_SEP_UM,
    min_bw_um: float = MIN_BANDWIDTH_UM,
    max_bw_um: float = MAX_BANDWIDTH_UM,
    selected_components: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merge all per-component peaks into a global pool, score and greedily select a deconflicted
    set of candidate filters with center λ and suggested FWHM.

    Scoring: score = EVR(component) * (prominence / max_abs_in_component).
    Greedy selection: sort by score desc, then accept if ≥ global_min_sep_um away from already selected.

    Returns
    -------
    candidates_df : pd.DataFrame
        All peaks (across selected components), with columns:
          - component, lambda_um, amplitude, abs_amplitude, sign
          - prominence, width_um, left_um, right_um, rel_height
          - max_abs_in_component
          - evr_component
          - norm_prominence = prominence / max_abs_in_component
          - score = evr_component * norm_prominence
          - selected_for_set: bool
          - selection_rank: rank among selected (1..K); NaN for non-selected
          - suggested_FWHM_um: width clipped to [min_bw_um, max_bw_um]
    """
    records = []
    comp_list = list(peaks_by_comp.keys()) if selected_components is None else list(selected_components)
    evr_use = None
    if evr is not None:
        # Normalize EVR series to include zeros for missing
        evr_use = evr.reindex(comp_list).fillna(0.0).astype(float)

    for comp in comp_list:
        df = peaks_by_comp.get(comp, None)
        if df is None or df.empty:
            continue
        evr_c = float(evr_use.loc[comp]) if evr_use is not None and comp in evr_use.index else 0.0
        for _, r in df.iterrows():
            max_abs = float(r["max_abs_in_component"]) if "max_abs_in_component" in r else max(1e-12, float(r["abs_amplitude"]))
            norm_prom = float(r["prominence"]) / max(max_abs, 1e-12)
            score = evr_c * norm_prom
            w = float(r["width_um"])
            w_clip = float(np.clip(w, min_bw_um, max_bw_um))
            records.append(
                dict(
                    component=comp,
                    lambda_um=float(r["lambda_um"]),
                    amplitude=float(r["amplitude"]),
                    abs_amplitude=float(r["abs_amplitude"]),
                    sign=int(r["sign"]),
                    prominence=float(r["prominence"]),
                    left_um=float(r["left_um"]),
                    right_um=float(r["right_um"]),
                    width_um=float(r["width_um"]),
                    rel_height=float(r["rel_height"]),
                    max_abs_in_component=max_abs,
                    evr_component=evr_c,
                    norm_prominence=norm_prom,
                    score=score,
                    suggested_FWHM_um=w_clip,
                )
            )
    if len(records) == 0:
        return pd.DataFrame(columns=[
            "component","lambda_um","amplitude","abs_amplitude","sign","prominence",
            "left_um","right_um","width_um","rel_height","max_abs_in_component",
            "evr_component","norm_prominence","score",
            "selected_for_set","selection_rank","suggested_FWHM_um"
        ])

    pool = pd.DataFrame.from_records(records)
    pool = pool.sort_values(by=["score","prominence","abs_amplitude"], ascending=[False, False, False]).reset_index(drop=True)

    # Greedy global selection with minimum spacing
    selected_flags = np.zeros(len(pool), dtype=bool)
    selection_rank = np.full(len(pool), fill_value=np.nan, dtype=float)
    selected_centers: List[float] = []
    rank = 0
    for i in range(len(pool)):
        if rank >= target_count:
            break
        lam_i = float(pool.loc[i, "lambda_um"])
        if all(abs(lam_i - c) >= global_min_sep_um for c in selected_centers):
            selected_flags[i] = True
            rank += 1
            selection_rank[i] = rank
            selected_centers.append(lam_i)

    pool["selected_for_set"] = selected_flags
    pool["selection_rank"] = selection_rank
    return pool


# --------------------------------------------------------
# Function 6 — Figures (overlapped and separate with peaks)
# Title: plot_eigenspectra_with_peaks
# --------------------------------------------------------
def plot_eigenspectra_with_peaks(
    lam: np.ndarray,
    E_df: pd.DataFrame,
    peaks_by_comp: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Make two diagnostic plots:
      - Eigensensitivity_overlapped_with_peaks.png
      - Eigensensitivity_separate_with_peaks.png
    """
    # Overlapped
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    cmap = plt.get_cmap("tab10")

    y_min = float(np.min(E_df.to_numpy()))
    y_max = float(np.max(E_df.to_numpy()))
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    for i, comp in enumerate(E_df.columns):
        col = cmap(i % 10)
        ax.plot(lam, E_df[comp].to_numpy(), lw=1.2, color=col, label=comp)
        # Peaks
        dfp = peaks_by_comp.get(comp, None)
        if dfp is not None and not dfp.empty:
            # Up triangle for positive, down for negative
            pos = dfp[dfp["sign"] >= 0]
            neg = dfp[dfp["sign"] < 0]
            if not pos.empty:
                ax.plot(pos["lambda_um"], E_df.loc[pos["lambda_um"], comp], "^", color=col, ms=5, alpha=0.9)
            if not neg.empty:
                ax.plot(neg["lambda_um"], E_df.loc[neg["lambda_um"], comp], "v", color=col, ms=5, alpha=0.9)

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
    ax.set_xlim(float(lam.min()), float(lam.max()))
    ax.set_xlabel("Wavelength λ [µm]")
    ax.set_ylabel("Eigensensitivity E_j(λ) [arb. units]")
    ax.set_title("Eigensensitivity Spectra — overlapped with peaks")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=min(3, E_df.shape[1]), fontsize=9)
    fig.tight_layout()
    p_over = output_dir / "Eigensensitivity_overlapped_with_peaks.png"
    fig.savefig(p_over, bbox_inches="tight")
    plt.close(fig)

    # Separate
    ncomp = E_df.shape[1]
    ncols = 3
    nrows = int(np.ceil(ncomp / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(9.3, 2.9 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for i, comp in enumerate(E_df.columns):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        ax.plot(lam, E_df[comp].to_numpy(), lw=1.2, color="tab:blue")
        dfp = peaks_by_comp.get(comp, None)
        if dfp is not None and not dfp.empty:
            pos = dfp[dfp["sign"] >= 0]
            neg = dfp[dfp["sign"] < 0]
            if not pos.empty:
                ax.plot(pos["lambda_um"], E_df.loc[pos["lambda_um"], comp], "^", color="tab:orange", ms=5, alpha=0.95)
            if not neg.empty:
                ax.plot(neg["lambda_um"], E_df.loc[neg["lambda_um"], comp], "v", color="tab:red", ms=5, alpha=0.95)
        ax.axhline(0.0, color="k", lw=0.7, alpha=0.6)
        ax.set_title(comp)
        ax.grid(True, alpha=0.25)
        if r == nrows - 1:
            ax.set_xlabel("λ [µm]")
        if c == 0:
            ax.set_ylabel("E_j(λ)")

    # Hide unused axes
    for idx in range(ncomp, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    fig.suptitle("Eigensensitivity Spectra — separate panels with peaks", y=0.995, fontsize=12)
    fig.tight_layout()
    p_sep = output_dir / "Eigensensitivity_separate_with_peaks.png"
    fig.savefig(p_sep, bbox_inches="tight")
    plt.close(fig)

    return p_over, p_sep


# -------------------------------------------------------------------
# Function 7 — Figure with selected candidate filters (bands overlay)
# Title: plot_candidate_filters_overlay
# -------------------------------------------------------------------
def plot_candidate_filters_overlay(
    lam: np.ndarray,
    E_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Overlapped E_j(λ) with vertical bands marking selected candidate filters
    (center λ and suggested FWHM).
    """
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    cmap = plt.get_cmap("tab10")

    y_min = float(np.min(E_df.to_numpy()))
    y_max = float(np.max(E_df.to_numpy()))
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    for i, comp in enumerate(E_df.columns):
        col = cmap(i % 10)
        ax.plot(lam, E_df[comp].to_numpy(), lw=1.2, color=col, label=comp)

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)

    sel = candidates_df[candidates_df["selected_for_set"] == True].copy()
    sel = sel.sort_values(by="selection_rank", ascending=True)
    for _, r in sel.iterrows():
        center = float(r["lambda_um"])
        bw = float(r["suggested_FWHM_um"])
        left = center - 0.5 * bw
        right = center + 0.5 * bw
        ax.axvspan(left, right, color="gold", alpha=0.18)
        ax.axvline(center, color="goldenrod", lw=1.0, alpha=0.75)
        ax.text(center, y_max + 0.02 * (y_max - y_min), f"{center:.2f} µm", rotation=90,
                va="bottom", ha="center", fontsize=8, color="goldenrod")

    ax.set_xlim(float(lam.min()), float(lam.max()))
    ax.set_xlabel("Wavelength λ [µm]")
    ax.set_ylabel("Eigensensitivity E_j(λ) [arb. units]")
    ax.set_title("Selected Candidate Filters on Eigensensitivity Spectra")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=min(3, E_df.shape[1]), fontsize=9)
    fig.tight_layout()
    p = output_dir / "CandidateFilters_on_overlapped.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# --------------------------------------
# Function 8 — I/O
# Title: save_step4_outputs
# --------------------------------------
def save_step4_outputs(
    output_dir: Path,
    peaks_by_comp: Dict[str, pd.DataFrame],
    candidates_df: pd.DataFrame,
    fig_paths: List[Path],
) -> List[Path]:
    """
    Save CSVs and figures for Step 4. Returns list of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # Per-component peaks: concatenate with a 'component' column
    rows = []
    for comp, df in peaks_by_comp.items():
        if df is None or df.empty:
            continue
        dfc = df.copy()
        dfc.insert(0, "component", comp)
        # Use λ as the index for CSV compatibility with previous steps
        dfc = dfc.set_index("lambda_um")
        rows.append(dfc)
    if rows:
        peaks_all = pd.concat(rows, axis=0)
    else:
        peaks_all = pd.DataFrame(columns=["component"])

    path_peaks = output_dir / "Peaks_per_component.csv"
    peaks_all.to_csv(path_peaks, index=True)
    saved.append(path_peaks)

    # Candidate filters (keep λ as index for consistency)
    cand_out = candidates_df.copy()
    if "lambda_um" in cand_out.columns:
        cand_out = cand_out.set_index("lambda_um")
    path_cands = output_dir / "CandidateFilters_ranked.csv"
    cand_out.to_csv(path_cands, index=True)
    saved.append(path_cands)

    for p in fig_paths:
        saved.append(p)

    return saved


# --------------------------------------
# Function 9 — Logging
# Title: write_detailed_log_step4
# --------------------------------------
def write_detailed_log_step4(
    output_dir: Path,
    input_scores_path: Path,
    input_evr_path: Optional[Path],
    wavelength_index: pd.Index,
    E_df: pd.DataFrame,
    selected_components: List[str],
    peaks_by_comp: Dict[str, pd.DataFrame],
    candidates_df: pd.DataFrame,
    saved_paths: List[Path],
) -> Path:
    """
    Write a detailed log describing inputs, parameter choices, outputs, and meanings for Step 4.
    """
    log_lines: List[str] = []

    log_lines.append("Project: Optimal Infrared Filter Selection via Eigenspectra Analysis")
    log_lines.append("Step 4: Objective Identification of Optimal Wavelengths — peak finding and filter synthesis.")
    log_lines.append("E_scores columns (E1..En) are the eigensensitivity spectra E(λ) vs wavelength λ.")
    log_lines.append("This step reads Step 2 outputs, detects peaks on |E_j(λ)|, and proposes candidate filters.")
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
    log_lines.append(f"  - Peaks_by_component: {sum((0 if (v is None or v.empty) else len(v)) for v in peaks_by_comp.values())} total peaks across {len(peaks_by_comp)} components.")
    log_lines.append(f"  - Candidate filters pool: {candidates_df.shape} rows (includes selected and non-selected).")
    log_lines.append("")

    # Parameters
    log_lines.append("Key parameters:")
    log_lines.append(f"  - Component selection method: {SELECT_METHOD}; CUM_EVR_THRESHOLD={CUM_EVR_THRESHOLD}; TOP_N_COMPONENTS={TOP_N_COMPONENTS}")
    log_lines.append(f"  - Peak detection: MIN_PROMINENCE_FRAC={MIN_PROMINENCE_FRAC}, MIN_HEIGHT_FRAC={MIN_HEIGHT_FRAC}, MIN_PEAK_SEP_UM={MIN_PEAK_SEP_UM} µm")
    log_lines.append(f"  - Global filter synthesis: TARGET_FILTER_COUNT={TARGET_FILTER_COUNT}, GLOBAL_MIN_SEP_UM={GLOBAL_MIN_SEP_UM} µm")
    log_lines.append(f"  - Bandwidth clipping: MIN_BANDWIDTH_UM={MIN_BANDWIDTH_UM} µm, MAX_BANDWIDTH_UM={MAX_BANDWIDTH_UM} µm")
    log_lines.append(f"  - Selected components (priority order): {', '.join(selected_components)}")
    log_lines.append("")

    # Outputs
    log_lines.append("Outputs (what and where):")
    for p in saved_paths:
        fname = p.name
        if fname == "Peaks_per_component.csv":
            fdesc = ("Per-component peaks on |E_j(λ)| with metrics: center λ [µm], signed amplitude, sign, "
                     "prominence, FWHM-like width [µm] (rel_height=0.5), and bounds. Useful to inspect "
                     "dominant spectral lobes per eigenspectrum.")
        elif fname == "CandidateFilters_ranked.csv":
            fdesc = ("Global, deconflicted candidate filter list. Columns: parent component, λ [µm] (index), "
                     "prominence, EVR(component), normalized prominence, score, suggested_FWHM [µm], and "
                     "selection flags/ranks. Sorted by descending score; selection enforces global min spacing.")
        elif fname == "Eigensensitivity_overlapped_with_peaks.png":
            fdesc = "Overlapped E_j(λ) plot with peak markers (triangles up/down indicate the sign of E_j at λ*)."
        elif fname == "Eigensensitivity_separate_with_peaks.png":
            fdesc = "Separate subplots of each E_j(λ) with annotated peak markers for clear per-component diagnostics."
        elif fname == "CandidateFilters_on_overlapped.png":
            fdesc = "Overlapped E_j(λ) with vertical bands at selected filter centers and suggested FWHM for placement."
        else:
            fdesc = "Additional output."
        log_lines.append(f"  - {p.resolve()}")
        log_lines.append(f"    Meaning: {fdesc}")

    log_lines.append("")
    log_lines.append("Notes:")
    log_lines.append("- Peaks are detected on |E_j(λ)| to capture sensitivity magnitude irrespective of sign; the sign at λ* is recorded.")
    log_lines.append("- Prominence provides a robust measure of peak distinctness relative to neighboring troughs.")
    log_lines.append("- Widths are computed at rel_height=0.5 on |E_j|, analogous to FWHM; suggested bandwidths are clipped to user bounds.")
    log_lines.append("- Global candidate selection balances component importance (EVR) and per-peak strength (normalized prominence) while enforcing spectral separation.")
    log_lines.append("")

    log_path = output_dir / "processing_log_step4.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    return log_path


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load E(λ) and optional EVR
    wavelength_index, E_df, evr = load_eigenspectra_and_evr(INPUT_E_SCORES_CSV, evr_csv=INPUT_EVR_CSV)
    lam = wavelength_index.to_numpy(dtype=float)

    # Determine priority components
    selected_components = select_components_by_evr(evr, list(E_df.columns), method=SELECT_METHOD,
                                                   cum_threshold=CUM_EVR_THRESHOLD, top_n=TOP_N_COMPONENTS)

    # Step 4a: Per-component peaks on |E_j(λ)|
    peaks_by_comp = find_peaks_per_component(
        lam, E_df,
        min_prom_frac=MIN_PROMINENCE_FRAC,
        min_height_frac=MIN_HEIGHT_FRAC,
        min_sep_um=MIN_PEAK_SEP_UM,
    )

    # Step 4b: Aggregate and propose candidate filters
    candidates_df = aggregate_and_rank_candidate_filters(
        peaks_by_comp,
        evr=evr,
        target_count=TARGET_FILTER_COUNT,
        global_min_sep_um=GLOBAL_MIN_SEP_UM,
        min_bw_um=MIN_BANDWIDTH_UM,
        max_bw_um=MAX_BANDWIDTH_UM,
        selected_components=selected_components,
    )

    # Figures
    fig_paths: List[Path] = []
    p_over, p_sep = plot_eigenspectra_with_peaks(lam, E_df, peaks_by_comp, OUTPUT_DIR)
    fig_paths.extend([p_over, p_sep])

    p_cand = plot_candidate_filters_overlay(lam, E_df, candidates_df, OUTPUT_DIR)
    fig_paths.append(p_cand)

    # Save outputs
    saved_paths = save_step4_outputs(OUTPUT_DIR, peaks_by_comp, candidates_df, fig_paths)

    # Log
    log_path = write_detailed_log_step4(
        OUTPUT_DIR,
        INPUT_E_SCORES_CSV,
        INPUT_EVR_CSV if INPUT_EVR_CSV.exists() else None,
        wavelength_index,
        E_df,
        selected_components,
        peaks_by_comp,
        candidates_df,
        saved_paths,
    )

    print(f"Step 4 complete. Outputs saved to: {OUTPUT_DIR.resolve()}")
    print(f"Detailed log: {log_path.resolve()}")


if __name__ == "__main__":
    main()