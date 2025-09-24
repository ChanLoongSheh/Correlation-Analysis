# Project Proposal: Optimal Infrared Filter Selection via Eigenspectra Analysis of PCA-based Weighting Functions

**1. Project Overview**

The principal objective of this project is to define a set of spectrally independent, high-sensitivity channels for a ground-based infrared radiometer. We have successfully derived a set of nine physically-representative weighting functions, $U_i(\lambda)$, which describe the spectral radiance change $$\partial I(\lambda) / \partial c_i$$ for each of the nine principal modes of atmospheric variation.

However, as the provided correlation matrix `PC_pairwise_correlation_U.csv` clearly shows, these sensitivity spectra are themselves highly correlated. This is expected, as different physical processes (e.g., changes in temperature vs. water vapor at different altitudes) can produce radiance changes in similar spectral regions (e.g., the 6.3 µm water vapor band). A direct selection of filters based on the peaks of individual $U_i(\lambda)$ spectra would result in a redundant and suboptimal measurement system.

This project will transform the correlated sensitivity spectra into a new, orthogonal basis of "eigensensitivity spectra." The spectral features of this new basis will directly and objectively inform the optimal number, placement, and characteristics of the instrument's filters, ensuring maximal information retrieval.

**2. Technology Stack**

We will continue with the Python ecosystem, leveraging its powerful data analysis and visualization libraries.

*   **Core Libraries:**
    *   **Pandas:** For loading and initial handling of the `.csv` data.
    *   **NumPy:** For all numerical and matrix operations, including the calculation of the covariance matrix.
    *   **Scikit-learn:** Its `PCA` and `StandardScaler` modules will be used to perform the core transformation.
    *   **Matplotlib & Seaborn:** For generating the heatmap and for high-quality plots of the final eigensensitivity spectra.
    *   **SciPy:** The `signal` module will be invaluable for objectively identifying peaks in the spectra to guide filter placement.

*   **Functional Module Breakdown:**
    *   **Module 1 (Data Loading):** Realized using `Pandas`.
    *   **Module 2 (Sensitivity Visualization):** Realized using `Seaborn` for the heatmap.
    *   **Module 3 (PCA Transformation):** Realized using `scikit-learn.decomposition.PCA` and `scikit-learn.preprocessing.StandardScaler`.
    *   **Module 4 (Eigenspectra Analysis):** Realized using `Matplotlib` for plotting and `SciPy.signal` for peak finding.
    *   **Module 5 (Filter Definition):** A synthesis of the results from Module 4 to propose a final filter set.

The modules are sequential. The output of one module serves as the direct input for the next, creating a clear and logical workflow from data to instrument design recommendation.

**3. Detailed Implementation Plan**

Here is a step-by-step plan to execute the project.

---

### **Step 1: Data Loading and Initial Visualization**

**Objective:** Load the unit-coefficient sensitivity data and visualize its structure using a heatmap.

**Description:** We will begin by loading the `PC_U_unit_sensitivity.csv` file into a Pandas DataFrame and then converting it to a NumPy array, $U$, of shape (3501, 9).

The first task is to generate a heatmap of this matrix $U$.
*   The y-axis will represent the wavelength (2.5-20 µm).
*   The x-axis will represent the Principal Component index (PC1 to PC9).
*   The color intensity will represent the sensitivity $\partial I(\lambda) / \partial c_i$. It is important to use a diverging colormap (e.g., `coolwarm` or `RdBu_r`) to properly represent both positive and negative sensitivities, as both are physically significant.

This heatmap will give us a rough, qualitative overview of which spectral regions are most sensitive to atmospheric changes.

---

### **Step 2: Perform PCA on the Sensitivity Spectra**

**Objective:** To transform the correlated sensitivity spectra $U$ into a set of orthogonal eigensensitivity spectra $E$.

**Description:** This is the core analytical step.
1.  **Preprocessing:** The $U$ matrix must be preprocessed. We will use `StandardScaler` from scikit-learn to center the data (subtract the mean of each column) and scale it to unit variance. Scaling is crucial here to ensure that each PC's sensitivity spectrum contributes equally to the analysis, preventing a single, high-magnitude spectrum from dominating the PCA results.
2.  **PCA Execution:** Apply PCA to the scaled (3501, 9) matrix. Since we want to define 9 orthogonal channels, we will set `n_components=9`.
3.  **Outputs:** The PCA will produce the principal components, which are our new **eigensensitivity spectra**. Let us call this matrix $E$, of shape (3501, 9). The columns of $E$ are our new, orthogonal basis vectors for the instrument's spectral response. We will also analyze the `explained_variance_ratio_` to understand the relative importance of each eigensensitivity spectrum.

---

### **Step 3: Analysis of Eigensensitivity Spectra**

**Objective:** To visualize and interpret the physical meaning of the new, orthogonal eigensensitivity spectra.

**Description:** We will plot each of the 9 columns of the matrix $E$ against wavelength. These plots are the key to our analysis.

*   Each plot, $E_j(\lambda)$, shows a unique, orthogonal pattern of spectral sensitivity.
*   The **peaks (positive values)** represent wavelengths where an instrument channel would see an *increase* in radiance for this particular mode of sensitivity.
*   The **troughs (negative values)** represent wavelengths where a channel would see a *decrease* in radiance.
*   The magnitude (absolute value) at any wavelength indicates the strength of the sensitivity. The most prominent peaks and troughs are the prime candidates for filter placement.

---

### **Step. 4: Objective Identification of Optimal Wavelengths**

**Objective:** To move from qualitative visual inspection to an objective, data-driven selection of candidate filter wavelengths.

**Description:** For each of the most significant eigensensitivity spectra (e.g., the top 4 or 5, which will likely capture >99% of the sensitivity variance), we will use an algorithm to find the locations of the most prominent peaks.

The `scipy.signal.find_peaks` function is ideal for this. We will apply it to the absolute value of each eigenspectrum, $|E_j(\lambda)|$, to identify the wavelengths of maximum sensitivity, regardless of sign. We can set parameters like `height` and `distance` to ensure we only select the most significant and spectrally distinct peaks, avoiding minor ripples.

The output of this step will be a list of candidate center wavelengths for our filters, ranked by importance according to which eigenspectrum they were derived from and the peak's prominence.

---

### **Step 5: Proposal for an Optimal Filter Set**

**Objective:** Synthesize the analysis into a concrete recommendation for a set of 9 to 11 instrument filters.

**Description:** Based on the ranked list of candidate wavelengths from Step 4, we will propose a final filter set. The strategy will be:

1.  Select the most prominent, non-overlapping spectral peaks from the first few (e.g., 3-4) eigensensitivity spectra. This ensures we capture the most significant and independent channels of information.
2.  If more filters are needed to reach our target of 9-11, we will select secondary peaks from these same spectra or primary peaks from the subsequent, less dominant eigenspectra.
3.  For each selected center wavelength, we will also analyze the *width* of the corresponding peak in the eigenspectrum. This width provides a physically-grounded starting point for defining the optimal **filter bandwidth** (FWHM). A sharper peak suggests a narrower filter is required, while a broader peak can be covered by a wider filter.

The final deliverable of this project will be a report recommending a specific set of 9-11 filter center wavelengths and initial bandwidth estimates, complete with the supporting plots of the eigensensitivity spectra that justify these choices. This provides a robust, physically-meaningful, and data-driven design for the instrument.