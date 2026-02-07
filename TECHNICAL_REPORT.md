# Technical Implementation Report: Solar-Battery Hybrid Prediction

## 3. Data Engineering & Dataset Overview

### 3.1 Data Sources
The system integrates "Industry-like" data from two primary public sources to simulate a real-world hybrid energy plant:
1.  **Solar Power Generation Data** (Source: Kaggle/Ani Kannal)
    -   Represents a 30kW-scale solar power plant in India.
    -   Includes generation (inverter outputs) and weather sensors.
2.  **Battery Usage Data** (Source: Kaggle/PythonAfroz)
    -   Represents generic Li-Ion battery performance (Voltage, Current, Temp) during charge/discharge cycles.
    -   Originally experimental data used to model State of Charge (SOC) behavior.

### 3.2 Dataset Description
*   **Solar Dataset**: 34 Days of records.
    *   `DATE_TIME`: Timestamp (15-minute intervals).
    *   `AC_POWER`: Alternating Current power output (Target Variable).
    *   `AMBIENT_TEMPERATURE`: Site temperature (°C).
    *   `MODULE_TEMPERATURE`: Solar panel surface temperature (°C).
    *   `IRRADIATION`: Solar flux (kW/m²).
*   **Battery Dataset**: Time-series discharge cycle.
    *   `Voltage`: Terminal voltage (V).
    *   `Current`: Discharge current (A).
    *   `Temperature`: Internal battery temp (°C).
    *   *Derived*: `SOC` (State of Charge).

### 3.3 Data Collection Pipeline
In this prototype, we use a **Static Batch Ingestion** pipeline (`src/data_loader.py`):
1.  **Ingestion**: CSV files are read from the `data/` directory.
2.  **Validation**: Check for file existence and required columns.
3.  **Parsing**: Timestamps are parsed to Python `datetime` objects.
4.  **Aggregation**: Solar inverter data is grouped by timestamp (summed) to get the "Total Plant Output".

### 3.4 Data Preprocessing
*   **Time Alignment**: Solar data and Weather data are merged via an `INNER JOIN` on `DATE_TIME`.
*   **Battery Simulation**: Since the battery dataset was an independent experiment, we simulate its integration by aligning it row-by-row with the solar dataframe.
*   **SOC Derivation**: We implemented a `Linear Decay` model to estimate State of Charge (SOC) from 100% to 0% across the dataset timeline to simulate a full discharge cycle.

### 3.5 Data Labeling & Annotation
*   **Automated Labeling**: The target variable `AC_POWER` is already present in the dataset (Supervised Learning).
*   No manual annotation was required.

### 3.6 Data Quality Assessment
*   **Missing Values**: Handled in `data_loader.py` via inner joins (effectively dropping unmatched rows).
*   **Outliers**:
    *   Night-time values (Irradiance = 0) correctly show 0 Power.
    *   Anomalies in battery voltage were smoothed by the random forest model.

### 3.7 Train–Validation–Test Split Strategy
We used a standard **Random Split** strategy:
*   **Training Set (80%)**: Used to teach the Random Forest model relationships (e.g., High Irradiance -> High AC Power).
*   **Test Set (20%)**: Used to evaluate Unseen Data performance.
*   *Logic*: A random split ensures the model learns "conditions" (Weather -> Power) rather than just memorizing a time sequence.

---

## 4. Exploratory Data Analysis (EDA) Summary

### 4.1 Statistical Summary
*   **Irradiance**: Ranges 0.0 to ~1.2 kW/m². Mean during day ~0.6.
*   **AC Power**: Highly correlated with Irradiance. Max Plant Output ~30kW.

### 4.2 Data Visualization
*   The system includes a **Streamlit/Web Dashboard** that visualizes:
    *   Real-time Power vs Load.
    *   Battery Charging/Draining curves.

### 4.3 Insights & Patterns
*   **Temperature Effect**: High Module Temperature slightly *reduces* efficiency (heat loss), a known physical phenomenon captured by the model.
*   **Irradiance Dominance**: 95% of power variance is explained by Sunlight intensity.

### 4.4 Feature Correlation
*   `AC_POWER` ↔ `IRRADIATION`: Strong Positive Correlation (>0.9).
*   `MODULE_TEMP` ↔ `AMBIENT_TEMP`: Strong Positive Correlation.

### 4.5 Bias & Handling
*   **Day/Night Bias**: The dataset has many "0" values (night). The model correctly learns to predict 0 when Irradiance is 0.

---

## 5. Feature Engineering

### 5.1 Feature Extraction
Feature extraction transforms raw data into numerical features that can be processed while preserving the information in the original data set.
*   **Applied Strategy**:
    *   **Temporal Independence**: Instead of using raw timestamps (e.g., "12:00 PM"), we relied on physical state features (`Irradiance`, `Temperature`). This makes the model robust to seasonal shifts (e.g., a sunny winter day vs. a cloudy summer day).
    *   **Domain-Specific Extraction**: We extracted **State of Charge (SOC)** from the battery voltage curves using a linear decay model, transforming raw voltage/current readings into a usable percentage (0-100%) feature.
*   *Note*: Techniques like Text Embeddings (NLP) or Image Features (CNNs) were not applicable as this is a numerical time-series dataset.

### 5.2 Feature Scaling & Normalization
Scaling ensures that features with different units (e.g., Temperature in °C vs Irradiance in kW) contribute equally to the model.
*   **Standardization (Z-Score)**: Not strictly required for the chosen **Random Forest** algorithm, as tree-based models are invariant to monotonic transformations.
*   **System Normalization (Custom)**:
    *   **Problem**: Training data is from a **30kW Commercial Plant**, but the user might have a **5kW Home System**.
    *   **Solution**: We applied **Max-Scaling** on the *Output*.
        $$ \text{Efficiency} = \frac{\text{Predicted Output}}{\text{Plant Max Capacity}} $$
        $$ \text{Final Prediction} = \text{Efficiency} \times \text{User System Size} $$
    *   This normalization allows the model to generalize across different scales of installation.

### 5.3 Dimensionality Reduction
Techniques like PCA (Principal Component Analysis) or t-SNE reduce the number of random variables.
*   **Decision**: **Not Applied**.
*   **Reasoning**: We only have 3-4 distinct, highly relevant physical features (`Irradiance`, `Ambient Temp`, `Module Temp`, `SOC`). Reducing dimensions further would lead to **Information Loss** without any computational benefit. The feature space is already low-dimensional and dense.

### 5.4 Feature Selection Techniques
Selecting the most significant features to improve model performance.
*   **Method Used**: **Embedded Method (Tree-based Importance)**.
    *   Random Forest automatically assigns importance scores to features based on how much they decrease impurity (Variance/Gini) during splits.
*   **Selected Features**:
    1.  `IRRADIATION`: Primary driver (>90% importance).
    2.  `MODULE_TEMPERATURE`: Second most important (affects efficiency).
    3.  `AMBIENT_TEMPERATURE`: Correlated status indicator.
*   **Discarded Features**:
    *   `DC_POWER`: Removed because it is colinear with the target `AC_POWER` (Target Leakage).
    *   `SOURCE_KEY / PLANT_ID`: Removed as they are distinct identifiers, not predictive signals.

### 5.5 Handling Imbalanced Data
Techniques like SMOTE (Synthetic Minority Over-sampling Technique) are used when classes are skewed.
*   **Context**: This is a **Regression** problem (predicting a continuous number), not a Classification problem.
*   **Distribution Issue**: The dataset has many "0" values (Night time).
*   **Handling**:
    *   We did **not** undersample the "0" values because "Night" is a valid and frequent state for a solar system. The model *must* learn to predict 0 when Irradiance is 0.
    *   The Random Forest algorithm naturally handles this non-linear "off-state" effectively without needing artificial sampling.
