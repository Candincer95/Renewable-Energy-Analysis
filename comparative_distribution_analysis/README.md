# Comparative Distribution and Temporal Analysis Module

This module implements the complete data visualization pipeline for the global "Renewable Power Plants" dataset as specified in the multi-disciplinary interim report. The analysis evaluates spatial distribution across nations, temporal infrastructure growth tracking, and empirical validation against engineering literature.

## Module Structure
* `Comparative_Distribution_Analysis.ipynb`: Main Jupyter Notebook containing the data ingestion, preprocessing (melt transformation), and visualization cells.
* `outputs/`: Directory containing high-resolution static exports of all generated charts.

## Visualizations and Analytical Content

### 1. Comparative Distribution Analysis (Stacked Bar Chart)
* **Objective:** Compares cumulative installed renewable capacity across different countries, segmented by specific energy technologies (Solar, Wind, Bioenergy, Hydro, etc.).
* **Data Filter:** Uses the final available operational year (2020) threshold to evaluate recent geopolitical asset distributions.
* **Output File:** `outputs/stacked_bar_chart.png`

### 2. Temporal Infrastructure Expansion Analysis (Heatmap)
* **Objective:** Correlates capacity metrics against temporal intervals (years vs. months) to identify historical windows of rapid structural growth and energy sector investments.
* **Data Filter:** Evaluates Germany (DE) as a representative regional asset sample.
* **Output File:** `outputs/temporal_heatmap.png`

### 3. Growth Trajectory Analysis (Line Chart)
* **Objective:** Tracks the individual chronological development and acceleration velocity of different renewable technologies over the entire historical timeline.
* **Output File:** `outputs/line_chart_trend.png`

### 4. Cumulative Energy Mix Evolution (Stacked Area Chart)
* **Objective:** Visualizes the volumetric changes and shifting market shares among different generation technologies within the global renewable energy mix over time.
* **Output File:** `outputs/stacked_area_chart.png`

### 5. Theoretical Literature Validation (Scatter Plot)
* **Objective:** Maps empirical rüzgar hızı ($m/s$) versus output power ($kW$) measurements to structurally validate the non-linear, sigmoid-shaped turbine power curve models outlined in the report's literature review.
* **Output File:** `outputs/turbine_power_curve.png`

## Dependencies
The notebook utilizes standard engineering and data science libraries including:
* `pandas` for data structuring and melt transformations
* `plotly.express` for interactive distribution, line, and area plotting
* `seaborn` and `matplotlib.pyplot` for statistical matrices and correlation visualizations
