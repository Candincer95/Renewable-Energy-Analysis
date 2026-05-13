# 🌍 Renewable Energy: Multidimensional & Hierarchical Analysis

This project provides a sophisticated analytical look at the European renewable energy landscape using high-level data visualization techniques. Moving beyond simple trends, it employs **Multidimensional (Mekko)** and **Hierarchical (Sunburst)** charts to reveal market structures, national portfolio strategies, and technical specifications.

## 🚀 Overview
While standard charts show simple quantities, this project focuses on the *interplay* between different variables:
- **Mekko Chart:** Analyzes global market share (width) vs. national energy mix (height) simultaneously.
- **Sunburst Chart:** Explores the internal anatomy of national grids, specifically focusing on the UK's regional distribution and the technical split of Wind energy (Onshore vs. Offshore).

## 🛠️ Technologies Used
- **Python**: Core data processing.
- **Streamlit**: Interactive web dashboard framework.
- **Plotly**: Advanced interactive visualization engine (used for high-performance Mekko and Sunburst rendering).
- **Pandas & NumPy**: Data manipulation and statistical normalization.

## 📊 Visualizations & Analytical Depth

### 1. Mekko Chart (Multidimensional Market Share)
- **X-Axis (Width):** Represents the country's total renewable capacity relative to the total market.
- **Y-Axis (Height):** Displays the internal percentage distribution of energy sources within that specific country.
- **Insight:** Allows for immediate comparison of market leaders (Germany, UK) vs. specialized markets (Denmark).

### 2. Sunburst Chart (Hierarchical Drill-down)
- **Hierarchy:** Country > Region (UK specific) > Energy Source > Technical Detail (Wind Spec).
- **Interactive Drill-down:** Users can click on a country to isolate its national grid or explore the Onshore/Offshore balance of the wind sector.
- **Insight:** Reveals the asymmetrical nature of power grids and regional infrastructure focus.

## 📂 Dataset
The analysis is based on the **Renewable Power Plants** time-series data, focusing on the latest recorded market snapshot (Nov 2020) to provide an up-to-date structural analysis.
- [Dataset Source (Kaggle)](https://www.kaggle.com/datasets/eugeniyosetrov/renewable-power-plants)

## 💻 Run the Project
To launch the interactive dashboard, ensure you have the requirements installed and run:

```bash
# To view the Mekko Chart analysis
streamlit run mekko_chart.py

# To view the Hierarchical Sunburst analysis
streamlit run sunburst_chart.py
