# Geospatial Analysis of European Renewable Energy Infrastructure

This submodule focuses on the spatial and temporal distribution of renewable energy power plants across Europe. The analysis investigates the topographical clustering of different renewable energy sources and tracks the infrastructural expansion trends from 1990 to 2020.

- **Language:** Python 3.14
- **Libraries:** Pandas, GeoPandas, Matplotlib
- **Data Source:** [Kaggle - Renewable Power Plants (Europe)](https://www.kaggle.com/datasets/vitorpinto/renewable-power-plants-europe)
- **Base Map Source:** [Natural Earth - Admin 0 Countries](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/)

## Generated Maps & Visualizations

| Map Type | File Name | Description |
| :--- | :--- | :--- |
| **Choropleth** | `choropleth_map.png` | Illustrates aggregated total renewable energy capacity (GW) per country |
| **Categorical** | `categorical_energy_map.png` | Point distribution categorized by sub-types |
| **Heatmap** | `heatmap_clusters.png` | Identifies high-density clusters of renewable installations using KDE density mapping. |
| **Spatiotemporal** | `spatiotemporal_expansion_map.png` | A 3-era facet map demonstrating the aggressive geographical expansion of infrastructure from 1990 to 2020. |

⚠️ Data Limitation Note: Missing Dataset Coverage (Poland)

During the visualization phase (prominently visible across the Choropleth Capacity Map, Proportional Bubble Map, and Kernel Density Heatmap), a sharp contrast is observed between high-density clusters in Central/Western Europe and the complete absence of data points within Polish borders.

