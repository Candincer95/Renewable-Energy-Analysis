import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# ======================================
# Streamlit UI
# Multi Country Temporal Trend Analysis
# ======================================

st.set_page_config(
    page_title="Temporal and Trend Analysis",
    layout="wide"
)

st.title("Temporal and Trend Analysis")
st.subheader("Multi-Line Chart - Renewable Energy Capacity Growth")

country_files = {
    "Germany": "renewable_power_plants_DE.csv",
    "France": "renewable_power_plants_FR.csv",
    "United Kingdom": "renewable_power_plants_UK.csv",
    "Denmark": "renewable_power_plants_DK.csv",
    "Switzerland": "renewable_power_plants_CH.csv",
    "Czech Republic": "renewable_power_plants_CZ.csv",
    "Poland": "renewable_power_plants_PL.csv",
    "Sweden": "renewable_power_plants_SE.csv"
}

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    list(country_files.keys()),
    default=["Germany", "France"]
)

start_year = st.sidebar.slider(
    "Select Start Year",
    min_value=1900,
    max_value=2025,
    value=1990
)

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

def read_csv_safely(data_path):
    try:
        return pd.read_csv(data_path, low_memory=False)
    except pd.errors.ParserError:
        return pd.read_csv(
            data_path,
            sep=';',
            engine='python',
            on_bad_lines='skip'
        )

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(14, 7))

plotted_any_country = False

for country in selected_countries:
    file_name = country_files[country]
    data_path = f"../data/{file_name}"

    df = read_csv_safely(data_path)

    required_columns = ['commissioning_date', 'electrical_capacity']

    if not all(col in df.columns for col in required_columns):
        st.warning(f"{country} skipped. Required columns not found.")
        st.write(f"{country} columns:", list(df.columns))
        continue

    df = df[required_columns]
    df = df.dropna()

    df['commissioning_date'] = pd.to_datetime(
        df['commissioning_date'],
        errors='coerce'
    )

    df = df.dropna(subset=['commissioning_date'])

    df['commissioning_year'] = df['commissioning_date'].dt.year

    df['electrical_capacity'] = pd.to_numeric(
        df['electrical_capacity'],
        errors='coerce'
    )

    df = df.dropna(subset=['electrical_capacity'])

    capacity_by_year = (
        df.groupby('commissioning_year')['electrical_capacity']
        .sum()
        .sort_index()
    )

    capacity_by_year = capacity_by_year[
        capacity_by_year.index >= start_year
    ]

    if capacity_by_year.empty:
        st.warning(f"{country} has no data after {start_year}.")
        continue

    ax.plot(
        capacity_by_year.index,
        capacity_by_year.values,
        marker='o',
        linewidth=3,
        label=country
    )

    plotted_any_country = True

ax.set_title(
    "Multi-Line Chart - Renewable Energy Capacity Growth",
    fontsize=18,
    fontweight='bold'
)

ax.set_xlabel("Year")
ax.set_ylabel("Total Electrical Capacity")
ax.tick_params(axis='x', rotation=45)

if plotted_any_country:
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    output_path = f"{output_folder}/multi_line_chart_capacity_growth.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')

    st.success(f"Graph saved successfully: {output_path}")
else:
    st.error("No country could be plotted. Please select countries with valid data.")

st.markdown("""
### Analysis

This multi-line chart compares renewable energy capacity growth across selected countries over time.

- Multiple countries can be selected from the sidebar.
- The start year can be adjusted dynamically.
- Countries with incompatible columns are skipped automatically.
""")