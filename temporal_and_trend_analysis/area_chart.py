import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os

# ======================================
# Area Chart - Cumulative Installations
# ======================================

st.set_page_config(
    page_title="Area Chart Analysis",
    layout="wide"
)

st.title("Temporal and Trend Analysis")
st.subheader("Area Chart - Cumulative Renewable Installations")

country_files = {
    "Germany": "renewable_power_plants_DE.csv",
    "France": "renewable_power_plants_FR.csv",
    "Denmark": "renewable_power_plants_DK.csv",
    "Switzerland": "renewable_power_plants_CH.csv",
    "Czech Republic": "renewable_power_plants_CZ.csv",
    "Poland": "renewable_power_plants_PL.csv",
    "Sweden": "renewable_power_plants_SE.csv",
    "United Kingdom": "renewable_power_plants_UK.csv"
}

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    list(country_files.keys()),
    default=["Germany", "France", "Denmark"]
)

start_year = st.sidebar.slider(
    "Select Start Year",
    min_value=1900,
    max_value=2025,
    value=1990
)

output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

sns.set_style("whitegrid")


def read_csv_safely(data_path):
    try:
        return pd.read_csv(data_path, low_memory=False)
    except pd.errors.ParserError:
        return pd.read_csv(
            data_path,
            sep=";",
            engine="python",
            on_bad_lines="skip"
        )


all_data = []

for country in selected_countries:
    file_name = country_files[country]
    data_path = f"../data/{file_name}"

    df = read_csv_safely(data_path)

    required_columns = ["commissioning_date"]

    if not all(col in df.columns for col in required_columns):
        st.warning(f"{country} skipped. Required columns not found.")
        st.write(f"{country} columns:", list(df.columns))
        continue

    df = df[required_columns].dropna()

    df["commissioning_date"] = pd.to_datetime(
        df["commissioning_date"],
        errors="coerce"
    )

    df = df.dropna(subset=["commissioning_date"])

    df["commissioning_year"] = df["commissioning_date"].dt.year

    df = df[df["commissioning_year"] >= start_year]

    if df.empty:
        st.warning(f"{country} has no data after {start_year}.")
        continue

    df["country"] = country
    all_data.append(df)


if len(all_data) == 0:
    st.error("No valid data found for selected countries.")
    st.stop()


df_all = pd.concat(all_data, ignore_index=True)

fig, ax = plt.subplots(figsize=(14, 7))

# Her ülke için ayrı cumulative area çiz
for country in df_all["country"].unique():
    df_country = df_all[df_all["country"] == country]

    installations_by_year = (
        df_country.groupby("commissioning_year")
        .size()
        .sort_index()
    )

    cumulative_installations = installations_by_year.cumsum()

    ax.fill_between(
        cumulative_installations.index,
        cumulative_installations.values,
        alpha=0.25
    )

    ax.plot(
        cumulative_installations.index,
        cumulative_installations.values,
        linewidth=3,
        label=country
    )

ax.set_title(
    "Area Chart - Cumulative Renewable Installations",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Year")
ax.set_ylabel("Cumulative Installations")
ax.tick_params(axis="x", rotation=45)
ax.legend()

plt.tight_layout()

st.pyplot(fig)

output_path = f"{output_folder}/area_chart.png"

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

st.success(f"Graph saved successfully: {output_path}")

st.markdown("""
### Analysis

This area chart visualizes the cumulative growth of renewable energy installations over time.

- Each selected country is displayed as a separate cumulative trend.
- The chart helps compare long-term installation growth between countries.
- The start year filter allows focusing on modern renewable energy expansion.
""")