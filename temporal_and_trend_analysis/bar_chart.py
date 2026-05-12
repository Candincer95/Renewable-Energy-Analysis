import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import numpy as np

# ======================================
# Bar Chart - Yearly Plant Installations
# ======================================

st.set_page_config(
    page_title="Bar Chart Analysis",
    layout="wide"
)

st.title("Temporal and Trend Analysis")
st.subheader("Bar Chart - Yearly Renewable Plant Installations by Country")

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

year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=1900,
    max_value=2025,
    value=(2000, 2020)
)

start_year = year_range[0]
end_year = year_range[1]

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

    df = df[
        (df["commissioning_year"] >= start_year) &
        (df["commissioning_year"] <= end_year)
    ]

    if df.empty:
        st.warning(f"{country} has no data between {start_year} and {end_year}.")
        continue

    df["country"] = country
    all_data.append(df)


if len(all_data) == 0:
    st.error("No valid data found for selected countries and year range.")
    st.stop()


df_all = pd.concat(all_data, ignore_index=True)

# Ülke ve yıla göre kurulum sayısı
country_year_installations = (
    df_all.groupby(["commissioning_year", "country"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

fig, ax = plt.subplots(figsize=(14, 7))

years = country_year_installations.index.to_list()
countries = country_year_installations.columns.to_list()

x = np.arange(len(years))
bar_width = 0.8 / len(countries)

# Her ülkeyi ayrı renkte yan yana bar olarak çiz
for i, country in enumerate(countries):
    ax.bar(
        x + i * bar_width,
        country_year_installations[country].values,
        width=bar_width,
        label=country
    )

ax.set_title(
    f"Bar Chart - Yearly Renewable Plant Installations by Country ({start_year}-{end_year})",
    fontsize=18,
    fontweight="bold"
)

ax.set_xlabel("Year")
ax.set_ylabel("Number of Installations")

ax.set_xticks(x + bar_width * (len(countries) - 1) / 2)
ax.set_xticklabels(years, rotation=45)

ax.legend(title="Country")

plt.tight_layout()

st.pyplot(fig)

output_path = f"{output_folder}/bar_chart.png"

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

st.success(f"Graph saved successfully: {output_path}")

st.markdown("""
### Analysis

This bar chart compares yearly renewable power plant installations by country.

- Each country is shown with a separate color.
- Bars are grouped by year.
- The chart helps compare yearly installation activity across selected countries.
""")