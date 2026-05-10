import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import glob
import os

print("--- LOADING DATA FOR CATEGORICAL MAP ---")
file_paths = glob.glob('renewable_power_plants_*.csv')
df_list = []
target_cols = ['lat', 'lon', 'energy_source_level_2']

for file in file_paths:
    if 'EU' in file:
        continue
    try:
        # Check for column availability (Poland is skipped)
        available_cols = pd.read_csv(file, nrows=0, engine='python').columns.tolist()
        cols_to_use = [col for col in target_cols if col in available_cols]

        if all(x in cols_to_use for x in ['lat', 'lon', 'energy_source_level_2']):
            temp_df = pd.read_csv(file, usecols=cols_to_use, engine='python', on_bad_lines='skip')
            df_list.append(temp_df)

    except Exception as e:
        print(f"Error reading {file}: {e}")

combined_df = pd.concat(df_list, ignore_index=True)

print("--- DATA CLEANING ---")
for col in ['lat', 'lon']:
    combined_df[col] = combined_df[col].astype(str).str.replace(',', '.')
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

combined_df = combined_df.dropna(subset=['lat', 'lon', 'energy_source_level_2'])

europe_data = combined_df[
    (combined_df['lat'] >= 35) & (combined_df['lat'] <=70) &
    (combined_df['lon'] >= -15) & (combined_df['lon'] <= 35)
]

print("--- DRAWING CATEGORICAL MAP ---")
world = gpd.read_file("zip://ne_110m_admin_0_countries.zip")
europe_base = world[world['CONTINENT'] == 'Europe']

fig, ax = plt.subplots(figsize=(16, 14))

europe_base.plot(ax=ax, color='#555555', edgecolor='white', linewidth=0.8)
source_counts = europe_data['energy_source_level_2'].value_counts()
energy_sources = source_counts.index

cmap = plt.colormaps['tab10']

for idx, source in enumerate(energy_sources):
    subset = europe_data[europe_data['energy_source_level_2'] == source]

    count = source_counts[source]
    label_text = f"{source} ({count:,})"

    ax.scatter(
        subset['lon'],
        subset['lat'],
        s=35,
        color=cmap(idx % 10),
        alpha=0.85,
        edgecolors='black',
        label=label_text
    )
plt.legend(
    title="Legend & Installation Counts", 
    loc="upper left", 
    markerscale=1.5, 
    fontsize=12, 
    title_fontsize=14, 
    frameon=True,
    facecolor='white',
    edgecolor='black'
)
plt.title('Topographical Distribution of Renewable Energy Sources', fontsize=20, fontweight='bold', pad=20)

ax.set_xlim([-15, 30])
ax.set_ylim([35, 70])

ax.grid(True, linestyle='-', alpha=0.2, color='white')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', which='both', length=0)


output_filename = 'categorical_energy_map.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='#eaeaea')
print(f"Success Categorical map saved as: {output_filename}")