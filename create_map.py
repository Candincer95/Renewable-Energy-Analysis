import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import glob
import os

print("--- READING AND AGGREGATING DATA ---")
file_paths = glob.glob('renewable_power_plants_*.csv')
df_list = []

for file in file_paths:
    file_name = os.path.basename(file)
    if 'EU' in file_name:
        continue

    try:
        # Optimization: Only load columns we need to save memory and processing time
        temp_df = pd.read_csv(file, usecols=['electrical_capacity', 'energy_source_level_1'], engine='python', on_bad_lines='skip')
        temp_df['Country_Code'] = file_name.split('_')[-1].split('.')[0]
        df_list.append(temp_df)

    except Exception as e:
        print(f"Error reading {file_name}: {e}")

combined_df = pd.concat(df_list, ignore_index=True)

print("--- DATA CLEANING & TYPE CONVERSION ---")
combined_df['electrical_capacity'] = combined_df['electrical_capacity'].astype(str).str.replace(',', '.')
combined_df['electrical_capacity'] = pd.to_numeric(combined_df['electrical_capacity'], errors='coerce').fillna(0)

# Calculate total capacity per country
capacity_by_country = combined_df.groupby('Country_Code')['electrical_capacity'].sum().reset_index()

# Convert MW to GW
capacity_by_country['Capacity_GW'] = capacity_by_country['electrical_capacity'] / 1000

print("--- GEOSPATIAL MAPPING ---")
# Dictionary to map 2-letter codes from our CSV to Alpha-3 codes used by Geopandas
iso_mapping = {
    'SE': 'SWE', 'PL': 'POL', 'UK': 'GBR', 'FR': 'FRA',
    'CZ': 'CZE', 'DE': 'DEU', 'CH': 'CHE', 'DK': 'DNK' 
}
capacity_by_country['iso_a3'] = capacity_by_country['Country_Code'].map(iso_mapping)

# Load the built-in world map from file
world = gpd.read_file("zip://ne_110m_admin_0_countries.zip")

# Merge the map boundaries with calculated data
europe_map = world.merge(capacity_by_country, left_on='ISO_A3', right_on='iso_a3', how='left')

# Create the visualization figure
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Draw the map for Europe
europe_base = world[world['CONTINENT'] == 'Europe']
europe_base.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)

# Overlay the Choropleth map with data
europe_map.dropna(subset=['Capacity_GW']).plot(
    column='Capacity_GW',
    ax=ax,
    cmap='YlGnBu',
    legend=True,
    legend_kwds={'label': "Total Renewable Capacity (GW)", 'orientation': "horizontal", 'shrink': 0.6},
    edgecolor='black',
    linewidth=0.8
)
# Zoom in on the where data points are
ax.set_xlim([-15, 30])
ax.set_ylim([35, 70])
ax.set_axis_off() # Remove coordinate axes

plt.title('Geospatial Distribution of Total Renewable Energy Capacity', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()

# Save the high-resolution output
output_filename = 'choropleth_map.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS The geospatial map has been saved as: {output_filename}")
