import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import glob
import os

print("--- CALCULATING TOTAL CAPACITY PER COUNTRY ---")
file_paths = glob.glob('renewable_power_plants_*.csv')
country_capacities = {}

for file in file_paths:
    if 'EU' in file: 
        continue
    try:
        
        filename = os.path.basename(file)
        country_code = filename.split('_')[-1].split('.')[0]
        
        
        temp_df = pd.read_csv(file, usecols=['electrical_capacity'], engine='python', on_bad_lines='skip')
        temp_df['electrical_capacity'] = temp_df['electrical_capacity'].astype(str).str.replace(',', '.')
        temp_df['electrical_capacity'] = pd.to_numeric(temp_df['electrical_capacity'], errors='coerce')
        
        # MegaWatt to GigaWatt
        total_mw = temp_df['electrical_capacity'].sum()
        if total_mw > 0:
            country_capacities[country_code] = total_mw / 1000  
            print(f"Aggregated: {country_code} -> {country_capacities[country_code]:.2f} GW")
            
    except Exception as e:
        print(f"Error reading {file}: {e}")


cap_df = pd.DataFrame(list(country_capacities.items()), columns=['Country_Code', 'Capacity_GW'])

print("--- DRAWING ENHANCED CHOROPLETH MAP ---")

world = gpd.read_file("zip://ne_110m_admin_0_countries.zip")
europe_base = world[world['CONTINENT'] == 'Europe'].copy()


europe_merged = europe_base.merge(cap_df, how='left', left_on='ISO_A2_EH', right_on='Country_Code')

fig, ax = plt.subplots(1, 1, figsize=(14, 12))


europe_base.plot(ax=ax, color='#d3d3d3', edgecolor='#666666', linewidth=0.5)


europe_merged.dropna(subset=['Capacity_GW']).plot(
    column='Capacity_GW', 
    ax=ax, 
    cmap='YlGnBu', 
    edgecolor='#333333', 
    linewidth=0.8,
    legend=True,
    legend_kwds={
        'label': "Total Renewable Capacity (GW)",
        'orientation': "horizontal",
        'shrink': 0.7,
        'pad': 0.05
    }
)


ax.set_title('Geospatial Distribution of Total Renewable Energy Capacity', fontsize=20, fontweight='bold', pad=20)
ax.set_xlim([-15, 35])
ax.set_ylim([35, 70])
ax.set_axis_off() 

output_filename = 'choropleth_map.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSuccess Choropleth map saved as: {output_filename}")
