import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import glob
import os

print("--- LOADING DATA FOR BUBBLE MAP ---")
file_paths = glob.glob('renewable_power_plants_*.csv')
df_list = []
target_cols = ['lat', 'lon', 'electrical_capacity', 'energy_source_level_1']

for file in file_paths:
    if 'EU' in file: 
        continue
    try:
        # Check for column availability before loading
        available_cols = pd.read_csv(file, nrows=0, engine='python').columns.tolist()
        cols_to_use = [col for col in target_cols if col in available_cols]
        
        if all(x in cols_to_use for x in ['lat', 'lon', 'electrical_capacity']):
            temp_df = pd.read_csv(file, usecols=cols_to_use, engine='python', on_bad_lines='skip')
            df_list.append(temp_df)
        else:
            print(f"Skipping {os.path.basename(file)}: Essential columns missing.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

combined_df = pd.concat(df_list, ignore_index=True)

print("--- CLEANING AND TRANSFORMING DATA ---")
# Convert coordinates and capacity to numeric
for col in ['lat', 'lon', 'electrical_capacity']:
    combined_df[col] = combined_df[col].astype(str).str.replace(',', '.')
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

combined_df = combined_df.dropna(subset=['lat', 'lon', 'electrical_capacity'])
combined_df = combined_df[combined_df['electrical_capacity'] > 0]

# Filter for European mainland bounding box
europe_data = combined_df[
    (combined_df['lat'] >= 35) & (combined_df['lat'] <= 70) &
    (combined_df['lon'] >= -15) & (combined_df['lon'] <= 35)
]

print("--- GENERATING PROPORTIONAL SYMBOL MAP ---")
world = gpd.read_file("zip://ne_110m_admin_0_countries.zip")
europe_base = world[world['CONTINENT'] == 'Europe']

fig, ax = plt.subplots(figsize=(15, 12))

# Plot base map
europe_base.plot(ax=ax, color='#f0f0f0', edgecolor='#bcbcbc', linewidth=0.5)

# We use a square root scaling for the bubble area to ensure visual proportionality.
sizes = europe_data['electrical_capacity'] * 0.5 

scatter = ax.scatter(
    europe_data['lon'], 
    europe_data['lat'],
    s=sizes,
    c=europe_data['electrical_capacity'],
    cmap='viridis',
    alpha=0.6,
    edgecolors='w',
    linewidth=0.3,
    label='Power Plants'
)
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('Capacity (MW) - Color Scale', fontsize=11, fontweight='bold')

# Legend for bubble sizes
for cap in [10, 100, 500]:
    ax.scatter([], [], s=cap * 0.5, c='gray', alpha=0.5, label=f'{cap} MW', edgecolors='k')

plt.legend(title="Plant Capacity", loc="lower left", labelspacing=1.2, borderpad=1.5)
plt.title('Geospatial Infrastructure Deployment: Plant Capacity Scale Map', fontsize=16, fontweight='bold')

# Zoom to Europe
ax.set_xlim([-15, 30])
ax.set_ylim([35, 70])
ax.set_axis_off()

output_filename = 'bubble_map_capacity.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nSuccess Bubble map saved as: {output_filename}")

