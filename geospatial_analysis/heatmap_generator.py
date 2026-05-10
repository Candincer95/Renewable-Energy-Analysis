import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import glob
import os

print("--- READING COORDINATE DATA ---")
file_paths = glob.glob('renewable_power_plants_*.csv')
df_list = []

target_cols = ['lat', 'lon', 'energy_source_level_1']

for file in file_paths:
    if 'EU' in file: 
        continue
        
    try:
        # Step 1: Read ONLY the header (nrows=0) to see what columns actually exist
        available_cols = pd.read_csv(file, nrows=0, engine='python').columns.tolist()
        
        # Step 2: Find the intersection of what we want vs what the file has
        cols_to_use = [col for col in target_cols if col in available_cols]
        
        # Step 3: If the file doesn't even have lat/lon, completely skip it
        if 'lat' not in cols_to_use or 'lon' not in cols_to_use:
            print(f"Skipping {file}: Missing 'lat' or 'lon' columns.")
            continue
            
        # Step 4: Safely load the file using the exact columns we verified
        temp_df = pd.read_csv(file, usecols=cols_to_use, engine='python', on_bad_lines='skip')
        df_list.append(temp_df)
        print(f"Successfully loaded: {file}")
        
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Prevent the 'No objects to concatenate' error by checking if list is empty
if not df_list:
    raise ValueError("CRITICAL ERROR: No valid coordinate data found in any of the CSV files!")

combined_df = pd.concat(df_list, ignore_index=True)

print("--- DATA CLEANING AND FILTERING ---")
# Convert European comma to standard decimal points
for col in ['lat', 'lon']:
    combined_df[col] = combined_df[col].astype(str).str.replace(',', '.')
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

# Drop rows with missing coordinate data
combined_df = combined_df.dropna(subset=['lat', 'lon'])

# Filter coordinates within European bounding
# Latitude: 35 to 70, Longitude: -15 to 35
europe_bounds = combined_df[
    (combined_df['lat'] >= 35) & (combined_df['lat'] <= 70) &
    (combined_df['lon'] >= -15) & (combined_df['lon'] <= 35)
]

print("--- DRAWING HEXBIN HEATMAP ---")
# Load the local map data
world = gpd.read_file("zip://ne_110m_admin_0_countries.zip")
europe_base = world[world['CONTINENT'] == 'Europe']

fig, ax = plt.subplots(1, 1, figsize=(14,10))
# Plot the background map of Europe
europe_base.plot(ax=ax, color='#333333', edgecolor='#555555', linewidth=0.5)

# Hexbin Heatmap parameters
# gridsize: Resolution of the hexagons
# cmap='inferno': Color palette that simulates heat
hb = ax.hexbin(
    europe_bounds['lon'],
    europe_bounds['lat'],
    gridsize=120,
    cmap='inferno',
    mincnt=1, # Only draw hexagons that contain at least 1 power plant
    alpha=0.85,
    zorder=2
)

# Add a colorbar (Legend)
cb = fig.colorbar(hb, ax=ax, shrink=0.5, orientation='horizontal', pad=0.02)
cb.set_label('Plant Density (Number of Power Plants per Hexagon)', fontsize=12, color='black', fontweight='bold')

# Set map limits and remove axes for a cleaner look
ax.set_xlim([-15, 30])
ax.set_ylim([35, 70])
ax.set_axis_off()

plt.title('Geospatial Heatmap: Regional Energy Clusters in Europe', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()

output_filename = 'heatmap_clusters.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Success Hexbin heatmap saved as {output_filename}")