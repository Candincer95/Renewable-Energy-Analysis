import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import glob
import os

print("--- LOADING TEMPORAL-SPATIAL DATA ---")
file_paths = glob.glob('renewable_power_plants_*.csv')
df_list = []

target_cols = ['lat', 'lon', 'commissioning_date']

for file in file_paths:
    if 'EU' in file:
        continue
    try:
        available_cols = pd.read_csv(file, nrows=0, engine='python').columns.tolist()
        cols_to_use = [col for col in target_cols if col in available_cols]

        if 'lat' in cols_to_use and 'lon' in cols_to_use:
            temp_df = pd.read_csv(file, usecols=cols_to_use, engine='python', on_bad_lines='skip')
            df_list.append(temp_df)
            print(f"Loaded: {os.path.basename(file)} | Found: {cols_to_use}")
        else:
            print(f"Skipped: {os.path.basename(file)} | Missing lat/lon")

    except Exception as e:
        print(f"Error reading {file}: {e}")
if not df_list:
    raise ValueError("CRITICAL ERROR: No files contained coordinate data")

combined_df = pd.concat(df_list, ignore_index=True)
if 'commissioning_date' not in combined_df.columns:
    raise KeyError("CRITICAL ERROR: 'commissioning_date' column is missing from all loaded files")

print("--- DATA CLEANING & TIME PARSING ---")

for col in ['lat', 'lon']:
    combined_df[col] = combined_df[col].astype(str).str.replace(',', '.')
    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

combined_df['commissioning_date'] = pd.to_datetime(combined_df['commissioning_date'], errors='coerce')
combined_df = combined_df.dropna(subset=['lat', 'lon', 'commissioning_date'])
combined_df['Year'] = combined_df['commissioning_date'].dt.year

europe_data = combined_df[
    (combined_df['lat'] >= 35) & (combined_df['lat'] <= 70) &
    (combined_df['lon'] >= -15) & (combined_df['lon'] <= 35) &
    (combined_df['Year'] >= 1990) & (combined_df['Year'] <= 2020)
]

print("--- DRAWING SPATIOTEMPORAL FACET MAP ---")
world = gpd.read_file("zip://ne_110m_admin_0_countries.zip")
europe_base = world[world['CONTINENT'] =='Europe']

periods = [
    ('Era 1: Early Adopters\n(1990 - 2000)', 1990, 2000, '#1f77b4'), 
    ('Era 2: Initial Boom\n(2001 - 2010)', 2001, 2010, '#ff7f0e'),  
    ('Era 3: Expansion\n(2011 - 2020)', 2011, 2020, '#d62728')       
]

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle('Spatiotemporal Expansion of Renewable Energy Infrastructure', fontsize=22, fontweight='bold', y=1.02)

for i, (title, start_year, end_year, color) in enumerate(periods):
    ax = axes[i]
    europe_base.plot(ax=ax, color='#e9ecef', edgecolor='white', linewidth=0.8)
    period_data = europe_data[(europe_data['Year'] >= start_year) & (europe_data['Year'] <= end_year)]

    ax.scatter(
        period_data['lon'], 
        period_data['lat'],
        s=12, 
        color=color,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.3,
        label=f'{len(period_data):,} New Plants' 
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim([-15, 30])
    ax.set_ylim([35, 70])
    ax.set_axis_off()
    
    # Lejantı her haritanın sol üstüne ekle
    ax.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', edgecolor='black')

plt.tight_layout()
output_filename = 'spatiotemporal_expansion_map.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSuccess Spatiotemporal map saved as: {output_filename}")