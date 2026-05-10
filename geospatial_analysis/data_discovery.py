import pandas as pd
import glob
import os
# All csv files starting with 'renewable_power_plants'
file_paths = glob.glob('renewable_power_plants_*.csv')

df_list = []

print("--- READING FILES ---")
for file in file_paths:
    file_name = os.path.basename(file)

    # Skip the total Europe data
    if 'EU' in file_name:
        print(f"warning: {file_name} skipped.")
        continue
    # Read csv  
    try:   
        temp_df = pd.read_csv(file, engine='python', on_bad_lines='skip')

        # Extract thr country code and add it as a new column
        country_code = file_name.split('_')[-1].split('.')[0] 
        temp_df['Country_Code'] = country_code

        df_list.append(temp_df)
        print(f"Success: {file_name} added. (Rows: {len(temp_df)})")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
# Combine all country tables into a single dataframe
if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)

    print("\n--- MERGED PROCESS COMPLETED ---")
    print(f"Total number of Rows: {combined_df.shape[0]}")
    print(f"Merged Country Codes: {combined_df['Country_Code'].unique()}")
    print("\n--- COLUMN NAMES OF THE COMBINED TABLE ---")
    print(combined_df.columns.tolist())
else:
    print("\nNo data was successfully loaded.")





