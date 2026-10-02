# This script loads the original solar irradiance data and aligns it to be exactly every 15 minutes. 
# It also calculates the solar current from the irradiance data using a conversion factor, and saves the aligned drift-free current dataset to a new CSV file. 
# Finally, it identifies and removes any days where the solar current never drops near zero (indicating a possible sensor error or data drift), and saves this final clean dataset to another CSV file.

import pandas as pd

# 1. Load data from CSV file
print("Loading data...")
df_solar = pd.read_csv('solarRadiation/GN45_Solar_radiation.csv', sep=',')

# 2. Convert Time column to datetime
df_solar['Time'] = pd.to_datetime(df_solar['Time'], format='%d/%m/%Y %H:%M')

# 3. Handle duplicates by averaging
if df_solar['Time'].duplicated().any():
    print("Duplicate timestamps found. Resolving...")
    df_solar = df_solar.groupby('Time').mean().reset_index()

# 4. Set Time as the index and sort it
df_solar.set_index('Time', inplace=True)
df_solar.sort_index(inplace=True)

# 5. Create a perfect 15-minute grid
print("Creating perfect 15-minute intervals...")
start_time = df_solar.index.min().round('15min')
end_time = df_solar.index.max().round('15min')

# Generate a DatetimeIndex exactly every 15 minutes
perfect_15min_index = pd.date_range(start=start_time, end=end_time, freq='15min')

# 6. Reindex and Interpolate
combined_index = df_solar.index.union(perfect_15min_index)
df_combined = df_solar.reindex(combined_index)
df_interpolated = df_combined.interpolate(method='time')

# Filter the dataframe to ONLY include our perfect 15-minute timestamps
df_perfect = df_interpolated.reindex(perfect_15min_index)
df_perfect.index.name = 'Time'

# 7. Add Vectorized Solar Current Calculation
print("Calculating solar current...")
# We grab the first column dynamically, assuming it is the irradiance data
irradiance_col_name = df_perfect.columns[0] 
conversion_factor = 0.000093942 #We arrive at this value by taking into account the following factors
# Solar panel dimensions in mm: 60.1 x 41.3 --> 0.00248213
# Solar efficiency between 15% and 22% --> we use 0.185
# Solar angle effect approximation --> 0.75
# --> 0.00248213 x 0.185 x 0.75 = 0.00034444
# Now we still have to go from W to A --> At this point we get resistive power
# --> we divide by Vt-max (= 3.3)
# --> 0.00034444/3.3 = 0.00010438
# --> Take into account 0.9 PMU efficiency factor
# --> 0.00010438 x 0.9 = 0.000093942

# Multiply the entire irradiance column by the conversion factor at once
df_perfect['solar_current_A'] = df_perfect[irradiance_col_name] * conversion_factor

# 8. Rounding the data
# We keep the irradiance rounded to 3 decimals, but rounded the new current to 6 decimals.
# Because your conversion factor is so small, rounding it to 3 decimals might turn all your data to 0.000.
df_perfect = df_perfect.round({irradiance_col_name: 6, 'solar_current_A': 6})
df_perfect_current_only = df_perfect.drop(columns=[irradiance_col_name]) # If you only want the solar current column in the final CSV file, we can drop the irradiance column here.


# 9. Save the fixed dataset to a new CSV file
output_filename = 'solar_current_exact_15min_with_time.csv'
df_perfect_current_only.to_csv(output_filename, index=True)
print(f"Done! Drift-free data and calculated current saved to {output_filename}")

df = pd.read_csv('solar_current_exact_15min_with_time.csv')
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

# --- CLEANUP PHASE (3:00 AM to 3:00 AM Chunks) ---

col_name = df.columns[0]

# 1. The Time Shift Trick
# Subtract 3 hours from the index. (e.g., 03:00 AM becomes 00:00 Midnight)
# We don't overwrite the original df.index, we just use this for grouping!
shifted_index = df.index - pd.Timedelta(hours=3)

# 2. Group by the shifted 'days' and find the minimum
# Now, a "day" runs from exactly 03:00:00 to 02:45:00 the next day.
daily_mins = df.groupby(shifted_index.normalize())[col_name].min()

threshold = 0.0001 

# 3. Report the bad days
bad_days = daily_mins[daily_mins > threshold]
print(f"Found {len(bad_days)} suspicious 24h blocks (3AM-3AM) where the sun 'never sets':")
for date, val in bad_days.items():
    # Adding back the 3 hours just so the printout makes sense to human eyes
    real_start_time = date + pd.Timedelta(hours=3)
    print(f"- Block starting {real_start_time.strftime('%Y-%m-%d %H:%M')}: Lowest value was {val}")

# 4. Identify the GOOD days (using the shifted midnight labels)
good_days = daily_mins[daily_mins <= threshold].index

# 5. Filter the dataframe using the shifted index
# If the shifted row's "midnight" is in the good_days list, we keep the original row.
df_clean = df[shifted_index.normalize().isin(good_days)]

# 6. Report the cleanup results
print(f"\n--- Cleanup Summary ---")
print(f"Original data rows: {len(df)}")
print(f"Cleaned data rows:  {len(df_clean)}")
print(f"Removed {len(df) - len(df_clean)} faulty rows.")

# 7. Save the pristine dataset
output_file = 'solar_current_clean_days_only.csv'
df_clean.to_csv(output_file, index=False) # Change index to True if we want timestamps
print(f"\nSuccess! Cleaned up, 3AM-aligned data saved to {output_file}")