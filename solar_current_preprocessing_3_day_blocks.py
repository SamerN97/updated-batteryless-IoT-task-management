# This script takes the clean solar current dataset (with only the days where the current drops near zero, so no drift) and fills in the missing data points by interpolation.
# Furthermore, it splits the data into 72h blocks, randomly shuffles these blocks, and then splits the shuffled data into a training set (80%) and an inference/validation set (20%). 
# Finally, it saves the shuffled full dataset, the shuffled training dataset, and the shuffled inference dataset to new CSV files.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fill_with_interpolation(array, nrOfInterpolationPoints):
    final_array = []

    for i in range(len(array) - 1):
        # Append the current original number
        final_array.append(array[i])
        # Add 4 interpolated values between this and the next original number
        interpolated_values = np.linspace(array[i], array[i + 1], nrOfInterpolationPoints + 2)[1:-1]
        final_array.extend(interpolated_values)

    # Append the last original number
    final_array.append(array[-1])

    # Convert to a numpy array for consistency
    final_array = np.array(final_array)

    return final_array

nrOfInterpolationPoints = 89 # We go from 15 min to 10 sec
dataList = pd.read_csv('solar_current_clean_days_only.csv', sep = ',') 
dataList = dataList.iloc[51:, 0]  # We only want the solar current data from the CSV file, starting at 3AM
dataList = dataList.to_numpy()
dataList = fill_with_interpolation(dataList, nrOfInterpolationPoints) 
print("Max number in dataset: " + str(dataList.max()))

# # --- Chunking, Shuffling, and Splitting ---

chunk_size = 8640 * 3 # We want 3 day blocks for training

# 1. Truncate the array slightly so it is perfectly divisible by 8640*3
num_chunks = len(dataList) // chunk_size
trimmed_data = dataList[:num_chunks * chunk_size]
print(f"Original data length: {len(dataList)}")
print(f"Trimmed data length:  {len(trimmed_data)} (perfectly divisible by {chunk_size})")

# 2. Reshape into a 2D array: (number_of_blocks, 8640*3)
# Now, each row represents exactly one 3-day period.
chunks = trimmed_data.copy().reshape((num_chunks, chunk_size))

# --- The Stitch-Checker ---
# This ensures every chunk starts and ends near zero so they connect smoothly
valid_chunks = []
edge_tolerance = 0.001 # Maximum allowed current at 3:00 AM

for i, chunk in enumerate(chunks):
    if chunk[0] < edge_tolerance and chunk[-1] < edge_tolerance:
        valid_chunks.append(chunk)
    else:
        print(f"Discarded chunk {i} to prevent a discontinuity! (Start: {chunk[0]:.4f}, End: {chunk[-1]:.4f})")

chunks = np.array(valid_chunks)
num_valid_chunks = len(chunks)
print(f"Proceeding with {num_valid_chunks} perfectly stitchable 72h blocks.")

# # 3. Shuffle the 72h chunks randomly
np.random.seed(42) # Uncomment this if you want the exact same "random" shuffle every time you run the script
np.random.shuffle(chunks)

# 4. Calculate the 80% split index
split_index = int(0.8 * num_chunks)

# 5. Split the chunks into Training (80%) and Validation (20%)
train_chunks = chunks[:split_index]
val_chunks = chunks[split_index:]

# 6. Flatten the 2D arrays back into continuous 1D sequences
train_data = train_chunks.flatten()
val_data = val_chunks.flatten()

print(f"Total 72h blocks: {num_chunks}")
print(f"Training data size:   {len(train_data)} points ({len(train_chunks)} days)")
print(f"Validation data size: {len(val_data)} points ({len(val_chunks)} days)")



# --- Plotting the new Training Data to verify ---

# We look at a slice of the shuffled training data
shortDataList = train_data
# shortDataList = dataList[:1000000] # We look at the same slice of the original data for comparison
plt.plot(shortDataList)
plt.plot()


# Add a clear red vertical line every 8640 points
for i in range(chunk_size, len(shortDataList), chunk_size):
    plt.axvline(x=i, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

plt.title("Shuffled 72h Blocks (80% Training Data)")
plt.show()


# We look at a slice of the shuffled training data
shortDataList = val_data
# shortDataList = dataList[:1000000] # We look at the same slice of the original data for comparison
plt.plot(shortDataList)
plt.plot()


# Add a clear red vertical line every 8640 points
for i in range(chunk_size, len(shortDataList), chunk_size):
    plt.axvline(x=i, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

plt.title("Shuffled 72h Blocks (20% Validation Data)")
plt.show()

# --- Saving the new datasets to CSV files ---   
df2 = pd.DataFrame({"Shuffled Solar Training Data (3-day)": train_data})
df2.to_csv("3_day_shuffled_solar_training_dataset.csv", index=False)
df3 = pd.DataFrame({"Shuffled Solar Validation Data (3-day)": val_data})
df3.to_csv("3_day_shuffled_solar_validation_dataset.csv", index=False)
