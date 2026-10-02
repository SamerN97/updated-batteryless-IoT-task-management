import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

downSampleFactor = 3
# Based on your 30s interval: 2880 steps = 1 day
STEPS_PER_DAY = 2880 
title = 'Solar Harvesting Current Validation Data'

# Code to read
dataList = pd.read_csv('shuffled_solar_validation_dataset.csv', sep=',')
dataList = dataList.iloc[:, 0].to_numpy()
dataList = dataList[::downSampleFactor]

# Create X-axis in units of Days
# len(dataList) gives total steps, dividing by STEPS_PER_DAY converts to days
x_days = np.arange(len(dataList)) / STEPS_PER_DAY

plt.figure(figsize=(8, 4))
plt.plot(x_days, dataList, color='green', linewidth=1.5, label='Harvested Current ($I_H$)')

# plt.title(title, fontsize=16, fontweight='bold')
plt.xlabel('Time (Days)', fontsize=14)
# Added 'Unit' to Y-label for academic rigor
plt.ylabel('Harvesting Current (A)', fontsize=14) 

# Optional: Set ticks every 5 or 10 days for better readability
plt.xticks(np.arange(0, x_days.max() + 1, 5), fontsize=12) 
plt.yticks(fontsize=12)

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlim(0, x_days.max()) # Ensure plot starts exactly at Day 0

# Scientific plots usually look better with a tight layout
# plt.legend(loc='upper right', fontsize=10, framealpha=0.85)

filepath = os.path.join('validation_solar_data', 'validation_plot_days.pdf')
# Ensure directory exists
os.makedirs(os.path.dirname(filepath), exist_ok=True)

plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved trend plot with Days on X-axis: {filepath}")