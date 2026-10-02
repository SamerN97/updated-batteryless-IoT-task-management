# This script calculates the leakage current for a given capacitance based on the data from the CDE DGH 5.5V datasheet. (https://www.cde.com/resources/catalogs/DGH.pdf)
# It performs a linear regression to find the relationship between capacitance and leakage current, and then provides a function to estimate the leakage current for any capacitance value.

import numpy as np
import matplotlib.pyplot as plt

# 1. The data from the CDE DGH 5.5V datasheet
capacitance_values = np.array([0.5, 1.0, 1.5, 2.5, 3.5, 5.0]) # Farads
leakage_current_values = np.array([8, 10, 12, 16, 20, 30])    # Microamps (uA)

# 2. Perform Linear Regression
slope, intercept = np.polyfit(capacitance_values, leakage_current_values, 1)

# 3. Create the plot
# Scatter the actual datasheet points
plt.scatter(capacitance_values, leakage_current_values, color='red', label='Datasheet Values', zorder=5)

# Generate points for the regression line (extending up to 10.5F for visibility)
line_x = np.linspace(0, 10.5, 100)
line_y = slope * line_x + intercept

print(f"Regression completed!")
print(f"Slope (m): {slope:.4f}")
print(f"Intercept (b): {intercept:.4f}")
print(f"Formula: LC = {slope:.4f} * C + {intercept:.4f}\n")

# Plot the regression line
plt.plot(line_x, line_y, color='blue', label=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}')

# 4. Formatting the plot
plt.title('Capacitance vs. Leakage Current (5.5V DGH Series)')
plt.xlabel('Capacitance (Farads)')
plt.ylabel('Leakage Current ($\mu A$)')

# Set bounds to clearly see the extrapolation to 10F
plt.xlim(0, 10.5)
plt.ylim(0, max(line_y) + 5)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('leakage_regression.png')