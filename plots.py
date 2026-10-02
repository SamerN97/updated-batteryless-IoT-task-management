import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# Function to read data from a CSV file and plot the CDF
def plot_cdf(csv_file):
    # Load the data from the CSV file
    data = pd.read_csv(csv_file, header=None)
    
    # Extract the data from the single column
    values = data[0].dropna().sort_values()
    
    # Calculate the CDF
    cdf = values.rank(method='first') / len(values)
    
    # Plot the CDF
    plt.figure(figsize=(10, 6))
    plt.plot(values, cdf, marker='.', linestyle='none')
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function (CDF)')
    plt.grid(True)
    plt.show()

# Function to read data from a CSV file and plot the data
def plot_data(csv_file):
    # Load the data from the CSV file
    data = pd.read_csv(csv_file, header=None)
    
    # Extract the data from the first (and only) column
    values = data[0].dropna()
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, values, marker='.', linestyle='none')
    plt.xlabel('Row Number')
    plt.ylabel('Value')
    plt.title('Data Plot')
    plt.grid(True)
    plt.show()


def plot_multiple_graphs_from_csv(csv_files, titles, x_col, y_col, nrows, ncols, figsize=(15, 10)):
    """
    Plots multiple graphs from multiple CSV files in one overview.

    Parameters:
    - csv_files: List of paths to CSV files.
    - titles: List of titles for each subplot.
    - x_col: The column name to be used for x values.
    - y_col: The column name to be used for y values.
    - nrows: Number of rows in the subplot grid.
    - ncols: Number of columns in the subplot grid.
    - figsize: Tuple specifying the figure size (default is (15, 10)).
    """
    if len(csv_files) != len(titles):
        raise ValueError("The number of CSV files must match the number of titles.")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    axes = axes.flatten()  # Flatten the array of axes for easy iteration

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file, header=None)  # Read CSV without header
        y_values = df.iloc[:, 0].values  # Assume single column of y values
        x_values = range(len(y_values))  # Generate x values as indices
        ax = axes[i]
        ax.plot(x_values, y_values)
        ax.set_title(titles[i])
        ax.grid(True)

    # Hide any unused subplots
    for i in range(len(csv_files), nrows * ncols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_multiple_cdf_from_csv(csv_files, titles, nrows, ncols, figsize=(15, 10)):
    """
    Plots multiple CDF graphs from multiple CSV files in one overview where each file contains one column with one row per data point.

    Parameters:
    - csv_files: List of paths to CSV files.
    - titles: List of titles for each subplot.
    - nrows: Number of rows in the subplot grid.
    - ncols: Number of columns in the subplot grid.
    - figsize: Tuple specifying the figure size (default is (15, 10)).
    """
    if len(csv_files) != len(titles):
        raise ValueError("The number of CSV files must match the number of titles.")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    axes = axes.flatten()  # Flatten the array of axes for easy iteration

    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file, header=None)  # Read CSV without header
        data = df.iloc[:, 0].values  # Assume single column of data points
        sorted_data = np.sort(data)
        y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)  # CDF values
        
        ax = axes[i]
        ax.plot(sorted_data, y_values)
        ax.set_title(titles[i])
        ax.set_xlabel('Data Values')
        ax.set_ylabel('CDF')
        ax.grid(True)

    # Hide any unused subplots
    for i in range(len(csv_files), nrows * ncols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# Function to read data from a CSV file and plot a histogram
def plot_histogram_from_csv(file_path):
    # Read the CSV file without headers
    data = pd.read_csv(file_path, header=None)
    
    # Check if the data is in one column
    if data.shape[1] != 1:
        print("The CSV file should have only one column.")
        return

    # Extract the data series (first column)
    data_series = data[0]
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data_series, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# Function to read data from multiple CSV files and plot histograms
def plot_histograms_from_multiple_csv(file_paths, nrows, ncols):
    # Number of CSV files
    num_files = len(file_paths)

    if num_files > nrows * ncols:
        print("The number of subplots is less than the number of CSV files. Some files will not be plotted.")
        return

    # Set up the figure for subplots
    plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, file_path in enumerate(file_paths):
        # Read the CSV file without headers
        data = pd.read_csv(file_path, header=None)
        
        # Check if the data is in one column
        if data.shape[1] != 1:
            print(f"The CSV file '{file_path}' should have only one column.")
            continue

        # Extract the data series (first column)
        data_series = data[0]

        # Plot the histogram in the specified subplot position
        plt.subplot(nrows, ncols, i + 1)
        plt.hist(data_series, bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Histogram from {file_path}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


# Function to read data from multiple CSV files and plot histograms side by side
def plot_side_by_side_histograms(file_paths, bar_width_factor=7):
    # Number of CSV files
    num_files = len(file_paths)

    # Set up the figure
    plt.figure(figsize=(10, 6))

    # List to store the histogram data
    histograms = []

    # Read data and compute histograms
    for file_path in file_paths:
        # Read the CSV file without headers
        data = pd.read_csv(file_path, header=None)
        
        # Check if the data is in one column
        if data.shape[1] != 1:
            print(f"The CSV file '{file_path}' should have only one column.")
            continue

        # Extract the data series (first column)
        data_series = data[0]
        
        # Compute the histogram
        hist, bin_edges = np.histogram(data_series, bins=30)
        histograms.append((hist, bin_edges, file_path))

    # Width of each bar
    bin_width = histograms[0][1][1] - histograms[0][1][0]
    bar_width = bin_width / num_files * bar_width_factor

    # Plot each histogram
    for i, (hist, bin_edges, file_path) in enumerate(histograms):
        bin_centers = bin_edges[:-1] + (i - (num_files - 1) / 2) * bar_width
        plt.bar(bin_centers, hist, width=bar_width, alpha=0.7, label=file_path, edgecolor='black')

    # Add legend, title, and labels
    plt.title('Side by Side Histograms')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc=(0.05,0.8))
    plt.grid(True)

    # Show the plot
    plt.show()


def fill_with_interpolation(array, nrOfInterpolationPoints):
    # Step 2: Create the final array with 4 interpolated values between each original number
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


# Main block to call the function
if __name__ == "__main__":

    capSizeStr = "10"
    trainingCapSizeStr = "random_0.5_10"
    trainingPayload = "daily_random_20_255"
    payloadOption = "daily_random_20_255"

    # Choose jitter optimized or off time optimized
    agent_option = "JITTER_OPT" 
    # agent_option = "OFF_OPT"

    tsf_max = 1000
    neg_inaction_reward = -0.5
    training_neg_inaction_reward = -0.5
    gamma = 0.99    
    V_thresh = 3

    training_parameter_suffix = agent_option + "_" + str(training_neg_inaction_reward) + "_INACTION_REWARD_ADR_" + trainingPayload + "_BYTES_" + trainingCapSizeStr + "FARAD_TSF_" + str(tsf_max) + "_GAMMA_" + str(gamma)  # naming convention: POSREWARD_NEGREWARD_ALPHA_MU_SPREADING_SIGMA_HISTORYSIZE
    inference_parameter_suffix = agent_option + "_" + str(training_neg_inaction_reward) + "_INACTION_REWARD_ADR_" + payloadOption + "_BYTES_" + capSizeStr + "FARAD_TSF_" + str(tsf_max)  # naming convention: POSREWARD_NEGREWARD_ALPHA_MU_SPREADING_SIGMA_HISTORYSIZE
    baseline_parameter_suffix = "ADR_" + payloadOption + "_BYTES_" + capSizeStr + "_FARAD_" + str(V_thresh) + "V_THRESH" # naming convention: _ALPHA_MU_SPREADING_SIGMA_HISTORYSIZE
    predictive_parameter_suffix = "ADR_" + payloadOption + "_BYTES_" + capSizeStr + "FARAD"
    oracle_parameter_suffix = "ADR_" + payloadOption + "_BYTES_" + capSizeStr + "FARAD" 
    astar_parameter_suffix = "ADR_" + payloadOption + "_BYTES_" + capSizeStr + "FARAD"

    titles3 = ["Actions", "Harvesting current", "Capacitor Voltage"]

    titles4 = ["Rewards", "Harvesting current", "Capacitor Voltage"]

    titles5 = ["Rewards", "Harvesting current", "Capacitor Voltage", "Actions", "Harvesting current", "Capacitor Voltage"]

    titles6 = ["Rewards", "Actions", "Harvesting current", "Harvesting current", "Capacitor voltage", "Capacitor voltage"]

    titles7 = ["Rewards", "Feasibility", "Device state", "restart", "Harvesting current", "Capacitor Voltage"]

    x_col = "timesteps"

    y_col = "test"



    # multiplot_training = ['experiments_combined/training_experiments/rewardList' + training_parameter_suffix + '.csv', 'experiments_combined/training_experiments/feasibilityList' + training_parameter_suffix + '.csv', 'experiments_combined/training_experiments/deviceStateList' + training_parameter_suffix + '.csv', 'experiments_combined/training_experiments/restartList' + training_parameter_suffix + '.csv', 'experiments_combined/training_experiments/harvestingCurrentList' + training_parameter_suffix + '.csv', 'experiments_combined/training_experiments/voltageList' + training_parameter_suffix + '.csv']

    multiplot_inference = ['RLModelData/experiments_solar/inference_experiments/rewardList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'RLModelData/experiments_solar/inference_experiments/harvestingCurrentList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'RLModelData/experiments_solar/inference_experiments/voltageList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv']

    # multiplot_inference_vs_baseline = ['RLModelData/experiments_combined/inference_experiments/rewardList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'baselineModelData/actionList' + baseline_parameter_suffix + '.csv', 'RLModelData/experiments_combined/inference_experiments/harvestingCurrentList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'baselineModelData/currentList' + baseline_parameter_suffix + '.csv', 'RLModelData/experiments_combined/inference_experiments/voltageList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'baselineModelData/voltageList' + baseline_parameter_suffix + '.csv']

    # multiplot_inference_vs_success_baseline = ['experiments_combined/inference_experiments/rewardList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'baselineDataLists/successList' + baseline_parameter_suffix + '.csv', 'experiments_combined/inference_experiments/harvestingCurrentList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'baselineDataLists/currentList' + baseline_parameter_suffix + '.csv', 'experiments_combined/inference_experiments/voltageList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'baselineDataLists/voltageList' + baseline_parameter_suffix + '.csv']

    # multiplot_inference_vs_inference = ['RLModelData/experiments_combined/inference_experiments/rewardList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'RLModelData/experiments_solar/inference_experiments/rewardList' + test_suffix + '__' + test_suffix + '.csv', 'RLModelData/experiments_combined/inference_experiments/harvestingCurrentList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'RLModelData/experiments_solar/inference_experiments/harvestingCurrentList' + test_suffix + '__' + test_suffix + '.csv', 'RLModelData/experiments_combined/inference_experiments/voltageList' + inference_parameter_suffix + '__' + training_parameter_suffix + '.csv', 'RLModelData/experiments_solar/inference_experiments/voltageList' + test_suffix + '__' + test_suffix + '.csv']

    # multiplot_baseline_vs_baseline = ['baselineDataLists/actionList' + test_suffix + '.csv', 'baselineDataLists/actionList' + baseline_parameter_suffix + '.csv', 'baselineDataLists/currentList' + test_suffix + '.csv', 'baselineDataLists/currentList' + baseline_parameter_suffix + '.csv', 'baselineDataLists/voltageList' + test_suffix + '.csv', 'baselineDataLists/voltageList' + baseline_parameter_suffix + '.csv']

    multiplot_training = ['RLModelData/experiments_combined/training_experiments/rewardList' + training_parameter_suffix + '.csv', 'RLModelData/experiments_combined/training_experiments/harvestingCurrentList' + training_parameter_suffix + '.csv', 'RLModelData/experiments_combined/training_experiments/voltageList' + training_parameter_suffix + '.csv']

    # multiplot_baseline = ['baselineModelData/actionList' + baseline_parameter_suffix + '.csv', 'baselineModelData/currentList' + baseline_parameter_suffix + '.csv', 'baselineModelData/voltageList' + baseline_parameter_suffix + '.csv']
    
    multiplot_predictive = ['APModelData/actionList' + predictive_parameter_suffix + '.csv', 'mathModelData/currentList' + predictive_parameter_suffix + '.csv', 'mathModelData/voltageList' + predictive_parameter_suffix + '.csv']

    multiplot_astar = ['astarModelData/actionList' + astar_parameter_suffix + '.csv', 'astarModelData/currentList' + astar_parameter_suffix + '.csv', 'astarModelData/voltageList' + astar_parameter_suffix + '.csv']

    # multiplot_predictive = ['oracleDataLists/actionList' + oracle_parameter_suffix + '.csv', 'oracleDataLists/currentList' + oracle_parameter_suffix + '.csv', 'oracleDataLists/voltageList' + oracle_parameter_suffix + '.csv']

    # multiplot_goodIT = ['goodITData/actionList' + goodIT_parameter_suffix + '.csv', 'goodITData/currentList' + goodIT_parameter_suffix + '.csv', 'goodITData/voltageList' + goodIT_parameter_suffix + '.csv']

    # multiplot_baseline = ['baselineModelData/actionList' + baseline_parameter_suffix + '.csv', 'baselineModelData/currentList' + baseline_parameter_suffix + '.csv', 'baselineModelData/voltageList' + baseline_parameter_suffix + '.csv']


    # Plot the data
    # plot_multiple_graphs_from_csv(multiplot_inference_vs_inference, titles6, x_col, y_col, nrows=3, ncols=2)
    
    # plot_multiple_graphs_from_csv(multiplot_inference_vs_baseline, titles6, x_col, y_col, nrows=3, ncols=2)

    # plot_multiple_graphs_from_csv(multiplot_inference_vs_success_baseline, titles6, x_col, y_col, nrows=3, ncols=2)

    # plot_multiple_graphs_from_csv(multiplot_baseline, titles3, x_col, y_col, nrows=3, ncols=1)
    
    # plot_multiple_graphs_from_csv(multiplot_inference, titles4, x_col, y_col, nrows=3, ncols=1)
    
    plot_multiple_graphs_from_csv(multiplot_inference, titles4, x_col, y_col, nrows=3, ncols=1)

    # plot_multiple_graphs_from_csv(multiplot_training, titles4, x_col, y_col, nrows=6, ncols=1)

    # plot_multiple_graphs_from_csv(multiplot_predictive, titles3, x_col, y_col, nrows=3, ncols=1)

    # plot_multiple_graphs_from_csv(multiplot_astar, titles3, x_col, y_col, nrows=3, ncols=1)

    # plot_multiple_graphs_from_csv(multiplot_oracle, titles3, x_col, y_col, nrows=3, ncols=1)

    # plot_multiple_graphs_from_csv(multiplot_goodIT, titles3, x_col, y_col, nrows=3, ncols=1)

    



