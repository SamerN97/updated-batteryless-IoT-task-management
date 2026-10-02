import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib

# Force background rendering for speed
matplotlib.use('Agg')

# ---------------------------------------------------------
# 1. Global Configurations
# ---------------------------------------------------------
cap_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 
             5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

thresh_map = {
    10.0: 1.9, 9.5: 1.9, 9.0: 1.9, 8.5: 1.9, 8.0: 1.9, 
    7.5: 1.9, 7.0: 1.9, 6.5: 1.9, 6.0: 1.9, 5.5: 1.95, 
    5.0: 1.95, 4.5: 1.95, 4.0: 2.0, 3.5: 2.0, 3.0: 2.05, 
    2.5: 2.05, 2.0: 2.15, 1.5: 2.25, 1.0: 2.5, 0.5: 3.45
}

payloadOption = "daily_random_20_255"
trainingPayload = "daily_random_20_255"
trainingCapSizeStr = "random_0.5_10"
tsf_max = 1000
training_neg_inaction_reward = -0.5
gamma = 0.99
pos_reward = 1.0
off_reward = 6.0
steps_per_day = 2880

# Labels and Styling (Updated Terminology & Accessibility)
labels_all = ['Agent (ITI)', 'Agent (Off-Time)', 'Opt. Static Thresh.', 'Static (1.9V)', 'Static (3.45V)', 'Approx. Pred.', 'ST Oracle', 'AsTAR']

# Okabe-Ito Color Palette (Colorblind-friendly)
colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

# Unique markers for every line
markers = ['o', 's', '^', 'v', '<', 'D', 'p', 'X'] 
linestyles = ['-', '-', '-', '--', '--', '-', '-', '-']

# Dictionaries to hold our trend data across all sizes
trend_data = {
    'mean_iti': {label: [] for label in labels_all},
    'mean_daily_success': {label: [] for label in labels_all},
    'median_survival': {label: [] for label in labels_all},          
    'median_max_daily_iti': {label: [] for label in labels_all},     
    'median_off_time_duration': {label: [] for label in labels_all}  # Updated key
}

# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------
def load_agent_data(path):
    try:
        return pd.read_csv(path, header=None).iloc[:, 0].to_numpy()
    except Exception:
        return np.array([])

def get_tbs(success_array, agent_type=None, eps=0.01):
    if agent_type == "jitter": # Keeping backend type mapping intact
        idx = np.where((success_array > pos_reward - eps) & (success_array < 6.0))[0]
    elif agent_type == "off_time":
        idx = np.where((success_array > 0.0) & (success_array < pos_reward + eps))[0]
    else:
        idx = np.where(success_array == 1)[0]
    return np.diff(idx) if len(idx) >= 2 else np.array([])

def get_daily_success_counts(success_array, agent_type=None, eps=0.01):
    if agent_type == "jitter":
        binary_success = ((success_array > pos_reward - eps) & (success_array < 6.0)).astype(int)
    elif agent_type == "off_time":
        binary_success = ((success_array > 0.0) & (success_array < pos_reward + eps)).astype(int)
    else:
        binary_success = np.copy(success_array)
    
    num_days = len(binary_success) // steps_per_day
    if num_days == 0: return np.array([]) 
    truncated = binary_success[:num_days * steps_per_day]
    return np.sum(truncated.reshape((num_days, steps_per_day)), axis=1)

def get_daily_max_iti(success_array, agent_type=None, eps=0.01):
    if agent_type == "jitter":
        binary_success = ((success_array > pos_reward - eps) & (success_array < 6.0)).astype(int)
    elif agent_type == "off_time":
        binary_success = ((success_array > 0.0) & (success_array < pos_reward + eps)).astype(int)
    else:
        binary_success = np.copy(success_array)
    
    local_num_days = len(binary_success) // steps_per_day
    if local_num_days == 0: return np.array([]) 
    
    truncated = binary_success[:local_num_days * steps_per_day]
    daily_chunks = truncated.reshape((local_num_days, steps_per_day))
    
    daily_max_itis = []
    for day_chunk in daily_chunks:
        success_indices = np.where(day_chunk == 1)[0]
        if len(success_indices) == 0:
            daily_max_itis.append(steps_per_day)
        elif len(success_indices) == 1:
            max_gap = max(success_indices[0], steps_per_day - success_indices[0])
            daily_max_itis.append(max_gap)
        else:
            gaps = np.diff(success_indices)
            first_gap = success_indices[0] 
            last_gap = steps_per_day - success_indices[-1] 
            daily_max_itis.append(max(np.max(gaps), first_gap, last_gap))
    return np.array(daily_max_itis)

def get_survival_time(state_array, agent_type=None, off_r=6.0, eps=0.01):
    if agent_type in ["jitter", "off_time"]:
        is_on = ~((state_array > off_r - eps) & (state_array < off_r + eps))
    else:
        is_on = np.array(state_array).astype(bool)
        
    padded = np.pad(is_on.astype(int), (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    return np.where(diffs == -1)[0] - np.where(diffs == 1)[0]

def get_off_time_durations(state_array, agent_type=None, off_r=6.0, eps=0.01):
    if agent_type in ["jitter", "off_time"]:
        is_off = (state_array > off_r - eps) & (state_array < off_r + eps)
    else:
        is_off = ~np.array(state_array).astype(bool)
        
    padded = np.pad(is_off.astype(int), (1, 1), mode='constant', constant_values=0)
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return ends - starts

# ---------------------------------------------------------
# 3. Main Extraction Loop
# ---------------------------------------------------------
print("Extracting trend data across all capacitor sizes...")

for c in cap_sizes:
    V_thresh = thresh_map[c]
    
    capSizeStr = str(int(c)) if c.is_integer() else str(c)
    V_thresh_str = str(V_thresh)
    
    base_suffix = f"ORIGINAL_ADR_{payloadOption}_BYTES_{capSizeStr}"
    baseline_suffix_opt = f"{base_suffix}_FARAD_{V_thresh_str}V_THRESH"
    baseline_suffix_1_9 = f"{base_suffix}_FARAD_1.9V_THRESH"
    baseline_suffix_3_45 = f"{base_suffix}_FARAD_3.45V_THRESH"
    other_suffix = f"{base_suffix}FARAD"
    
    train_jitter = f"ORIGINAL_JITTER_OPT_{training_neg_inaction_reward}_INACTION_REWARD_ORIGINAL_ADR_{trainingPayload}_BYTES_{trainingCapSizeStr}FARAD_TSF_{tsf_max}_GAMMA_{gamma}"
    infer_jitter = f"ORIGINAL_JITTER_OPT_{training_neg_inaction_reward}_INACTION_REWARD_ORIGINAL_ADR_{payloadOption}_BYTES_{capSizeStr}FARAD_TSF_{tsf_max}"
    
    train_off = f"ORIGINAL_OFF_OPT_{training_neg_inaction_reward}_INACTION_REWARD_ORIGINAL_ADR_{trainingPayload}_BYTES_{trainingCapSizeStr}FARAD_TSF_{tsf_max}_GAMMA_{gamma}"
    infer_off = f"ORIGINAL_OFF_OPT_{training_neg_inaction_reward}_INACTION_REWARD_ORIGINAL_ADR_{payloadOption}_BYTES_{capSizeStr}FARAD_TSF_{tsf_max}"

    folder = 'RLModelData/experiments_solar/inference_experiments'
    
    try:
        ag_jit = load_agent_data(f'{folder}/rewardList{infer_jitter}__{train_jitter}.csv')
        print(f"Successfully loaded agent jitter data for cap size {c} with shape {ag_jit.shape}")
        ag_off = load_agent_data(f'{folder}/rewardList{infer_off}__{train_off}.csv')
        print(f"Successfully loaded agent off-time data for cap size {c} with shape {ag_off.shape}")
        
        if len(ag_jit) == 0 or len(ag_off) == 0:
            raise ValueError("Agent data is empty.")

        b_succ = np.loadtxt(f'staticThreshModelData/successList{baseline_suffix_opt}.csv', delimiter='\t')
        b_state = pd.read_csv(f"staticThreshModelData/deviceStateList{baseline_suffix_opt}.csv", header=None).iloc[:, 0].astype(bool)
        
        b_succ_1_9 = np.loadtxt(f'staticThreshModelData/successList{baseline_suffix_1_9}.csv', delimiter='\t')
        b_state_1_9 = pd.read_csv(f"staticThreshModelData/deviceStateList{baseline_suffix_1_9}.csv", header=None).iloc[:, 0].astype(bool)

        b_succ_3_45 = np.loadtxt(f'staticThreshModelData/successList{baseline_suffix_3_45}.csv', delimiter='\t')
        b_state_3_45 = pd.read_csv(f"staticThreshModelData/deviceStateList{baseline_suffix_3_45}.csv", header=None).iloc[:, 0].astype(bool)

        p_succ = np.loadtxt(f'APModelData/successList{other_suffix}.csv', delimiter='\t')
        p_state = pd.read_csv(f"APModelData/deviceStateList{other_suffix}.csv", header=None).iloc[:, 0].astype(bool)
        
        o_succ = np.loadtxt(f'oracleModelData/successList{other_suffix}.csv', delimiter='\t')
        o_state = pd.read_csv(f"oracleModelData/deviceStateList{other_suffix}.csv", header=None).iloc[:, 0].astype(bool)
        
        a_succ = np.loadtxt(f'astarModelData/successList{other_suffix}.csv', delimiter='\t')
        a_state = pd.read_csv(f"astarModelData/deviceStateList{other_suffix}.csv", header=None).iloc[:, 0].astype(bool)
        
        target_len = len(p_succ)
        ag_jit = ag_jit[:target_len] if len(ag_jit) > target_len else ag_jit
        ag_off = ag_off[:target_len] if len(ag_off) > target_len else ag_off

        succ_arrays = [ag_jit, ag_off, b_succ, b_succ_1_9, b_succ_3_45, p_succ, o_succ, a_succ]
        state_arrays = [ag_jit, ag_off, b_state, b_state_1_9, b_state_3_45, p_state, o_state, a_state]
        types = ["jitter", "off_time", None, None, None, None, None, None]

        for i, label in enumerate(labels_all):
            tbs = get_tbs(succ_arrays[i], agent_type=types[i])
            trend_data['mean_iti'][label].append(np.mean(tbs) if len(tbs) > 0 else np.nan)
            
            daily = get_daily_success_counts(succ_arrays[i], agent_type=types[i])
            trend_data['mean_daily_success'][label].append(np.mean(daily) if len(daily) > 0 else 0)
            
            surv = get_survival_time(state_arrays[i], agent_type=types[i])
            trend_data['median_survival'][label].append(np.median(surv) if len(surv) > 0 else 0)

            daily_max = get_daily_max_iti(succ_arrays[i], agent_type=types[i])
            trend_data['median_max_daily_iti'][label].append(np.median(daily_max) if len(daily_max) > 0 else np.nan)
            
            b_dur = get_off_time_durations(state_arrays[i], agent_type=types[i])
            trend_data['median_off_time_duration'][label].append(np.median(b_dur) if len(b_dur) > 0 else np.nan)

    except Exception as e:
        print(f"Skipping cap size {c} due to missing data: {e}")
        for label in labels_all:
            trend_data['mean_iti'][label].append(np.nan)
            trend_data['mean_daily_success'][label].append(np.nan)
            trend_data['median_survival'][label].append(np.nan)
            trend_data['median_max_daily_iti'][label].append(np.nan)
            trend_data['median_off_time_duration'][label].append(np.nan)

print("Data extraction complete. Generating plots...")

# ---------------------------------------------------------
# 4. Plotting the Trends
# ---------------------------------------------------------
out_dir = "graphs_original/comparison_all/trend_lines"
os.makedirs(out_dir, exist_ok=True)

def plot_trend(metric_key, title, ylabel, filename, use_log=False, ymin=None, ymax=None):
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels_all):
        y_data = trend_data[metric_key][label]
        
        plt.plot(cap_sizes, y_data, label=label.replace('\n', ' '), 
                 color=colors[i], marker=markers[i], linestyle=linestyles[i],
                 linewidth=3, markersize=8, alpha=0.9)
    
    if use_log: 
        plt.yscale('log')
    
    if ymin is not None or ymax is not None:
        plt.ylim(bottom=ymin, top=ymax)
        
    # plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Capacitor Size (Farads)', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    plt.xticks(cap_sizes, rotation=45, fontsize=12) 
    plt.yticks(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.legend(loc='best', fontsize=12, framealpha=0.85)

    plt.tight_layout()
    
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved trend plot: {filepath}")

# 1. Mean ITI 
plot_trend('mean_iti', 'Trend: Mean Inter-Task Interval vs. Capacitor Size', 
           'Mean Time Steps Between Successes', 'trend_mean_iti.pdf', use_log=False, ymin=0)

# 2. Mean Daily Successes
plot_trend('mean_daily_success', 'Trend: Mean Daily Executions vs. Capacitor Size', 
           'Mean Successful Transmissions per 24h', 'trend_daily_success.pdf', use_log=False)

# 3. Median Time Between Off-times
plot_trend('median_survival', 'Trend: Median Time Between Off-States vs. Capacitor Size', 
           'Median Continuous ON-State Duration (Steps, Log)', 'trend_time_between_off_states.pdf', use_log=True)

# 4. Median Daily Max ITI 
plot_trend('median_max_daily_iti', 'Trend: Median of Daily Max ITI vs. Capacitor Size', 
           'Median Max Daily ITI (Steps, Log)', 'trend_max_daily_iti.pdf', use_log=True, ymin=100)

# 5. Median Off-time Duration
plot_trend('median_off_time_duration', 'Trend: Median Off-State Duration vs. Capacitor Size', 
           'Median Continuous OFF-State Duration (Steps, Log)', 'trend_off_state_duration.pdf', use_log=True)

print("All 5 trend lines generated successfully as PDFs!")