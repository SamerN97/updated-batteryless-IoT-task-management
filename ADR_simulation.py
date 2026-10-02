import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import random

# -------------------------------
# 1. Simulation Parameters
# -------------------------------
training_data = False  # Set to False for testing data generation
payload_size = 0  # User-defined payload
payload_option = "daily_random_20_255" 
voltage = 3.3
total_inference_steps = 151488
total_training_steps = 1817857 # for training data
base_rssi = -110  
rssi_range = (-135, -90)
rssi_std_dev = 2.5
steps_in_24h = 2880

if training_data:
    total_steps = total_training_steps
else:    
    total_steps = total_inference_steps

# RX parameters (SX1262 typical)
rx_time_ms = 50 
rx_current_mA = 12 

# -------------------------------
# 2. LoRa Air Time Calculator
# -------------------------------
def calculate_air_time(sf, payload_bytes, bw=125000):
    """Calculates Air Time (ms) based on Semtech LoRa formulas"""
    t_symbol = (2**sf) / bw
    t_preamble = (8 + 4.25) * t_symbol
    
    # Low Data Rate Optimization (DE) mandatory for SF11/SF12 at 125kHz
    de = 1 if sf >= 11 else 0
    
    # Payload symbol count formula
    term = math.ceil((8 * payload_bytes - 4 * sf + 28 + 16) / (4 * (sf - 2 * de)))
    n_payload = 8 + max(term * 5, 0)
    
    return (t_preamble + (n_payload * t_symbol)) * 1000

# -------------------------------
# 3. RSSI → ADR Mapping (Datasheet-Driven)
# -------------------------------
def get_adr_params(rssi, pl_size):
    """
    Determines SF and TX Current based on RSSI.
    Current values sourced from SX1262 Typical (3.3V DC-DC optimal).
    """
    if rssi <= -125:
        sf, ma = 12, 118  # SF12 @ +22dBm
    elif rssi <= -120:
        sf, ma = 10, 58   # SF10 @ +17dBm
    elif rssi <= -110:
        sf, ma = 9, 45    # SF9  @ +14dBm
    elif rssi <= -100:
        sf, ma = 8, 25.5  # SF8  @ +14dBm
    else:
        sf, ma = 7, 15    # SF7  @ +10dBm
        
    ms = calculate_air_time(sf, pl_size)
    return round(ms, 2), ma, sf

# -------------------------------
# 4. Generate Trace & Compute Energy
# -------------------------------
if training_data:
    np.random.seed(42)
    random.seed(42)
else:
    np.random.seed(0)  # Different seed for testing data to ensure variability
    random.seed(0)
rssi_trace = [base_rssi]
for _ in range(total_steps - 1):
    next_rssi = np.clip(rssi_trace[-1] + np.random.normal(0, rssi_std_dev), *rssi_range)
    rssi_trace.append(next_rssi)

data = []

payload_size = random.randint(20, 255)
for step, rssi in enumerate(rssi_trace):
    # Every 2880 steps (1 physical day), pick a new payload size
    if step > 0 and step % steps_in_24h == 0:
        payload_size = random.randint(20, 255)
    tx_time_ms, tx_current_mA, sf = get_adr_params(rssi, payload_size)

    # Compute Energy Components
    energy_tx = (tx_current_mA / 1000) * (tx_time_ms / 1000) * voltage
    energy_rx = (rx_current_mA / 1000) * (rx_time_ms / 1000) * voltage
    total_energy = energy_tx + energy_rx
    
    # Matching the desired CSV Structure
    data.append({
        "Step": step,
        "RSSI_dBm": rssi,
        "TxTime_ms": tx_time_ms,
        "RxTime_ms": rx_time_ms,
        "TotalTime_ms": tx_time_ms + rx_time_ms,
        "TxCurrent_mA": tx_current_mA,
        "RxCurrent_mA": rx_current_mA,
        "TotalCurrent_mA": tx_current_mA + rx_current_mA,
        "Energy_TX_J": energy_tx,
        "Energy_RX_J": energy_rx,
        "Energy_Total_J": total_energy,
        "ADR_State": f"SF{sf}_{tx_current_mA}mA", # Modified to include SF for clarity
        "Payload_Size": payload_size,
        "Spreading_Factor": sf 
    })

df = pd.DataFrame(data)

# -------------------------------
# 5. Save to CSV
# -------------------------------
if training_data:
    csv_path = "training_adr_simulation_" + payload_option + "_bytes_payload.csv"
else:    
    csv_path = "validation_adr_simulation_" + payload_option + "_bytes_payload.csv"
# df.to_csv(csv_path, index=False)
# print(f"CSV saved to: {csv_path}")

# -------------------------------
# 6. Plotting (4-panel version)
# -------------------------------
fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

axs[0].plot(df["Step"], df["RSSI_dBm"], color='black')
axs[0].set_ylabel("RSSI (dBm)")
axs[0].set_title("RSSI Over Time")

axs[1].plot(df["Step"], df["Energy_Total_J"], color='red')
axs[1].set_ylabel("Total Energy (J)")
axs[1].set_title(f"Total Energy per Packet (Payload: {payload_option} bytes)")

axs[2].plot(df["Step"], df["TotalTime_ms"], color='orange')
axs[2].set_ylabel("Total Time (ms)")
axs[2].set_title("Total Radio Time (TX + RX)")

axs[3].plot(df["Step"], df["TxCurrent_mA"], label="TX Current", color='green')
axs[3].plot(df["Step"], df["RxCurrent_mA"], label="RX Current", color='blue', linestyle='--')
axs[3].set_ylabel("Current (mA)")
axs[3].legend()
axs[3].set_title("Current Consumption")

axs[4].plot(df["Step"], df["Payload_Size"], color='purple')
axs[4].set_ylabel("Payload Size (Bytes)")
axs[4].set_title("Payload Size")

plt.tight_layout()
plt.show()