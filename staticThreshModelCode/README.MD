# Static Thresholding Approach Manager

This directory contains the Python implementation of the **Static Thresholding** policy for batteryless, energy-harvesting IoT nodes.

Static thresholding represents the most simple operational strategy within the simulation framework. Task execution decisions are governed by a fixed, predefined voltage rule ($V_{thresh}$). At each wake-up interval, the system directly compares the current capacitor voltage ($V_t$) to this constant value to determine whether to perform a task or conserve energy.

---

## Working Principle & Operational Profiles

At each 30-second wake-up step, the algorithm executes one of two deterministic paths based on the available energy buffer:

1. **Sufficient Charge ($V_t \ge V_{thresh}$):** The system triggers full peripheral execution, performing the temperature sensor readout (`rSHT`, `tSHT`) followed immediately by the active communication cycle (`rCycle`, `tCycle`).


2. **Insufficient Charge ($V_t < V_{thresh}$):** The system circumvents the high-power communication cycle. It still performs the (low-power) sensor read-outs, but then immediately transitions the MCU into a low-power sleep state (`rSleep`) for the remainder of the interval to systematically harvest ambient energy.



## Threshold Modes

The script includes configurations to evaluate three specific threshold behaviors discussed in the research paper:

* **Aggressive Baseline (`optimizedThresholds = False`, `highThreshold = False`):** Sets $V_{thresh} = 1.9\text{ V}$. This maximizes short-term data collection frequency but places the node at a high risk of catastrophic brownouts under volatile workloads.


* **Conservative Baseline (`highThreshold = True`):** Fixed at $V_{thresh} = 3.45\text{ V}$. This prioritizes device survival across zero-harvesting gaps but leaves substantial ambient energy unutilized.


* **Statically Optimized Mapping (`optimizedThresholds = True`):** Assigns an optimized static voltage target from `thresh_map` (ranging from $1.9\text{ V}$ up to $3.45\text{ V}$) mapped precisely to the node's physical capacitor size, based on experimental findings.



---

## Key Parameters & Simulation Design

* **`c`**: Storage capacitor volume in Farads (evaluations typically span discrete bounds from $0.5\text{ F}$ to $10.0\text{ F}$).


* **`adrSimulation = True`**: Enables variable load profile modeling using LoRa Adaptive Data Rate (ADR) parameters. Current demands (`iCycle`) swing from $15\text{ mA}$ up to $118\text{ mA}$ based on active RSSI signal attenuation traces.


* **System Operating Limits**: Configured around real-world component limits with a maximum capacitor capacity $V_{max} = 5.5\text{ V}$, boot turn-on voltage $V_{to} = 2.3\text{ V}$, and the hardware brownout point $V_{min} = 1.8\text{ V}$.



---

## Algorithmic Footprint & Leakage Modeling

As a primary baseline approach, Static Thresholding represents a zero-overhead policy requiring minimal hardware processing or memory resources.

* **Capacitor Self-Discharge:** Parasitic leakage current ($I_{leakage}$) is accurately evaluated at every loop step using the discrete supercapacitor regression model derived from component datasheets: $$I_{leakage} = 4.7442 \cdot C + 4.9302 \, (\mu\text{A})$$


* **Hardware Currents:** Core processing draws match standard device parameters, setting active ADC reads to $0.311\text{ mA}$ and the underlying MCU sleep standby to a low current of $0.00065\text{ mA}$.



---

## Execution and Output Dataset Mapping

To execute the static benchmark loop run:

```bash
python static_threshold_model.py

```

Upon processing the continuous 45-day validation dataset , diagnostic simulation files are systematically generated inside the `staticThreshModelData/` subfolder for benchmark comparison against adaptive scheduling methods:

* `voltageList_*.csv`: Time-series capacitor potential recordings ($V_t$).

* `currentList_*.csv`: Logged input harvesting current vectors ($I_H$) driven by real-world validation data.
* `actionList_*.csv` & `successList_*.csv`: Staged transmission attempts and verified successful packets.
* `deviceStateList_*.csv`: Evaluates continuous device operational modes (ON vs. OFF power failures).
* `tCycleList_*.csv` & `iCycleList_*.csv`: Tracks the active duration and current profiles matching the LoRa channel fading simulation.