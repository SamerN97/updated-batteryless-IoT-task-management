# Approximated Prediction (AP) Task Manager

This directory contains the Python implementation of the **Approximated Prediction (AP)** strategy for batteryless, energy-harvesting IoT nodes.

It uses an approximation of the transmission task load to predict the feasibility, and make decisions on the execution.

---

## Working Principle & Architecture

The AP approach operates in three primary phases within our discrete simulation environment:


1. **Prediction (Predict):** At each wake-up interval ($t_i = 30\text{ s}$), the MCU utilizes its cached, approximated equivalent load parameters ($R_{eq}$) to solve the closed-form RC step response equation: $$V(t+\Delta t) = I_H R_{eq} \left(1 - e^{-\frac{\Delta t}{R_{eq}C}}\right) + V(t)e^{-\frac{\Delta t}{R_{eq}C}}$$
It estimates if a pending task (e.g., LoRa transmission) will cause the capacitor voltage to violate a dynamically padded minimum safe threshold: $V_{min\_padded} = V_{min} + 1.5 \cdot I_{task}$.



2. **Execution Decision:** If the predicted state is safe ($V_{predict} > V_{min\_padded}$), the task executes; otherwise, the device returns to sleep mode to store energy.


3. **Verification & Recalibration (Verify):** Following an execution, if the discrepancy between the actual voltage ($V_{actual}$) and $V_{predict}$ exceeds a dynamic error threshold ($E_{thresh} = 10 \cdot V_{supply} / R_{eq}$), or if a brownout occurs, a **Newton-Raphson root-finding solver** is invoked to dynamically recalculate the drifting resistance parameters.



---

## Key Code Parameters & Configurations

The script handles specific hardware profiles, environmental dataset options, and safe bounds:

* **`c`**: Storage capacitor volume in Farads (evaluations typically span discrete bounds from $0.5\text{ F}$ to $10.0\text{ F}$).


* **`adrSimulation = True`**: Enables highly variable execution tasks mimicking LoRa Adaptive Data Rate (ADR) current draws ($15\text{ mA}$ to $118\text{ mA}$) linked to log-normal environmental fading traces.


* **`payloadOption = "daily_random_20_255"`**: Sets application demand variability by randomizing payload lengths between 20 and 255 bytes on a daily basis.


* **`shuffledData = True`**: Drives the model using, linearly interpolated 30-second intervals from real-world solar irradiance traces collected in Hveragerði, Iceland.


* **Hardware Thresholds**: Bounded symmetrically around a maximum voltage $V_{max} = 5.5\text{ V}$, boot turn-on threshold $V_{to} = 2.3\text{ V}$, and brownout voltage $V_{min} = 1.8\text{ V}$.



---

## Algorithmic Cost & Overheads

Unlike the model-free RL agent, the AP tracker acts as a mathematically direct, lightweight solver. However, its mathematical logic incurs small processing costs that are directly modeled inside the state logic loop:

* **Simulation Verification Step:** Approximated to consume ~300 clock cycles ($0.3\text{ ms}$ at $1\text{ MHz}$) per prediction.


* **Newton-Raphson Iteration:** Each corrective iteration runs for ~3 ms drawing $0.091\text{ mA}$ active current. To avoid infinite numerical loops, iterations are strictly capped at a maximum of 100 steps.


* **Capacitor Leakage:** Dynamically scaled to match physical self-discharge profiles of the Cornell Dubilier DGH supercapacitor series via linear regression: $$I_{leakage} = 4.7442 \cdot C + 4.9302 \, (\mu\text{A})$$



---

## Execution and Output Mapping

To run the simulation file from this directory:

```bash
python AP_model.py

```

Upon a successful 45-day evaluation execution tracking through the validation datasets, it logs the performance tracking metrics and exports comma-separated datasets into the `APModelData/` directory for analytical benchmarking:

* `voltageList_*.csv`: Time-series capacitor potential recordings ($V_t$).

* `currentList_*.csv`: Logged input harvesting current vectors ($I_H$) driven by real-world validation data.
* `actionList_*.csv` & `successList_*.csv`: Staged transmission attempts and verified successful packets.
* `deviceStateList_*.csv`: Evaluates continuous device operational modes (ON vs. OFF power failures).
* `tCycleList_*.csv` & `iCycleList_*.csv`: Tracks the active duration and current profiles matching the LoRa channel fading simulation.