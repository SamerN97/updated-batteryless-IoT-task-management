# AsTAR (Adaptive Task Rate) Approach

This directory contains the Python implementation of the **AsTAR** (Adaptive Task Rate) execution algorithm for batteryless, energy-harvesting IoT nodes.

The implementation is based on the work of Yang et al. which can be found through this [link](https://dl.acm.org/doi/10.1145/3467894)

Unlike greedy optimization approaches that maximize immediate task throughput at all costs, AsTAR centers its execution policies around a specific target storage potential ($V_{optimal}$). By dynamically modulating the execution rate via **Additive Increase Multiplicative Decrease (AIMD)** and **Multiplicative Increase Additive Decrease (MIAD)** rules, it focuses on ensuring stable pacing and sustaining lifetime consistency across prolonged energy gaps (such as nighttime communication blackouts).

---

## Working Principle & Adaptation Logic

At each valid evaluation trigger point, the system compares the current capacitor voltage ($V_t$) against an acceptable margin ($m = \text{hysteresis}$) around the preset $V_{optimal}$. The script handles task rates via the following dynamic control regimes in `get_adaptation_rate()`:

* Low Voltage State ($V_t < V_{optimal} - m$):


    * If the voltage is recovering ($V_t > V_{prev}$), the task rate is **incrementally increased** (+1) to cautiously step up activity.


    * If the voltage is dropping ($V_t \le V_{prev}$), the task rate is **divided by 2** (Multiplicative Decrease), aggressively safeguarding the device from brownout.




* High Voltage State ($V_t > V_{optimal} + m$):


    * If the voltage is expanding ($V_t \ge V_{prev}$), the system executes **Multiplicative Increase** ($\times 2$) to aggressively consume overflow energy before saturation.


    * If the voltage is declining ($V_t < V_{prev}$), it relies on an **additive decrease** (-1) to safely step down operations.




* Optimal Voltage State : The execution pacing rate remains unchanged.



The rate-limiting interval is enforced globally by converting the adapted hourly transmission metrics directly into programmatic sleep steps: `execTimer = math.ceil(max_rate_per_hour / rate)`.

---

## Key Parameters & Simulation Setup

* **`V_optimal = 4`**: The benchmark baseline voltage threshold (in Volts) around which AsTAR balances stabilization parameters.


* **`hysteresis = 0.05`**: Defines the boundary margin window ($m$) flanking $V_{optimal}$ to prevent system oscillation.


* **`c`**: Storage capacitor volume in Farads (evaluations typically span discrete bounds from $0.5\text{ F}$ to $10.0\text{ F}$).


* **`adrSimulation = True`**: Links operations to highly variable LoRa Adaptive Data Rate communication loads, updating airtimes (`tCycle`) and current demands (`iCycle`) based on realistic fading datasets.


* **Hardware Operating Limits**: Matches real-world component specifications with a maximum ceiling $V_{max} = 5.5\text{ V}$, cold-boot turn-on voltage $V_{to} = 2.3\text{ V}$, and low-voltage cutoff threshold $V_{min} = 1.8\text{ V}$.



---

## Algorithmic Cost & Capacitor Leakage

AsTAR represents a highly elegant, lightweight tracking architecture that bypasses heavy mathematical solvers or continuous neural network inference processing overheads.

* **Self-Discharge Modelling:** Supercapacitor baseline leakage current is updated continuously at every evaluation step using the experimental regression trace of the DGH component series:



$$I_{leakage} = 4.7442 \cdot C + 4.9302 \, (\mu\text{A})$$


* **Task Constraints:** Baseline idle power configurations default to a standby consumption current of $0.00065\text{ mA}$ at a regular $3.3\text{ V}$ supply bus.



---

## Execution and Output Dataset Mapping

Execute the script from your terminal:

```bash
python astar_model.py

```

Upon executing across the 45-day validation harvesting current tracks, analytical diagnostic trends are output inside the `astarModelData/` directory for visual plotting and performance multi-criteria benchmarking:

* `voltageList_*.csv`: Time-series capacitor potential recordings ($V_t$).

* `currentList_*.csv`: Logged input harvesting current vectors ($I_H$) driven by real-world validation data.
* `actionList_*.csv` & `successList_*.csv`: Staged transmission attempts and verified successful packets.
* `deviceStateList_*.csv`: Evaluates continuous device operational modes (ON vs. OFF power failures).
* `tCycleList_*.csv` & `iCycleList_*.csv`: Tracks the active duration and current profiles matching the LoRa channel fading simulation.