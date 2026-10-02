# Short-Term (ST) Oracle Baseline Manager

This directory contains the Python simulation code for the **Short-Term (ST) Oracle** scheduling policy.

The ST Oracle acts strictly as a **comparative upper-bound baseline** for evaluating your dynamic scheduling algorithms. Because it operates with perfect, zero-overhead knowledge of the immediate future, it cannot be deployed in real-world environments. It establishes a performance ceiling by evaluating exactly whether a pending task execution will survive the current epoch without causing a brownout.

---

## Working Principle: Look-Ahead Feasibility

At the beginning of each 30-second interval, the ST Oracle executes a sequence of forward-looking calculations to verify system stability. It models the expected voltage drops of a full execution duty cycle inside the `Vt_check` variable before committing the real physical state:

1. **Task Impact:** It projects the consecutive voltage drops across the sensor reading (`rSHT`, `tSHT`) and the high-power communication burst (`rCycle`, `tCycle`).
2. **Interval Sleep Impact:** It chains the subsequent decay during the remaining low-power sleep slot: `ti - tSense - tSHT - tCycle`.
3. **Peripheral Read Impact:** It factorizes final voltage tracking stages (`rADC`, `rMCUI2CCoulomb`).

* **`feasible = 1`**: If the projected voltage path never falls below or equal to the minimum operating limit ($V_{min} = 1.8\text{ V}$) at any point in the cycle, the Oracle executes the transmission task.


* **`feasible = 0`**: If any subtask triggers $V_{predict} \le V_{min}$, the task is skipped entirely to protect the hardware, forcing the node into a standard safe energy-accumulation sleep period.

---

## Idealized Characteristics & Performance Constraints

While the ST Oracle provides maximum immediate data-generation throughput, it highlights critical design boundaries described in the paper:

* **Limits:** The baseline is short-term and not globally optimal; it does not calculate long-term cumulative depletion across sequential days or extended nighttime energy gaps.


* **Zero Overhead Physics:** Unlike the neural networks or root-finding numerical solvers, this file assumes the look-ahead verification itself takes $0\text{ ms}$ and consumes $0\text{ nJ}$ of active processing energy.

---

## Key Parameters & Simulation Setup

* **`c`**: Storage capacitor volume in Farads (evaluations typically span discrete bounds from $0.5\text{ F}$ to $10.0\text{ F}$).
* **`adrSimulation = True`**: Simulates the full dynamic variability of the LoRa communication module, linking current bounds (`iCycle`) and transmission airtimes (`tCycle`) directly to network condition traces.


* **System Boundaries**: Built around a maximum storage ceiling $V_{max} = 5.5\text{ V}$, physical cold-boot threshold $V_{to} = 2.3\text{ V}$, and the hardware brownout point $V_{min} = 1.8\text{ V}$.



---

## Execution and Output Dataset Mapping

Execute the Oracle validation test block via:

```bash
python oracle_model.py

```

After validating across the solar harvesting current tracks, the script exports matching data frameworks into the `oracleModelData/` directory for further evaluation:

* `voltageList_*.csv`: Time-series capacitor potential recordings ($V_t$).

* `currentList_*.csv`: Logged input harvesting current vectors ($I_H$) driven by real-world validation data.
* `actionList_*.csv` & `successList_*.csv`: Staged transmission attempts and verified successful packets.
* `deviceStateList_*.csv`: Evaluates continuous device operational modes (ON vs. OFF power failures).
* `tCycleList_*.csv` & `iCycleList_*.csv`: Tracks the active duration and current profiles matching the LoRa channel fading simulation.