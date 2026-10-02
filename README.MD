# Batteryless-IoT-task-management

This repository contains the simulation framework for an ambiently powered, batteryless IoT node communicating via LoRa with an Adaptive Data Rate (ADR) implementation. It simulates and benchmarks energy-aware task execution logic across five distinct strategies:
* **RL-based:** A model-free Reinforcement Learning agent.
* **Approximated Prediction:** An on-the-fly analytical prediction solver.
* **AIMD-based (AsTAR):** An additive increase multiplicative decrease task rate adaptor.
* **Static Thresholding:** Predefined operational voltage baselines.
* **Short-term Oracle:** A short-term look-ahead performance baseline.

---

## 🚀 Getting Started & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SamerN97/batteryless-IoT-task-management.git
cd batteryless-IoT-task-management
```

### 2. Set Up a Virtual Environment
* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **On macOS/Linux:**
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Simulation Execution Workflow

To run the full validation baseline, follow the data generation pipelines in order:

### Step 1: Solar Data Preprocessing
Generate the interpolated validation timeline segments by running the data scripts sequentially:
```bash
python solar_data_preprocessing.py
python solar_current_preprocessing_1_day_blocks.py
python solar_current_preprocessing_3_day_blocks.py
```

### Step 2: Generate LoRa ADR Traces
1. Set `training_data = True` inside `ADR_simulation.py` and run it to create training fading distributions.
2. Set `training_data = False` inside `ADR_simulation.py` and run it to produce validation profiles.

### Step 3: Run Deterministic Schedulers
Execute the following models across all capacitance increments from $0.5\text{ F}$ up to $10.0\text{ F}$ (in $0.5\text{ F}$ steps):
```bash
python AP_model.py
python astar_model.py
python oracle_model.py
```

Run `static_threshold_model.py` across the same capacitor bounds for each of these three target parameters:

1. `optimizedThresholds = True` 

2. `optimizedThresholds = False` and `highThreshold = False`

3. `optimizedThresholds = False` and `highThreshold = True`

---

## 🧠 RL Agent Pipeline

### Training the Policy
For each target reward configuration $o$ (where $o$ is either `"jitter"` or `"off_time"`):
1. **Configure `RL_model.py`**: Set `training = True` and `optimization_metric = o`.
2. **Configure `training.py`**: Set `optimisation_metric = o`.
3. **Execute Training**: 
   ```bash
   python training.py
   ```

### Running Agent Inference
1. Open `RL_model.py` and toggle `training = False`.
2. For each optimization target configuration $o$ (`"jitter"` or `"off_time"`):

   * Set `optimization_metric = o` inside `RL_model.py`.

   * For each discrete capacitor step validation size $x$ ($0.5\text{ F} \to 10.0\text{ F}$):

     * Set `cap_size = x` in `RL_model.py`.

     * Run the inference file:
       ```bash
       python inference.py
       ```

---

## 📊 Evaluating & Plotting Results

To parse data files and output the analytical multi-criteria benchmarking charts, run:
```bash
python evaluation.py
```

This generates continuous parameter trends across all evaluated capacitor bounds for:
* **Mean Inter-Task Interval (ITI)** 
* **Mean Daily Successful Executions** 
* **Median Continuous ON-State Survival Time** 
* **Median Daily Maximum ITI (Pacing under harvesting gaps)** 
* **Median Continuous OFF-State Recovery Duration**

---

## ⚠️ Paper Results Replication

To identically reproduce the figures and numerical bounds presented in the publication, implement these overrides:

* **ADR Dataset Swap:** Inside all the simulation files for the different approaches, change the active layout file reference from:
  ```python
  adrList = pd.read_csv('validation_adr_simulation_' + payloadOption + '_bytes_payload.csv', sep = ',')
  ```
  to the original immutable trace:
  ```python
  adrList = pd.read_csv('original_validation_adr_simulation_' + payloadOption + '_bytes_payload.csv', sep = ',')

* **RL-model Swap:** Change `OFF_OPT` and `JITTER_OPT` to `ORIGINAL_OFF_OPT` and `ORIGINAL_JITTER_OPT` in the suffix/additional_text for both `RL_model.py` and `inference.py`