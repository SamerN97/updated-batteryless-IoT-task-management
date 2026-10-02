# Reinforcement Learning (RL) Approach
Within the RL approach, there are 3 main files:
* `RL_model.py`: The RL environment
* `training.py`: The script used for training the agent
* `inference.py`: The script used for running inference with the trained model

---

## `RL_model.py` RL Environment

This file contains the custom **Gymnasium** environment implementation (`FullyTaskedBatteryLessWorldEnv`) modeling a fully-tasked, batteryless IoT environment for Reinforcement Learning.
The environment frames the dynamic voltage thresholding task as a Markov Decision Process (MDP). It trains a model-free agent (PPO) to evaluate system variables holistically and navigate the operational trade-offs between energetic survival and high task throughput without specialized hardware profiling.

---

### Observation and Action Spaces

The RL-environment uses the following setup:

* **Action Space:** A discrete space (`spaces.Discrete(2)`) defining two system pathways:


    * `0` (**No Execute / NE**): Conserves energy during sleep cycles.


    * `1` (**Execute / E**): Triggers a high-power LoRa packet transmission.




* **Observation Space:** A structured sequence box of dimension `(9, 10)` mapping $k=10$ sequential context timesteps (history) across **9 specialized features** normalized strictly between `0.0` and `1.0`:


    1. **Capacitor Voltage ($V_t$)** $\rightarrow$ Primary charge indicator.


    2. **Harvesting Current ($I_H$)** $\rightarrow$ Instataneous input solar trace.


    3. **Cycle Time ($t_{Cycle}$)** $\rightarrow$ Last measured transmission time.


    4. **Device State** $\rightarrow$ Binary availability trace indicating if the hardware is currently operational or dead.

    5. **Time Since Failure (TSF)** $\rightarrow$ Continuous survival streak tracker acting as a risk decay variable.


    6. **Payload Size (PL)** $\rightarrow$ Application demand bounds varying dynamically between 20 and 255 bytes.


    7. **Spreading Factor (SF)** $\rightarrow$ Communication parameter regulated by the channel's ADR mechanics.


    8. **Capacitor Size ($C$)** $\rightarrow$ Evaluates physical capacitor size across mutations.


    9. **Stored Energy ($E_t$)** $\rightarrow$ Explicit energetic footprint calculated directly via $E(t) = \frac{1}{2}CV(t)^2$.





---

### Multi-Objective Optimization Regimes

Using `optimization_metric`, the agent's behavior can be customized via distinct reward functions tailored to specific application constraints:

#### 1. Jitter Optimization (`"jitter"`)

Enforces consistent data transmission pacing by minimizing inter-task intervals (ITI) and bounding maximum communication gaps.

* **Positive Task Reward (PTR):** Scaled by the device's continuous survival record: $$\text{PTR} = 1.0 + 4.0 \cdot \left(\frac{\text{TSF}}{\text{TSF}_{max}}\right)$$


* **Negative Inaction Penalty (NIR):** Aggressively amplified during high energy storage and high solar irradiance to discourage idle hoarding: $$\text{NIR} = -0.5 \cdot (1.0 + 4.0 \cdot E_{norm} \cdot I_{norm})$$



#### 2. Off-Time Optimization (`"off_time"`)

Prioritizes basic hardware protection and survivability, aiming to eliminate power failure blackouts entirely.

* **Positive Task Reward (PTR):** Linked directly to capacitor charge to encourage transmissions safely buffering peak conditions:



$$\text{PTR} = 1.0 \cdot E_{norm}$$


* **Negative Inaction Penalty (NIR):** Scales with available energy and environmental harvesting capacity; no penalty is applied if ambient energy is zero:

$$\text{NIR} = -0.5 \cdot E_{norm} \cdot I_{norm}$$



---

### Physics-Guided Time Distortion & Domain Randomization

The environment embeds severe physical hardware constraints and specialized edge-computing paradigms:

* **Hardware-Agnostic Training:** During training (`training = True`), the script applies strict **domain randomization**. At each `reset()`, it uniformly samples the capacitor size ($C$) from a discrete set between $0.5\text{ F}$ and $10.0\text{ F}$ (in $0.5\text{ F}$ steps). This teaches the policy underlying electrical principles rather than locked system trajectories.


* **Electrical Load Modeling:** Supercapacitor leakage current changes dynamically following physical capacitance adjustments: $I_{leakage} = 4.7442 \cdot C + 4.9302 \, (\mu\text{A})$.


* **Time-Distorted Power Failures (`off_simulation`):** If voltage drops below $V_{min} = 1.8\text{ V}$ due to an unsafe decision, the device shuts down. Because an edge MCU cannot perceive time while unpowered, the environment decouples the agent from individual steps. The script runs a fast-forward loop aggregating ambient charging until $V_t \ge V_{to} (2.3\text{ V})$, wrapping the entire downtime into a single, heavily penalized environment step for the agent.


* **Inference Overhead:** Includes active drawing parameter drops modeled on running mathematical model calls, factoring network evaluation time ($t_{agent} \approx 58\text{ ms}$) into the capacitor physics loop.



---

### Execution and Structure

The setup within `RL_model.py` is used in `training.py` and `inference.py`.

---

## `training.py` Training Script

This file contains the training pipeline orchestrating the reinforcement learning agent's training loop. It hooks the custom `FullyTaskedBatteryLessWorldEnv` environment into **Stable-Baselines3 (SB3)** to train a hardware-agnostic policy.

---

### Execution Flow

The training script executes the following discrete operations:

1. **Environment Initialization:** Instantiates the customized batteryless world environment. (Make sure `training = True` in `RL-model.py`).

2. **Algorithm Setup:** Configures an SB3 **Proximal Policy Optimization (PPO)** agent using a multi-layer perceptron (`MlpPolicy`) neural network architecture. The optimization process is parameterized with a discount factor ($\gamma = 0.99$).


3. **Evaluation Callbacks:** Registers an SB3 `EvalCallback` executing 4 deterministic test evaluation episodes every 10 training episodes (28,800 steps)  to log model learning convergence without exploration noise.


4. **Agent Training:** Executes model optimization across a training horizon of **1,000,000 total environment timesteps**, printing a real-time progress bar and logging telemetry via TensorBoard.


5. **Serialization:** Exports the finalized actor-critic policy weights to compressed archives and flushes the full episodic history metrics to CSV files.

6. **Network Profiling:** Parses the internal PyTorch policy graph to generate a layer-by-layer architectural printout and output a precise mathematical count of the network parameters.



---

### Telemetry & Output Mapping

To launch the model optimization training pipeline run:

```bash
python training.py

```

Binary model checkpoints and tracking telemetry are compartmentalized:

* **TensorBoard Tracks:** Stored in `./RLModelData/experiments_solar/PPO_experiments/` for monitoring performance trends.

* **Model Binaries:** Output to `./RLModelData/experiments_solar/models/rl_model_final*.zip`.

* **Parameter Manifest:** Outputs the layer configurations confirming the structural limits of the MCU's deployment profile.

---

## `inference.py` Inference Script

This file contains the deployment pipeline used to run continuous inference evaluations using your trained PPO models. It validates model generalizability by running the frozen policy network deterministically across the complete 45-day validation dataset.

---

### Execution Flow

The inference script processes execution testing through the following discrete actions (Make sure `training = False` and `c` is set to your desired capacitor size in `RL-model.py`):

1. **Model Serialization Load:** Locates and reloads the optimized, pre-trained neural network parameters (`rl_model_final*.zip`) matching the targeted `optimization_metric` policy.


2. **Continuous Episode Wrapping:** Syncs step metrics directly to the exact length of the high-resolution tracking traces (129,600 total evaluation steps)


3. **Deterministic Policy Evaluation:** Invokes the Stable-Baselines3 `evaluate_policy` engine to process the input features through the policy graph deterministically (`deterministic=True`), tracking continuous device up-time, reward metrics, and output logs.


4. **Analytical Dataset Diagnostics:** Strips away the Gymnasium wrappers (`TimeLimit` and `Monitor`) using the `.unwrapped` property to execute `save_results()`, outputting 13 distinct performance validation CSV logs.



---

### Execution and Telemetry Logging

To launch the evaluation pass using your trained policy network:

```bash
python inference.py

```

The script monitors average reward totals (`mean_reward5`) and outputs the complete benchmarking metrics inside the matching `RLModelData/inference_experiments/` folder directory.