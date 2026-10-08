from RL_model import FullyTaskedBatteryLessWorldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit


combined = False
shuffledData = True
trainingPayload = "daily_random_20_255"
trainingCapSizeStr = "random_0.5_10"
tsf_max = 1000
neg_inaction_reward = -0.5
gamma = 0.99
optimization_metric = "off_time" # Change to "off_time" or "jitter" based on what you want to optimize for (off_time = minimize off time, jitter = variability in time between successful transmissions)

if optimization_metric == "jitter":
    additional_text = "JITTER_OPT_" + str(neg_inaction_reward) + "_INACTION_REWARD_ORIGINAL_ADR_" + trainingPayload + "_BYTES_" + trainingCapSizeStr + "FARAD_TSF_" + str(tsf_max) + "_GAMMA_" + str(gamma)  # naming convention: POSREWARD_NEGREWARD_ALPHA_MU_SPREADING_SIGMA_CAPSIZE
else:
    additional_text = "OFF_OPT_" + str(neg_inaction_reward) + "_INACTION_REWARD_ORIGINAL_ADR_" + trainingPayload + "_BYTES_" + trainingCapSizeStr + "FARAD_TSF_" + str(tsf_max) + "_GAMMA_" + str(gamma)  # naming convention: POSREWARD_NEGREWARD_ALPHA_MU_SPREADING_SIGMA_CAPSIZE

nrOfInterpolationPoints = 29
downSampleFactor = 3

if shuffledData == False:   
    max_inference_steps = (5338 * nrOfInterpolationPoints) + 5339
else:
    max_inference_steps = 388800 / downSampleFactor # total shuffled solar validation dataset  / amount of downsampling
    max_inference_steps = int(max_inference_steps)

seeds = [42, 123, 456, 789, 999]
# seeds = [0, 222, 333, 444, 555]

# caps = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Capacitor sizes to evaluate

for seed in seeds:
    print(f"\n--- Running Inference for Seed {seed} ---")
    
    # 1. Target the specific seeded model and data suffixes
    seed_additional_text = additional_text + f"_SEED_{seed}"
    
    if combined == True:
        model_path = f"./RLModelData/experiments_combined/models/rl_model_final{seed_additional_text}"
    else:
        model_path = f"./RLModelData/experiments_solar/models/rl_model_final{seed_additional_text}"
        
    model = PPO.load(model_path)

    # 2. Setup the environment for this specific seed
    env = FullyTaskedBatteryLessWorldEnv()
    env.training = False
    env.combined = combined
    env.shuffledData = shuffledData
    env.downSampleFactor = downSampleFactor
    
    # IMPORTANT: Set BOTH suffixes so save_results() names the files correctly for evaluation.py
    env.training_parameter_suffix = seed_additional_text
    # env.inference_parameter_suffix = "INF"
    
    env.final_step = max_inference_steps 
    env.episode_length = max_inference_steps

    env = TimeLimit(env, max_episode_steps=max_inference_steps) 
    env = Monitor(env)

    # 3. Pass the seed to reset to lock the testing environment
    obs, info = env.reset(seed=seed)
    done = False
    truncated = False

    # 4. Manually step through the entire inference dataset
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
    # 5. Export the inference CSVs (including the new successList)
    env.unwrapped.save_results()