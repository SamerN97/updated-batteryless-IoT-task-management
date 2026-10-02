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
optimization_metric = "jitter" # Change to "off_time" or "jitter" based on what you want to optimize for (off_time = minimize off time, jitter = variability in time between successful transmissions)

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


if combined == True:
    model5 = PPO.load("./RLModelData/experiments_combined/models/rl_model_final" + additional_text)
else:
    model5 = PPO.load("./RLModelData/experiments_solar/models/rl_model_final" + additional_text)


env = FullyTaskedBatteryLessWorldEnv()

env.training = False
env.combined = combined
env.shuffledData = shuffledData
env.downSampleFactor = downSampleFactor
# Sync the suffixes so the filenames match your model
env.training_parameter_suffix = additional_text

# Sync the math exactly
env.final_step = max_inference_steps 
env.episode_length = max_inference_steps

env = TimeLimit(env, max_episode_steps=max_inference_steps) 
env = Monitor(env)

print("parameters: " + additional_text)

# Evaluate the trained agent
mean_reward5, std_reward5 = evaluate_policy(model5, env, n_eval_episodes=1, deterministic=True)

print(f"mean_reward after inference = {mean_reward5:.2f}")

# Access the underlying environment to trigger the save
# Since you wrapped it in Monitor and TimeLimit, use .unwrapped
env.unwrapped.save_results()