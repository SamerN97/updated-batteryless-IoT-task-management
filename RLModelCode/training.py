from RL_model import FullyTaskedBatteryLessWorldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TimeLimit
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback


combined = False # Change to false when only using solar
trainingPayload = "daily_random_20_255"
capSizeStr = "random_0.5_10"
tsf_max = 1000
neg_inaction_reward = -0.5
gam = 0.99
optimization_metric = "jitter" # Change to "off_time" or "jitter" based on what you want to optimize for (off_time = minimize off time, jitter = variability in time between successful transmissions)


if optimization_metric == "jitter":
    suffix = "JITTER_OPT_" + str(neg_inaction_reward) + "_INACTION_REWARD_ORIGINAL_ADR_" + trainingPayload + "_BYTES_" + capSizeStr + "FARAD_TSF_" + str(tsf_max) + "_GAMMA_" + str(gam)  # naming convention: POSREWARD_NEGREWARD_ALPHA_MU_SPREADING_SIGMA_CAPSIZE
else:
    suffix = "OFF_OPT_" + str(neg_inaction_reward) + "_INACTION_REWARD_ORIGINAL_ADR_" + trainingPayload + "_BYTES_" + capSizeStr + "FARAD_TSF_" + str(tsf_max) + "_GAMMA_" + str(gam)  # naming convention: POSREWARD_NEGREWARD_ALPHA_MU_SPREADING_SIGMA_CAPSIZE

steps_in_24_h = 2880 # e.g. 24 * 60 * 2 to go from 30 sec to 24 h (DEPENDANT ON ti)
days_per_episode = 3
episode_length = steps_in_24_h * days_per_episode # This is the number of steps
# env = BatteryLessWorldEnv()
env = FullyTaskedBatteryLessWorldEnv()

# Sync parameters to the instance (NOT WORKING, CHANGE IN ENVIRONMENT ITSELF!)
env.training = True  # MUST BE TRUE FOR TRAINING REWARDS AND DATA SHUFFLING
env.combined = combined
env.trainingPayload = trainingPayload
env.training_parameter_suffix = suffix
# Optional: sync other values to ensure consistency
env.neg_inaction_reward = neg_inaction_reward



if combined == True:
    model = PPO("MlpPolicy", env, gamma=gam, verbose=0, tensorboard_log="./RLModelData/experiments_combined/PPO_experiments/")
else:
    model = PPO("MlpPolicy", env, gamma=gam, verbose=0, tensorboard_log="./RLModelData/experiments_solar/PPO_experiments/")


eval_callback = EvalCallback(model.env, eval_freq = episode_length * 10, n_eval_episodes=4, deterministic=True, render=False) 



model.learn(total_timesteps=1000000, tb_log_name='PPO_' + suffix, progress_bar=True, callback= eval_callback) # TEST NEW WAY OF TRAINING FOR NEW DATA 1 MINUTE INTERVAL WITH INTERPOLATION --> TOTAl NR OF STEPS - EVAL STEPS (15 * 10 * 500)


if combined == True:   
    model.save('./RLModelData/experiments_combined/models/rl_model_final' + suffix) 
else:
    model.save('./RLModelData/experiments_solar/models/rl_model_final' + suffix)


# Save the training CSVs
print("Saving training logs to CSV...")
env.unwrapped.save_results()


from typing import List, Any, Tuple
import torch



def summary(network):
    print("======================================")
    print("= Network Summary                    =")
    print("======================================")
    print(print_layers([(network.__class__.__name__, extract_parameters(network.parameters()), find_layers(network))]))
    print(network.parameters())

def find_layers(network) -> List[Tuple[str, Any]]:
    layers = []
    for child in network.children():
        children = list(child.children())
        if not len(children):
            layers.append((child.__class__.__name__, extract_parameters(child.parameters())))
        else:
            layers.append((child.__class__.__name__, extract_parameters(child.parameters()), find_layers(child)))
    return layers

def extract_parameters(parameters) -> int:
    total = 0
    for param in parameters:
        if isinstance(param, torch.Tensor):
            total += torch.prod(torch.LongTensor(list(param.size())))
        else:
            total += extract_parameters(param)
    return int(total)


def print_layers(layers) -> str:
    result = ""
    for layer in layers:
        result += "- " + layer[0]
        if len(layer) == 2:
            result += f': {layer[1]:n}\n'
        else:
            result += f': {layer[1]:n}\n'
            for line in print_layers(layer[2]).splitlines():
                result += "   %s\n" % line
    return result

summary(model.policy)



