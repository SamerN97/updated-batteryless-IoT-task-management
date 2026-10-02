

import gymnasium
from gymnasium import spaces
import numpy as np
import csv
import math as math
import pandas as pd
import random
import os

class FullyTaskedBatteryLessWorldEnv(gymnasium.Env):

    training = False # Change to False when running inference.py
    cap_size = 10 # Capacitor size
    optimization_metric = "jitter" # Change to "off_time" or "jitter" based on what you want to optimize for (off_time = minimize off time, jitter = variability in time between successful transmissions)

    adrSimulation = True
    multipleMu = False # This is used for going over different cycle times (5, 10, 15 and 20) during inference
    shuffledData = True # True when using the augmented data from data_augmentation.py
    randomCycleTime = False # True when using random cycle time (for training)
    combined = False # Change to False when only using solar data
    payloadOption = "daily_random_20_255"
    trainingPayload = "daily_random_20_255"
    training_cap_size = 1
    trainingCapSizeStr = "random_0.5_10"
    capSizeStr = str(cap_size)
    tsf_max = 1000
    neg_inaction_reward = -0.5
    gamma = 0.99

    if optimization_metric == "jitter":
        suffix = "JITTER_OPT" + str(neg_inaction_reward) + "_INACTION_REWARD_ADR_" + trainingPayload + "_BYTES_" + capSizeStr + "FARAD_TSF_" + str(tsf_max) + "_GAMMA_" + str(gamma)  
        inference_parameter_suffix = "JITTER_OPT_" + str(neg_inaction_reward) + "_INACTION_REWARD_ADR_" + payloadOption + "_BYTES_" + capSizeStr + "FARAD_TSF_" + str(tsf_max)
    else:
        suffix = "OFF_OPT_" + str(neg_inaction_reward) + "_INACTION_REWARD_ADR_" + trainingPayload + "_BYTES_" + capSizeStr + "FARAD_TSF_" + str(tsf_max) + "_GAMMA_" + str(gamma)  
        inference_parameter_suffix = "OFF_OPT_" + str(neg_inaction_reward) + "_INACTION_REWARD_ADR_" + payloadOption + "_BYTES_" + capSizeStr + "FARAD_TSF_" + str(tsf_max)
    ti = 30
    downSampleFactor = 3 # Change based on ti and dataset sample frequency (for shuffled dataset frequency is every 10 sec) --> e.g. if we want to sample every 30 sec, we only keep every 3 samples
    nrOfInterpolationPoints = 29 # Change based on ti and dataset sample frequency (for our teg + solar dataset frequency is every 15 min)
    steps_in_24_h = 2880 # e.g. 24 * 60 * 2 to go from 30 sec to 24 h (DEPENDANT ON ti)
    days_per_episode = 3
    episode_length = steps_in_24_h * days_per_episode # This is the number of steps in an episode (e.g. 3 days with 30 sec intervals --> 2880 * 3)
    eval_freq = 10 * episode_length
    nr_of_evals = 4
    pos_reward = 1
    neg_reward = -10 # Put the actual negative value here (as opposed to the naming convention)
    rewardWeight = 0
    mu = 5

    tCycleMin = 5
    tCycleMax = 20

    off_min = 0
    off_max = 1

    c_min = 0.5
    c_max = 10

    spreading = 1

    sigma = 0.4
    lower = mu - spreading
    upper = mu + spreading

    if adrSimulation == True:
        mu = 0.247 # We use mu to decide tCycle
        sigma = 0.00000001
        spreading = 0.000001 # This decides where the output of the normal distribution gets clipped
        lower = mu - spreading
        upper = mu + spreading

    # --- NORMALIZATION CONSTANTS ---
    # Max possible Capacitor Voltage (Vt_max is 5.5 V)
    V_max_norm = 5.5
    # Max possible Harvesting Current
    if combined == True:
        I_max_norm = 0.08106409804416 # This is the max harvesting current we get from the combined solar + teg dataset
    else:
        I_max_norm =  0.089834 # This is the max harvesting current we get from the solar-only dataset
    # Max possible Cycle Time
    CT_max_norm = tCycleMax + spreading
    # Max possible Device State (off_max is 1)
    OFF_max_norm = 1.0
    # Max possible Time Since Failure (tsf_max is 1000)
    TSF_max_norm = 1000
    # Max payload size
    PL_max_norm = 255
    # Min payload size
    PL_min_norm = 20
    # Min Spreading Factor
    SF_min_norm = 7
    # Max Spreading Factor
    SF_max_norm = 12
    # Min capacitor size
    C_min_norm = 0.5
    # Max capacitor size
    C_max_norm = 10
    # --- END NORMALIZATION CONSTANTS ---

    cummulativeReward = 0
    cummulativeRewardList = list()
    ewmaCapVoltageList = list() # uncomment when using ewma
    finalHarvestingCurrentList = list()
    harvestingCurrentList = list()
    rewardList = list()
    global_counter = 0
    capVoltageList = list()
    irradianceList = list()
    soilTempList = list()
    airTempList = list()
    dataList = list()
    tCycleList = list()
    feasibilityList = list()
    deviceStateList = list()
    restartList = list()
    failList = list()

    final_step = 0

    n = 0  
    total_n = 0

    
    fail = 0
    off_simulation_happened = False
    rewardBeforeOffState = 0
    first_iteration = False
    already_in_off = False
    sumOffReward = 0

    adrCount = 0


    onCount = 0



    def initialize(self):
        self.OFF = []
        self.I = []
        self.Ih = 0.02
        self.It = 0
        self.It_max = 3
        self.It_min = 0
        self.V_to = 2.3
        self.history_size = 10 
        self.V = [self.V_to] * self.history_size
        self.Vt_max = 5.5
        self.V_supply = 3.3
        self.Vt_min = 1.8
        self.Vt_min_obs = 0
        self.b1 = 0
        self.b2 = 0
        self.feasible = 0
        self.Vt_check = 0
        if self.training == True:
            self.c = self.training_cap_size
        else:
            self.c = self.cap_size

        self.combinerEfficiency = 0.88
        self.ctr = 0
        self.iCycle = 0.022671 # Using recalculated power profiling of processing on IoWater device and NB-IoT on CO2 Box (see PPT)
        self.iSleep = 0.00003
        self.iStartup = 0.003        
        self.off = False

        self.rCycle = self.V_supply/self.iCycle
        self.rSleep = self.V_supply/self.iSleep
        self.rStartup = self.V_supply/self.iStartup
        # self.rOff = self.V_supply/self.iLeakage 
        self.reward = 0
        self.tCycle = self.mu
        self.tStartup = 0.001


        self.v = 3.3 
        self.Vt = self.capacitor_voltage(self.It, self.rStartup, self.tStartup, self.c, self.V_to)

        self.tCycle_min = 0
        self.CT = []
        self.tCycle_init = self.mu
        self.noisy_tCycle = self.mu

        self.done = False

        self.truncated = False

        self.step_counter = 0

        self.sim_steps_this_episode = 0

        self.final_training_step = 1000000 + math.floor(1000000 / self.eval_freq) * self.nr_of_evals * self.episode_length # This is for using episodes of 2880 steps with 4 evaluation episodes every 10 training episodes

        self.restartWillHappen = False

        # 'Time since failure parameter' serves as a survival streak counter for the device
        self.tsf = 0 # Initialize TSF to 0 (meaning it just recovered from failure)
        self.TSF = [0] * self.history_size # History array for TSF
        self.tsf_min = 0

        self.C = [self.c] * self.history_size

        self.saved_V = [2.7] * self.history_size
        self.saved_I = [0] * self.history_size
        self.saved_CT = [self.tCycle_init] * self.history_size
        self.saved_OFF = [1] * self.history_size
        self.saved_TSF = self.TSF
        self.saved_C = self.C

        # Initialize PL
        self.pl = 20
        self.PL = [self.pl] * self.history_size # History array for PL
        self.plList = []
        self.saved_PL = self.PL

        # Initialize SF
        self.sf = 7
        self.SF = [self.sf] * self.history_size # History array for SF
        self.sfList = []
        self.saved_SF = self.SF

        # Initialize tsf list
        self.tsfList = []


        ## General Current Values
        self.iCycle = 0
        self.iSense = 0
        self.iLeakage = 0.0000047442 * self.c + 0.0000049302 # Based on linear regression of the leakage current for different capacitor sizes from datasheet (see calculate_leakage_current.py)


        ## Current Consumption Parameters Per Subtask
        # SHT30
        self.iSHT = 0.0006

        # MCU Sleep
        self.iSleep = 0.00000065 

        # NB-IoT Module
        self.iNBIoT = 0.02065

        # MCU Active
        self.iADC = 0.000311
        self.iActiveMCU = 0.000091
        self.iMCUI2CCoulomb = self.iActiveMCU
        self.iAgent = self.iActiveMCU


        ## Equivalent Resistance Values Per Task
        self.rAgent = self.V_supply / (self.iAgent + self.iLeakage)
        print(self.rAgent)
        self.rSHT = self.V_supply / (self.iSHT + self.iActiveMCU + self.iLeakage)
        print(self.rSHT)
        self.rSleep = self.V_supply / (self.iSleep + self.iLeakage)
        print(self.rSleep)
        self.rNBIoT = self.V_supply / (self.iNBIoT + self.iActiveMCU + self.iLeakage)
        print(self.rNBIoT)
        self.rADC = self.V_supply / (self.iADC + self.iLeakage)
        print(self.rADC)
        self.rMCUI2CCoulomb = self.V_supply / (self.iMCUI2CCoulomb + self.iLeakage)
        print(self.rMCUI2CCoulomb)


        ## General Equivalent Resistance Values
        self.rSense = self.rADC
        self.rCycle = self.rNBIoT
        self.rOff = self.V_supply / self.iLeakage

        ## Duration Values
        # SHT Module
        self.tSHTI2C = 0.000325 # (calculated based on 2-byte command to initiate measurement + 6-byte read for temp and hum + protocol overhead and CPU overhead at 400kbps bus speed)
        self.tSHTMeasure = 0.0055 # (power up 1ms + sensing 4.5ms (see datasheet SHT30))
        self.tSHT = self.tSHTMeasure + self.tSHTI2C

        # NB-IoT Module
        self.tNBIoT = 7.89

        # MCU Active
        self.tADC = 0.00005
        self.tMCUI2CCoulomb = 0.00023
        self.tReadModel = 0.00504
        self.tRunModel = 0.0529992 
        self.tObservationMemoryOverhead = 0.001
        self.tAgent = self.tReadModel + self.tRunModel + self.tObservationMemoryOverhead

        # General
        self.tSense = self.tADC + self.tMCUI2CCoulomb
        # tCycle = tNBIoT

        self.recoverySteps = 0

        self.lifetime_days = 0
        self.days_per_lifetime = 1 # Increase c every day (1 episode)

        print("Loading RSSI and ADR data")
        if self.training == True:
            self.adrList = pd.read_csv('training_adr_simulation_' + self.trainingPayload + '_bytes_payload.csv', sep = ',') # Test with augmented combined data
        else:
            self.adrList = pd.read_csv('validation_adr_simulation_' + self.payloadOption + '_bytes_payload.csv', sep = ',') # Test with augmented combined data

        self.adrCurrentList = self.adrList.iloc[:, 7]
        self.adrCurrentList = self.adrCurrentList.to_numpy()
        self.adrCurrentList = self.adrCurrentList/1000 # From mA to A
        self.adrTimeList = self.adrList.iloc[:, 4]
        self.adrTimeList = self.adrTimeList.to_numpy()
        self.adrTimeList = self.adrTimeList /1000 # From ms to s
        self.adrPLList = self.adrList.iloc[:, -2]
        self.adrPLList = self.adrPLList.to_numpy()
        self.adrSFList = self.adrList.iloc[:, -1]
        self.adrSFList = self.adrSFList.to_numpy()
        self.adrDataSize = len(self.adrList)
        if self.adrSimulation == True:
            self.mu = self.adrTimeList[self.adrCount]
            self.lower = self.mu - self.spreading
            self.upper = self.mu + self.spreading
            self.iCycle = self.adrCurrentList[self.adrCount]
            self.rCycle = self.V_supply/self.iCycle
            self.pl = self.adrPLList[self.adrCount]
            self.sf = self.adrSFList[self.adrCount]
            self.adrCount = 1

        
        if self.shuffledData == False:
            self.final_inference_step = (5338 * self.nrOfInterpolationPoints) + 5339 # Based on interpolation calculations: ((size of original data - 1) * nrOfInterpolationPoints) + size of original data
        else:
            self.final_inference_step = 388800 / self.downSampleFactor # (Total shuffled solar validationdataset length / amount of downsampling

        # In case of training we use the harvestingCurrentList64 file, in case of inference we use the harvestingCurrentList68 file (deprecated)
        if self.training == True:
            print("Training is True")
            self.episode_length = self.episode_length # This is already defined at the beginning of the class, but we redefine it here in case we want to use a different episode length for training and inference (e.g. for training we might want to use 3 days episodes, for inference we want to use 1 day episodes to better evaluate the performance of the agent on shorter term)
            
            if self.shuffledData == True:
                if self.combined == True:
                    self.dataList = pd.read_csv('shuffled_training_dataset.csv', sep = ',')
                    self.dataList = self.dataList.iloc[:, 0]
                    self.dataList = self.dataList.to_numpy()
                    self.dataList = self.dataList[::self.downSampleFactor]
                    self.harvestingCurrentList = self.dataList 
                else:
                    print("Using solar-only shuffled data for training")
                    self.dataList = pd.read_csv('3_day_shuffled_solar_training_dataset.csv', sep = ',') 
                    self.dataList = self.dataList.iloc[:, 0]
                    self.dataList = self.dataList.to_numpy()
                    self.dataList = self.dataList[::self.downSampleFactor]
                    self.harvestingCurrentList = self.dataList 
            
            else:
                self.dataList = pd.read_csv('solar_and_teg_current_data.csv', sep = ',')
    
                self.solarCurrentList = self.dataList.iloc[:20000, 0] 
                self.solarCurrentList = self.solarCurrentList.to_numpy() 

                self.tegCurrentList = self.dataList.iloc[:20000, 1]  
                self.tegCurrentList = self.tegCurrentList.to_numpy() 
                if self.combined == True:
                    self.harvestingCurrentList = (self.solarCurrentList + self.tegCurrentList) * self.combinerEfficiency
                else:
                    self.harvestingCurrentList = self.solarCurrentList

                self.harvestingCurrentList = self.fill_with_interpolation(self.harvestingCurrentList, self.nrOfInterpolationPoints) # Original data is every 15 min and we want every 3 min final array size --> n + (n-1) * nrOfInterpolationPoints


        else:
            print("Training is False")
            self.episode_length = self.final_inference_step
            
            if self.shuffledData == True:
                if self.combined == True:
                    self.dataList = pd.read_csv('shuffled_inference_dataset.csv', sep = ',') 
                    self.dataList = self.dataList.iloc[:, 0]
                    self.dataList = self.dataList.to_numpy()
                    self.dataList = self.dataList[::self.downSampleFactor]
                    self.harvestingCurrentList = self.dataList 
                else:
                    print("Using solar-only shuffled data for inference")
                    self.dataList = pd.read_csv('shuffled_solar_validation_dataset.csv', sep = ',')
                    self.dataList = self.dataList.iloc[:, 0]
                    self.dataList = self.dataList.to_numpy()
                    self.dataList = self.dataList[::self.downSampleFactor]
                    self.harvestingCurrentList = self.dataList 
            
            else:
                self.dataList = pd.read_csv('solar_and_teg_current_data.csv', sep = ',') 
    
                self.solarCurrentList = self.dataList.iloc[20000:25339, 0]  
                self.solarCurrentList = self.solarCurrentList.to_numpy() 

                self.tegCurrentList = self.dataList.iloc[20000:25339, 1]  
                self.tegCurrentList = self.tegCurrentList.to_numpy() 
                if self.combined == True:
                    self.harvestingCurrentList = (self.solarCurrentList + self.tegCurrentList) * self.combinerEfficiency
                else:
                    self.harvestingCurrentList = self.solarCurrentList

                self.harvestingCurrentList = self.fill_with_interpolation(self.harvestingCurrentList, self.nrOfInterpolationPoints) # Original data is every 15 min and we want every 3 min final array size --> n + (n-1) * nrOfInterpolationPoints
            
        if self.training == True:
            self.final_step = self.final_training_step
        else:
            self.final_step = self.final_inference_step

        self.tCycleIncreaseThresh = math.floor(self.final_step/4)
        self.tCycleIncreaseCounter = 0
        
    def __init__(self):
        self.initialize()


        # We have 9 observations: Vt, It, tCycle, device state (on/off), time since failure, payload size, spreading factor, capacitor size, and energy. We also include the history of these 9 features for the past 10 steps, so the shape of the observation space is (9, history_size).
        # Because _get_obs normalizes ALL features to be strictly between 0.0 and 1.0,
        # Our observation space is simply a uniformly bounded box of [0.0, 1.0].
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(9, self.history_size), 
            dtype=np.float32  # Note: PyTorch & SB3 prefer float32 over default float64
        )

        # We have 2 actions, corresponding to "no execution" and "execution"
        self.action_space = spaces.Discrete(2)


    def write_to_csv(self, filename, data):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for num in data:
                writer.writerow([num])



    # Forward calculation for the next capacitor voltage
    def capacitor_voltage(self, ih, req, t, c, v0):
        self.v = ((ih*req*(1-math.exp((-t)/(req*c))))) + (v0*math.exp((-t)/(req*c)))
        self.v = min(self.v, self.Vt_max)
        return self.v

    # Shifts the elements of the array to the left and replaces the last element
    def shift_and_replace(self, arr, new_value):
        # Convert arr to a list so we can use append
        arr = list(arr)
        # Shift all values to the left by slicing from index 1 to the end
        arr = arr[1:]
        # Replace the last element with the new value
        arr.append(new_value)
        return arr



    def read_from_harvesting_current_list(self, n):
        desired_value = self.harvestingCurrentList[n]
        self.n += 1
        self.total_n += 1
        return desired_value

        
    def generate_normal_in_range(self, mu, sigma, lower, upper):
        while True:
            # Use the environment's internal seeded generator
            number = self.np_random.normal(mu, sigma)
            if lower <= number <= upper:
                # number = round(number, 2)
                return number


    def fill_with_interpolation(self, array, nrOfInterpolationPoints):
        #Create the final array with 4 interpolated values between each original number
        final_array = []

        for i in range(len(array) - 1):
            # Append the current original number
            final_array.append(array[i])
            # Add 4 interpolated values between this and the next original number
            interpolated_values = np.linspace(array[i], array[i + 1], nrOfInterpolationPoints + 2)[1:-1]
            final_array.extend(interpolated_values)

        # Append the last original number
        final_array.append(array[-1])

        # Convert to a numpy array for consistency
        final_array = np.array(final_array)

        return final_array
    
    def update_leakage_current(self, c):
        print("Updating leakage current based on capacitor size")
        self.iLeakage = 0.0000047442 * c + 0.0000049302 # Based on linear regression of the leakage current for different capacitor sizes from datasheet (see calculate_leakage_current.py)
        print("new leakage current: " + str(self.iLeakage) + " A for capacitor size: " + str(c) + " F")
        # We also update the equivalent resistance values that depend on the leakage current
        ## Equivalent Resistance Values Per Task
        self.rAgent = self.V_supply / (self.iAgent + self.iLeakage)
        self.rSHT = self.V_supply / (self.iSHT + self.iActiveMCU + self.iLeakage)
        self.rSleep = self.V_supply / (self.iSleep + self.iLeakage)
        self.rADC = self.V_supply / (self.iADC + self.iLeakage)
        self.rMCUI2CCoulomb = self.V_supply / (self.iMCUI2CCoulomb + self.iLeakage)
        ## General Equivalent Resistance Values
        self.rSense = self.rADC
        self.rOff = self.V_supply / self.iLeakage
        return self.iLeakage

    def determine_reward(self):

        # Calculate CURRENT Energy and Current Harvesting metrics
        E_current = 0.5 * self.c * (self.Vt ** 2)
        E_max = 0.5 * self.C_max_norm * (self.Vt_max ** 2)
        E_norm = min(max(E_current / E_max, 0.0), 1.0)
        
        I_norm = min(max(self.It / self.I_max_norm, 0.0), 1.0)

        if self.optimization_metric == 'jitter':
            # Positive reward still scales with survival streak (TSF)
            tsf_multiplier = 1.0 + (4.0 * (self.tsf / self.tsf_max))
            scaled_pos_reward = self.pos_reward * tsf_multiplier
 
            # Only punishes the agent heavily if it has energy AND the sun is shining!
            # If I_norm approaches 0 (dusk/night), the penalty gracefully drops to -0.5.
            scaled_neg_inaction = self.neg_inaction_reward * (1 + (4.0 * E_norm * I_norm))
        if self.optimization_metric == 'off_time':
            I_norm = min(max(self.It / self.I_max_norm, 0.0), 1.0)
            scaled_pos_reward = self.pos_reward * E_norm # We want to really encourage the agent to complete cycles when energy levels are high

            scaled_neg_inaction = self.neg_inaction_reward * E_norm * I_norm # We want to really encourage the agent to take action when the harvesting conditions are good, so we scale the negative inaction reward with the harvesting current (which is a proxy for good harvesting conditions). If the harvesting current is 0, there is no penalty for inaction, which makes sense because the agent cannot do anything about the fact that there is no energy to harvest.

        # -- REWARD SYSTEM
        if self.training == True:
            # If the agent does nothing while a task is feasible it gets a negative reward
            if self.b1 == 0 and self.b2 == 0 and self.feasible == 1:
                self.reward = scaled_neg_inaction
            # If the agent does nothing while a task is not feasible it gets a positive reward
            if self.b1 == 0 and self.b2 == 0 and self.feasible == 0:
                self.reward = self.pos_reward
            # If the agent successfully completes a task while a task is feasible it gets a positive reward
            if self.b1 == 0 and self.b2 == 1 and self.feasible == 1:
                self.reward = scaled_pos_reward
            # The agent should not be able to successfully complete a task when the task is not feasible
            if self.b1 == 0 and self.b2 == 1 and self.feasible == 0:
                print('impossible combination 010')
            # The agent should not fail to complete a task when the task is feasible
            if self.b1 == 1 and self.b2 == 0 and self.feasible == 1:
                print('impossible combination 101')
            # If the agent tries to perform a task (and fails) while the task is not feasible it gets a negative reward
            if self.b1 == 1 and self.b2 == 0 and self.feasible == 0:
                self.reward = self.neg_reward
                self.fail = 1
            # The agent cannot both fail and succeed at completing the task
            if self.b1 == 1 and self.b2 == 1:
                print('impossible combination 11')
        else:
            # If the agent does nothing while a task is feasible it gets a negative reward
            if self.b1 == 0 and self.b2 == 0 and self.feasible == 1:
                self.reward = scaled_neg_inaction
            # If the agent does nothing while a task is not feasible it gets a positive reward
            if self.b1 == 0 and self.b2 == 0 and self.feasible == 0:
                if self.optimization_metric == 'jitter':
                    self.reward = self.pos_reward - 0.1 # We add a small penalty to just doing nothing when the task is not feasible, to encourage the agent to at least try to complete a cycle sometimes (e.g. in case the agent has learned something useful that allows it to complete a cycle even when the task is not feasible, which could be the case if the agent has learned to predict when the harvesting conditions will be good and can therefore sometimes successfully complete a cycle even when the task is not feasible on average)
                else:   
                    self.reward = self.pos_reward + 0.1
            # If the agent successfully completes a task while a task is feasible it gets a positive reward
            if self.b1 == 0 and self.b2 == 1 and self.feasible == 1:
                self.reward = scaled_pos_reward
            # The agent should not be able to successfully complete a task when the task is not feasible
            if self.b1 == 0 and self.b2 == 1 and self.feasible == 0:
                print('impossible combination 010')
            # The agent should not fail to complete a task when the task is feasible
            if self.b1 == 1 and self.b2 == 0 and self.feasible == 1:
                print('impossible combination 101')
            # If the agent tries to perform a task (and fails) while the task is not feasible it gets a negative reward
            if self.b1 == 1 and self.b2 == 0 and self.feasible == 0:
                self.reward = self.neg_reward
            # The agent cannot both fail and succeed at completing the task
            if self.b1 == 1 and self.b2 == 1:
                print('impossible combination 11')
        return self.reward


    def off_simulation(self):
        self.loop_ran = False
        if self.already_in_off == False:
            self.first_iteration = True
            self.rewardBeforeOffState = self.determine_reward()
            self.tsf = 0 # Reset TSF to 0 at the moment of failure
        else:
            self.first_iteration = False

        # We reset the onCount because the device has failed.
        self.onCount = 0
        safety_counter = 0
            
        while self.Vt < self.V_to:
            self.loop_ran = True
            safety_counter += 1
            if safety_counter > 10000: # Safety break (approx 3.4 days of failure)
                print("Safety break triggered: Device stuck in OFF state for too long.")
                if self.training:
                    print("Safety break triggered: Device stuck in OFF state too long.")
                    self.done = True
                    break
                else:
                    # Allow it to stay dead during inference; do not set done = True
                    pass
            # Safely break if we run out of data during the OFF period
            # --- DATASET BOUNDARY CHECK ---
            if self.n >= len(self.harvestingCurrentList) - 1:
                if self.training:
                    # Training: Loop seamlessly and shuffle
                    # self.harvestingCurrentList = self.shuffle_data(self.harvestingCurrentList)
                    self.n = 0 
                else:
                    # Inference: End the simulation immediately
                    self.done = True
                    self.truncated = False
                    break 
            # ------------------------------

            # Episode Boundary Check (72 Hours of Real Time)
            if self.training and self.sim_steps_this_episode >= self.episode_length:
                self.done = False
                self.truncated = True
                break # The 72-hour day ended while the device was dead.

            # We are advancing time, so increment our episode physical clock
            self.sim_steps_this_episode += 1

            if self.first_iteration == False:
                self.Vt = self.capacitor_voltage(self.It, self.rOff, self.ti, self.c, self.Vt)  
                self.feasible = 0
            self.first_iteration = False
            if self.training == True:
                self.reward = 0
            else:
                self.reward = 6
            self.cummulativeReward = self.cummulativeReward + self.reward
            self.sumOffReward -= 1

            self.capVoltageList.append(self.Vt)
            self.rewardList.append(self.reward)
            self.cummulativeRewardList.append(self.cummulativeReward)
            self.tCycleList.append(self.tCycle)
            self.feasibilityList.append(self.feasible)
            self.deviceStateList.append(not self.off)
            self.restartList.append(self.restartWillHappen)
            self.failList.append(self.fail)
            self.plList.append(self.pl)
            self.sfList.append(self.sf)
            self.tsfList.append(self.tsf)

            self.It = self.read_from_harvesting_current_list(self.n)
            self.finalHarvestingCurrentList.append(self.It)
            

        self.already_in_off = False
        if self.multipleMu == True and self.training == False:
            self.tCycleIncreaseCounter += 1
            if self.tCycleIncreaseCounter > self.tCycleIncreaseThresh:
                self.tCycleIncreaseCounter = 0
                self.mu = self.mu + 5
                self.lower = self.mu - self.spreading
                self.upper = self.mu + self.spreading
        
        self.off_simulation_happened = True 
        self.restartWillHappen = True


    def _get_obs(self):
        # 1. Voltage: 0V to 5.5V -> 0.0 to 1.0
        V_norm = np.clip(np.array(self.V) / self.V_max_norm, 0, 1)
        
        # 2. Current: 0A to I_max -> 0.0 to 1.0
        I_norm = np.clip(np.array(self.I) / self.I_max_norm, 0, 1)
        
        # 3. Cycle Time: 0s to 21s -> 0.0 to 1.0
        CT_norm = np.clip(np.array(self.CT) / self.CT_max_norm, 0, 1)
        
        # 4. Device State: 0 (On) or 1 (Off)
        OFF_norm = np.array(self.OFF, dtype=np.float32) # Already binary
        
        # 5. Survival Streak: 0 to 1000 -> 0.0 to 1.0
        TSF_norm = np.clip(np.array(self.TSF) / self.TSF_max_norm, 0, 1)
        
        # 6. Payload: 20 to 255 -> 0.0 to 1.0 (Min-Max)
        PL_norm = np.clip((np.array(self.PL) - self.PL_min_norm) / (self.PL_max_norm - self.PL_min_norm), 0, 1)
        
        # 7. Spreading Factor: 7 to 12 -> 0.0 to 1.0 (Min-Max)
        SF_norm = np.clip((np.array(self.SF) - self.SF_min_norm) / (self.SF_max_norm - self.SF_min_norm), 0, 1)
        
        # 8. Cap Size: 0.5F to 10F -> 0.0 to 1.0 (Min-Max)
        C_norm = np.clip((np.array(self.C) - self.C_min_norm) / (self.C_max_norm - self.C_min_norm), 0, 1)

        # 9. Stored Energy (E = 0.5 * C * V^2)
        E_current = 0.5 * np.array(self.C) * (np.array(self.V) ** 2)
        E_max_possible = 0.5 * self.C_max_norm * (self.V_max_norm ** 2) # Absolute physical max
        E_norm = np.clip(E_current / E_max_possible, 0.0, 1.0)

        # Final check: Ensure we stack all 9 features correctly!
        obs = np.stack([V_norm, I_norm, CT_norm, OFF_norm, TSF_norm, PL_norm, SF_norm, C_norm, E_norm]).astype(np.float32)
        return obs



    def reset(self, seed=None, options=None):
        # 1. This initializes self.np_random using the seed
        super().reset(seed=seed)

        self.done = False
        self.truncated = False

        self.step_counter = 0
        self.sim_steps_this_episode = 0 

        # --- RANDOMIZE CAPACITOR SIZE FOR TRAINING ---
        if self.training:
            # np_random.integers is the seeded version of random.randint
            self.c = self.np_random.integers(1, 21) / 2
            self.iLeakage = self.update_leakage_current(self.c)
            print(f"--- RESET: New training episode. Cap size: {self.c:.2f}F ---")
        else:
            # During inference, use the strictly defined static size
            self.c = self.cap_size

        print("in reset")
        print("Current cycle time is " + str(self.tCycle))

        if self.n >= len(self.harvestingCurrentList):  
            print("in re-init case")
            self.n = 0

        # Read the current for step 0
        self.It = self.read_from_harvesting_current_list(self.n)
        # --- WIPE DEVICE STATE (Physics & RL Baseline) ---
        # We DO NOT use saved values here. We start the new episode with a baseline state.
        self.Vt = self.V_to
        self.tsf = 0
        self.off = False 
        
        # Completely flush and rebuild the history arrays with the baseline values
        self.V = [self.V_to] * self.history_size
        self.I = [self.It] * self.history_size
        self.CT = [self.tCycle_init] * self.history_size
        self.OFF = [0] * self.history_size # 0 means ON in your normal operation
        self.TSF = [0] * self.history_size # Reset the survival streak to 0!
        self.SF = [self.sf] * self.history_size 
        self.PL = [self.pl] * self.history_size 
        self.C = [self.c] * self.history_size # Match the newly rolled capacitor
        observation = self._get_obs()
        return observation, {}
    

    def save_results(self):
    
        # 1. Determine the path logic
        training_suffix = str(self.training_parameter_suffix)
        inference_suffix = self.inference_parameter_suffix
        
        base_dir = "RLModelData/experiments_combined" if self.combined else "RLModelData/experiments_solar"
        sub_folder = "training_experiments" if self.training else "inference_experiments"
        
        # 2. Ensure directories exist
        target_path = os.path.join(base_dir, sub_folder)
        os.makedirs(target_path, exist_ok=True)
        
        # 3. Define the filename suffix
        # Inference uses the combined naming convention from your code
        suffix = f"{inference_suffix}__{training_suffix}" if not self.training else training_suffix
        
        print(f"--- Saving results to {target_path} ---")

        # 4. Full List of CSV Exports
        data_map = {
            "voltageList": self.capVoltageList,
            "ewmaList": self.ewmaCapVoltageList,
            "rewardList": self.rewardList,
            "harvestingCurrentList": self.finalHarvestingCurrentList,
            "cummulativeRewardList": self.cummulativeRewardList,
            "tCycleList": self.tCycleList,
            "feasibilityList": self.feasibilityList,
            "deviceStateList": self.deviceStateList,
            "restartList": self.restartList,
            "failList": self.failList,
            "payloadList": self.plList,
            "sfList": self.sfList,
            "tsfList": self.tsfList
        }

        for filename, data_list in data_map.items():
            # Only write if the list actually has data to avoid empty files
            if data_list:
                full_file_path = os.path.join(target_path, f"{filename}{suffix}.csv")
                self.write_to_csv(full_file_path, data_list)
            else:
                print(f"Skipping {filename}: List is empty.")

        print("--- All data successfully saved ---")


    def step(self, action):

        if self.training == False:
            if action == 1:
                if self.randomCycleTime == False:
                    self.tCycle = self.generate_normal_in_range(self.mu, self.sigma, self.lower, self.upper)
                else:
                    self.tCycle = self.np_random.uniform(self.tCycleMin, self.tCycleMax) 

        if self.off == False:  # We first check if the device is currently in the on state
                               # If it is, then we follow the usual logic for granting the rewards and performing cycles etc. 
            if action == 0:
                if self.Vt > self.Vt_min:
                    self.Vt_check = self.capacitor_voltage(self.It, self.rSHT, self.tSHT, self.c, self.Vt)
                    self.Vt_check = self.capacitor_voltage(self.It, self.rCycle, self.tCycle, self.c, self.Vt_check)
                    if self.Vt_check <= self.Vt_min:
                        self.feasible = 0
                    else:
                        self.Vt_check = self.capacitor_voltage(self.It, self.rSleep, (self.ti - self.tAgent - self.tSense - self.tSHT -self.tCycle), self.c, self.Vt_check)
                        if self.Vt_check <= self.Vt_min:
                            self.feasible = 0
                        else:
                            self.Vt_check = self.capacitor_voltage(self.It, self.rADC, self.tADC, self.c, self.Vt_check)
                            self.Vt_check = self.capacitor_voltage(self.It, self.rMCUI2CCoulomb, self.tMCUI2CCoulomb, self.c, self.Vt_check)
                            self.Vt_check = self.capacitor_voltage(self.It, self.rAgent, self.tAgent, self.c, self.Vt_check)
                            if self.Vt_check <= self.Vt_min:
                                self.feasible = 0
                            else:
                                self.feasible = 1
                    self.Vt = self.capacitor_voltage(self.It, self.rSHT, self.tSHT, self.c, self.Vt)
                    self.Vt = self.capacitor_voltage(self.It, self.rSleep, (self.ti - self.tSHT - self.tAgent - self.tSense), self.c, self.Vt)
                    if self.Vt > self.Vt_min:
                        self.Vt = self.capacitor_voltage(self.It, self.rADC, self.tADC, self.c, self.Vt)
                        self.Vt = self.capacitor_voltage(self.It, self.rMCUI2CCoulomb, self.tMCUI2CCoulomb, self.c, self.Vt)
                        self.Vt = self.capacitor_voltage(self.It, self.rAgent, self.tAgent, self.c, self.Vt)
                    else:
                        self.Vt = self.capacitor_voltage(self.It, self.rOff, (self.tAgent + self.tSense), self.c, self.Vt)
                        self.off = True
                        self.off_simulation()
                else:
                    self.Vt = self.capacitor_voltage(self.It, self.rOff, self.ti, self.c, self.Vt)
                    self.off = True
                    self.feasible = 0
                    self.off_simulation()

            # If cycle then do both with the different time intervals and r and Vt values
            else:
                self.Vt_check = self.capacitor_voltage(self.It, self.rSHT, self.tSHT, self.c, self.Vt)
                self.Vt_check = self.capacitor_voltage(self.It, self.rCycle, self.tCycle, self.c, self.Vt_check)
                if self.Vt_check <= self.Vt_min:
                    self.feasible = 0
                else:
                    self.Vt_check = self.capacitor_voltage(self.It, self.rSleep, (self.ti - self.tAgent - self.tSense - self.tSHT - self.tCycle), self.c, self.Vt_check)
                    if self.Vt_check <= self.Vt_min:
                        self.feasible = 0
                    else:
                        self.Vt_check = self.capacitor_voltage(self.It, self.rADC, self.tADC, self.c, self.Vt_check)
                        self.Vt_check = self.capacitor_voltage(self.It, self.rMCUI2CCoulomb, self.tMCUI2CCoulomb, self.c, self.Vt_check)
                        self.Vt_check = self.capacitor_voltage(self.It, self.rAgent, self.tAgent, self.c, self.Vt_check)
                        if self.Vt_check <= self.Vt_min:
                            self.feasible = 0
                        else:
                            self.feasible = 1
                self.Vt = self.capacitor_voltage(self.It, self.rSHT, self.tSHT, self.c, self.Vt)
                self.Vt = self.capacitor_voltage(self.It, self.rCycle, self.tCycle, self.c, self.Vt)
                

                if self.Vt <= self.Vt_min: # Cycle has failed
                    self.Vt = self.Vt_min # If the voltage after the cycle is below the threshold, then we clip it at the threshold since it will just turn off
                    self.b1 = 1  # If the task fails, bit 1 in flash will be high
                    self.Vt = self.capacitor_voltage(self.It, self.rOff, (self.ti - self.tSHT - self.tCycle), self.c, self.Vt) #Technically, if the cycle is cut early beacuase it goes below V_min, then this time estimation is underestimated, but if we also keep it like this for the baseline, then the comparison is still fair
                    self.off = True
                    self.off_simulation()

                else:
                    # self.b2 = 1  # If the task succeeds, bit 2 in flash will be high
                    self.Vt = self.capacitor_voltage(self.It, self.rSleep, (self.ti - self.tSHT - self.tCycle - self.tAgent - self.tSense), self.c, self.Vt)
                    if self.Vt <= self.Vt_min:
                        self.b1 = 1  # If the task fails, bit 1 in flash will be high
                        self.Vt = self.capacitor_voltage(self.It, self.rOff, (self.tAgent + self.tSense), self.c, self.Vt) # If we are below V_min then we should not perform the agent running anymore and just turn off
                        self.off = True
                        self.off_simulation()
                    else:
                        self.Vt = self.capacitor_voltage(self.It, self.rADC, self.tADC, self.c, self.Vt)
                        self.Vt = self.capacitor_voltage(self.It, self.rMCUI2CCoulomb, self.tMCUI2CCoulomb, self.c, self.Vt)
                        self.Vt = self.capacitor_voltage(self.It, self.rAgent, self.tAgent, self.c, self.Vt)
                        if self.Vt <= self.Vt_min:
                            self.b1 = 1  # If the task fails, bit 1 in flash will be high
                        else:
                            self.b2 = 1  # If the full interval with task succeeds, bit 2 in flash will be high
                            if self.adrSimulation == True:
                                self.mu = self.adrTimeList[self.adrCount]
                                self.lower = self.mu - self.spreading
                                self.upper = self.mu + self.spreading
                                self.iCycle = self.adrCurrentList[self.adrCount]
                                self.rCycle = self.V_supply/self.iCycle
                                self.pl = self.adrPLList[self.adrCount]
                                self.sf = self.adrSFList[self.adrCount]
                                if self.adrCount < (self.adrDataSize - 1):
                                    self.adrCount += 1
                                else:
                                    self.adrCount = 0

        else:   # If the device is in the off state, it should do nothing if the voltage is under V_to
                # and it should restart the device if the voltage is higher or equal to V_to (and perform an action if needed)
            if self.Vt < self.V_to:
                self.already_in_off = True
                print("Triggered already off state")
                self.off_simulation()              
            else:
                self.off = False
                if action == 0:
                    if self.Vt > self.Vt_min:
                        self.Vt_check = self.capacitor_voltage(self.It, self.rSHT, self.tSHT, self.c, self.Vt)
                        self.Vt_check = self.capacitor_voltage(self.It, self.rCycle, self.tCycle, self.c, self.Vt_check)
                        if self.Vt_check <= self.Vt_min:
                            self.feasible = 0
                        else:
                            self.Vt_check = self.capacitor_voltage(self.It, self.rSleep, (self.ti - self.tAgent - self.tSense - self.tCycle - self.tSHT), self.c, self.Vt_check)
                            if self.Vt_check <= self.Vt_min:
                                self.feasible = 0
                            else:
                                self.Vt_check = self.capacitor_voltage(self.It, self.rADC, self.tADC, self.c, self.Vt_check)
                                self.Vt_check = self.capacitor_voltage(self.It, self.rMCUI2CCoulomb, self.tMCUI2CCoulomb, self.c, self.Vt_check)
                                self.Vt_check = self.capacitor_voltage(self.It, self.rAgent, self.tAgent, self.c, self.Vt_check)
                                if self.Vt_check <= self.Vt_min:
                                    self.feasible = 0
                                else:
                                    self.feasible = 1
                        self.Vt = self.capacitor_voltage(self.It, self.rSHT, self.tSHT, self.c, self.Vt)
                        self.Vt = self.capacitor_voltage(self.It, self.rSleep, (self.ti - self.tSHT - self.tAgent - self.tSense), self.c, self.Vt)
                        if self.Vt > self.Vt_min:
                            self.Vt = self.capacitor_voltage(self.It, self.rADC, self.tADC, self.c, self.Vt)
                            self.Vt = self.capacitor_voltage(self.It, self.rMCUI2CCoulomb, self.tMCUI2CCoulomb, self.c, self.Vt)
                            self.Vt = self.capacitor_voltage(self.It, self.rAgent, self.tAgent, self.c, self.Vt)
                        else:
                            self.Vt = self.capacitor_voltage(self.It, self.rOff, (self.tAgent + self.tSense), self.c, self.Vt)
                            self.off = True
                            self.off_simulation()
                    else:
                        self.feasible = 0
                        self.Vt = self.capacitor_voltage(self.It, self.rOff, self.ti, self.c, self.Vt)
                        self.off = True
                        self.off_simulation()

                # If cycle then do both with the different time intervals and r and Vt values
                else:
                    self.Vt_check = self.capacitor_voltage(self.It, self.rSHT, self.tSHT, self.c, self.Vt)
                    self.Vt_check = self.capacitor_voltage(self.It, self.rCycle, self.tCycle, self.c, self.Vt_check)
                    if self.Vt_check <= self.Vt_min:
                        self.feasible = 0
                    else:
                        self.Vt_check = self.capacitor_voltage(self.It, self.rSleep, (self.ti - self.tAgent - self.tSense - self.tCycle - self.tSHT), self.c, self.Vt_check)
                        if self.Vt_check <= self.Vt_min:
                            self.feasible = 0
                        else:
                            self.Vt_check = self.capacitor_voltage(self.It, self.rADC, self.tADC, self.c, self.Vt_check)
                            self.Vt_check = self.capacitor_voltage(self.It, self.rMCUI2CCoulomb, self.tMCUI2CCoulomb, self.c, self.Vt_check)
                            self.Vt_check = self.capacitor_voltage(self.It, self.rAgent, self.tAgent, self.c, self.Vt_check)
                            if self.Vt_check <= self.Vt_min:
                                self.feasible = 0
                            else:
                                self.feasible = 1
                    self.Vt = self.capacitor_voltage(self.It, self.rSHT, self.tSHT, self.c, self.Vt)
                    self.Vt = self.capacitor_voltage(self.It, self.rCycle, self.tCycle, self.c, self.Vt)
                    

                    if self.Vt <= self.Vt_min:
                        self.Vt = self.Vt_min # If the voltage after the cycle is below the threshold, then we clip it at the threshold since it will just turn off
                        self.b1 = 1  # If the task fails, bit 1 in flash will be high
                        self.Vt = self.capacitor_voltage(self.It, self.rOff, (self.ti - self.tCycle - self.tSHT), self.c, self.Vt) #Technically, if the cycle is cut early beacuase it goes below V_min, then this time estimation is underestimated, but if we also keep it like this for the baseline, then the comparison is still fair
                        self.off = True
                        self.off_simulation()
                    else:
                        self.Vt = self.capacitor_voltage(self.It, self.rSleep, (self.ti - self.tCycle - self.tAgent - self.tSense - self.tSHT), self.c, self.Vt)
                        if self.Vt <= self.Vt_min:
                            self.b1 = 1  # If the task fails, bit 1 in flash will be high
                            self.Vt = self.capacitor_voltage(self.It, self.rOff, (self.tAgent + self.tSense), self.c, self.Vt) # If we are below V_min then we should not perform the agent running anymore and just turn off
                            self.off = True
                            self.off_simulation()
                        else:
                            self.Vt = self.capacitor_voltage(self.It, self.rADC, self.tADC, self.c, self.Vt)
                            self.Vt = self.capacitor_voltage(self.It, self.rMCUI2CCoulomb, self.tMCUI2CCoulomb, self.c, self.Vt)
                            self.Vt = self.capacitor_voltage(self.It, self.rAgent, self.tAgent, self.c, self.Vt)
                            if self.Vt <= self.Vt_min:
                                self.b1 = 1  # If the task fails, bit 1 in flash will be high
                            else:
                                self.b2 = 1  # If the full interval with task succeeds, bit 2 in flash will be high
                                if self.adrSimulation == True:
                                    self.mu = self.adrTimeList[self.adrCount]
                                    self.lower = self.mu - self.spreading
                                    self.upper = self.mu + self.spreading
                                    self.iCycle = self.adrCurrentList[self.adrCount]
                                    self.rCycle = self.V_supply/self.iCycle
                                    self.pl = self.adrPLList[self.adrCount]
                                    self.sf = self.adrSFList[self.adrCount]
                                    if self.adrCount < (self.adrDataSize -1):
                                        self.adrCount += 1
                                    else:
                                        self.adrCount = 0


        self.determine_reward()


        # 1. Handle the Reward and Time Updates based on state
        if self.off_simulation_happened:
            # Device DIED and off_simulation already fast-forwarded time.
            # We ONLY calculate the penalty and assign it to the agent.
            if self.training == True:
                scaled_sum_off_reward = max(self.sumOffReward * 0.1, -1000.0)
                self.reward = self.rewardBeforeOffState + scaled_sum_off_reward
            else:
                self.reward = self.neg_reward

            # Only overwrite if the while loop actually ran and appended data
            if getattr(self, 'loop_ran', False):
                if len(self.rewardList) > 0:
                    self.rewardList[-1] = self.reward

        if not self.off_simulation_happened or not getattr(self, 'loop_ran', False):
            # Device STAYED ON. We process exactly 1 normal timestep.
            self.cummulativeReward = self.cummulativeReward + self.reward
            self.capVoltageList.append(self.Vt)
            self.rewardList.append(self.reward)
            self.cummulativeRewardList.append(self.cummulativeReward)
            self.tCycleList.append(self.tCycle)
            self.feasibilityList.append(self.feasible)
            self.deviceStateList.append(not self.off)
            self.restartList.append(self.restartWillHappen)
            self.failList.append(self.fail)
            self.plList.append(self.pl)
            self.sfList.append(self.sf)
            self.tsfList.append(self.tsf)

            # Advance dataset clock
            self.It = self.read_from_harvesting_current_list(self.n)
            self.finalHarvestingCurrentList.append(self.It)

            # Advance episode clocks
            self.sim_steps_this_episode += 1
            self.global_counter += 1
            
            # --- TSF Update Logic ---
            # Increase the survival streak because the device survived this step
            self.tsf = min(self.tsf + 1, self.tsf_max)

        # Update cycle time for next step observation
        if self.training == True and action == 1:
            if self.randomCycleTime == False:
                self.tCycle = self.generate_normal_in_range(self.mu, self.sigma, self.lower, self.upper)
            else:
                self.tCycle = random.uniform(self.tCycleMin, self.tCycleMax)

        # Reset flags for the next step
        self.b1 = 0
        self.b2 = 0
        self.off_simulation_happened = False
        self.sumOffReward = 0        
        self.restartWillHappen = False
        self.fail = 0
        self.feasible = 0

        # 1. Check if the physical dataset ran out (Loop the data)
        if self.n >= len(self.harvestingCurrentList) - 1:
            if self.training:
                self.n = 0
            else:
                print("Terminating because inference reached the end of the dataset at n:", self.n)
                self.done = True
                self.truncated = False
        
        
        # 2. Check if the 24-hour episode is over
        if self.training and self.sim_steps_this_episode >= self.episode_length:
            self.done = False
            self.truncated = True
            
            self.lifetime_days += 1


        # Shift history arrays so the final step is recorded
        self.V = self.shift_and_replace(self.V, self.Vt) 
        self.noisy_tCycle = self.generate_normal_in_range(self.tCycle, self.sigma, self.tCycle - self.spreading, self.tCycle + self.spreading) 
        self.CT = self.shift_and_replace(self.CT, self.noisy_tCycle)
        self.TSF = self.shift_and_replace(self.TSF, self.tsf)
        self.I = self.shift_and_replace(self.I, self.It)
        self.OFF = self.shift_and_replace(self.OFF, self.off)
        self.SF = self.shift_and_replace(self.SF, self.sf)
        self.PL = self.shift_and_replace(self.PL, self.pl)
        self.C = self.shift_and_replace(self.C, self.c)

        observation = self._get_obs()

        return observation, self.reward, self.done, self.truncated, {}

if __name__ == "__main__":
    world = FullyTaskedBatteryLessWorldEnv()

    world.reset()
    for i in range (50):
        for i in range(7):
            world.step(0)
        world.step(1)
        for i in range(5):
            world.step(0)
