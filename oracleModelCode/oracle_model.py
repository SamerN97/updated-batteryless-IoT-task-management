import numpy as np
import csv
import math as math
import pandas as pd
from scipy.optimize import fsolve


adrSimulation = True
multipleMu = False
day_test = False
combined = False 
shuffledData = True
payloadSize = 50
payloadOption = "daily_random_20_255"
c = 10 # Cap size
capSizeStr = str(c)


additional_text = "ADR_" + payloadOption + "_BYTES_" + capSizeStr + "FARAD"

ti = 30 #Interval time
downSampleFactor = 3 # Change based on ti and dataset sample frequency (for shuffled dataset frequency is every 10 sec) --> e.g. if we want to sample every 30 sec, we only keep every 3 samples
nrOfInterpolationPoints = 29 # Needs to be adapted to INTERVAL (when not using shuffledData)
if shuffledData == False:   
    max_steps = (5338 * nrOfInterpolationPoints) + 5339
else:
    max_steps = int(388800 / downSampleFactor) # total shuffled solar validation dataset  / amount of downsampling
if adrSimulation == False:
    mu = 5 # We use mu to decide tCycle
    sigma = 0.4
    spreading = 1 # This decides where the output of the normal distribution gets clipped
    lower = mu - spreading
    upper = mu + spreading
else:
    mu = 0.247 # We use mu to decide tCycle
    sigma = 0.00000001
    spreading = 0.000001 # This decides where the output of the normal distribution gets clipped
    lower = mu - spreading
    upper = mu + spreading

adrCount = 0

tCycleIncreaseThresh = math.floor(max_steps/4)
tCycleIncreaseCounter = 0

harvestingCurrentList = list()
finalHarvestingCurrentList = list()
actionList = list()
successList = list()
stepCounter = 0
capVoltageList = list()
deviceStateList = list()
tCycleList = list()
iCycleList = list()
I = []
It = 0
V = []
V_to = 2.3
Vt_max = 5.5
V_supply = 3.3
Vt_min = 1.8
Vt_check = 0
feasible = 0
tCycle = 0

action = 0
action_ctr = 0
ctr = 0
success = 0
success_ctr = 0
n = 0  
off = False
tCycle = mu 
Vt = 2.7
V_0 = 0
combinerEfficiency = 0.88 
fail = False
fail_ctr = 0
calc_ctr = 0


## General Current Values
iCycle = 0
iSense = 0
iLeakage = 0.0000047442 * c + 0.0000049302 # Based on linear regression of the leakage current for different capacitor sizes from datasheet (see calculate_leakage_current.py)

## Current Consumption Parameters Per Subtask
# SHT30
iSHT = 0.0006

# MCU Sleep
iSleep = 0.00000065 

# NB-IoT Module
iNBIoT = 0.02065

# MCU Active
iADC = 0.000311
iActiveMCU = 0.000091
iMCUI2CCoulomb = iActiveMCU


## Equivalent Resistance Values Per Task
rSHT = V_supply / (iSHT + iActiveMCU + iLeakage)
print(rSHT)
rSleep = V_supply / (iSleep + iLeakage)
print(rSleep)
rNBIoT = V_supply / (iNBIoT + iActiveMCU + iLeakage)
print(rNBIoT)
rADC = V_supply / (iADC + iLeakage)
print(rADC)
rMCUI2CCoulomb = V_supply / (iMCUI2CCoulomb + iLeakage)
print(rMCUI2CCoulomb)


## General Equivalent Resistance Values
rSense = rADC
rCycle = rNBIoT
rOff = V_supply / iLeakage

## Duration Values
# SHT Module
tSHTI2C = 0.000325 # (calculated based on 2-byte command to initiate measurement + 6-byte read for temp and hum + protocol overhead and CPU overhead at 400kbps bus speed)
tSHTMeasure = 0.0055 # (power up 1ms + sensing 4.5ms (see datasheet SHT30))
tSHT = tSHTMeasure + tSHTI2C

# NB-IoT Module
tNBIoT = 7.89

# MCU Active
tADC = 0.00005
tMCUI2CCoulomb = 0.00023

# General
tSense = tADC + tMCUI2CCoulomb
# tCycle = tNBIoT


if combined == True:
    print("Using shuffled dataset")
    dataList = pd.read_csv('shuffled_inference_dataset.csv', sep = ',') # Test with augmented combined data
    dataList = dataList.iloc[:, 0]
    dataList = dataList.to_numpy()
    dataList = dataList[::downSampleFactor]
    harvestingCurrentList = dataList # Test with augmented combined current data
else:
    print("Using shuffled solar dataset")
    dataList = pd.read_csv('shuffled_solar_validation_dataset.csv', sep = ',') # Test with augmented solar current data
    dataList = dataList.iloc[:, 0]
    dataList = dataList.to_numpy()
    dataList = dataList[::downSampleFactor]
    harvestingCurrentList = dataList # Test with augmented solar current data

print("Loading RSSI and ADR data")
adrList = pd.read_csv('validation_adr_simulation_' + payloadOption + '_bytes_payload.csv', sep = ',') # 
adrCurrentList = adrList.iloc[:, 7]
adrCurrentList = adrCurrentList.to_numpy()
adrCurrentList = adrCurrentList/1000 # From mA to A
adrTimeList = adrList.iloc[:, 4]
adrTimeList = adrTimeList.to_numpy()
adrTimeList = adrTimeList /1000 # From ms to s
if adrSimulation == True:
    mu = adrTimeList[adrCount]
    lower = mu - spreading
    upper = mu + spreading
    iCycle = adrCurrentList[adrCount]
    rCycle = V_supply/iCycle
    adrCount += 1

# Forward calculation for the next capacitor voltage
def capacitor_voltage(ih, req, t, c, v0):
    v = ((ih*req*(1-math.exp((-t)/(req*c))))) + (v0*math.exp((-t)/(req*c)))
    v = min(v, Vt_max)
    return v

def read_from_harvesting_current_list(n):
    desired_value = harvestingCurrentList[n]
    return desired_value


def generate_normal_in_range(mu, sigma, lower, upper):
    while True:
        number = np.random.normal(mu, sigma)
        if lower <= number <= upper:
            return number


def write_to_csv(filename, data):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for num in data:
            writer.writerow([num])

def fill_with_interpolation(array, nrOfInterpolationPoints):
    # Create the final array with 4 interpolated values between each original number
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

if shuffledData == False:
    harvestingCurrentList = fill_with_interpolation(harvestingCurrentList, nrOfInterpolationPoints) # Original data is every 15 min and we want every 3 min (final array size with interpolation = array size + (array size - 1) * nrOfInterpolationPoints
    if day_test == True:
        harvestingCurrentList = harvestingCurrentList[162700:165050]
        max_steps = len(harvestingCurrentList)


print(max_steps)
for i in range(max_steps): 

    It = read_from_harvesting_current_list(n)
    n += 1
    if off == False:
        if Vt <= Vt_min: # Should be smaller or equal 
            Vt = capacitor_voltage(It, rOff, ti, c, Vt)
            off = True
        else:
            Vt_check = capacitor_voltage(It, rSHT, tSHT, c, Vt)
            Vt_check = capacitor_voltage(It, rCycle, tCycle, c, Vt_check)
            if Vt_check <= Vt_min:
                feasible = 0
            else:
                Vt_check = capacitor_voltage(It, rSleep, (ti - tSense - tSHT - tCycle), c, Vt_check)
                if Vt_check <= Vt_min:
                    feasible = 0
                else:
                    Vt_check = capacitor_voltage(It, rADC, tADC, c, Vt_check)
                    Vt_check = capacitor_voltage(It, rMCUI2CCoulomb, tMCUI2CCoulomb, c, Vt_check)
                    if Vt_check <= Vt_min:
                        feasible = 0
                    else:
                        feasible = 1

            if feasible == 0: 
                Vt = capacitor_voltage(It, rSHT, tSHT, c, Vt)
                Vt = capacitor_voltage(It, rSleep, (ti - tSHT - tSense), c, Vt)
                Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                Vt = capacitor_voltage(It, rMCUI2CCoulomb, tMCUI2CCoulomb, c, Vt)
            else:
                Vt = capacitor_voltage(It, rSHT, tSHT, c, Vt)
                Vt = capacitor_voltage(It, rCycle, tCycle, c, Vt)
                if Vt > Vt_min:
                    Vt = capacitor_voltage(It, rSleep, (ti - tSHT - tCycle - tSense), c, Vt)
                    if Vt > Vt_min:
                        Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                        Vt = capacitor_voltage(It, rMCUI2CCoulomb, tMCUI2CCoulomb, c, Vt)
                        if Vt > Vt_min:
                            success = 1
                            success_ctr += 1
                            if adrSimulation == True:
                                mu = adrTimeList[adrCount]
                                lower = mu - spreading
                                upper = mu + spreading
                                iCycle = adrCurrentList[adrCount]
                                rCycle = V_supply/iCycle
                                if adrCount < 151000:
                                    adrCount += 1
                                else:
                                    adrCount = 0
                    else:
                        Vt = capacitor_voltage(It, rOff, tSense, c, Vt)
                        off = True
                        fail = True
                        fail_ctr += 1
                        cum_fail_ctr += 1
                else:
                    Vt = Vt_min
                    Vt = capacitor_voltage(It, rOff, ti - tSHT - tCycle, c, Vt)
                    off = True
                    fail = True
                    fail_ctr += 1
                    cum_fail_ctr += 1
                action = 1
                action_ctr += 1

    else:
        if Vt < V_to:
            Vt = capacitor_voltage(It, rOff, ti, c, Vt)
        else:
            off = False

            Vt_check = capacitor_voltage(It, rSHT, tSHT, c, Vt)
            Vt_check = capacitor_voltage(It, rCycle, tCycle, c, Vt_check)
            if Vt_check <= Vt_min:
                feasible = 0
            else:
                Vt_check = capacitor_voltage(It, rSleep, (ti - tSense - tSHT - tCycle), c, Vt_check)
                if Vt_check <= Vt_min:
                    feasible = 0
                else:
                    Vt_check = capacitor_voltage(It, rADC, tADC, c, Vt_check)
                    Vt_check = capacitor_voltage(It, rMCUI2CCoulomb, tMCUI2CCoulomb, c, Vt_check)
                    if Vt_check <= Vt_min:
                        feasible = 0
                    else:
                        feasible = 1

            if feasible == 1:
                Vt = capacitor_voltage(It, rSHT, tSHT, c, Vt)
                Vt = capacitor_voltage(It, rCycle, tCycle, c, Vt)
                if Vt > Vt_min:
                    Vt = capacitor_voltage(It, rSleep, (ti - tSHT - tCycle - tSense), c, Vt)
                    if Vt > Vt_min:
                        Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                        Vt = capacitor_voltage(It, rMCUI2CCoulomb, tMCUI2CCoulomb, c, Vt)
                        if Vt > Vt_min:
                            success = 1
                            success_ctr += 1
                            if adrSimulation == True:
                                mu = adrTimeList[adrCount]
                                lower = mu - spreading
                                upper = mu + spreading
                                iCycle = adrCurrentList[adrCount]
                                rCycle = V_supply/iCycle
                                if adrCount < 151000:
                                    adrCount += 1
                                else:
                                    adrCount = 0

                    else:
                        Vt = capacitor_voltage(It, rOff, tSense, c, Vt)
                        off = True
                        fail = True
                        fail_ctr += 1
                        cum_fail_ctr += 1
                else:
                    Vt = Vt_min
                    Vt = capacitor_voltage(It, rOff, ti - tSHT - tCycle, c, Vt)
                    off = True
                    fail = True
                    fail_ctr += 1
                    cum_fail_ctr += 1
                action = 1
                action_ctr += 1
            else:
                Vt = capacitor_voltage(It, rSHT, tSHT, c, Vt)
                Vt = capacitor_voltage(It, rSleep, (ti - tSHT - tSense), c, Vt)
                if Vt > Vt_min:
                    Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                    Vt = capacitor_voltage(It, rMCUI2CCoulomb, tMCUI2CCoulomb, c, Vt)
                else:
                    Vt = capacitor_voltage(It, rOff, tSense, c, Vt)
                    off = True

    deviceStateList.append(not off)
    capVoltageList.append(Vt)
    finalHarvestingCurrentList.append(It)
    actionList.append(action)
    successList.append(success)
    tCycleList.append(tCycle)
    iCycleList.append(iCycle)
    action = 0
    success = 0
    if multipleMu == True:
        tCycleIncreaseCounter += 1
        if tCycleIncreaseCounter > tCycleIncreaseThresh:
            tCycleIncreaseCounter = 0
            mu = mu + 5
            lower = mu - spreading
            upper = mu + spreading
    # We update the cycle time at the end of the step so we have a fair comparison with the agent
    tCycle = generate_normal_in_range(mu, sigma, lower, upper)

write_to_csv("oracleModelData/voltageList" + additional_text + ".csv", capVoltageList)
write_to_csv("oracleModelData/currentList" + additional_text + ".csv", finalHarvestingCurrentList)
write_to_csv("oracleModelData/actionList" + additional_text + ".csv", actionList)
write_to_csv("oracleModelData/successList" + additional_text + ".csv", successList)
write_to_csv("oracleModelData/deviceStateList" + additional_text + ".csv", deviceStateList)
write_to_csv("oracleModelData/tCycleList" + additional_text + ".csv", tCycleList)
write_to_csv("oracleModelData/iCycleList" + additional_text + ".csv", iCycleList)
print(action_ctr)
print(success_ctr)
print(calc_ctr)