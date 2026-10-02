import numpy as np
import csv
import math as math
import pandas as pd

adrSimulation = True
optimizedThresholds = False
highThreshold = False
multipleMu = False
day_test = False
combined = False 
shuffledData = True
payloadSize = 50
payloadOption = "daily_random_20_255"
c = 10 # Cap size
capSizeStr = str(c)

# Mapping for optimized thresholds for different capacitor sizes (key = cap size, value = threshold)
cap_sizes = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 
             5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

thresh_map = {
    10.0: 1.9, 9.5: 1.9, 9.0: 1.9, 8.5: 1.9, 8.0: 1.9, 
    7.5: 1.9, 7.0: 1.9, 6.5: 1.9, 6.0: 1.9, 5.5: 1.95, 
    5.0: 1.95, 4.5: 1.95, 4.0: 2.0, 3.5: 2.0, 3.0: 2.05, 
    2.5: 2.05, 2.0: 2.15, 1.5: 2.25, 1.0: 2.5, 0.5: 3.45
}

if optimizedThresholds == True:
    V_thresh = thresh_map.get(c) 
    print("V_thresh value is:" + str(V_thresh))

else:
    if highThreshold == True:
        V_thresh = 3.45
        print("V_thresh value is:" + str(V_thresh))
    else:
        V_thresh = 1.9
        print("V_thresh value is:" + str(V_thresh))


additional_text = "ADR_" + payloadOption + "_BYTES_" + capSizeStr + "_FARAD_" + str(V_thresh) + "V_THRESH"
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
# We use mu to decide tCycle

tCycleIncreaseThresh = math.floor(max_steps/4)
tCycleIncreaseCounter = 0

adrCount = 0


harvestingCurrentList = list()
finalHarvestingCurrentList = list()
actionList = list()
successList = list()
stepCounter = 0
capVoltageList = list()
deviceStateList = list()
tCycleList = list()
tCycleList = list()
iCycleList = list()
I = []
Ih = 0.02
It = 0
It_max = 3
It_min = 0
V = []
V_to = 2.3
Vt_max = 5.5
V_supply = 3.3
Vt_min = 1.8

action = 0
action_ctr = 0
ctr = 0
success = 0
success_ctr = 0


n = 0  
off = False


tCycle = mu 

Vt = 2.7

combinerEfficiency = 0.88


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




# -- IN CASE OF IRRADIANCE VALUES CSV FILE

if shuffledData == False:
    dataList = pd.read_csv('solar_and_teg_current_data.csv', sep = ',') # WHEN WE USE A 3 MIN TIME INTERVAL WITH THE PREPROCESSED DATA (see data_preprocessing.py)

    if day_test == False:
        # solarCurrentList = dataList.iloc[17516:25339, 0]  # We only want the solar current data from the CSV file (training part)
        solarCurrentList = dataList.iloc[20000:25339, 0]  # Testing new data split
        solarCurrentList = solarCurrentList.to_numpy() 

        # tegCurrentList = dataList.iloc[17516:25339, 1]  # We only want the TEG current data from the CSV file (training part)
        tegCurrentList = dataList.iloc[20000:25339, 1]  # Testing new data split
        tegCurrentList = tegCurrentList.to_numpy() 
    else:
         # solarCurrentList = dataList.iloc[17516:25339, 0]  # We only want the solar current data from the CSV file (training part)
        solarCurrentList = dataList.iloc[:, 0]  # Testing new data split
        solarCurrentList = solarCurrentList.to_numpy() 

        # tegCurrentList = dataList.iloc[17516:25339, 1]  # We only want the TEG current data from the CSV file (training part)
        tegCurrentList = dataList.iloc[:, 1]  # Testing new data split
        tegCurrentList = tegCurrentList.to_numpy() 

        nrOfInterpolationPoints = 14

    if combined == True:
        harvestingCurrentList = (solarCurrentList + tegCurrentList) * combinerEfficiency
    else:
        harvestingCurrentList = solarCurrentList
else:
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
# print(len(harvestingCurrentList))

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
            # number = round(number, 2)
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



# - DATA INTERPOLATION OPTION
if shuffledData == False:
    harvestingCurrentList = fill_with_interpolation(harvestingCurrentList, nrOfInterpolationPoints) # Original data is every 15 min and we want every 3 min (final array size with interpolation = array size + (array size - 1) * nrOfInterpolationPoints
    if day_test == True:
        harvestingCurrentList = harvestingCurrentList[162700:165050]
        max_steps = len(harvestingCurrentList)
# print("Length is " + str(len(harvestingCurrentList)))

print(max_steps)
for i in range(max_steps):  

    It = read_from_harvesting_current_list(n)
    n += 1
        

    if off == False:
        if Vt <= Vt_min: 
            Vt = capacitor_voltage(It, rOff, ti, c, Vt)
            off = True
        else:
            if Vt < V_thresh: 
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
                else:
                    Vt = Vt_min
                    Vt = capacitor_voltage(It, rOff, ti - tSHT - tCycle, c, Vt)
                    off = True
                action = 1
                action_ctr += 1

    else:
        if Vt < V_to:
            Vt = capacitor_voltage(It, rOff, ti, c, Vt)
        else:
            off = False
            if Vt >= V_thresh:
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
                else:
                    Vt = Vt_min
                    Vt = capacitor_voltage(It, rOff, ti - tSHT - tCycle, c, Vt)
                    off = True
                action = 1
                action_ctr += 1
            else:
                Vt = capacitor_voltage(It, rSleep, (ti - tSense), c, Vt)
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

write_to_csv("staticThreshModelData/voltageList" + additional_text + ".csv", capVoltageList)
write_to_csv("staticThreshModelData/currentList" + additional_text + ".csv", finalHarvestingCurrentList)
write_to_csv("staticThreshModelData/actionList" + additional_text + ".csv", actionList)
write_to_csv("staticThreshModelData/successList" + additional_text + ".csv", successList)
write_to_csv("staticThreshModelData/deviceStateList" + additional_text + ".csv", deviceStateList)
write_to_csv("staticThreshModelData/tCycleList" + additional_text + ".csv", tCycleList)
write_to_csv("staticThreshModelData/iCycleList" + additional_text + ".csv", iCycleList)
print(action_ctr)
print(success_ctr)


