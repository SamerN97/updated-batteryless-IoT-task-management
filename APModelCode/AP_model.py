import sys
import os
# Add the root directory of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import csv
import math as math
import pandas as pd
from scipy.optimize import fsolve
from solvers.newton_raphson import newton_raphson


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
V_thresh = 3.2
if adrSimulation == False:
    mu = 10 # We use mu to decide tCycle
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
tCycleCalc = 0
V_0_cycle = 0
V_1_cycle = 0
tSleepCalc = 0
V_0_sleep = 0
V_1_sleep = 0
tSenseCalc = 0
iCalc = 0
V_0_sense = 0
V_1_sense = 0
valuesMeasured = False
Vt_check = 0
feasible = 0
prev_tCycle = 0
cum_fail_ctr = 0
Vt_min_check = Vt_min
recalc = False

V_0_sht = 0
V_1_sht = 0
tSHTCalc = 0


tCycleIncreaseThresh = math.floor(max_steps/4)
tCycleIncreaseCounter = 0

action = 0
action_ctr = 0
ctr = 0
success = 0
success_ctr = 0
n = 0  
off = False
tCycle = mu 
tStartup = 0.001
Vt = 2.7
V_0 = 0
combinerEfficiency = 0.88 
calibrate = True
rSenseCalc = 0
rSleepCalc = 0
rStartupCalc = 0
fail = False
fail_ctr = 0
calc_ctr = 0
error_thresh = 0
iCycleCalc = 0


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

# Solver
# MCU at 1MHz consumes ~100uA according to the datasheet
# Total current = MCU Active + Leakage
iSolver = iActiveMCU + iLeakage 
rSolver = V_supply / iSolver

iSimulate = iSolver
rSimulate = rSolver

# Time per iteration at 1MHz (estimated 3000 cycles = 3ms)
tPerIteration = 0.003

tPerCalculation = 0.0003 # We approximate the calculation (capacitor equation for chcking) to take up around 300 clock cycles (at 1MHz --> 0.0003s)

tSolver = 0

calcIterations = 0

V_prev = 0

drop = 0



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
adrTimeList = adrTimeList/1000 # From ms to s
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

    
def equation(R_eq, v, v_0, I_h, C, t):
    # Physics guard: The argument of the log must be positive
    # and we must avoid division by zero (v_0 == I_h * R_eq)

    try:

        # Cast R_eq to float to handle scipy passing it as an array
        R_eq_scalar = float(R_eq)

        numerator = v - I_h * R_eq_scalar
        denominator = v_0 - I_h * R_eq_scalar
        
        if numerator <= 0 or denominator <= 0:
            return 1e6 # Return a large error to push the solver back
            
        return -(t / (C * math.log(numerator / denominator))) - R_eq
    except (ValueError, ZeroDivisionError, TypeError):
        return 1e6


def derivative(R_eq, v, v_0, I_h, C, t):

    try:

        # Cast R_eq to float to handle scipy passing it as an array
        R_eq_scalar = float(R_eq)

        term1 = v - I_h * R_eq_scalar
        term2 = v_0 - I_h * R_eq_scalar
        
        if term1 <= 0 or term2 <= 0:
            return -1 # Slope to push solver away
            
        log_term = math.log(term1 / term2)
        return t * I_h * (v - v_0) / (C * term1 * term2 * (log_term ** 2)) - 1
    except (ValueError, ZeroDivisionError, TypeError):
        return -1

def calculate_req(v, v_0, I_h, C, t):
    # Initial guess for R_eq (can be chosen based on expected value)
    R_eq_initial_guess = 3.3
    # Use fsolve to find the root of the equation
    R_eq_solution = fsolve(equation, R_eq_initial_guess, args=(v, v_0, I_h, C, t))
    print("For the values v, v_0, I_h, C, t we have: " + str(v) + "," + str(v_0) + "," + str(I_h) + "," + str(C) +  "," + str(t) )
    print("The original solution is: " + str(R_eq_solution[0]))
    return R_eq_solution[0]


def calculate_req_newton_raphson(v, v_0, I_h, C, t):
    R_eq_guess = 5
    # find solution using newton-raphson's method (see source code for more info)
    nrsolution, iterations = newton_raphson(equation, derivative, R_eq_guess, args=(v, v_0, I_h, C, t))
    if iterations > 99:
        print("NR did not converge within 100 steps")
    print("Newton-Raphson Solution:", nrsolution)
    print("Number of iterations:", iterations)
    return nrsolution, iterations


def update_error_thresh(cycleCurrent):
    error_thresh = 10 * cycleCurrent
    error_thresh = max(0.01, error_thresh)
    return error_thresh


if shuffledData == False:
    harvestingCurrentList = fill_with_interpolation(harvestingCurrentList, nrOfInterpolationPoints) # Original data is every 15 min and we want every 3 min (final array size with interpolation = array size + (array size - 1) * nrOfInterpolationPoints
    if day_test == True:
        harvestingCurrentList = harvestingCurrentList[162700:165050]
        max_steps = len(harvestingCurrentList)


print(max_steps)
for i in range(max_steps): 

    if calibrate == True:
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
                    if valuesMeasured == False:
                        # Take into acccount the energy consumption of the measurement of V_0 itself
                        Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                        V_0 = Vt
                        Vt = capacitor_voltage(It, rSHT, tSHT, c, Vt)
                        tSHTCalc = tSHT
                        V_0_sht = V_0
                        V_1_sht = Vt
                        # Take into acccount the energy consumption of the measurement of V_0 itself
                        Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                        V_0 = Vt
                        Vt = capacitor_voltage(It, rCycle, tCycle, c, Vt)
                        tCycleCalc = tCycle
                        V_0_cycle = V_0
                        V_1_cycle = Vt
                        iCalc = It
                        if Vt > Vt_min:
                            # Take into acccount the energy consumption of the measurement of V_0 itself
                            Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                            V_0 = Vt
                            Vt = capacitor_voltage(It, rSleep, (ti - tSHT - tCycle - tSense), c, Vt)
                            tSleepCalc = (ti - tSHT - tCycle - tSense)
                            V_0_sleep = V_0
                            V_1_sleep = Vt
                            iCalc = It
                            if Vt > Vt_min:
                                # Take into acccount the energy consumption of the measurement of V_0 itself
                                Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                                V_0 = Vt
                                Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                                Vt = capacitor_voltage(It, rMCUI2CCoulomb, tMCUI2CCoulomb, c, Vt)
                                tSenseCalc = tSense
                                V_0_sense = V_0
                                V_1_sense = Vt
                                iCalc = It
                                if Vt > Vt_min:
                                    success = 1
                                    success_ctr += 1
                                    valuesMeasured = True
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
                        if recalc == False:
                            rSHTCalc = calculate_req(V_1_sht, V_0_sht, iCalc, c, tSHTCalc)
                            rSenseCalc, calcIterations = calculate_req_newton_raphson(V_1_sense, V_0_sense, iCalc, c, tSenseCalc)
                            tSolver = tPerIteration * calcIterations
                            V_prev = Vt
                            Vt = capacitor_voltage(It, rSolver, tSolver, c, Vt)
                            drop = V_prev - Vt
                            print("Voltage drop because of solver(sense):", drop)

                            # Check for brownout
                            if Vt <= Vt_min:
                                off = True

                            if not off:
                                rSleepCalc, calcIterations = calculate_req_newton_raphson(V_1_sleep, V_0_sleep, iCalc, c, tSleepCalc)
                                tSolver = tPerIteration * calcIterations
                                V_prev = Vt
                                Vt = capacitor_voltage(It, rSolver, tSolver, c, Vt)
                                drop = V_prev - Vt
                                print("Voltage drop because of solver(sleep):", drop)

                                # Check for brownout
                                if Vt <= Vt_min:
                                    off = True

                        if not off:
                            rCycleCalc, calcIterations = calculate_req_newton_raphson(V_1_cycle, V_0_cycle, iCalc, c, tCycleCalc)
                            tSolver = tPerIteration * calcIterations
                            V_prev = Vt
                            Vt = capacitor_voltage(It, rSolver, tSolver, c, Vt)
                            drop = V_prev - Vt
                            print("Voltage drop because of solver(cycle):", drop)

                            # Final brownout check
                            if Vt <= Vt_min:
                                off = True
                            elif rCycleCalc > 3.31:
                                calibrate = False
                                recalc = False
                                iCycleCalc = V_supply/rCycleCalc
                                Vt_min_check = Vt_min + 1.5 * iCycleCalc

                        # Handle Device Death
                        if off:
                            # If the device died during the approximation, we record the failure
                            fail = True
                            fail_ctr += 1
                            cum_fail_ctr += 1
                            print("Device browned out while running solver iterations!")

                        valuesMeasured = False
                        calc_ctr += 1

        else:
            if Vt < V_to:
                Vt = capacitor_voltage(It, rOff, ti, c, Vt)
            else:
                off = False
                if Vt >= V_thresh:
                    if valuesMeasured == False:
                        # Take into acccount the energy consumption of the measurement of V_0 itself
                        Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                        V_0 = Vt
                        Vt = capacitor_voltage(It, rSHT, tSHT, c, Vt)
                        tSHTCalc = tSHT
                        V_0_sht = V_0
                        V_1_sht = Vt
                        # Take into acccount the energy consumption of the measurement of V_0 itself
                        Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                        V_0 = Vt
                        Vt = capacitor_voltage(It, rCycle, tCycle, c, Vt)
                        tCycleCalc = tCycle
                        V_0_cycle = V_0
                        V_1_cycle = Vt
                        iCalc = It
                        if Vt > Vt_min:
                            # Take into acccount the energy consumption of the measurement of V_0 itself
                            Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                            V_0 = Vt
                            Vt = capacitor_voltage(It, rSleep, (ti - tSHT -  tCycle - tSense), c, Vt)
                            tSleepCalc = (ti - tSHT - tCycle - tSense)
                            V_0_sleep = V_0
                            V_1_sleep = Vt
                            iCalc = It
                            if Vt > Vt_min:
                                # Take into acccount the energy consumption of the measurement of V_0 itself
                                Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                                V_0 = Vt
                                Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                                Vt = capacitor_voltage(It, rMCUI2CCoulomb, tMCUI2CCoulomb, c, Vt)
                                tSenseCalc = tSense
                                V_0_sense = V_0
                                V_1_sense = Vt
                                iCalc = It
                                if Vt > Vt_min:
                                    success = 1
                                    success_ctr += 1
                                    valuesMeasured = True
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
                        rSHTCalc = calculate_req(V_1_sht, V_0_sht, iCalc, c, tSHTCalc)

                        rSenseCalc, calcIterations = calculate_req_newton_raphson(V_1_sense, V_0_sense, iCalc, c, tSenseCalc)
                        tSolver = tPerIteration * calcIterations 
                        V_prev = Vt
                        Vt = capacitor_voltage(It, rSolver, tSolver, c, Vt)
                        drop = V_prev - Vt
                        print("Voltage drop because of solver(sense):", drop)

                        # Check for brownout
                        if Vt <= Vt_min:
                            off = True

                        # 2. Cycle Calculation (only if device hasn't browned out)
                        if not off:
                            rCycleCalc, calcIterations = calculate_req_newton_raphson(V_1_cycle, V_0_cycle, iCalc, c, tCycleCalc)
                            tSolver = tPerIteration * calcIterations 
                            V_prev = Vt
                            Vt = capacitor_voltage(It, rSolver, tSolver, c, Vt)
                            drop = V_prev - Vt
                            print("Voltage drop because of solver(cycle):", drop)

                            # Check for brownout
                            if Vt <= Vt_min:
                                off = True

                        # 3. Sleep Calculation (only if device hasn't browned out)
                        if not off:
                            rSleepCalc, calcIterations = calculate_req_newton_raphson(V_1_sleep, V_0_sleep, iCalc, c, tSleepCalc)
                            tSolver = tPerIteration * calcIterations 
                            V_prev = Vt
                            Vt = capacitor_voltage(It, rSolver, tSolver, c, Vt)
                            drop = V_prev - Vt
                            print("Voltage drop because of solver(sleep):", drop)

                            # Final brownout check
                            if Vt <= Vt_min:
                                off = True

                        # Handle Device Death vs. Success
                        if off:
                            # Device died during the solver approximation
                            fail = True
                            fail_ctr += 1
                            cum_fail_ctr += 1
                            print("Device browned out while running solver iterations in recovery phase!")
                        else:
                            # Only end calibration successfully if it survived all iterations
                            calibrate = False
                        
                        valuesMeasured = False
                        calc_ctr += 1
                else:
                    Vt = capacitor_voltage(It, rSleep, (ti - tSense), c, Vt)
                    if Vt > Vt_min:
                        Vt = capacitor_voltage(It, rADC, tADC, c, Vt)
                        Vt = capacitor_voltage(It, rMCUI2CCoulomb, tMCUI2CCoulomb, c, Vt)
                    else:
                        Vt = capacitor_voltage(It, rOff, tSense, c, Vt)
                        off = True
    else:
        It = read_from_harvesting_current_list(n)
        n += 1
        if off == False:
            if Vt <= Vt_min: # Should be smaller or equal 
                Vt = capacitor_voltage(It, rOff, ti, c, Vt)
                off = True
            else:
                Vt_check = capacitor_voltage(It, rSHTCalc, tSHT, c, Vt)
                Vt_check = capacitor_voltage(It, rCycleCalc, prev_tCycle, c, Vt_check)
                # Also account for power consumption due to these calculations:
                Vt = capacitor_voltage(It, rSimulate, tPerCalculation, c, Vt)
                Vt = capacitor_voltage(It, rSimulate, tPerCalculation, c, Vt)
                if Vt_check <= Vt_min_check:
                    feasible = 0
                else:
                    Vt_check = capacitor_voltage(It, rSleepCalc, (ti - tSense - tSHT - prev_tCycle), c, Vt_check)
                    # Also account for power consumption due to these calculations:
                    Vt = capacitor_voltage(It, rSimulate, tPerCalculation, c, Vt)
                    if Vt_check <= Vt_min_check:
                        feasible = 0
                    else:
                        Vt_check = capacitor_voltage(It, rSenseCalc, tSense, c, Vt_check)
                        # Also account for power consumption due to these calculations:
                        Vt = capacitor_voltage(It, rSimulate, tPerCalculation, c, Vt)
                        if Vt_check <= Vt_min_check:
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
                                diff = Vt - Vt_check
                                if diff > error_thresh:
                                    calibrate = True
                                    recalc = True
                                    print("DIFFERENCE CALIBRATION:")
                                    print(diff)
                                    t_diff = tCycle - prev_tCycle
                                    t_diff_calc = tCycle - tCycleCalc
                                    print(t_diff)
                                    print(t_diff_calc)
                                    print("n: ")
                                    print(n)
                                    print("\n")
                                    iCycleCalc = V_supply/rCycleCalc
                                    error_thresh = update_error_thresh(iCycleCalc)
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
                Vt_check = capacitor_voltage(It, rSHTCalc, tSHT, c, Vt)
                Vt_check = capacitor_voltage(It, rCycleCalc, prev_tCycle, c, Vt_check)
                # Also account for power consumption due to these calculations:
                Vt = capacitor_voltage(It, rSimulate, tPerCalculation, c, Vt)
                Vt = capacitor_voltage(It, rSimulate, tPerCalculation, c, Vt)
                if Vt_check <= Vt_min_check:
                    feasible = 0
                else:
                    Vt_check = capacitor_voltage(It, rSleepCalc, (ti - tSense - tSHT - prev_tCycle), c, Vt_check)
                    # Also account for power consumption due to these calculations:
                    Vt = capacitor_voltage(It, rSimulate, tPerCalculation, c, Vt)
                    if Vt_check <= Vt_min_check:
                        feasible = 0
                    else:
                        Vt_check = capacitor_voltage(It, rSenseCalc, tSense, c, Vt_check)
                        # Also account for power consumption due to these calculations:
                        Vt = capacitor_voltage(It, rSimulate, tPerCalculation, c, Vt)
                        if Vt_check <= Vt_min_check:
                            feasible = 0
                        else:
                            feasible = 1

                if feasible == 1:
                    Vt = capacitor_voltage(It, rSHT, tSHT, c, Vt)
                    Vt = capacitor_voltage(It, rCycle, tCycle, c, Vt)
                    if Vt > Vt_min:
                        Vt = capacitor_voltage(It, rSleep, (ti - tCycle - tSHT - tSense), c, Vt)
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
                                diff = Vt - Vt_check
                                if diff > error_thresh:
                                    calibrate = True
                                    recalc = True
                                    print("DIFFERENCE CALIBRATION:")
                                    print(diff)
                                    t_diff = tCycle - prev_tCycle
                                    t_diff_calc = tCycle - tCycleCalc
                                    print(t_diff)
                                    print(t_diff_calc)
                                    print("n: ")
                                    print(n)
                                    print("\n")
                                    iCycleCalc = V_supply/rCycleCalc
                                    error_thresh = update_error_thresh(iCycleCalc)
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
    iCycleList.append(iCycle)
    action = 0
    success = 0
    prev_tCycle = tCycle
    if multipleMu == True:
        tCycleIncreaseCounter += 1
        if tCycleIncreaseCounter > tCycleIncreaseThresh:
            tCycleIncreaseCounter = 0
            mu = mu + 5
            lower = mu - spreading
            upper = mu + spreading
    # We update the cycle time at the end of the step so we have a fair comparison with the agent
    tCycle = generate_normal_in_range(mu, sigma, lower, upper)
    if fail == True:
        fail = False
        calibrate = True
        recalc = True
        print("FAILURE CALIBRATION")
        print("n: ")
        print(n)
        print("\n")

print(n)
write_to_csv("APModelData/voltageList" + additional_text + ".csv", capVoltageList)
write_to_csv("APModelData/currentList" + additional_text + ".csv", finalHarvestingCurrentList)
write_to_csv("APModelData/actionList" + additional_text + ".csv", actionList)
write_to_csv("APModelData/successList" + additional_text + ".csv", successList)
write_to_csv("APModelData/deviceStateList" + additional_text + ".csv", deviceStateList)
write_to_csv("APModelData/tCycleList" + additional_text + ".csv", tCycleList)
write_to_csv("APModelData/iCycleList" + additional_text + ".csv", iCycleList)
print(action_ctr)
print(success_ctr)
print(cum_fail_ctr)
print(calc_ctr)