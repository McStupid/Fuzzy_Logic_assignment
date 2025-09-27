# FIS1_balanced_tuned_for_report.py — Soft Computing Assignment 2
# Goal: Equalize waiting times between main and side streets (Tuned for Simulator + Report Figures)
# Author: Aiden Moses u3239240
# Date: 20/09/2025

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import trafficSimulator  # compiled .pyc file

# -------------------------------
# 1. Inputs & Outputs
# -------------------------------
waitingTraffic = ctrl.Antecedent(np.arange(0, 101, 1), 'waiting')      # Side street queue
incomingTraffic = ctrl.Antecedent(np.arange(0, 101, 1), 'incoming')    # Main street queue
waitDuration = ctrl.Consequent(np.arange(0, 121, 1), 'wait duration')  # Main green
openDuration = ctrl.Consequent(np.arange(0, 121, 1), 'open duration')  # Side green

# -------------------------------
# 2. Membership functions
# -------------------------------
waitingTraffic['low'] = fuzz.trapmf(waitingTraffic.universe, [0, 0, 10, 25])
waitingTraffic['medium'] = fuzz.trimf(waitingTraffic.universe, [20, 40, 60])
waitingTraffic['high'] = fuzz.trapmf(waitingTraffic.universe, [50, 70, 100, 100])

incomingTraffic['low'] = fuzz.trapmf(incomingTraffic.universe, [0, 0, 15, 35])
incomingTraffic['medium'] = fuzz.trimf(incomingTraffic.universe, [25, 50, 75])
incomingTraffic['high'] = fuzz.trapmf(incomingTraffic.universe, [65, 80, 100, 100])

waitDuration['short'] = fuzz.trimf(waitDuration.universe, [0, 20, 40])
waitDuration['medium'] = fuzz.trimf(waitDuration.universe, [35, 60, 85])
waitDuration['long'] = fuzz.trapmf(waitDuration.universe, [70, 90, 120, 120])

openDuration['short'] = fuzz.trapmf(openDuration.universe, [0, 0, 5, 15])
openDuration['medium'] = fuzz.trimf(openDuration.universe, [10, 20, 35])
openDuration['long'] = fuzz.trimf(openDuration.universe, [25, 40, 55])

# -------------------------------
# 3. Rule Base — tuned for balance
# -------------------------------
rules = []

# Main low
rules.append(ctrl.Rule(incomingTraffic['low'] & waitingTraffic['low'], [waitDuration['short'], openDuration['short']]))
rules.append(ctrl.Rule(incomingTraffic['low'] & waitingTraffic['medium'], [waitDuration['short'], openDuration['medium']]))
rules.append(ctrl.Rule(incomingTraffic['low'] & waitingTraffic['high'], [waitDuration['medium'], openDuration['long']]))

# Main medium
rules.append(ctrl.Rule(incomingTraffic['medium'] & waitingTraffic['low'], [waitDuration['medium'], openDuration['short']]))
rules.append(ctrl.Rule(incomingTraffic['medium'] & waitingTraffic['medium'], [waitDuration['medium'], openDuration['medium']]))
rules.append(ctrl.Rule(incomingTraffic['medium'] & waitingTraffic['high'], [waitDuration['long'], openDuration['medium']]))

# Main high
rules.append(ctrl.Rule(incomingTraffic['high'] & waitingTraffic['low'], [waitDuration['long'], openDuration['short']]))
rules.append(ctrl.Rule(incomingTraffic['high'] & waitingTraffic['medium'], [waitDuration['long'], openDuration['medium']]))
rules.append(ctrl.Rule(incomingTraffic['high'] & waitingTraffic['high'], [waitDuration['long'], openDuration['medium']]))

# -------------------------------
# 4. Build FIS
# -------------------------------
waitDuration.defuzzify_method = 'centroid'
openDuration.defuzzify_method = 'centroid'

fis = ctrl.ControlSystem(rules)
fis_simulator = ctrl.ControlSystemSimulation(fis)

# -------------------------------
# 5. Run Simulator
# -------------------------------
num_cars_on_main, num_cars_on_side, wait_times_main, wait_times_side = trafficSimulator.simulate(fis_simulator, verbose=False)

mean_main = np.mean(wait_times_main)
mean_side = np.mean(wait_times_side)

print("=== Simulator Output (FIS1 Balanced Tuned) ===")
print("Mean waiting time - main street:", round(mean_main, 2))
print("Mean waiting time - side street:", round(mean_side, 2))
print("Difference:", round(abs(mean_main - mean_side), 2))
print("Ratio (main/side):", round(mean_main / mean_side, 2))

# -------------------------------
# 6. Figures for Report
# -------------------------------

# Figure 1: Number of cars waiting over time
plt.figure(figsize=(8, 5))
plt.plot(num_cars_on_main, label="Main Street", color="blue")
plt.plot(num_cars_on_side, label="Side Street", color="red")
plt.xlabel("Time Step")
plt.ylabel("Number of Cars Waiting")
plt.title("Figure 1: Number of Cars Waiting Over Time (FIS1 Balanced)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("fig1_traffic_over_time.png")

# Figure 2: Boxplot of waiting times
plt.figure(figsize=(6, 5))
plt.boxplot([wait_times_main, wait_times_side], labels=["Main Street", "Side Street"])
plt.ylabel("Waiting Time (s)")
plt.title("Figure 2: Boxplot of Waiting Times (FIS1 Balanced)")
plt.grid(alpha=0.3)
plt.savefig("fig2_waiting_times_boxplot.png")

# Figure 3: Bar chart of mean waiting times
plt.figure(figsize=(6, 5))
plt.bar(["Main Street", "Side Street"], [mean_main, mean_side], color=["blue", "red"])
plt.ylabel("Mean Waiting Time (s)")
plt.title("Figure 3: Mean Waiting Times (FIS1 Balanced)")
plt.grid(alpha=0.3)
plt.savefig("fig3_mean_wait_times.png")
plt.show()

# -------------------------------
# 7. Test Cases Table (Table 1)
# -------------------------------
test_cases = [
    {'incoming': 10, 'waiting': 5},
    {'incoming': 50, 'waiting': 30},
    {'incoming': 80, 'waiting': 70},
    {'incoming': 0, 'waiting': 90},
    {'incoming': 90, 'waiting': 10}
]

results = []
print("\n=== Test Cases ===")
for case in test_cases:
    fis_simulator.input['incoming'] = case['incoming']
    fis_simulator.input['waiting'] = case['waiting']
    fis_simulator.compute()
    wait_dur = fis_simulator.output['wait duration']
    open_dur = fis_simulator.output['open duration']
    ratio_case = wait_dur / open_dur if open_dur > 0 else 0
    results.append([case['incoming'], case['waiting'], round(wait_dur,1), round(open_dur,1), round(ratio_case,2)])
    print(f"Incoming: {case['incoming']:3d}, Waiting: {case['waiting']:3d} -> "
          f"Main Green: {wait_dur:5.1f}s, Side Green: {open_dur:5.1f}s, Ratio: {ratio_case:.2f}")

# Save Test Cases Table for report
df_results = pd.DataFrame(results, columns=["Incoming (main)", "Waiting (side)", "Main Green (s)", "Side Green (s)", "Ratio"])
df_results.to_csv("Table1_fis1_test_cases.csv", index=False)
print("\nTest case table saved as Table1_fis1_test_cases.csv")
