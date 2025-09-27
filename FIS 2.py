# FIS2_main_priority_for_report.py — Soft Computing Assignment 2
# Goal: Minimize waiting time for the main street (Aggressive)
# Author: Aiden Moses u3239240
# Date: 20/09/2025

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import trafficSimulator

# -------------------------------
# 1. Define inputs & outputs
# -------------------------------
waitingTraffic  = ctrl.Antecedent(np.arange(0, 101, 1), 'waiting')      # Side street traffic
incomingTraffic = ctrl.Antecedent(np.arange(0, 101, 1), 'incoming')     # Main street traffic
waitDuration    = ctrl.Consequent(np.arange(0, 121, 1), 'wait duration') # Main street green
openDuration    = ctrl.Consequent(np.arange(0, 121, 1), 'open duration') # Side street green

# -------------------------------
# 2. Membership functions
# -------------------------------
waitingTraffic['low']    = fuzz.trapmf(waitingTraffic.universe, [0, 0, 10, 30])
waitingTraffic['medium'] = fuzz.trimf(waitingTraffic.universe, [20, 50, 80])
waitingTraffic['high']   = fuzz.trapmf(waitingTraffic.universe, [60, 80, 100, 100])

incomingTraffic['low']    = fuzz.trapmf(incomingTraffic.universe, [0, 0, 15, 35])
incomingTraffic['medium'] = fuzz.trimf(incomingTraffic.universe, [25, 50, 75])
incomingTraffic['high']   = fuzz.trapmf(incomingTraffic.universe, [65, 80, 100, 100])

waitDuration['short']     = fuzz.trimf(waitDuration.universe, [0, 15, 35])
waitDuration['medium']    = fuzz.trimf(waitDuration.universe, [30, 60, 90])
waitDuration['long']      = fuzz.trimf(waitDuration.universe, [80, 100, 120])
waitDuration['very_long'] = fuzz.trapmf(waitDuration.universe, [100, 110, 120, 120])

openDuration['short']     = fuzz.trapmf(openDuration.universe, [0, 0, 5, 15])
openDuration['medium']    = fuzz.trimf(openDuration.universe, [10, 20, 30])
openDuration['long']      = fuzz.trimf(openDuration.universe, [25, 35, 45])
openDuration['very_long'] = fuzz.trapmf(openDuration.universe, [40, 50, 60, 60])

# -------------------------------
# 3. Aggressive Rule Base (favor main street heavily)
# -------------------------------
rules = []
rules.append(ctrl.Rule(incomingTraffic['high'], [waitDuration['very_long'], openDuration['short']]))
rules.append(ctrl.Rule(incomingTraffic['medium'] & waitingTraffic['low'],    [waitDuration['long'], openDuration['short']]))
rules.append(ctrl.Rule(incomingTraffic['medium'] & waitingTraffic['medium'], [waitDuration['long'], openDuration['medium']]))
rules.append(ctrl.Rule(incomingTraffic['medium'] & waitingTraffic['high'],   [waitDuration['medium'], openDuration['medium']]))
rules.append(ctrl.Rule(incomingTraffic['low'] & waitingTraffic['low'],       [waitDuration['short'], openDuration['short']]))
rules.append(ctrl.Rule(incomingTraffic['low'] & waitingTraffic['medium'],    [waitDuration['medium'], openDuration['medium']]))
rules.append(ctrl.Rule(incomingTraffic['low'] & waitingTraffic['high'],      [waitDuration['medium'], openDuration['long']]))

# -------------------------------
# 4. Build FIS system
# -------------------------------
fis2 = ctrl.ControlSystem(rules)
fis2_sim = ctrl.ControlSystemSimulation(fis2)

# -------------------------------
# 5. Run simulator
# -------------------------------
num_cars_on_main, num_cars_on_side, wait_times_main, wait_times_side = trafficSimulator.simulate(
    fis2_sim, verbose=False
)

mean_main = np.mean(wait_times_main)
mean_side = np.mean(wait_times_side)

print("\n=== Simulator Output (FIS2 Aggressive Final) ===")
print("Mean waiting time - main street:", round(mean_main, 2))
print("Mean waiting time - side street:", round(mean_side, 2))
print("Difference:", round(mean_main - mean_side, 2))
print("Ratio (main/side):", round(mean_main / mean_side, 2))

# -------------------------------
# 6. Figures for Report
# -------------------------------

# Figure 4: Number of cars waiting over time
plt.figure(figsize=(8, 5))
plt.plot(num_cars_on_main, label="Main Street", color="blue")
plt.plot(num_cars_on_side, label="Side Street", color="red")
plt.xlabel("Time Step")
plt.ylabel("Number of Cars Waiting")
plt.title("Figure 4: Number of Cars Waiting Over Time (FIS2 Aggressive)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("fig4_fis2_cars_over_time.png")

# Figure 5: Boxplot of waiting times
plt.figure(figsize=(6, 5))
plt.boxplot([wait_times_main, wait_times_side], labels=["Main Street", "Side Street"])
plt.ylabel("Waiting Time (s)")
plt.title("Figure 5: Distribution of Waiting Times (FIS2 Aggressive)")
plt.grid(alpha=0.3)
plt.savefig("fig5_fis2_wait_times_boxplot.png")

# Figure 6: Bar chart of mean waiting times
plt.figure(figsize=(6, 5))
plt.bar(["Main Street", "Side Street"], [mean_main, mean_side], color=["blue", "red"])
plt.ylabel("Mean Waiting Time (s)")
plt.title("Figure 6: Mean Waiting Times (FIS2 Aggressive)")
plt.grid(alpha=0.3)
plt.savefig("fig6_fis2_mean_wait_bar.png")
plt.show()

# -------------------------------
# 7. Test specific input cases (Table 2)
# -------------------------------
test_cases = [
    {'incoming': 10, 'waiting': 5},
    {'incoming': 50, 'waiting': 30},
    {'incoming': 80, 'waiting': 70},
    {'incoming': 0, 'waiting': 90},
    {'incoming': 90, 'waiting': 10},
]

results = []
print("\n=== Test Cases ===")
for case in test_cases:
    fis2_sim.input['incoming'] = case['incoming']
    fis2_sim.input['waiting'] = case['waiting']
    fis2_sim.compute()
    wait_dur = fis2_sim.output['wait duration']
    open_dur = fis2_sim.output['open duration']
    ratio_case = wait_dur / open_dur if open_dur > 0 else 0
    results.append([case['incoming'], case['waiting'], round(wait_dur,1), round(open_dur,1), round(ratio_case,2)])
    print(f"Incoming: {case['incoming']:3d}, Waiting: {case['waiting']:3d} -> "
          f"Main Green: {wait_dur:5.1f}s, Side Green: {open_dur:5.1f}s, Ratio: {ratio_case:.2f}")

# Save Test Cases Table for report
df_results = pd.DataFrame(results, columns=["Incoming (main)", "Waiting (side)", "Main Green (s)", "Side Green (s)", "Ratio"])
df_results.to_csv("Table2_fis2_test_cases.csv", index=False)
print("\nTest case table saved as Table2_fis2_test_cases.csv")
