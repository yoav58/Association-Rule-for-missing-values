import subprocess
import matplotlib.pyplot as plt
import numpy as np


import os
from AprioriForNullMachineLearningPrediction import association_rule_method
from SimpleMachineLearningPrediction import simpleMethod
import sys
import os

# association_rule_method(0.6)

thresholds = [0.6,0.7,0.8,0.9,1]
association_rule_results = []
number_of_iterations = 1
for threshold in thresholds:
    value = []
    for i in range(number_of_iterations):
        print("thresholds number:" + str(threshold) + ", iteration number: " + str(i))
        result = association_rule_method(threshold)
        value.append(result)
    arr = np.array(value)
    value = np.mean(arr)
    association_rule_results.append(value)

conventional_result_temp = []
for i in range(number_of_iterations):
    result = simpleMethod()
    conventional_result_temp.append(result)
arr = np.array(conventional_result_temp)
conventional_result_temp = arr.mean()
conventional_result = [conventional_result_temp for i in range(5)]

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize=(12, 8))

# set height of bar
IT = association_rule_results
CSE = conventional_result

# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]

# Make the plot
plt.bar(br1, IT, color='r', width=barWidth,
        edgecolor='grey', label='Asocciation Rule Method')
plt.bar(br2, CSE, color='b', width=barWidth,
        edgecolor='grey', label='Conventional Method')

# Adding Xticks
plt.xlabel('Threshold', fontweight='bold', fontsize=15)
plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(IT))],
           ['0.6', '0.7', '0.8', '0.9', '0.10'])

plt.legend()
plt.show()