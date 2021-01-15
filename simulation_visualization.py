#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 21:58:01 2020

@author: geetinderpardesi
"""

import glob
import pandas as pd
import numpy as np
import math
from itertools import islice
from pollster_name_changes import pollster_changes
from scipy import stats
import matplotlib.pyplot as plt
import random

# Determining Number of trials in simulation
num_sims = 1000


df_all = pd.read_csv("dataForSimulation.csv")


# Clean State Polling Text File
filepath = "StatePollingData/*.txt"
statePollingFileName = glob.glob(filepath)

### Simulation ###
# Estimated Probability of Biden Win
def simulate_returns(polls_distrbs, state, num_sims):
    prob = pd.DataFrame(index=[state])

    for k in range(num_sims):
        sim_prob = polls_distrbs.rvs()

        trial_str = "Trial" + str(k + 1)
        prob[trial_str] = sim_prob

    return prob
   
prob_all_list = []

for i in statePollingFileName:  # Loop in all the text files
    state = i.replace("/", "?").replace(".", "?").split("?")[1]
    simulation_data = df_all[df_all.State == state]
    
    # Sort grade_new and poll_date column in ascending and descending order resspectively
    simulation_data = simulation_data.sort_values(by=["grade_new", "Poll_Date"], ascending=[True, False])

    # Create list on the basis of letter grade and if no grade then recent date
    row_count = 0
    sim_list = []
    for x in simulation_data.grade_new:
        if x == "A":
            prob = simulation_data.iloc[row_count, 6]
            sd = simulation_data.iloc[row_count, 7]
            sim_list.append([prob, sd])
            row_count += 1
        else:
            prob = simulation_data.iloc[0, 6]
            sd = simulation_data.iloc[0, 7]
            sim_list.append([prob,sd])

    # Choose Pollster Source randomly based on grade_new and if grade_new other than A then, choose recent date
    polls_distrbs = stats.norm(random.choice(sim_list)[0],random.choice(sim_list)[1])
    prob = simulate_returns(polls_distrbs, state, num_sims)
    prob_all_list.append(prob)

prob_all = pd.concat(prob_all_list)

print(prob_all)


### Visualisation ###

# Get state electoral votes
d = {}
temp = []
with open("ELECTORAL COLLEGE.txt") as f:
    data = f.read().splitlines()
    for i in data:
        temp.append(i.split("\t"))

for i in temp:
    d[i[0]] = i[1]
    d[i[2]] = i[3]

d["District of Columbia"] = 3

# Count the votes Biden gets in each trial

win_count = []
for n in range(num_sims):
    votes = 0
    n += 1
    for i in statePollingFileName:  # get state name
        state = i.replace("/", "?").replace(".", "?").split("?")[1]
        trial_count = str("Trial"+str(n))
        if prob_all.loc[state,trial_count] > 0.5:
            votes += int(d[state])
    win_count.append(votes)

# Histogram of Electoral Votes Distribution 
fig,ax = plt.subplots()
ax.hist(win_count,edgecolor='white',color="skyblue")


plt.xlabel('Total votes Biden wins',fontsize=10)
plt.ylabel('Frequency',fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('Final Electoral College Scores Distribution Histogram',fontsize=15)


plt.show()


# Count the votes Biden gets in each state in all trials
win_percent = {}
for i in statePollingFileName:  # get state name
    state = i.replace("/", "?").replace(".", "?").split("?")[1]
    votes_state = 0
    for n in range(num_sims):
        n += 1
        trial_count = str("Trial"+str(n))
        if prob_all.loc[state,trial_count] > 0.5:
            votes_state += 1
    win_percent[state]=votes_state/num_sims

# Convert the dictionary into df
win_percent_df=pd.DataFrame(win_percent.items(),columns=["STATE_NAME","percentage"])


# Bar Chart for respective Candidate winning probability in each state
win_percent_stack = win_percent_df.set_index("STATE_NAME")

def get_Trump_percent(x):
    return 1-x

win_percent_stack.rename(columns={'percentage': 'Biden'}, inplace=True)
win_percent_stack["Trump"]=win_percent_stack["Biden"].apply(get_Trump_percent)

win_percent_stack=win_percent_stack.sort_values(by="Biden")

win_percent_stack.plot(kind='barh', color={"lightcoral","skyblue"},stacked=True,figsize=(10,10))
plt.ylabel("State")
plt.xlabel("Probability to win")
plt.show()


# Conclusion - overall Biden Win Percentage
num_win=0
for i in win_count:
    if i > 270:
        num_win += 1

overall_win_percent = num_win/num_sims

print(overall_win_percent)