import glob
import pandas as pd
import numpy as np
import math
from itertools import islice
from pollster_name_changes import pollster_changes
from scipy import stats
import matplotlib.pyplot as plt
import random




### Cleaning Data ###
# Clean Pollster Rating Text File
with open("FiveThirtyEighty.txt") as f:
    pollster_data = f.read().splitlines()
    temp = []
    count = 8
    for num in range(len(pollster_data)):
        if num == count: # insert np.nan if no value in bias
            if "+" not in pollster_data[num]:
                temp.append(np.nan)
                temp.append(pollster_data[num])
                count += 8
            else:
                temp.append(pollster_data[num])
                count += 9
        else:
            temp.append(pollster_data[num])
    if "R+" not in temp[-1] and "D+" not in temp[-1]:
        temp.append(np.nan)

    # Extract Grade and Bias from Pollster Rating Data
    grade = [line for line in islice(temp, 7, len(temp), 9)]
    pollster = [line for line in islice(temp, 0, len(temp), 9)]
    bias = [line for line in islice(temp, 8, len(temp), 9)]

    # Create DataFrame
    df_pollster = pd.DataFrame(columns=["Poll_Source","grade", "bias"])
    df_pollster["Poll_Source"] = pollster
    df_pollster["grade"] = grade
    df_pollster["bias"] = bias

    # Add MISSING DATA value in Pollster Source Column of Pollster Rating Data to merge with Polling Data in next steps
    df_missing = pd.DataFrame([["MISSING DATA",np.nan,np.nan]], columns=["Poll_Source","grade", "bias"])
    df_pollster=df_pollster.append(df_missing)

# Clean State Polling Text File
filepath = "StatePollingData/*.txt"
statePollingFileName = glob.glob(filepath)

# Create dataframe by adding columns
df_all = pd.DataFrame(columns=["State", "Poll_Source", "Poll_Date", "Sample_Size", "Biden_Proportion", "Trump_Proportion","Est_Prob_Biden_Win"])


for i in statePollingFileName:
    state = i.replace("/", "?").replace(".","?").split("?")[1] # Get All State Name
    with open(i) as f:
        statePollingData = f.read().splitlines()
        statePollingData = [x.split("\t") for x in statePollingData]

        # Remove empty string at the beginning of each state polling file
        for q in statePollingData:
            if q[0] =="":
                del q[0]


        for num in statePollingData:
            # Remove space before the Pollster Source name
            num[0] = num[0].strip()

            # Drop polls without sample size information
            if "N/A" in num[2]:
                statePollingData.remove(num)


        for num in statePollingData:
            # Remove RV and LV in sample size column
            num[2] = int(num[2][:num[2].index(" ")].replace(',', ''))
            # Remove % symbol in columns Biden Proportion & Trump Proportion and convert the value into float
            num[3] = float(num[3].replace("%", "")) / 100
            num[4] = float(num[4].replace("%", "")) / 100
            num[5] = float(num[5].replace("%", "")) / 100
            # Assign equally the Other Candidate Proportion to each Biden & Trump
            split = num[3] + num[5] / 2
            num[5] = split
            # Change Pollster Name based on pollster_name_changes file and label the missing values as MISSING DATA
            if num[0] in pollster_changes.keys():
                num[0] = pollster_changes[num[0]]
            else:
                num[0] = "MISSING DATA"

        # Add the state name in first column of DataFrame
        for x in statePollingData:
            x.insert(0, state)
    # Add values to the Dataframe created above and calculate the standard deviation         
    df_polls = pd.DataFrame(statePollingData,columns=["State", "Poll_Source", "Poll_Date", "Sample_Size", "Biden_Proportion", "Trump_Proportion","Est_Prob_Biden_Win"])
    df_polls["Est_Standard_Deviation"] = np.sqrt(df_polls["Est_Prob_Biden_Win"] * (1 - df_polls["Est_Prob_Biden_Win"]) / df_polls["Sample_Size"])
    df = pd.merge(df_polls,df_pollster,on="Poll_Source")
    df_all= pd.concat([df,df_all])


    
# Adjust the Est_Prob_Biden_Win column for bias 
def get_bias_nubmer(x):
    # Return * for null value in column bias
    m=pd.DataFrame([x])
    if pd.isnull(m.iloc[0,0]):
        return "*"
    else:    # Get the value for Bias which has to be adjusted in Est_Prob_Biden_Win
        temp = x.split("+")[1]
        if x.split("+")[0] == "D":
            num = 1-float(temp)/100
            return num
        else:
            return 1+float(x.split("+")[1])/100
# Add another column "bias_adj" for Adjusted Bias 
df_all["bias_adj"] = df_all['bias'].apply(get_bias_nubmer)

# Change the data type of Est_Prob_Biden_Win to float
df["Est_Prob_Biden_Win"] = pd.to_numeric(df["Est_Prob_Biden_Win"],downcast = "float")



# Adjust the Est_Prob_Biden_Win for adjusted bias 
row_num =0
for i in df_all["bias_adj"]:
    if i != "*":
        df_all.iloc[row_num,6] = df_all.iloc[row_num,6] * df_all.iloc[row_num,10]
    row_num += 1



# Change data type of Poll_Date to Date
df_all["Poll_Date"]=pd.to_datetime(df_all["Poll_Date"])

# Change Grade column by assigning single letter grade to multiple grades of the same letter 
def get_new_grade(x):
    # If no value in Grade, return np.nan
    m=pd.DataFrame([x])
    if pd.isnull(m.iloc[0,0]):
        return x
    else:
        return x

df_all["grade_new"]=df_all['grade'].apply(get_new_grade)

df_all.to_csv("dataForSimulation.csv",index=None)

pollster_changes = {'CNBC/Change Research' : 'Change Research',
                    'Public Policy' : 'Public Policy Polling',
                    'Axios / SurveyMonkey' : 'SurveyMonkey',
                    'NY Times / Siena College' : 'Siena College/The New York Times Upshot',
                    'Quinnipiac' : 'Quinnipiac University',
                    'YouGov/CBS News' : 'YouGov',
                    'Ipsos/Reuters' : 'Ipsos',
                    'Fox News' : 'Fox News/Beacon Research/Shaw & Co. Research',
                    'Baldwin Wallace Univ.' : 'Baldwin Wallace University',
                    'NBC News/Marist' : 'Marist College',
                    'Mason-Dixon' : 'Mason-Dixon Polling & Strategy',
                    'CNN/SSRS' : 'CNN/Opinion Research Corp.',
                    'Marquette Law' : 'Marquette University Law School',
                    'East Carolina Univ.' : 'East Carolina University',
                    'Benenson / GS Strategy' : 'Benenson Strategy Group',
                    'Florida Atlantic Univ.' : 'Florida Atlantic University',
                    'Rasmussen Reports' : 'Rasmussen Reports/Pulse Opinion Research',
                    'UT Tyler' : 'University of Texas at Tyler',
                    'VCU' : 'Virginia Commonwealth University',
                    'Landmark Comm.' : 'Landmark Communications',
                    'UMass Lowell' : 'University of Massachusetts Lowell',
                    'Remington Research' : 'Remington Research Group',
                    'Susquehanna' : 'Susquehanna Polling & Research Inc.',
                    'WPA Intelligence' : 'WPA Intelligence (WPAi)',
                    '1892' : '1892 Polling',
                    'ABC News / Wash. Post' : 'ABC News/The Washington Post',
                    'Christopher Newport Univ.' : 'Christopher Newport University',
                    'East Tennessee State' : 'East Tennessee State University',
                    'Emer' : 'Emerson College',
                    'Fabrizio Lee' : 'Fabrizio, Lee & Associates',
                    'Fairleigh Dickinson' : 'Fairleigh Dickinson University (PublicMind)',
                    'Franklin & Marshall' : 'Franklin & Marshall College',
                    'GQR Research' : 'GQR Research (GQRR)',
                    'Garin-Hart-Yang' : 'Garin-Hart-Yang Research Group',
                    'Gonzales Research' : 'Gonzales Research & Marketing Strategies Inc.',
                    'HighGround' : 'HighGround Inc.',
                    'InsiderAdvantage' : 'Opinion Savvy/InsiderAdvantage',
                    'MRG' : 'MRG Research',
                    'MassINC' : 'MassINC Polling Group',
                    'Meeting Street Insights' : 'Meeting Street Research',
                    'Mitchell Research' : 'Mitchell Research & Communications',
                    'Montana State U.' : 'Montana State University Billings',
                    'Morning Call / Muhlenberg' : 'Muhlenberg College',
                    'PPIC' : 'Public Policy Institute of California',
                    'Research America' : 'Research America Inc.',
                    'Rutgers-Eagleton' : 'Rutgers University',
                    'SLU/YouGov' : 'Saint Leo University',
                    'Selzer & Company' : 'Selzer & Co.',
                    'Sooner Poll' : 'SoonerPoll.com',
                    'Sooner Survey' : 'SoonerPoll.com',
                    'St. Leo University' : 'Saint Leo University',
                    'THPF/Rice Univ.' : 'Rice University',
                    'TargetSmart' : 'TargetSmart/William & Mary',
                    'UC Berkeley' : 'University of California, Berkeley',
                    'UMass Amherst/WCVB' : 'University of Massachusetts Amherst',
                    'USC' : 'USC Dornsife/Los Angeles Times',
                    'Univ. of Colorado' : 'University of Colorado',
                    'Univ. of New Hampshire' : 'University of New Hampshire',
                    'Univ. of Georgia' : 'University of Georgia',
                    'Univ. of North Florida' : 'University of North Florida',
                    'Univ. of Texas / Texas Tribune' : 'University of Texas at Tyler',
                    'Univ. of Wisconsin-Madison' : 'University of Wisconsin (Badger Poll)',
                    'Univision/ASU' : 'Arizona State University',
                    'Univision/CMAS UH' : 'Univision/University of Houston/Latino Decisions',
                    'YouGov' : 'Yahoo/YouGov'
                    }
import glob
import pandas as pd
import numpy as np
import math
from itertools import islice
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