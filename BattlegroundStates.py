import pandas as pd
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import random
from dataCleaning import df_all




fig, axs = plt.subplots(1,3,figsize=(20,5))


# Georgia

GA_BG=df_all[df_all["State"]=="Georgia"].sort_values(by=["Poll_Date"])

GA_allDates = GA_BG[["Poll_Date","Biden_Proportion","Trump_Proportion"]]

GA = GA_allDates[GA_allDates["Poll_Date"]>= "2020-4-1"].sort_values(by="Poll_Date")

axs[0].plot(GA["Poll_Date"],GA["Biden_Proportion"], color = "blue",label = "Biden")
axs[0].plot(GA["Poll_Date"],GA["Trump_Proportion"],color = "red",label = "Trump")
axs[0].set_ylabel("Proportion")
axs[0].title.set_text("Georgia Votes Trend")



# Ohio

OH_BG=df_all[df_all["State"]=="Ohio"].sort_values(by=["Poll_Date"])

OH_allDates = OH_BG[["Poll_Date","Biden_Proportion","Trump_Proportion"]]

OH = OH_allDates[OH_allDates["Poll_Date"]>= "2020-4-1"].sort_values(by="Poll_Date")

axs[1].plot(OH["Poll_Date"],OH["Biden_Proportion"], color = "blue",label = "Biden")
axs[1].plot(OH["Poll_Date"],OH["Trump_Proportion"],color = "red",label = "Trump")
axs[1].title.set_text("Ohio Votes Trend")
axs[1].legend()



# Arizona

AZ_BG=df_all[df_all["State"]=="Arizona"].sort_values(by=["Poll_Date"])

AZ_allDates = AZ_BG[["Poll_Date","Biden_Proportion","Trump_Proportion"]]

AZ = AZ_allDates[AZ_allDates["Poll_Date"]>= "2020-4-1"].sort_values(by="Poll_Date")

axs[2].plot(AZ["Poll_Date"],AZ["Biden_Proportion"], color = "blue",label = "Biden")
axs[2].plot(AZ["Poll_Date"],AZ["Trump_Proportion"],color = "red",label = "Trump")
axs[2].set_xlabel("Dates")
axs[2].title.set_text("Arizona Votes Trend")
plt.show()
