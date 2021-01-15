"""
Geopandas map of US States for use in group project 1.  Adds inset axes for 
Alaska and Hawaii in lower left.
"""

import geopandas
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from simulation_visualization import win_percent_df
from simulation_visualization import d

fontcolor="white"
facecolor="silver"
linewidth=0.8

states = geopandas.read_file('states/states.shp')

#Convert to Mercator projection
states = states.to_crs("EPSG:3395")

continental = states[states.STATE_ABBR.isin(['AK', 'HI']) == False]

continental = pd.merge(continental,win_percent_df,on="STATE_NAME")


def win_percent_size(df):
    if df.percentage >= 0.75:
        return 75
    elif 0.6 <= df.percentage < 0.75:
        return 60
    elif 0.55 <= df.percentage < 0.6:
        return 50
    elif 0.45 <= df.percentage < 0.55:
        return 45
    elif 0.4 <= df.percentage < 0.45:
        return 40
    elif 0.15 <= df.percentage < 0.4:
        return 15
    else:
        return 10

continental.loc[:,'percentage'] = continental.apply(win_percent_size, axis=1)

fig, ax = plt.subplots(figsize=(14, 14))
# Continent map
continental_80 = continental[continental.percentage==75]
continental_60 = continental[continental.percentage==60]
continental_50 = continental[continental.percentage==50]
continental_45 = continental[continental.percentage==45]
continental_40 = continental[continental.percentage==40]
continental_20 = continental[continental.percentage==15]
continental_10 = continental[continental.percentage==10]

continental_80.plot(color="royalblue",edgecolor="white",linewidth=linewidth,ax=ax, legend=True)
continental_60.plot(color="cornflowerblue",edgecolor="white",linewidth=linewidth,ax=ax, legend=True)
continental_50.plot(color="lightsteelblue",edgecolor="white",linewidth=linewidth,ax=ax, legend=True)
continental_45.plot(color="peru",edgecolor="white",linewidth=linewidth,ax=ax, legend=True)
continental_40.plot(color="lightcoral",edgecolor="white",linewidth=linewidth,ax=ax, legend=True)
continental_20.plot(color="indianred",edgecolor="white",linewidth=linewidth,ax=ax, legend=True)
continental_10.plot(color="brown",edgecolor="white",linewidth=linewidth,ax=ax, legend=True)

for state in continental.index:
    statename =continental.loc[state,"STATE_NAME"]
    ax.annotate(s=continental.STATE_ABBR[state]+"\n"+str(d[statename]), xy = (continental.geometry[state].centroid.x,
                                                         continental.geometry[state].centroid.y),
                ha = 'center', va = 'center_baseline', fontsize=9,color=fontcolor)


plt.axis('off')




# HI map
ax_HI = inset_axes(ax, bbox_to_anchor=(.2, 0, 1, 1),
                   bbox_transform=ax.transAxes, width="15%", height="15%", loc="lower left", borderpad=0)
temp_HI = states[states['STATE_NAME'] == "Hawaii"]
HI = pd.merge(temp_HI, win_percent_df, on="STATE_NAME")
HI.loc[:,'percentage'] = HI.apply(win_percent_size, axis=1)

if HI.loc[0, "percentage"] == 75:
    HI.plot(ax=ax_HI, color="royalblue", linewidth=linewidth,edgecolor="white", legend=True)
elif HI.loc[0, "percentage"] == 60:
    HI.plot(ax=ax_HI, color="cornflowerblue",linewidth=linewidth, edgecolor="white", legend=True)
elif HI.loc[0, "percentage"] == 50:
    HI.plot(ax=ax_HI, color="lightsteelblue",linewidth=linewidth, edgecolor="white", legend=True)
elif HI.loc[0, "percentage"] == 45:
    HI.plot(ax=ax_HI, color="peru", linewidth=linewidth,edgecolor="white", legend=True)
elif HI.loc[0, "percentage"] == 40:
    HI.plot(ax=ax_HI, color="lightcoral", linewidth=linewidth,edgecolor="white", legend=True)
elif HI.loc[0, "percentage"] == 15:
    HI.plot(ax=ax_HI, color="indianred",linewidth=linewidth, edgecolor="white", legend=True)
elif HI.loc[0, "percentage"] == 10:
    HI.plot(ax=ax_HI,color="brown", linewidth=linewidth,edgecolor="white", legend=True)

ax_HI.annotate(s="HI"+"\n"+str(d["Hawaii"]), xy = (HI.geometry.centroid.x,
                                HI.geometry.centroid.y), ha = 'center', va = 'center_baseline',fontsize=10,color=fontcolor)

ax_HI.set_xticks([])
ax_HI.set_yticks([])
ax_HI.axis('off')


# AK map
ax_AK = inset_axes(ax, width="25%", height="25%", loc="lower left", borderpad=0)
temp_AK = states[states['STATE_NAME'] == "Alaska"]
AK = pd.merge(temp_AK, win_percent_df, on="STATE_NAME")
AK.loc[:,'percentage'] = AK.apply(win_percent_size, axis=1)


if AK.loc[0, "percentage"] == 75:
    AK.plot(ax=ax_AK, color="royalblue",linewidth=linewidth, edgecolor="white", legend=True)
elif AK.loc[0, "percentage"] == 60:
    AK.plot(ax=ax_AK,color="cornflowerblue", linewidth=linewidth,edgecolor="white", legend=True)
elif AK.loc[0, "percentage"] == 50:
    AK.plot(ax=ax_AK, color="lightsteelblue", linewidth=linewidth,edgecolor="white", legend=True)
elif AK.loc[0, "percentage"] == 45:
    AK.plot(ax=ax_AK,color="peru", linewidth=linewidth,edgecolor="white", legend=True)
elif AK.loc[0, "percentage"] == 40:
    AK.plot(ax=ax_AK,color="lightcoral", linewidth=linewidth,edgecolor="white", legend=True)
elif AK.loc[0, "percentage"] == 15:
    AK.plot(ax=ax_AK,color="indianred", linewidth=linewidth,edgecolor="white", legend=True)
elif AK.loc[0, "percentage"] == 10:
    AK.plot(ax=ax_AK,color="brown", linewidth=linewidth,edgecolor="white", legend=True)

ax_AK.annotate(s="AK"+"\n"+str(d["Alaska"]), xy = (AK.geometry.centroid.x,
                                AK.geometry.centroid.y), ha = 'center', va = 'center_baseline',fontsize=10,color=fontcolor)

ax_AK.set_xticks([])
ax_AK.set_yticks([])
ax_AK.axis('off')

fig.set_facecolor(facecolor)


plt.show()
#fig.savefig('usa.png', dpi=128)