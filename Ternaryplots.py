import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import seaborn as sb
import random
import plotly.express as px
from chemparse import parse_formula

df2 = pd.read_excel('G:/My Drive/Alkaline HOR/Screening/Db with no amine.xlsx', skiprows=None).dropna(axis=0,how="all")
#df2 = pd.read_excel('C:/Users/Jeremy/Desktop/Neural Network/SnPtRh NN prediction.xls', skiprows=None).dropna(axis=0,how="all")
df2 = df2.dropna(axis=1,how="all")
df2 = df2.dropna(axis=0,how="any")
#df2 = df2[df2['Onset potential H2 Ag/AgCl'] != 1.0]
db = pd.read_excel('G:/My Drive/CO2/ML/Database of elements.xlsx',index_col='Element')
tickvals = np.linspace(-0.30,0.8,12)
ticktext = ['-0.3','-0.2','-0.1','0.0','0.1','0.2','0.3',
            '0.4','0.5','0.6','0.7','0.8']

elements = parse_formula(df2.iloc[726]["Sample ID"])
elementss = list(elements.keys())
for e in range(len(elementss)):
  elementss[e] = "Percent " + elementss[e]
data = df2["Onset potential H2 Ag/AgCl"][726:726+66]
s = np.interp(-(data-0.9), (0, 1.2), (3, 29.))
fig = px.scatter_ternary(df2.iloc[726:726+66],a="Percent B", b="Percent A", c="Percent C", range_color=[-0.3,0.8],
    color="Onset potential H2 Ag/AgCl",size=s, size_max=np.max(s),
    color_continuous_scale=["red","orange","yellow","greenyellow","cornflowerblue"],
    labels={"Percent B":elementss[1],"Percent A":elementss[0],"Percent C":elementss[2]},template="simple_white")
fig.update_ternaries(aaxis_tickformat='%',baxis_tickformat='%',caxis_tickformat='%',baxis_tickangle=60,
                       aaxis_ticklen=0,baxis_ticklen=0,caxis_ticklen=0,
                       caxis_tickangle=-60,aaxis_tickfont_size=10,baxis_tickfont_size=10,caxis_tickfont_size=10,
                       aaxis={'tick0':0.1,'dtick':0.2},baxis={'tick0':0.1,'dtick':0.2},caxis={'tick0':0.1,'dtick':0.2},
                       aaxis_tickfont_color="crimson",aaxis_title_font_color="crimson",
                       baxis_tickfont_color="darkblue",baxis_title_font_color="darkblue",
                       caxis_tickfont_color="limegreen",caxis_title_font_color="limegreen")
fig.update_coloraxes(colorbar_x=0.89,colorbar_y=0.6,colorbar_tickvals=np.linspace(-0.3,0.8,11),
                       colorbar_tickformat=".2f",colorbar_title_text="Onset potential<br>vs Ag/AgCl")
#fig.show()
#plt.savefig('Onset potential H2 {}.png'.format(df2.iloc[990]["Sample ID"]),dpi=300,bbox_inches="tight")
fig.write_image('Onset potential H2 {} a-b.png'.format(df2.iloc[726]["Sample ID"]),scale=3.5)
"""
for i in range(len(df2)//66):
  title = df2.iloc[i*66]["Sample ID"]
  elements = parse_formula(df2.iloc[i*66]["Sample ID"])
  elementss = list(elements.keys())
  for e in range(len(elementss)):
      elementss[e] = "Percent " + elementss[e]
  data = df2["Onset potential H2 Ag/AgCl"][i*66:(i+1)*66]
  s = np.interp(-(data-0.9), (0, 1.2), (3, 29.))
  if np.max(s) == 3.0:
      maxsize = 10
  else:
      maxsize = np.max(s)
  fig = px.scatter_ternary(df2[i*66:(i+1)*66],a="Percent A", b="Percent B", c="Percent C", range_color=[-0.3,0.8],
        color="Onset potential H2 Ag/AgCl",size=s, size_max=maxsize,
        color_continuous_scale=["red","orange","yellow","greenyellow","cornflowerblue"],
        labels={"Percent A":elementss[0],"Percent B":elementss[1],"Percent C":elementss[2]},template="simple_white")
  #fig.update_layout({'ternary':{'aaxis':'tickformat':'%'}})
  fig.update_ternaries(aaxis_tickformat='%',baxis_tickformat='%',caxis_tickformat='%',baxis_tickangle=60,
                       aaxis_ticklen=0,baxis_ticklen=0,caxis_ticklen=0,
                       caxis_tickangle=-60,aaxis_tickfont_size=10,baxis_tickfont_size=10,caxis_tickfont_size=10,
                       aaxis={'tick0':0.1,'dtick':0.2},baxis={'tick0':0.1,'dtick':0.2},caxis={'tick0':0.1,'dtick':0.2},
                       aaxis_tickfont_color="crimson",aaxis_title_font_color="crimson",
                       baxis_tickfont_color="darkblue",baxis_title_font_color="darkblue",
                       caxis_tickfont_color="limegreen",caxis_title_font_color="limegreen")
  fig.update_coloraxes(colorbar_x=0.89,colorbar_y=0.6,colorbar_tickvals=tickvals,
                       colorbar_tickformat=".2f",colorbar_title_text="Onset potential<br>vs Ag/AgCl",
                       colorbar_tickmode='array',colorbar_ticktext=ticktext)
  #fig.show()
  fig.write_image('predicted onset potential in H2 {}.png'.format(df2.iloc[i*66]["Sample ID"]),scale=3.5)
  print(i+1,len(df2)//66)"""
