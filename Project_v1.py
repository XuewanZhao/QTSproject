# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:09:57 2018

@author: Xuewan Zhao
"""

from ReadData import *
from NNutils import *
from math import *
equity,commodity,bond,RE,currency = GetData()
print("Data Loaded.")
# Each object above is a dictionary, u can iterate it to train the data.
# Key is the symbol of the asset, the value is a dataframe containing Open, Close, High, Low

# MAperiod: moving average period
# fRperiod: fast RSI period, should be smaller than sRperiod
# sRperiod: short RSI period.
MAperiod,fRperiod,sRperiod = 5,7,14
lookback_period,holding_period = 90,30
markets = [equity,commodity,bond,RE,currency]
for market in markets:
    for key,value in market.items():
        print("Training asset:",str(key))
        market[key] = GenData(value,MAperiod,fRperiod,sRperiod)
        market[key] = RollingTrain(value,lookback_period,holding_period)

# We collect all information for assets and then check our return.
portfolio = pd.DataFrame()
port = pd.DataFrame()
for market in markets:
    for key,value in market.items():
        portfolio[key] = value.Close
        portfolio[key+'lr'] = np.log(value.Close/value.Close.shift(1))
        portfolio[key+'position'] = value.position
        portfolio[key+'return'] = portfolio[key+'lr']*portfolio[key+'position']
        port[key+'return'] = portfolio[key+'lr']*portfolio[key+'position']

# 5Y treasury rate used as risk-free rate.
rf = 0.0084
def stats(series,rf):
    stats = pd.DataFrame()
    stats.loc[0,'cumulative simple return'] = series.sum()
    stats.loc[0,'vol'] = series.std()
    stats.loc[0,'sharp ratio'] = (series.sum() - rf*5)/series.std()
    stats.loc[0,'Max DD'] = (series - series.cummax()).min()
    return stats

total_return = exp(np.sum(port.sum()))-1
port['portfolio'] = np.exp(port).sum(axis=1) - port.shape[1]
stat = stats(port.portfolio,rf)