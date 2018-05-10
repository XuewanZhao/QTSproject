# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:11:40 2018

@author: Xuewan Zhao
"""

import numpy as np
import pandas as pd

def GetData():
    equity = pd.read_excel("equity etf.xlsx",skiprows = 1,index_col = 0)
    equity = equity.reset_index(drop = True)
    equity_names = pd.read_excel("equity etf.xlsx",header = None,skip_footer = len(equity)+1)
    equities = dict()
    i = 0
    while i < equity_names.shape[1]:
        equities[equity_names.loc[0,i]] = equity.iloc[:,i:i+4]
        equities[equity_names.loc[0,i]].columns = ['Open','Close','High','Low']
        i += 6
    ###########
    commodity = pd.read_excel("commodity etf.xlsx",skiprows = 1,index_col = 0)
    commodity = commodity.reset_index(drop = True)
    commodity_names = pd.read_excel("commodity etf.xlsx",header = None,skip_footer = len(commodity)+1)
    commodities = dict()
    i = 0
    while i < commodity_names.shape[1]:
        commodities[commodity_names.loc[0,i]] = commodity.iloc[:,i:i+4]
        commodities[commodity_names.loc[0,i]].columns = ['Open','Close','High','Low']
        i += 6
    ###########    
    bond = pd.read_excel("bond etf.xlsx",skiprows = 1,index_col = 0)
    bond = bond.reset_index(drop = True)
    bond_names = pd.read_excel("bond etf.xlsx",header = None,skip_footer = len(bond)+1)
    bonds = dict()
    i = 0
    while i < bond_names.shape[1]:
        bonds[bond_names.loc[0,i]] = bond.iloc[:,i:i+4]
        bonds[bond_names.loc[0,i]].columns = ['Open','Close','High','Low']
        i += 6
    ###########    
    realestate = pd.read_excel("real estate etf.xlsx",skiprows = 1,index_col = 0)
    realestate = realestate.reset_index(drop = True)
    realestate_names = pd.read_excel("real estate etf.xlsx",header = None,skip_footer = len(realestate)+1)
    REs = dict()
    i = 0
    while i < realestate_names.shape[1]:
        REs[realestate_names.loc[0,i]] = realestate.iloc[:,i:i+4]
        REs[realestate_names.loc[0,i]].columns = ['Open','Close','High','Low']
        i += 6
    ###########    
    currency = pd.read_excel("currency etf.xlsx",skiprows = 1,index_col = 0)
    currency = currency.reset_index(drop = True)
    currency_names = pd.read_excel("currency etf.xlsx",header = None,skip_footer = len(currency)+1)
    currencies = dict()
    i = 0
    while i < currency_names.shape[1]:
        currencies[currency_names.loc[0,i]] = currency.iloc[:,i:i+4]
        currencies[currency_names.loc[0,i]].columns = ['Open','Close','High','Low']
        i += 6
    return equities,commodities,bonds,REs,currencies