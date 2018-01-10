"""@ autor: J.W.Spaak
This file brings together the data files created by generate_data files
"""
import numpy as np
import pandas as pd


def full_fact():
    ###########################################################################
    # full factorial data
    periods = ["10", "20","50","100"]
    pigments = ["real", "rand"]
    
    keys = [per+","+pig for per in periods for pig in pigments]
    datas = {key: pd.read_csv("data_period_"+key+".csv") for key in keys}
    
    tot_data = pd.concat(datas)
    tot_data["period"] = np.repeat([int(period) for period in periods],
                                    len(pigments)*12800)
    tot_data["pigments"] = np.tile(np.repeat(pigments,12800), len(periods))
    
    columns = ["r_pig", "r_spec", "r_pig_spec", "fac", "I_in", "case", "period"
               ,"pigments"] + [str(i+1) for i in range(10)]
    tot_data.to_csv("data_fullfact.csv",columns = columns)

def random(string,n):
    ###########################################################################
    # Original data, with linear interpolation of incoming light
    datas_all = pd.DataFrame()
    save = "data/data_random_{},{}.csv"
    for i in range(1,n+1):
        datas_all = datas_all.append(pd.read_csv(save.format(string,i)),
                                     ignore_index = True)
    del datas_all["Unnamed: 0"]
    datas_all.to_csv("data/data_random_{}_all.csv".format(string))
    return datas_all