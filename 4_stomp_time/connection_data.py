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

def random(string,n_start, n_end):
    ###########################################################################
    # Original data, with linear interpolation of incoming light
    n_start, n_end = int(n_start), int(n_end)
    if n_end is None:
        n_start, n_end = 1,n_start+1
    else:
        n_end +=1
    datas_all = pd.DataFrame()
    save = "data/data_random_{}{}.csv"
    try:
        datas_all = datas_all.append(pd.read_csv(save.format(string,"_all")),
                                     ignore_index = True)
    except OSError:
        print("File _all not found")
    for i in range(n_start, n_end):
        try:
            datas_all = datas_all.append(pd.read_csv(save.format(string,i)),
                                     ignore_index = True)
        except OSError:
            print("File {} not found".format(i))
    del datas_all["Unnamed: 0"]
    datas_all.to_csv("data/data_random_{}_all.csv".format(string))
    return datas_all

def pure_python(string, n_start, n_end = None):
    if n_end is None:
        n_start, n_end = 1,n_start+1
    else:
        n_end +=1
    save = "data/data_random_{}{}.csv"
    with open(save.format(string, "_new"),'a') as fout:
        with open(save.format(string,"_all")) as f:
            fout.write(f.read())
        for i in range(n_start, n_end):
            with open(save.format(string,i)) as f:
                next(f)
                fout.write(f.read())
                
                
import sys
random(sys.argv[1], sys.argv[2])