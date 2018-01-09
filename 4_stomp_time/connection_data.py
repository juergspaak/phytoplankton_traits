"""@ autor: J.W.Spaak
This file brings together the data files created by generate_data files
"""
import numpy as np
import pandas as pd

###############################################################################
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

###############################################################################
# Original data, with linear interpolation of incoming light
numbers = range(1,4)
max_count = [40000,40000,40000]
datas_rand = {}
for i,num in list(enumerate(numbers)):
    data = pd.read_csv("data_random_org"+str(num)+".csv")
    datas_rand[str(num)] = data[1:max_count[i]]
    
columns += ["loc1","loc2", "lux1", "lux2", "sigma1", "sigma2"]                
tot_data_rand = pd.concat(datas_rand)
tot_data_rand.to_csv("data_random_org.csv")

###############################################################################
# second data, with linear interpolation of complex 
numbers = range(1,3)
max_count = [8000,8000]
datas_rand = {}
for i,num in list(enumerate(numbers)):
    data = pd.read_csv("data_random_second"+str(num)+".csv")
    datas_rand[str(num)] = data[1:max_count[i]]
            
tot_data_rand = pd.concat(datas_rand)
tot_data_rand.to_csv("data_random_sec.csv")