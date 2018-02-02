"""@ autor: J.W.Spaak
This file brings together the data files created by generate_data files
Script is supposed to be called from the console
"""
import numpy as np
import pandas as pd

def random(string,n_start, n_end = None):
    """Connects all files of the form data_random_{string}i.csv into one
    
    i ranges from n_start to n_end. If n_end is not given n_start is assumed to
    be 1 and n_end is equal to n_start
    
    Creates the file data_random_{string}_all.csv in the folder data"""
    # n_end not given, swap n_start and n_end
    if n_end is None:
        n_start, n_end = 1,n_start+1
    else:
        # convert to strings as given by the console
        n_start, n_end = int(n_start), int(n_end)
        n_end +=1
    datas_all = pd.DataFrame() # to store all data in
    save = "data/data_random_{}{}.csv"
    try: # is there already an _all file where data should be appended?
        datas_all = datas_all.append(pd.read_csv(save.format(string,"_all")),
                                     ignore_index = True)
    except OSError:
        print("File _all not found")
    # concatenate all files
    for i in range(n_start, n_end):
        try:
            datas_all = datas_all.append(pd.read_csv(save.format(string,i)),
                                     ignore_index = True)
        except OSError:
            print("File {} not found".format(i))
    # save file
    del datas_all["Unnamed: 0"]
    # add averages of richness
    num_data = np.array(datas_all[[str(i+1) for i in range(10)]])
    ave_data = (np.arange(1,11)*num_data).sum(axis = -1)
    datas_all["s_div"] = ave_data
    datas_all = datas_all[np.isfinite(datas_all.s_div)]
    datas_all.to_csv("data/data_random_{}_all.csv".format(string))
    return datas_all

import sys
random(*sys.argv[1:])
