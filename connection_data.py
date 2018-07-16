"""
@author: J. W. Spaak, jurg.spaak@unamur.be

This file merges the csv files created from the sim* files into one single
csv file with the same name and "_all_" added to the file name
"""
import pandas as pd

def random(string,n_start, n_end = None):
    """Connects all files of the form data_{string}i.csv into one
    
    i ranges from n_start to n_end. If n_end is not given n_start is assumed to
    be 1 and n_end is equal to n_start
    
    Creates the file data_{string}_all.csv in the folder data"""
    # n_end not given, swap n_start and n_end
    if n_end is None:
        n_start, n_end = 1,n_start+1
    else:
        # convert to strings as given by the console
        n_start, n_end = int(n_start), int(n_end)
        n_end +=1
    datas_all = pd.DataFrame() # to store all data in
    save = "data/data_{}{}.csv"
    # concatenate all files
    for i in range(n_start, n_end):
        try:
            datas_all = datas_all.append(pd.read_csv(save.format(string,i)),
                                     ignore_index = True)
        except OSError:
            print("File {} not found".format(i))
    # save file
    del datas_all["Unnamed: 0"]

    datas_all.to_csv("data/data_{}_all.csv".format(string))
    return datas_all

import sys
random(*sys.argv[1:])
