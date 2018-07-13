"""
@author: J. W. Spaak, jurg.spaak@unamur.be
contains two functions to load/generate pigments
random_pigments: generates n random pigments
real_pigments: loads n predefined (in nature occuring) pigments
random_pigment and realpigments have the same 'in' and 'out' parameters"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

lambs, dlam = np.linspace(400,700,101, retstep = True)


gp_data = pd.read_csv("gp_kupper.csv")
absorptivity = pd.read_csv("absorptivity_kupper.csv")
names_pigments = list(absorptivity["Pigment"])

a = gp_data.iloc[::3,2:].values
xp = gp_data.iloc[1::3, 2:].values
sig = gp_data.iloc[2::3,2: ].values

kuepper = np.nansum(a*np.exp(-0.5*((xp-lambs.reshape(-1,1,1))/sig)**2),-1).T
kuepper *= absorptivity.iloc[:,1].reshape(-1,1) 

# load additional pogments from vaious authors
df_pigs = pd.read_csv("additional_pigments.csv") 
add_pigs = np.empty((df_pigs.shape[-1]//2, len(lambs)))
names_pigments.extend(df_pigs.columns[1::2])

for i,pigment in enumerate(df_pigs.columns[1::2]):
    x = df_pigs["lambs, " + pigment][np.isfinite(df_pigs["lambs, " + pigment])]
    y = df_pigs[pigment][np.isfinite(df_pigs[pigment])]
    add_pigs[i] = interp1d(x,y)(lambs)

# multiply with absorptivity
add_pigs /= np.nanmax(add_pigs, axis = 1, keepdims = True)
ref_chla = np.nanmax(kuepper[4])
absorptivity = ref_chla*np.array([2.0, 1, 1.5, 0.8,0.8,0.5,0.5,0.5,0.5])[:,np.newaxis]
    
pigments = np.append(kuepper, absorptivity*add_pigs, axis = 0)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.plot(pigments.T)