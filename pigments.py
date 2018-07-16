"""
@author: J. W. Spaak, jurg.spaak@unamur.be

Load absorption spectra of pigments found in nature
Load also pigments that are not phytosyntheticall active

Reference papers can be found in the Pigment_algae_table.csv
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# points used for simpson integration rule throught the programs
lambs, dlam = np.linspace(400,700,101, retstep = True)

# Load the gaussian peak method proposed by Kupper et al.
gp_data = pd.read_csv("pig_gp_kupper.csv")
absorptivity = pd.read_csv("pig_abs_kupper.csv")
names_pigments = list(absorptivity["Pigment"])

a = gp_data.iloc[::3,2:].values
xp = gp_data.iloc[1::3, 2:].values
sig = gp_data.iloc[2::3,2: ].values

kuepper = np.nansum(a*np.exp(-0.5*((xp-lambs.reshape(-1,1,1))/sig)**2),-1).T
kuepper *= absorptivity.iloc[:,1].reshape(-1,1) 

# load additional pogments from vaious authors
df_pigs = pd.read_csv("pig_additional.csv") 
add_pigs = np.empty((df_pigs.shape[-1]//2, len(lambs)))


for i,pigment in enumerate(df_pigs.columns[1::2]):
    x = df_pigs["lambs, " + pigment][np.isfinite(df_pigs["lambs, " + pigment])]
    y = df_pigs[pigment][np.isfinite(df_pigs[pigment])]
    add_pigs[i] = interp1d(x,y)(lambs)

# multiply with absorptivity
add_pigs /= np.nanmax(add_pigs, axis = 1, keepdims = True) #normalize
ref_chla = np.nanmax(kuepper[4]) # scale to absorptivity of chlorophyll a
absorptivity = ref_chla*np.array([1, 1.5, 0.8,0.8,0.5,0.5,0.5])[:,np.newaxis]

# combine the two pigment lists  
pigments = np.append(kuepper, absorptivity*add_pigs, axis = 0)
names_pigments.extend(df_pigs.columns[1::2])

if __name__ == "__main__":
    # plot the absorption spectra of all pigments for illustrations
    import matplotlib.pyplot as plt
    plt.plot(pigments.T)