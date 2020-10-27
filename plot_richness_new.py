import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', titlesize=16) 

fig = plt.figure(figsize = (9,4))

# load the dataset
try:
    data_photo
except NameError:
    data_photo = pd.read_csv("data/data_no_photoprotectionnew.csv")
    data_photo = pd.read_csv("data/data_no_photoprotection_old_pigments.csv")
    data_photo.r_pig_start = data_photo.r_pig_start.astype(int)
    
print(data_photo.shape)
plt.figure()
plt.hist(data_photo.r_pig_start, bins = np.arange(1,10))
    

fig = plt.figure()
key = "r_spec_equi"
r_pig = 1 + np.arange(max(data_photo.r_pig_start))
r_spec = 1+np.arange(max(data_photo[key]))

richness = np.empty((len(r_spec),len(r_pig)))
for i, rp in enumerate(r_pig):
    ind_p = data_photo.r_pig_start == rp
    print(i,rp, np.sum(ind_p))
    for j,rs in enumerate(r_spec):
        ind_s = data_photo[key] == rs
        richness[j,i] = np.sum(ind_s*ind_p)/np.sum(ind_p)
        
richness[richness<1e-3] = np.nan # raised warning concerns prexisting nan
# minimal value is 1 and not 0 as usually in plt.imshow
extent = list(r_pig[[0,-1]]+[-0.5,0.5]) + list(r_spec[[0,-1]]+[-0.5,0.5])
# dont plot the cases with ss = 0 or se = 0
im = plt.imshow(richness, interpolation = "nearest", extent = extent,
    origin = "lower",aspect = "auto",vmax = 1,vmin = 0)

# add averages to the figures
av_spe = np.nansum(richness*r_spec[:,np.newaxis],axis = 0)
plt.plot(r_pig[1:],av_spe[1:],'o', color = "blue")

plt.ylabel("Final species richness")
    
plt.xlabel("Initial pigment richness")

plt.xticks(r_pig)
plt.colorbar()