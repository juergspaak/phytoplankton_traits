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
    data_photo = pd.read_csv("data/data_photoprotection_cv_save.csv")
    data_photo.r_pig_start = data_photo.r_pig_start.astype(int)


r_pig = 1 + np.arange(max(data_photo.r_pig_start))

###############################################################################
# data analysis of the dataframe
print(data_photo.shape)
plt.figure()
plt.hist(data_photo.r_pig_start, bins = r_pig)
plt.xlabel("pigments richness")
plt.ylabel("number of communities")
###############################################################################
# plot relevant figure
fig, ax = plt.subplots(1,2,sharex = False, sharey = True, figsize = (7,5))


key = "r_spec_equi"

r_spec = 1+np.arange(max(data_photo[key])+1)

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
im = ax[0].imshow(richness, interpolation = "nearest", extent = extent,
    origin = "lower",aspect = "auto",vmax = 1,vmin = 0)

# add averages to the figures
av_spe = np.nansum(richness*r_spec[:,np.newaxis],axis = 0)
ax[0].plot(r_pig[1:],av_spe[1:],'o', color = "blue")

ax[0].set_ylabel("Final species richness")
    
ax[0].set_xlabel("Initial\npigment richness")

ax[0].set_xticks([2,5,10,r_pig[-1]])
ax[0].set_xlim([1.5,None])


###############################################################################
# effect of trait richness in size
x = "size_cv" # the x variable of interest
data_x = data_photo[data_photo[x] != 0]

bins_x = np.nanpercentile(data_x[x], [1,99])
bins_x, dx = np.linspace(*bins_x,27, retstep = True)

y = "r_spec_equi" # the y variable of interest
bins_y = np.arange(np.amax(data_x[y]+1))+1
richness, xedges, yedges = np.histogram2d(data_x[x],
                                          data_x[y],
                          bins = [bins_x, bins_y])

# normalize richness per starting size richness
richness = richness/np.sum(richness, axis = -1, keepdims = True)


# take running averages of the data to smoothen
weights = np.ones(2)/2
richness = np.array([np.convolve(rich, weights, mode = "valid")
        for rich in richness.T])
richness[richness<1e-3] = np.nan

cmap = ax[1].imshow(richness, vmin = 0, vmax = 1, aspect="auto",
       origin = "lower", extent = [*bins_x[[1,-1]],*(yedges[[0,-1]]-0.5)])

# show average of species richness
av_spe = np.nansum(richness.T*bins_y[:-1],axis = 1)
x_val = bins_x[:-1]+dx/2
ax[1].plot(bins_x[1:-1]+dx/2, av_spe, 'bo')

ax[1].set_xlabel("Initial\nCV(size)")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cmap, cax=cbar_ax)

fig.savefig("Figure_richness.pdf")