import matplotlib.pyplot as plt
import numpy as np

# load the dataset
try:
    data_org
except NameError:
    from load_data import data_org, max_spec

data_ND = data_org[data_org.r_spec_equi != 1]
   
fig, ax = plt.subplots(2,2, sharex = "col", sharey = False, figsize = (7,7))

n_pig = data_ND[["n_pig_sur_{}".format(i) for i in range(max_spec)]].values
ND = data_ND[["ND_{}".format(i) for i in range(max_spec)]].values
FD = data_ND[["FD_{}".format(i) for i in range(max_spec)]].values
size = data_ND[["size_sur_{}".format(i) for i in range(max_spec)]].values

ind = np.isfinite(FD) & np.isfinite(ND) & np.isfinite(size)

n_pig = n_pig[ind]
ND = ND[ind]
FD = FD[ind]
size = size[ind]

bins_pig = np.arange(min(n_pig), max(n_pig)+1)

ax[0,0].boxplot([ND[n_pig == i] for i in bins_pig], positions = bins_pig,
  sym = "")
ax[1,0].boxplot([FD[n_pig == i] for i in bins_pig], positions = bins_pig,
  sym = "")


ranges,dr = np.linspace(0, 12, 16,
                            retstep = True)

ND_box = []
FD_box = []

for i in range(len(ranges)-1):
    ind = (size>ranges[i]) & (size<ranges[i+1])
    ND_box.append(ND[ind])
    FD_box.append(FD[ind])

ax[0,1].boxplot(ND_box, positions = ranges[1:]-dr/2,
  sym = "")
ax[1,1].boxplot(FD_box, positions = ranges[1:]-dr/2,
  sym = "")

ax[1,1].set_xticks([0,5,10])
ax[1,1].set_xticklabels([0,5,10])

ax[1,1].set_xlabel("log(size)")
ax[1,0].set_xlabel("Pigments per species")
ax[1,0].set_ylabel(r"$\mathcal{F}_i$")
ax[0,0].set_ylabel(r"$\mathcal{N}_i$")

ax[0,0].set_title("A")
ax[1,0].set_title("B")
ax[0,1].set_title("C")
ax[1,1].set_title("D")
fig.savefig("Figure_S3_NFD_individually.pdf")