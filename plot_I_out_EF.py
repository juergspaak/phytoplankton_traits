import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset
try:
    data_photo
except NameError:
    data_photo = pd.read_csv("data/data_no_photoprotection0.csv")
    for col in data_photo.columns:
        if col[:2] == "EF":
            data_photo[col] *=1e-8
    
I_out_cols = ["I_out, t=240", "I_out, equi"]
biovol_cols = ["EF_t=240", "EF_equi"]

I_outs, biovol = np.empty((2,len(I_out_cols), max(data_photo.r_pig_start)))
r_pigs = 1+ np.arange(max(data_photo.r_pig_start))

for i, time in enumerate(I_out_cols):
    for j, rp in enumerate(r_pigs):
        I_outs[i,j] = np.nanmean((data_photo[I_out_cols[i]]/data_photo.lux)
                      [data_photo.r_pig_start == rp])
        biovol[i,j] = np.nanmean(
                data_photo[biovol_cols[i]][data_photo.r_pig_start == rp])
        
fig,ax = plt.subplots(1,2,sharex = True)
ax[0].set_xticks([2,5,10])

colors = ["lime", "green"]
for i in range(2):
    ax[1].plot(r_pigs, I_outs[i], '.', color = colors[i])
    ax[0].plot(r_pigs, biovol[i], '.', color = colors[i])
    
ax[0].semilogy()

ax[0].set_title("A")
ax[1].set_title("B")


ax[0].set_ylabel("Biovolume [$fl \,ml^{-1}$]")
ax[0].set_xlabel("initial pigment richness")
ax[1].set_xlabel("initial pigment richness")
ax[1].set_ylabel("Percentage of absorbed light")

fig.tight_layout()
fig.savefig("Figure_ecosystemfunction.pdf")