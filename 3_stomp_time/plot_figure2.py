# -*- coding: utf-8 -*-
"""
@author: J.W.Spaak
Plot the main Figure of the paper
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")

# load the dataset
spaak_data = pd.read_csv("data/data_random_EF_fluct_all.csv")

# add phylum diversity and species diversity at the beginning
spaak_data["phylum_diversity"] = [len(set(spec[1:-1].split())) 
                                        for spec in spaak_data.species]
                                        
spaak_data["species_diversity"] = [len(spec[1:-1].split()) 
                                        for spec in spaak_data.species]

# convert to integer
spaak_data.r_pig_start = spaak_data.r_pig_start.astype(int)                                        

# actual plotting                                      
fig,ax = plt.subplots(3,1, sharey = True, figsize = (9,9))

# only use one constant light
index = np.append(np.arange(0,len(spaak_data),5),
                      np.arange(4,len(spaak_data),5))
spaak_data = spaak_data.iloc[index,:]

# Plot pigment versus species richness
sns.violinplot(x = "r_pig_start", y = "r_spec_equi", data = spaak_data,
                 hue = "case",split = True,inner = None,cut = 0, ax = ax[0])
# change legend
L = ax[0].legend(loc = "upper left")
L.get_texts()[0].set_text("Constant light")
L.get_texts()[1].set_text("Fluctuating light")

# plot species richness at start vs. at end
sns.violinplot(x = "species_diversity", y = "r_spec_equi", data = spaak_data,
                 hue = "case",split = True,inner = None,cut = 0, ax = ax[1])
ax[1].legend_.remove()

# plot phylum richness vs. species richness
sns.violinplot(x = "phylum_diversity", y = "r_spec_equi", data = spaak_data,
                 hue = "case",split = True,inner = None,cut = 0, ax = ax[2])
ax[2].legend_.remove()

# change axes labeling
ax[0].set_ylabel("Species richness")
ax[1].set_ylabel("Species richness")
ax[2].set_ylabel("Species richness")

ax[0].set_xlabel("Pigment richness")
ax[1].set_xlabel("Species richness (start)")
ax[2].set_xlabel("Phylum richness (start)")
fig.savefig("Figure, trait species diversity.pdf")
