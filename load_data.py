import pandas as pd
import numpy as np

try:
    data_org
except NameError:
    data_org = pd.read_csv("data/data_photoprotection_size_all.csv")
    for col in data_org.columns:
        if col[:2] == "EF":
            data_org[col] *=1e-8
    data_org["complementarity"] *= 1e-8
    data_org["selection"] *= 1e-8
data_org.r_pig_start = data_org.r_pig_start.astype(int)
max_spec = int(np.nanpercentile(data_org.r_spec_equi, 99.9))



# indentify traits of surviving species
survivors = np.full((len(data_org), max_spec), np.nan)
for i in range(max_spec):
    survivors[data_org.r_spec_equi >= i+1, i] = 1



names = ["phi", "size", "abs", "n_pig"]    
for name in names:
    keys = ["{}_{}".format(name,i) for i in range(14)]
    
    # coefficient of variance of initial trait
    data_org[name + "_cv"] = (np.nanvar(data_org[keys], axis = 1)/
              np.nanmean(data_org[keys], axis = 1))
        
    # traits of survifing species
    for i in range(max_spec):
        data_org[name+ "_sur_{}".format(i)] = data_org[name + "_{}".format(i)]
    data_org[[name + "_sur_{}".format(i) for i in range(max_spec)]] *= survivors
    
    # coefficient of variance of traits at equilibrium
    keys = ["{}_sur_{}".format(name,i) for i in range(max_spec)]
    data_org[name + "_sur_cv"] = (np.nanvar(data_org[keys], axis = 1)/
              np.nanmean(data_org[keys], axis = 1)) 

# absolute value of fitness differences
for i in range(5):
    data_org["FD_abs_{}".format(i)] = np.abs(data_org["FD_{}".format(i)])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # plot summary statistics of data
    print(data_org.shape)
    
    fig, ax = plt.subplots(2,2, figsize = (7,7))
    ax = ax.flatten()
    
    for i, name in enumerate(names):
        keys = ["{}_{}".format(name,i) for i in range(14)]
        ax[i].set_title(name)
        bins = np.linspace(*np.nanpercentile(data_org[keys], [1,99]), 30)
        ax[i].hist(data_org[keys].values.flatten(), density = True, 
          label = "Initial", bins = bins)
        keys = ["{}_sur_{}".format(name,i) for i in range(max_spec)]
        ax[i].hist(data_org[keys].values.flatten(), density = True, 
          label = "Equilibrium", bins = bins, alpha = 0.5)
        ax[i].set_xticks(bins[[0,-1]])
    ax[0].legend()
        
    fig.tight_layout()
    
    phis = data_org[["phi_{}".format(i) for i in range(14)]].values
    abss = data_org[["abs_{}".format(i) for i in range(14)]].values
    fig, ax = plt.subplots(2, sharex = True, sharey = True)
    cmap = ax[0].hist2d(np.log(phis[np.isfinite(phis)]),
             np.log(abss[np.isfinite(abss)]),
      bins = 50, normed = True, vmin = 0.01)[-1]
    fig.colorbar(cmap, ax = ax[0])
    
    phis = data_org[["phi_sur_{}".format(i) for i in range(max_spec)]].values
    abss = data_org[["abs_sur_{}".format(i) for i in range(max_spec)]].values
    cmap = ax[1].hist2d(np.log(phis[np.isfinite(phis)]),
             np.log(abss[np.isfinite(abss)]),
      bins = 50, normed = True, vmin = 0.01)[-1]
    fig.colorbar(cmap, ax = ax[1])
    ax[1].set_xlabel("log(phi)")
    ax[0].set_ylabel("log(abs)")
    ax[1].set_ylabel("log(abs)")