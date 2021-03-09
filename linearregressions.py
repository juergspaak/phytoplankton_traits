import numpy as np
import pandas as pd
from scipy.stats import linregress

try:
    data_org
except NameError:
    from load_data import data_org, max_spec
    
cols = ["x", "y", "slope", "intercept", "R2", "err"]
df = pd.DataFrame(columns = cols)
for x in ["r_pig_start", "size_cv"]:
    for y in ["r_spec_t=240", "r_spec_equi", "EF_t=240", "EF_equi"]:
        res = linregress(data_org[x], data_org[y])
        df = df.append({"x": x, "y": y, "slope": res[0], "intercept": res[1],
                        "R2": res[2]**2, "err": res[4]}, ignore_index = True)
        
for x in ["r_pig_start", "size_cv"]:
    for val in ["ND_{}", "FD_{}"]:
        y = data_org[[val.format(i) for i in range(5)]].values
        if val == "FD_{}":
            y = np.abs(y)
        x_val = np.repeat(data_org[x].values[:,np.newaxis], 5, axis = 1)
        ind = np.isfinite(x_val*y)
        y = y[ind]
        x_val = x_val[ind]
        res = linregress(x_val, y)
        df = df.append({"x": x, "y": val, "slope": res[0], "intercept": res[1],
                        "R2": res[2]**2, "err": res[4]}, ignore_index = True)
        
    
###############################################################################
# renaming and layouting
df = df.sort_values("x")

df.loc[df.x == "r_pig_start", "x"] = "Initial pigment richness"
df.loc[df.x == "size_cv", "x"] = "Initial CV(size)"

df.loc[df.y == "EF_t=240", "y"] = "EF $t=10$"
df.loc[df.y == "EF_equi", "y"] = "EF equi"
df.loc[df.y == "r_spec_t=240", "y"] = "Species, $t=10$"
df.loc[df.y == "r_spec_equi", "y"] = "Species, equi"
df.loc[df.y == "ND_{}", "y"] = "$\mathcal{N}_i$"
df.loc[df.y == "FD_{}", "y"] = "$|\mathcal{F}_i|$"

df["smin"] = df.slope - 2*df.err
df["smax"] = df.slope + 2*df.err

df["intercept"] = np.round(df["intercept"], 3)
df["slope"] = np.round(df["slope"], 4)
df["smin"] = np.round(df["smin"], 4)
df["smax"] = np.round(df["smax"], 4)
df["R2"] = np.round(df["R2"], 3)
del df["err"]
print(df)

df["Reference"] = ["3A; light green", "3A; dark green", 
                  "3B; light green", "3B; dark green",
                    "4A", "4B",
                    "3C; light green", "3C; dark green", 
                  "3D; light green", "3D; dark green",
                    "4C", "4D",]

df.to_csv("Table_linregressions.csv", index = False)