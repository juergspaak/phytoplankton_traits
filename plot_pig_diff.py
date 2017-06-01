import numpy as np
import help_functions_chesson as ches
import different_pigments as dpig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import default_timer as timer

#parameters for the run
fit = 2**np.linspace(-1,1,100)
ratio = np.linspace(0,1,50)
avefit = 1.38e8
abs_fit = avefit/np.sqrt(fit)

#compute equilibriumdata
start = timer()
equis = dpig.equilibrium([ches.k_green, ches.k_red], ratio,abs_fit)
print(timer()-start)

#plot equilibrium data
fig = plt.figure(figsize = (7,7))
X,Y = np.meshgrid(np.log(abs_fit), ratio)
ax = fig.gca(projection = '3d')
ax.plot_wireframe(Y,X,equis)
ax.set_xlabel('ratio of pig 1')
ax.set_ylabel('Fitness (log)')
ax.set_zlabel('Equilibrium density')


#compute pigment distance
pd = dpig.pigments_distance([ches.k_green, ches.k_red], ratio,fit,approx = False
                           ,avefit = avefit)
coex_trip = np.where(pd>0) #case where invasion is possible
nocoex_trip = np.where(pd<0) #no coexistence possible
#invert species 1 and two to switch invasion and resident
nocoex_trip = [nocoex_trip[1], nocoex_trip[0], len(fit)-1-nocoex_trip[2]]
# find all points that are only in coex_trip and not in nocoex_trip
coex_set = set([tuple(x) for x in np.array(coex_trip).T])
nocoex_set = set([tuple(x) for x in np.array(nocoex_trip).T])
coex = np.array([x for x in coex_set-nocoex_set]).T
                
# converst to arrays for plotting                
xs = np.array([ratio[i] for i in coex[0]])
ys = np.array([ratio[i] for i in coex[1]])
zs = np.array([fit[i] for i in coex[2]])
#plot all points
"""
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111,projection= '3d')
ax.scatter(xs,ys,np.log(zs),s = 1, color = 'blue')
ax.set_xlabel('ratio invader')
ax.set_ylabel('ratio resident')
ax.set_zlabel('relative fitness (log)')
ax.set_zlim(np.amin(np.percentile(np.log(zs),10)), np.amax(np.percentile(np.log(zs),90)))
ax.set_xlim(0,1)
ax.set_ylim(0,1)"""

#to save the maximum values for fitness
maxz = len(fit)*np.ones([len(ratio), len(ratio),2])
co = coex.T
for i in range(len(ratio)):
    for j in range(len(ratio)):
        try: 
            maxz[i,j,0] = np.amin(co[np.logical_and(co[:,0]==i, co[:,1]==j)][:,2])
            maxz[i,j,1] = np.amax(co[np.logical_and(co[:,0]==i, co[:,1]==j)][:,2])
        except ValueError:
            pass

maxz = maxz.astype(int)        
fit = np.append(fit, np.nan)
fig = plt.figure(figsize = (7,7))
X,Y = np.meshgrid(ratio, ratio)
ax = fig.gca(projection = '3d')


ax.set_xlabel('ratio invader')
ax.set_ylabel('ratio resident')
ax.set_zlabel('relative fitness (log)')
ax.set_zlim(np.amin(np.percentile(np.log(zs),2)), np.amax(np.percentile(np.log(zs),98)))
ax.set_xlim(0,1)
ax.set_ylim(0,1)
#ax.plot_wireframe(X,Y,np.zeros(X.shape),color='red', rstride = 5, cstride = 5)
ax.plot_wireframe(X,Y,np.log(np.array([fit[i] for i in maxz[:,:,0]])), 
                     color = 'green', rstride = 5, cstride = 5)
ax.plot_wireframe(X,Y,np.log(np.array([fit[i] for i in maxz[:,:,1]]))
                , rstride = 5, cstride = 5)