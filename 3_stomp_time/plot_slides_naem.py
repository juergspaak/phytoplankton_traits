"""
creates the plots for the powerpoint of the NAEM talk"""

import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# First figure, Fourier decomposition of c-chord
x = np.linspace(0,4*np.pi,500)

C = 9*np.sin(1*x)
E = 3*np.sin(3*x)
G = 1*np.sin(9*x)



fig, ax = plt.subplots(figsize = (9,7))
plt.xticks(np.pi*np.arange(9)/2)
ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$',
                     r'$\pi$',r'$\frac{3\pi}{2}$',
                    r'$2\pi$',r'$\frac{5\pi}{2}$',
                    r'$3\pi$',r'$\frac{7\pi}{2}$',r'$4\pi$'], fontsize = 18)
plt.axis([0,x[-1],-10,10])
plt.plot(x, C,'g',label = "C")
plt.legend(loc = "upper right")
plt.savefig("Figure, C string.pdf")
plt.plot(x, E,'r', label = "E")
plt.plot(x, C+E,'b', linewidth = 2.0, label = "C+E")
plt.legend(loc = "upper right")
plt.savefig("Figure, C,E strings.pdf")

fig, ax = plt.subplots(figsize = (9,7))
plt.xticks(np.pi*np.arange(9)/2)
ax.set_xticklabels([r'$0$', r'$\frac{\pi}{2}$',
                     r'$\pi$',r'$\frac{3\pi}{2}$',
                    r'$2\pi$',r'$\frac{5\pi}{2}$',
                    r'$3\pi$',r'$\frac{7\pi}{2}$',r'$4\pi$'], fontsize = 18)


plt.plot(x, C,'g', label = "C")
plt.plot(x, E,'r', label = "E")
plt.plot(x, G,'c', label = "G")
plt.plot(x, C+E+G,'b',linewidth = 2.0, label = "C chord")
plt.legend(loc = "upper right")
plt.savefig("Figure, chord.pdf")



###############################################################################
# Second Figure, Fourier decomposition of pigments

lambs = np.linspace(400,700,301)


from scipy.interpolate import interp1d
path = "../../2_Data/3. Different pigments/"
phycoer = np.genfromtxt(path+"phycoer_stomp.csv", delimiter = ',').T
phycoer = interp1d(phycoer[0], phycoer[1],kind = "linear")(lambs[::6])
phycoer = interp1d(lambs[::6], phycoer,kind = "cubic")(lambs)

phycocy = np.genfromtxt(path+"phycocy_stomp.csv", delimiter = ',').T
phycocy = interp1d(phycocy[0], phycocy[1],kind = "linear")(lambs[::6])
phycocy = interp1d(lambs[::6], phycocy,kind = "cubic")(lambs)

chlo_a = np.genfromtxt(path+"chloro_stomp.csv", delimiter = ',').T
chlo_a = interp1d(chlo_a[0], chlo_a[1],kind = "linear")(lambs[::6])
chlo_a = interp1d(lambs[::6], chlo_a,kind = "cubic")(lambs)

fig, ax = plt.subplots(2,sharex = True, sharey = True, figsize = (9,7))
plt.axis([400,700,0,1.2])
ax[0].plot(lambs, phycocy,'c:', label = "Phycocyanin")
ax[0].plot(lambs, phycocy+chlo_a, 'b', linewidth = 2)
ax[0].plot(lambs, chlo_a-0.05,'g:')
plt.yticks([0.0,0.5,1.0])
plt.xticks([400,450,510,570,650,700])
ax[0].set_xticklabels([400,"blue", "green", "yellow", "red",700], fontsize = 18)


ax[1].plot(lambs, phycoer,'r:',label = "phycoerithrin")

ax[1].plot(lambs, chlo_a-0.05,'g:', label = "Chlo-A")
ax[1].plot(lambs, phycoer+chlo_a, 'b', linewidth = 2, label = "Absorption")
plt.legend(loc = "upper right")
plt.yticks([0.0,0.5,1.0])

plt.savefig("Figure, Fourier of pigments.pdf")

###############################################################################
# Third Figure, Real pigments occuring in nature

from load_pigments import real
plt.figure()
x = np.linspace(400,700,301)
gaus = lambda l_mean: 0.7*np.exp(-(l_mean-x)**2/200)
plt.plot(x, gaus(450))
plt.plot(x, gaus(500))
plt.plot(x, gaus(550))
plt.plot(x, gaus(600))
plt.plot(x, gaus(650))
plt.xticks([400,450,510,570,650,700])
plt.yticks([0,0.5,1.0])
plt.gca().set_xticklabels([400,"blue", "green",
                            "yellow", "red",700])
plt.savefig("Figure, optimal pigments.pdf")

plt.figure()
plt.plot(np.linspace(400,700,101),1.2e8*real.T)
plt.axis([400,700,0,1])
plt.xticks([400,450,510,570,650,700])
plt.yticks([0,0.5,1.0])
plt.gca().set_xticklabels([400,"blue", "green",
                            "yellow", "red",700])

plt.savefig("Figure, All real pigments.pdf")


###############################################################################
# Fourth Figure, species vs pigments richness
import pandas as pd
#datas = pd.read_csv("../3_stomp_time/data/data_random_continuous_all.csv")

datas = datas[datas.pigments == "real"]
datas = datas[datas.case == "Const1"]
# percentiles to be computed
perc = np.array([5,50,95])
percentiles = np.empty((21,len(perc)))
for i in range(21): # compute the percentiles
    index = datas.r_pig == i
    try:
        percentiles[i] = np.percentile(datas[index].s_div,perc)
    except IndexError:
        percentiles[i] = np.nan
percentiles[1] = 1

# plot the percentiles
color = ['orange', 'yellow', 'purple', 'blue','black']
plt.figure()
plt.xlim([0.8,20.2])
#plt.ylim([0.8,3])
plt.xticks([1,10,20])
plt.yticks([])
plt.ylim(0.8,20.2)
plt.xlabel("Pigments present in community")
plt.ylabel("Species presenent in community")
plt.savefig("Figure, numerical_data, empty.pdf")
plt.plot(np.arange(1,21),np.arange(1,21), label = "optimal pigments")
plt.yticks([1,10,20])
plt.savefig("Figure, numerical_data, optimal pigments.pdf")

plt.plot(np.arange(21),percentiles[:,1],'o',color = "black", label = "median, real")
plt.yticks([1,2,10,20])
plt.legend(loc = "upper left")
plt.savefig("Figure, numerical_data, median.pdf")

plt.plot(np.arange(21),percentiles[:,0],'go', label = "5%, real")

plt.plot(np.arange(21),percentiles[:,2],'ro', label = "95%, real")
plt.yticks([1,2,3,10,20])
plt.legend(loc = "upper left")
plt.savefig("Figure, numerical_data, all.pdf")



 