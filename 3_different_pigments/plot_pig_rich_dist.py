"""
@author: J.W.Spaak, jurg.spaak@unamur.be
plot the percentile curves of coexisting species depending on the pigments-
richness"""

import matplotlib.pyplot as plt
import numpy as np

import multispecies_functions as mc
from load_pigments import rand as l_pigs


def plt_ncoex(equis, pig_rich):
    """plots the amount of different species in a percentile curve
    
    equis: 2-dim array
        equis[i] should be the equilibrium densities of the species in
        iteration i
    
    Returns: None
    
    Plots: A percentile plot of the number of species that coexist in each
        iteration
        
    Example: 
        import load_pigments as pigments
        plt_ncoex(multispecies_equi(pigments.real)[0])"""
    spec_num = np.sum(equis>0,0)
    fig = plt.figure()
    plt.plot(np.linspace(0,100, len(spec_num)),sorted(spec_num))
    plt.ylabel("number coexisting species")
    plt.xlabel("percent")
    plt.title("pigment richness ="+str(pig_rich))
    plt.show()
    return fig
    
def plot_percentile_curves():
    """plots the percentile curves (5,25,50,75 and 95), of the number of coexisting
    species in dependence of the number of pigments"""
    equis = []
    unfixeds = []
    pigs_richness = np.arange(2,21,2)
    for i in pigs_richness:
        n_com = 500 # number of communities
        r_spec = 2*i # richness of species, regional richness
        r_pig = i # richness of pigments in community
        r_pig_spec = min(i,5) # richness of pigments in each species
        fac = 2 #factor by which fitness can differ
        phi = 2*1e8*np.random.uniform(1/fac, 1*fac,(r_spec,n_com)) #photosynthetic efficiency
        l = 0.014*np.random.uniform(1/fac, 1*fac,(r_spec,n_com)) # specific loss rate
        fitness = phi/l # fitness
        # spectrum of the species
        k_spec,alpha = mc.spectrum_species(l_pigs, r_pig, r_spec, 
                                           n_com, r_pig_spec) 
        equi, unfixed = mc.multispecies_equi(fitness, k_spec)
        equis.append(equi[:,np.logical_not(unfixed)]) #equilibrium density
        unfixeds.append(unfixed)
        plt_ncoex(equis[-1], i)
    spec_nums = [np.sum(equi>0,0) for equi in equis]    
    percents = np.array([[int(np.percentile(spec_num,per)) for per in [5,25,50,75,95]] for
                 spec_num in spec_nums])
    fig, ax = plt.subplots()
    leg = plt.plot(pigs_richness,percents, '.')
    ax.set_ylabel("number of coexisting species")
    ax.set_xlabel("number of pigments in the community")
    ax.legend(leg, ["5%","25%","50%","75%","95%"], loc = "upper left")
    ax.set_ybound(np.amin(percents)-0.1, np.amax(percents)+0.1)
    plt.figure()
    plt.plot(k_spec[:,:,0]) #plot a representative of the spectrum
    fig.savefig("Figure, pig_rich_dist.pdf")
    return equis, unfixeds, percents
    
plot_percentile_curves()