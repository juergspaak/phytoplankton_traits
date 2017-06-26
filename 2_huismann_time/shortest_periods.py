# -*- coding: utf-8 -*-
"""
Can compute throwth rates for the shortest periods
"""
import generate_communities as com
import numpy as np
from scipy import stats
import scipy.integrate as integrate

def coex_test(species, P = 1e-4):
    spec1, spec2 =list(species[0]), list(species[1])
    
    comp1, stor1, r_i_1 = shortest_r_i(species,P)
    species = np.array([spec2, spec1])
    comp2, stor2, r_i_2 = shortest_r_i(species,P)
    return comp1, stor1, r_i_1, comp2, stor2, r_i_2

def shortest_r_i(species = None,P = 1e-4,  I_in = np.array([50,200])):
    """returns the boundary growth rate of the speceis
    if analyt = True, the boundary growth rate is computed analytically
    if species or P are not given they are randomly chosen"""
    spec_return = False
    if species is None:
        species = com.find_species()
        spec_return = True
        
    mr,qr = fit_species(species[1], I_in)
    Iav = np.average(I_in)
    Wm, WM = mr*I_in+qr
    Wav = mr*Iav+qr
    
    mc, qc = fit_comp(species)
    compe = -species[0][-1]*(Iav*mc+qc-1)
    l = species[:,-1]
    dmp_dp = lambda P: mr*l[1]*np.exp(-l[1]*P)/((1+np.exp(-l[1]*P))**3
                        *(1-np.exp(-l[1]*P)))**(1/2)
    stor_prov = l[0]/l[1]*mc*(I_in[1]-I_in[0])**2/(24*Wav)
                  
    stor = dmp_dp(P)*stor_prov
    
    if spec_return:
        return compe, stor, stor-compe, species
    return  compe, stor, stor-compe


