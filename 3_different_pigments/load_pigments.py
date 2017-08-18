"""
@author: J. W. Spaak, jurg.spaak@unamur.be
contains two functions to load/generate pigments
random_pigments: generates n random pigments
real_pigments: loads n predefined (in nature occuring) pigments
random_pigment and realpigments have the same 'in' and 'out' parameters"""

import numpy as np
from scipy.interpolate import interp1d

def random_pigments(n):
    """randomly generates `n` absorption spectra
    
    n: number of absorption pigments to create
    
    Returns: pigs, a list of functions. Each function in pigs (`pigs[i]`) is
        the absorption spectrum of this pigment.
    `pig[i](lam)` = sum_peak(gamma*e^(-(lam-peak)**2/sigma)"""
    #number of peaks for each pigment:
    npeak = 2+np.random.randint(5,size = n)
    # location of peaks
    peak = [np.random.uniform(400,700,(1,npeak[i])) for i in range(n)]
    # shape of peack
    sigma = [np.random.uniform(100,900, size = (1,npeak[i])) for i in range(n)]
    # magnitude of peak
    gamma = [np.random.uniform(0,1, size = (1,npeak[i])) for i in range(n)]
    pigs = []
    for i in range(n):
        # pig[i](lam) = sum_peak(gamma*e^(-(lam-peak)**2/sigma)
        pig = lambda lam, i = i: np.sum(gamma[i]* #i=i to have different pigs
            np.exp(-(lam.reshape(lam.size,1)-peak[i])**2/sigma[i]),axis = 1)
        pigs.append(pig)
    return pigs

def real_pigments(n):
    """Loads n predefined pigments, similar to random_pigment_generator"""
    path = "../../2_Data/3. Different pigments/Pigs for python.csv"
    pig_data = (np.genfromtxt(path, delimiter = ',').T)
    lams = np.linspace(400,700,151)
    #i=i to have different pigs
    return [lambda lam, i=i: 10**-7*interp1d(lams, pig_data[i])(lam)
                        for i in range(n)]
    
real = real_pigments(29)
rand = random_pigments(29)