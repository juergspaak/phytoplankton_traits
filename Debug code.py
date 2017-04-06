# -*- coding: utf-8 -*-
"""
These file has possibly no stable functions, it is only to try stuff
"""

def res_absorb_growth(N,t,resident, precision = 0):
    """computes the growthrate when only one species is absorbing
    
    This is done by a tylor approximation up to 15 terms. The code is
    equivalent N*np.polyval(alphas[resident], N[resident]), but about
    10-20% faster"""
    print(1-N[0]/fun(t))
    N_values = fun(t)**exponent[precision:]
    return N*(sum(N_values*alphas[resident][precision:,:]))
    
def res_absorb_growth2(N,t,resident, precision = 0):
    """computes the growthrate when only one species is absorbing
    
    This is done by a tylor approximation up to 15 terms. The code is
    equivalent N*np.polyval(alphas[resident], N[resident]), but about
    10-20% faster"""
    N_values = fun(t)**range(16)[::-1]
    a=alphas[resident][precision:,:]
    return [N[0]*(sum(N_values*a[:,1])),0]
    

N_time2 = odeint(res_absorb_growth, [10.0**8,10.0**5],time[:25], args = (resi,))
