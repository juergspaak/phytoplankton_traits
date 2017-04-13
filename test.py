# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:46:20 2017

@author: spaakjue
"""
from scipy.integrate import ode

"""
N_timer = np.zeros([500,2])
N_timer[0] = N_start
itera = int(50)
h = 500.0/itera

for i in range(itera-1):
    N_timer[i+1] = N_timer[i]+h*growth(N_timer[i],1,resi)
"""   
    
class time_dependent():
    current = 100
    
    def time(self, t):
        if t%100 == 0:
            self.current = uni(50,200)
        return self.current
        
        
        
I_in = time_dependent()

print(I_in.time(1))
print(I_in.time(100))
print(I_in.time(150))
print(I_in.time(200))
