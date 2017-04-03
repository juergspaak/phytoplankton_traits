""" computes the partial fraction decomposition for a given polynomial"""
import numpy as np
from itertools import combinations

    
def poly_matrix(roots):
    n = len(roots)
    mat = np.array([np.poly(np.delete(roots,i)) for i in range(n)])
    return np.matrix.transpose(mat)    
    
def frac(roots):
    """returns the coefficients for the partial fraction decomposition"""
    mat = poly_matrix(roots)
    b = np.zeros(len(roots))
    b[-1]=1
    beta = np.linalg.solve(mat, b)
    return np.array(beta)

    
def integral_N(coefs):
    roots = np.roots(coefs)
    fracts = frac(roots)
    return lambda N: float(np.real(sum(fracts*np.log(N-roots)))/coefs[0])

rooter = np.roots(a)
fracts = frac(rooter)/a[0]
fprime_fun = lambda x: np.array([np.real(sum(fracts/(x-rooter)))])
f_relevant = integral_N(a)
f_solver = lambda x: f_relevant(x)-f_relevant(10**8)-1000
start = timer()
fsolve(f_solver, 4*10**8,fprime = fprime_fun)
print(timer()-start)
start = timer()
fsolve(f_solver, 4*10**8,fprime = lambda x: np.array([1]))
print(timer()-start)


