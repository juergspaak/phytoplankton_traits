""" computes the partial fraction decomposition for a given polynomial"""

from itertools import combinations

coefs = np.array([1,2,1])

roots = np.roots(coefs)

def ncr_arr(arr,n):
    if n == 0:
        return 1
    combi = list(combinations(arr,n))
    return sum(np.prod(combi, axis = 1))
    
def factor_poly(x,roots):
    return np.prod(x-roots)
    
def poly_expand(roots):
    l = len(roots)
    coefs = np.array([ncr_arr(-roots,i) for i in range(l+1)])
    print(coefs)
    print(np.poly(roots))
    return lambda x: sum([x**i*coefs[l-i] for i in range(l+1)])
    
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
    
frac(np.array([1,-1]))

def test(roots,x):
    print("true", 1/factor_poly(x,roots))
    print(sum(frac(roots)/(x-roots)))
    
test(np.array([1,5,4]),1.5)