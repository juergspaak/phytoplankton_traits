""" computes the partial fraction decomposition for a given polynomial"""

from itertools import combinations

coefs = np.array([1,-3,2])

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
    


def test(coefs):
    print(quad(lambda x: 1/np.polyval(coefs,x)/coefs[0],-11,-2))
    roots = np.roots(coefs)
    print(quad(lambda x: 1/factor_poly(x,roots),-11,-2))
    print(integral_N(roots)(-2)-integral_N(roots)(-11))
    
test(np.array([1,0,2]))
    
def integral_N(roots):
    print(frac(roots))
    return lambda N: np.real(sum(frac(roots)*np.log(np.abs(N-roots))))

