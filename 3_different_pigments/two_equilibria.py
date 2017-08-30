""" This program is to find a set of species, that allow two stable equilibria
with different species present at the equilibrium"""

def comp_beta(alpha):
    """compute the linear coefficients for the equation 0 = \sum \beta_r*k_r
    
    alpha: Coefficients of pigments in each species
        k_r = \sum_i alpha_(r,i)*pig_i
    alpha has shape (r_pig, r_pig+1,n_com), where r_pig is the richness of the pigments"""

    # create matrix:
    mat = np.ones((alpha.shape[0]+1, alpha.shape[1]+1, alpha.shape[2]))
    mat[1:,:-1] = alpha
    mat[1:,-1] = 0

    # Transform to upper triagnle matrix
    for i in range(len(mat)-1):
        mat[i+1:] -= mat[i+1:,i,None]/mat[i,i]*mat[i]

    
    # solve tirangular matrix
    beta = np.empty((len(mat), alpha.shape[-1]))
    for i in range(len(mat))[::-1]:

        beta[i] = (mat[i,-1]-np.sum(beta[i+1:]*mat[i,i+1:-1],axis=0))/mat[i,i]
    
    # check result
    print(np.amax(np.abs(np.sum(beta*alpha,1))))

pig_rich = 5
alpha = np.random.random((pig_rich, pig_rich+1,100))

comp_beta(alpha)