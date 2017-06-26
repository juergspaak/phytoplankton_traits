"""computes the \DeltaI part of r_i"""

def ode_for_autograd(fun, x,t,end_point,h,*args):
    """solves an ode with adams method,can be differentiated with autograd
    
    first two timesteps are done with midpoint rule (h/2).
    Solver is of order 3
    Returns F(end_point)
    fun must be a function
    x is the startcondition
    t0 is the starting time
    end_point is the end time"""
    f1 = fun(x,t)
    #compute starting values for adams with midpoint and h/2
    # because midpoint has order 2
    x1 = midpoint_rule(fun,midpoint_rule(fun,x,t,h/2,),t+h/2,h/2)
    x = midpoint_rule(fun,midpoint_rule(fun,x1,t+h,h/2),t+3*h/2,h/2)
    f2 = fun(x1,t+h)
    t = t+2*h
    #iteratively use adams until final timestep is smaller than h
    while end_point-t>h:
        x,f2,f1 = adams_method(x,t,h,fun,f1=f1,f2=f2)
        t = t+h
    #last step is done with midpoint, because adams cannot change stepsize
    return midpoint_rule(fun, x,t,end_point-t)
    
def adams_method(x,t,h,fun,*args,f1=None, f2=None):
    """computes one step of the adams method, order = 3
    
    x initial condition
    f1,f2 = diff(x_(n-1),t-h), diff(x_(n-2), t-2h))
        if f1,f2 is None, x must be [x_n, x_(n-1),x_(n-2)]
    t is the current time, F(t) = x
    h is the time step
    diff is the differential fucntion
    """
    if f1 is None:
        try:
            f1,f2 = fun(x[-3],t-2*h,*args), fun(x[-2], t-h,*args)
        except TypeError:
            raise InputError("you have to either specify f = "+
            "[diff(x_(n-1),t-h), diff(x_(n-2), t-2h))] or "+
            "x = [x_n, x_(n-1),x_(n-2)]" )
    
    f3 = fun(x,t)
    return x+h/12*(23*f3-16*f2+5*f1), f3,f2

def midpoint_rule(fun,x,t,h):
    """ implements the midpoint rule differential solve of order 2
    
    diff is the differential equation, must be a function
    x is the starting condition
    t is the current time
    h is the stepwidth
    *args are addittional arguments for the function
    butcher tableau is:
        """
    return x+h*fun(x+h/2*fun(x,t),t+h/2)
    
def slow_r_i(C,E, species,P,carbon):
    """computes r_i for the species[:,0] assuming species[:,1] at equilibrium
    
    similar to r_i, but uses own ode_solver, which is differentiable by
    autograd
    does this by solving the differential equations
    C should be the density of the resident species, E the incomming light
    P the period length, carbon = [carbon[0],carbon[1]] the carbon uptake 
    functions of both species
    
    returns r_i for this period"""
    end_dens = ode_for_autograd(dwdt, [1,C], [0,P], (species, E, carbon))
    return np.log(end_dens[0]/1)/P #divide by P, to compare different P
    
def dwdt(W,t,species, I_in,carbon):
    """computes the derivative, either for one species or for two
    biomass is the densities of the two species,
    species contains their parameters
    light0 is the incomming light at this given time"""
    dwdt = np.zeros(len(W))
    k,l = species[[0,-1]]
    abso = k[1]*W[1]
    dwdt[0] = k[0]*W[0]/abso*quad(lambda I: carbon[0](I)/(k[0]*I),I_in*math.exp(-abso), I_in)[0]-l[0]*W[0]   
    dwdt[1] = quad(lambda I: carbon[1](I)/(k[1]*I),I_in*math.exp(-abso), I_in)[0]-l[1]*W[1]
    return dwdt