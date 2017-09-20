"""
@author: Jurg W. Spaak
Computes the boundary growth for both species with analytical solution
and continuous time
Assumes that I_out = 0
"""

def cumsimps(y, x=None, dx=1.0, axis=-1, initial = None):
    """Cumulatively integrate y(x) using the composite simpson rule.
    
    Parameters:	
        y : array_like
            Values to integrate.
        x : array_like, optional
            The coordinate to integrate along. If None (default), use spacing 
            dx between consecutive elements in y.
        dx : int, optional
            Spacing between elements of y. Only used if x is None.
        axis : int, optional
            Specifies the axis to cumulate. Default is -1 (last axis).
        initial : scalar, optional
            If given, uses this value as the first value in the returned 
            result. Typically this value should be 0. Default is None, which
            means no value at x[0] is returned and res has one element less 
            than y along the axis of integration.
        
        Returns:	
            res : ndarray
                The result of cumulative integration of y along axis. 
                If initial is None, the shape is such that the axis of 
                integration has one less value than y. If initial is given, the
                shape is equal to that of y."""
    if y.shape[axis]%2==0:
        raise ValueError("y must have an odd number of elements in axis")
    if x is None:
        x = dx*np.ones(int((y.shape[axis]-1)/2))
        x.shape = (-1,)+(y.ndim-axis%y.ndim-1)*(1,)
    end = y.shape[axis]
    fa = np.take(y,range(0,end-1,2), axis = axis)
    fab = np.take(y,range(1,end,2), axis = axis)
    fb = np.take(y,range(2,end,2), axis = axis)
    Sf = x/6*(fa+4*fab+fb)
    if initial is None:
        out = np.cumsum(Sf, axis = axis)
    else:
        shape = list(Sf.shape)
        shape[axis] +=1
        out = np.full(shape,initial,dtype = float)
        idx = [slice(None)]*y.ndim
        idx[axis] = slice(1,None)
        out[tuple(idx)] = np.cumsum(Sf,axis = axis)
    return out