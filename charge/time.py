"""Routine for retarded time"""

import numpy as np
from scipy.optimize import root, root_scalar

def tr(r, t, rs, c):
    """Evaluate retarded time in atomic unit"""
    def g(dt, r, t, rs, c): return np.sqrt(np.square(r-rs(t-dt)).sum(-1)) - c*dt
    _dt_trial = 0.0
    _dt = None
    if abs(g(_dt_trial, r, t, rs, c)) < 1e-15: _dt = _dt_trial
    else:
        _dt_trial = np.sqrt(np.square(r-rs(t)).sum(-1)) / c 
        _root_results = root_scalar(g, args=(r,t,rs,c), method='secant', x0=0.0, x1=_dt_trial)
        #_result = root(g, _dt_trial, args=(r,t,rs,c))
        #try: _dt, = fsolve(g, 0.0, args=(r,t,rs,c))
        #except: raise Exception("Failed to find retarded time")

#        if not _result.success:
#            print("_dt_trial: {}".format(_dt_trial))
#            _mesg = "Failed to find retarded time with 'OptimizeResult':\n{}"
#            raise Exception(_mesg.format(_result))
#        _dt, = _result.x
        
        if not _root_results.converged:
            _mesg = "Failed to find retarded time with 'RootResults':\n{}"
            raise Exception(_mesg.format(_root_results))
        _dt = _root_results.root


    assert _dt is not None
    return t - _dt

def tr_arr(r_arr, t, rs, c):
    """Evaluate retarded time for each position vector given in atomic unit"""
    _ndim = 3
    assert isinstance(r_arr, np.ndarray)
    assert r_arr.ndim >= 1 and r_arr.shape[-1] == _ndim
    _vec3_arr_shape = r_arr.shape[:-1]
    _num_of_vec3 = int(np.prod(_vec3_arr_shape))
    _r_arr_reshape = r_arr.reshape((_num_of_vec3,_ndim))
    try: _tr_arr_list = [tr(_vec3, t, rs, c) for _vec3 in _r_arr_reshape]
    except: raise Exception("Failed to evaluate retarded time for array")
    _tr_arr = np.array(_tr_arr_list).reshape(_vec3_arr_shape)
    return _tr_arr

