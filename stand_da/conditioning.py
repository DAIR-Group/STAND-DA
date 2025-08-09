import numba.cuda as cuda
import numpy as np
from . import layers
from . import util

def solve_linear_inequality(u, v): #u + vz < 0
    u = float(u)
    v = float(v)
    
    if (v > -1e-16 and v < 1e-16):
        if (u <= 0):
            return [-np.inf, np.inf]
        else:
            return None
        
    if (v < 0):
        return [-u/v, np.inf]
    
    return [-np.inf, -u/v]

def intersect(itv1, itv2):
    if itv1 is None or itv2 is None:
        return None
    
    itv = [max(itv1[0], itv2[0]), min(itv1[1], itv2[1])]
    
    if itv[0] > itv[1]:
        return None    
    return itv

def get_dnn_interval(x, a, b, list_layers):
    itv = np.asarray([-np.inf, np.inf])
    itv = cuda.to_device(itv)
    a = cuda.to_device(a)
    b = cuda.to_device(b)
    x = cuda.to_device(x)
    
    streams = [cuda.stream() for _ in range(3)]
    for name, param in list_layers:
        if name == 'Linear Weight':
            a, b, x = layers.LinearWeight(a, b, x, param, streams)
            for stream in streams:
                stream.synchronize()
        elif name == 'Linear Bias':
            a, b, x = layers.LinearBias(a, b, x, param, streams)
            for stream in streams:
                stream.synchronize()
        elif name == 'ReLU':
            a, b, x, itv = layers.Relu(a, b, x, itv, streams)
            for stream in streams:
                stream.synchronize()
                
    itv = itv.copy_to_host()
    a = a.copy_to_host()
    b = b.copy_to_host()
    return itv, a, b

def get_ad_interval(X, X_hat, X_tilde, reconstruction_loss, a, b, wdgrl, ae, alpha):
    itv = [-np.inf, np.inf]
    
    sub_itv, u, v = get_dnn_interval(X, a, b, wdgrl)
    itv = intersect(itv, sub_itv)
    
    sub_itv, p, q = get_dnn_interval(X_hat, u, v, ae)
    itv = intersect(itv, sub_itv)
    
    s = np.sign(X_tilde - X_hat)
    
    u_args = s * (u - p)
    v_args = s * (v - q)
    
    all_lower_bounds = np.full(u_args.shape, -np.inf)
    all_upper_bounds = np.full(u_args.shape, np.inf)
    
    mask_v_pos = v_args > 1e-16   # v_arg > 0  => z < -u_arg/v_arg
    mask_v_neg = v_args < -1e-16   # v_arg < 0  => z > -u_arg/v_arg
    
    np.divide(-u_args, v_args, out=all_upper_bounds, where=mask_v_pos)
    np.divide(-u_args, v_args, out=all_lower_bounds, where=mask_v_neg)
    
    mask_v_zero_u_bad = (np.abs(v_args) <= 1e-16) & (u_args > 0)
    
    all_lower_bounds[mask_v_zero_u_bad] = np.inf
    all_upper_bounds[mask_v_zero_u_bad] = -np.inf
    
    final_lower_bound = np.max(all_lower_bounds)
    final_upper_bound = np.min(all_upper_bounds)
    
    itv = intersect(itv, [final_lower_bound, final_upper_bound])
    
    pivot = util.get_alpha_percent_greatest(reconstruction_loss, alpha)
    reconstruction_loss = np.array(reconstruction_loss)
    
    A = np.sum(s * (p - u), axis=1)
    B = np.sum(s * (q - v), axis=1)
    
    A_pivot = A[pivot]
    B_pivot = B[pivot]
    
    mask_lt = reconstruction_loss < reconstruction_loss[pivot]
    u_args = np.where(mask_lt, A - A_pivot, A_pivot - A)
    v_args = np.where(mask_lt, B - B_pivot, B_pivot - B)
    
    all_lower_bounds = np.full(u_args.shape, -np.inf)
    all_upper_bounds = np.full(u_args.shape, np.inf)

    mask_v_pos = v_args > 1e-16
    mask_v_neg = v_args < -1e-16

    np.divide(-u_args, v_args, out=all_upper_bounds, where=mask_v_pos)
    np.divide(-u_args, v_args, out=all_lower_bounds, where=mask_v_neg)

    mask_v_zero_u_bad = (np.abs(v_args) <= 1e-16) & (u_args > 0)
    all_lower_bounds[mask_v_zero_u_bad] = np.inf
    all_upper_bounds[mask_v_zero_u_bad] = -np.inf

    final_lower_bound = np.max(all_lower_bounds)
    final_upper_bound = np.min(all_upper_bounds)

    itv = intersect(itv, [final_lower_bound, final_upper_bound])
            
    return itv

def get_test_statistic_interval(ns, nt, d, xt, Oc, a, b, j):
    itv = [-np.inf, np.inf]
    for i in range(d):
        testj = xt[j, i]
        testOc = (1/len(Oc)) * np.sum(xt[Oc[k], i] for k in range(len(Oc)))
        if (testj - testOc) < 0:
            itv = intersect(itv, solve_linear_inequality(a[j * d + i + ns * d] - (1/len(Oc))*np.sum(a[Oc[k] * d + i + ns * d] for k in range(len(Oc))), b[j * d + i + ns * d] - (1/len(Oc))*np.sum(b[Oc[k] * d + i + ns * d] for k in range(len(Oc)))))
        else:
            itv = intersect(itv, solve_linear_inequality(-a[j * d + i + ns * d] + (1/len(Oc))*np.sum(a[Oc[k] * d + i + ns * d] for k in range(len(Oc))), -b[j * d + i + ns * d] + (1/len(Oc))*np.sum(b[Oc[k] * d + i + ns * d] for k in range(len(Oc)))))
    return itv