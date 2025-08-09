import torch
import numpy as np
import mpmath as mp
from . import operations
from . import util
from . import ad_da
from . import conditioning
from scipy.linalg import block_diag

def cdf(sigma, list_zk, list_Oz, etajTX, O):
    numerator = 0
    denominator = 0
    
    for each_interval in range(len(list_zk) - 1):
        al = list_zk[each_interval]
        ar = list_zk[each_interval + 1] - 1e-3

        if (np.array_equal(O, list_Oz[each_interval]) == False):
            continue

        denominator = denominator + mp.ncdf((ar)/sigma) - mp.ncdf((al)/sigma)
        if etajTX >= ar:
            numerator = numerator + mp.ncdf((ar)/sigma) - mp.ncdf((al)/sigma)
        elif (etajTX >= al) and (etajTX< ar):
            numerator = numerator + mp.ncdf((etajTX)/sigma) - mp.ncdf((al)/sigma)
    
    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None

def parametric_si(Xz, a, b, zk, wdgrl, ae, np_wdgrl, np_ae, alpha, ns, nt):
    Xz_hat = wdgrl.extract_feature(torch.DoubleTensor(Xz).to(wdgrl.device)).cpu().detach().numpy()
    Xz_tilde = ae.forward(torch.DoubleTensor(Xz_hat).to(ae.device)).cpu().detach().numpy()
    Xz_hat_tensor = torch.from_numpy(Xz_hat).to(device=ae.device, dtype=torch.double)
    reconstruction_loss_tensor = ae.reconstruction_loss(Xz_hat_tensor)
    reconstruction_loss = []
    for value in reconstruction_loss_tensor:
        reconstruction_loss.append(value.item())
    reconstruction_loss = np.asarray(reconstruction_loss)
    Oz, _ = ad_da.AE_AD(Xz_hat[:ns], Xz_hat[ns:], Xz_tilde, alpha)
    
    itv = conditioning.get_ad_interval(Xz, Xz_hat, Xz_tilde, reconstruction_loss, a, b, np_wdgrl, np_ae, alpha)
    
    return itv[1] - min(zk, itv[1]), Oz

def divide_and_conquer(a, b, threshold, wdgrl, ae, alpha, ns, nt):
    zk = threshold[0]

    np_wdgrl = operations.convert_network_to_numpy(wdgrl.generator)
    np_ae = operations.convert_network_to_numpy(ae)
    wdgrl.generator = wdgrl.generator.cuda()
    ae.net = ae.net.cuda()

    list_zk = [zk]
    list_Oz = []

    while zk < threshold[1]:
        Xz = util.compute_yz(zk, a, b)
        skz, Oz = parametric_si(Xz, a, b, zk, wdgrl, ae, np_wdgrl, np_ae, alpha, ns, nt)
        zk = zk + skz + 1e-3 
        
        list_zk.append(min(threshold[1], zk))
        list_Oz.append(Oz)
    return list_zk, list_Oz

def compute_p_value(Xs, Xt, sigma_s, sigma_t, wdgrl, ae, j, O, alpha=0.05):
    ns = Xs.shape[0]
    nt = Xt.shape[0]
    d = Xs.shape[1]
    
    if len(O) == 0 or len(O) == nt:
        return None
    
    sigma = block_diag(sigma_s, sigma_t)
    
    yt_hat = np.zeros(Xt.shape[0])
    yt_hat[O] = 1
    Oc = list(np.where(yt_hat == 0)[0])
    X = np.vstack((Xs.flatten().reshape((ns * d, 1)), Xt.flatten().reshape((nt * d, 1))))
    etaj, etajTX, etajTsigmaetaj = util.compute_etaj_etajTX_etajTsigmaetaj(j, d, ns, nt, Xt, Oc, X, sigma)
    a, b = util.compute_a_b(sigma, etaj, etajTsigmaetaj, X, ns, nt, d)
    
    itv = [-20 * etajTsigmaetaj[0][0], 20 * etajTsigmaetaj[0][0]]
    itv = conditioning.intersect(itv, conditioning.get_test_statistic_interval(ns, nt, d, Xt, Oc, a, b, j))
    
    list_zk, list_Oz = divide_and_conquer(a.reshape(-1, d), b.reshape(-1, d), 
                                          itv, wdgrl, ae, alpha, ns, nt)
    
    CDF = cdf(np.sqrt(etajTsigmaetaj[0][0]), list_zk, list_Oz, etajTX[0][0], O)
    p_value = 2 * min(CDF, 1 - CDF)

    return p_value