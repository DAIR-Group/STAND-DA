import numpy as np

def compute_yz(zk, a, b):
    Xz = a + b*zk
    return Xz

def get_alpha_percent_greatest(X, alpha):
    return np.argsort(X)[-int(alpha*len(X))]

def compute_etaj_etajTX_etajTsigmaetaj(j, d, ns, nt, xt, Oc, X, sigma):
    etj = np.zeros((nt * d, 1)) 
    for i in range(d):
        etj[j * d + i] = 1
    etOc = np.zeros((nt * d, 1))
    for i in Oc:
        for k in range(d):
            etOc[i * d + k] = 1
    s = np.zeros((ns * d + nt * d, 1))
    for i in range(d):
        testj = xt[j, i]
        testOc = (1/len(Oc)) * np.sum(xt[Oc[k], i] for k in range(len(Oc)))
        if np.sign(testj - testOc) == -1:
            etj[j * d + i] = -1
            for k in Oc:
                etOc[k * d + i] = -1
    etaj = np.vstack((np.zeros((ns * d, 1)), etj - (1/len(Oc))*etOc))
    etajTX = etaj.T.dot(X)
    etajTsigmaetaj = etaj.T.dot(sigma).dot(etaj)
    return etaj, etajTX, etajTsigmaetaj

def compute_a_b(sigma, etaj, etajTsigmaetaj, X, ns, nt, d):
    b = sigma.dot(etaj).dot(np.linalg.inv(etajTsigmaetaj))
    a = (np.identity(ns * d + nt * d) - b.dot(etaj.T)).dot(X)
    return a, b