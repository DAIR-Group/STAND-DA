import torch 
import numpy as np
from . import util

def AE_AD(Xs_hat, Xt_hat, X_tilde, alpha):
    X = np.concatenate((Xs_hat, Xt_hat), axis=0)
    reconstruction_loss = np.abs(X - X_tilde).sum(axis=1)
    
    O = np.argsort(reconstruction_loss)[-int(alpha*len(reconstruction_loss)):]
    O = [i - Xs_hat.shape[0] for i in O if i >= Xs_hat.shape[0]]
    return np.sort(O), reconstruction_loss

def AD_DA(Xs, Xt, wdgrl, ae, alpha=0.05):
    Xs = torch.DoubleTensor(Xs).to(wdgrl.device)
    Xt = torch.DoubleTensor(Xt).to(wdgrl.device)
    
    Xs_hat = wdgrl.extract_feature(Xs)
    Xt_hat = wdgrl.extract_feature(Xt)
    
    X_hat = torch.cat([Xs_hat, Xt_hat], dim=0)
    X_tilde = ae.forward(X_hat)
    
    Xs_hat = Xs_hat.cpu().detach().numpy()
    Xt_hat = Xt_hat.cpu().detach().numpy()
    X_tilde = X_tilde.cpu().detach().numpy()
    
    O, _ = AE_AD(Xs_hat, Xt_hat, X_tilde, alpha)
    return O