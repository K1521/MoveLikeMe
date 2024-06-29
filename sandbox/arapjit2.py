import numpy as np
from numba import njit, prange




def calculateS(i,j,Pij,P_):
    S=np.zeros((len(P_),3,3))

    Pij_=P_[i]-P_[j]
    c=Pij[:, :, None] * Pij_[:, None, :] #0.024 per call
    addat(S,i,c)
    #np.add.at(S, i, Pij[:, :, None] * Pij_[:, None, :]) 

    return S

@njit(cache=True)
def addat(a,idx,b):
    for i in prange(len(b)):
        a[idx[i]] += b[i]

import arap2 as arap
#arap.calculateb=calculateb
arap.calculateS=calculateS