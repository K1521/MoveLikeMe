import numpy as np
from numba import njit, prange

@njit(cache=True,inline="always")
def addat(a,idx,b):
    for i in prange(len(b)):
        a[idx[i]] += b[i]

@njit(cache=True)
def calculateS(i,j,Pij,P_):
    S=np.zeros((len(P_),3,3))

    Pij_=P_[i]-P_[j]
    c=Pij[:, :, None] * Pij_[:, None, :] #0.024 per call
    addat(S,i,c)
    #np.add.at(S, i, Pij[:, :, None] * Pij_[:, None, :]) 

    return S



# @njit(cache=True)
# def calculateR(S,N,W,P,P_):#slower
#     rotation_matrices=np.zeros((len(N),3,3))
#     for i in range(len(N)):
#         Ui, Si_singular_values, VTi = np.linalg.svd(S[i],full_matrices=False)
        
#         # Compute rotation matrix Ri
#         Ri = np.dot(VTi.T, Ui.T)#np.dot(Ui,VTi).T #np.dot(VTi.T, Ui.T)
#         if np.linalg.det(Ri) < 0:
#             Ui[:, -1] *= -1
#             Ri=np.dot(Ui,VTi).T
#         rotation_matrices[i] = Ri
#     return rotation_matrices

# @njit(cache=True)
# def calculateb(i,j,Pij,R):
#     b=np.zeros((len(R),3))
#     Rij=R[i]+R[j]
#     addat(b,i,(Rij@Pij[:,:,None]).squeeze())
#     return b*0.5


import arap2 as arap
#arap.calculateb=calculateb
arap.calculateS=calculateS
#arap.calculateR=calculateR