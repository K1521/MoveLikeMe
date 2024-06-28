
import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True,fastmath=True)
def calculateS(N, W, P, P_):

    n = len(N)
    S = np.zeros((n, 3, 3))
    
    for i in prange(n):
        pi=P[i]
        pi_=P_[i]
        for j in N[i]:
            diff_P = pi - P[j]
            diff_P_ =pi_ - P_[j]
            w=W[i,j]
            for k in range(3):
                for l in range(3):
                    S[i, k, l] += w * diff_P[k] * diff_P_[l]
    return S



@njit(parallel=True, cache=True,fastmath=True)
def calculateb(N,W,P,R):
    n=len(N)
    b=np.zeros((n,3))
    for i in prange(n):
        pi=P[i]
        ri=R[i]
        for j in N[i]:
            pj = P[j]
            rj = R[j]
            # Matrix-vector multiplication: (ri + rj) @ (pi - pj)
            mat_sum = ri + rj
            vec_diff = pi - pj
            w=W[i, j]
            #result = np.zeros(3)
            for k in range(3):
                rk=0
                for l in range(3):
                    #result[k] += mat_sum[k, l] * vec_diff[l]
                    rk += mat_sum[k, l] * vec_diff[l]
                b[i, k] += w * rk
    return 0.5 * b

  
import arap
arap.calculateb=calculateb
arap.calculateS=calculateS