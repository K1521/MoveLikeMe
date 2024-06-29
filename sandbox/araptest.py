import numpy as np

def Nij(N):
    return np.array([(i,j)for i,Nj in enumerate(N) for j in Nj])

# def calculateb(N,W,P,R):
#     b=np.zeros((len(N),3))
#     for i,j in Nij(N):
#         b[i]+=W[i,j]*((R[i]+R[j])@(P[i]-P[j]))
#     return 0.5*b
# def calculateb(N,W,P,R):
    
#     #print(R)
#     #I=np.eye(3)
#     b=np.zeros((len(N),3))
#     # for i,Ni in enumerate(N):
#     #     for j in Ni:
#     #         b[i]+=0.5*W[i,j]*((R[i]+R[j])@(P[i]-P[j]))
    
#     for i, Ni in enumerate(N):
#         #TODO understand this chatgpt code
#         P_diff = P[i] - P[Ni]  # Shape: (len(Ni), 3)
#         R_sum = R[i] + R[Ni]  # Shape: (len(Ni), 3, 3)
#         W_ij = W[i, Ni]  # Shape: (len(Ni),)

#         # Perform the weighted sum of the transformed differences
#         transformed_diff = np.einsum('ijk,ik->ij', R_sum, P_diff)
#         b[i] += 0.5 * np.sum(transformed_diff * W_ij[:, np.newaxis], axis=0)
#     return b
def calculateb(N,W,P,R):
    b=np.zeros((len(N),3))
    i,j=Nij(N).T
    Pij=P[i]-P[j]
    Wij=W[i,j]
    Rij=R[i]+R[j]
    
    print(f"{Rij.shape=}")
    print(f"{Pij.shape=}")

    np.add.at(b,i,Wij[:,None]*(Rij@Pij[:,:,None]).squeeze())
    return b*0.5
def calculateS(N,W,P,P_):
    S=np.zeros((len(N),3,3))
    #for i,j in Nij(N):
    i,j=Nij(N).T
    Pij=P[i]-P[j]
    Pij_=P_[i]-P_[j]
    Wij=W[i,j]

    #print(wij)
    #print(Pij[:,:,None]*Pij_[:,None,:])
    #print()
    #S[i]+=W[i,j][:,None,None]*Pij[:,None,:]*Pij_[:,:,None]
    #S[i,:,:]+=Wij[:,None,None]*Pij[:,:,None]*Pij_[:,None,:]
    np.add.at(S, i, Wij[:, None, None] * Pij[:, :, None] * Pij_[:, None, :]) 
    #for idx in range(len(i)):
        #S[i[idx]]+=Wij[idx,None,None]*Pij[idx,:,None]*Pij_[idx,None,:]#np.outer(Pij[idx],Pij_[idx])
    return S
# def calculateb(N,W,P,R):
#     b=np.zeros((len(N),3))
#     for i,Ni in enumerate(N):
#         for j in Ni:
#             b[i]+=0.5*W[i,j]*((R[i]+R[j])@(P[i]-P[j]))
#     return b

# def calculateS(N,W,P,P_):
#     S=np.zeros((len(N),3,3))
#     for i,Ni in enumerate(N):
#         for j in Ni:
#             S[i]+=W[i,j]*np.outer(P[i]-P[j],P_[i]-P_[j])
#     return S

import arap
arap.calculateb=calculateb
arap.calculateS=calculateS