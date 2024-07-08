import pyvista as pv
import numpy as np
import time
import copy
import scipy.sparse
import scipy.sparse.linalg

def gettriangles(mesh):
    faces = mesh.faces
    index = 0
    while index < len(faces):
        num_vertices = faces[index]
        #print(num_vertices)
        face=faces[index+1:index+num_vertices+1]
        if num_vertices==3:
            yield face
        else:
            # This is a polygon, perform triangulation
            # Using a simple fan triangulation
            #TODO Use better Triangulation
            for i in range(1, num_vertices - 1):
                triangle=face[[0,i,i+1]]
                yield triangle
        index+=num_vertices+1
    
def generateN(mesh):
    N=[set()for i in range(len(mesh.points))]# Neighbours
    for triangle in gettriangles(mesh):
        for i in range(3):
            N[triangle[i]].add(triangle[i-1])
            N[triangle[i-1]].add(triangle[i])
    return [np.array(list(x))for x in N]


def generateW2(mesh,):
    #TODO Maybe use sparse matrices?
    #This is the vectorised version of generateW
    W=np.zeros((len(mesh.points),len(mesh.points)))
    triangles=np.array(list(gettriangles(mesh)))
    #points=mesh.points[triangles]
    abci=[triangles[:,i] for i in range(3)]
    abc=[mesh.points[i] for i in abci]
    for i in range(3):
        a,b,c=np.roll(abc,i,axis=0)
        ai,bi,ci=np.roll(abci,i,axis=0)
        #cotalpha=cot(b-a,c-a)
        ab=b-a
        ac=c-a
        cotalpha=np.sum(ab*ac,axis=1) / np.linalg.norm(np.cross(ab, ac,axis=1),axis=1)
        

        #cosTheta = np.sum(ab*ac,axis=1) / (np.linalg.norm(ab,axis=1) * np.linalg.norm(ac,axis=1))
        #theta = np.arccos(cosTheta)
        #cotalpha = np.cos(theta) / np.sin(theta)#some code i copied from the internet to check if the other code works.worse version
        #cotalpha=np.maximum(0.1,cotalpha)
        W[bi,ci]+=cotalpha
        W[ci,bi]+=cotalpha
    W=W/2
    #assert np.allclose(W,generateW(mesh))
    return W

def generateL(W):
    return np.diag(np.sum(W,axis=1))-W



def calculateR(S,N,W,P,P_):
    

    # rotation_matrices=np.zeros((len(N),3,3))
    # for i in range(len(N)):
    #     Ui, Si_singular_values, VTi = np.linalg.svd(S[i],full_matrices=False)
        
    #     # Compute rotation matrix Ri
    #     Ri = np.dot(VTi.T, Ui.T)#np.dot(Ui,VTi).T #np.dot(VTi.T, Ui.T)
    #     if np.linalg.det(Ri) < 0:
    #         Ui[:, -1] *= -1
    #         Ri=np.dot(Ui,VTi).T
    #     rotation_matrices[i] = Ri
    #return rotation_matrices
    Ui, Si_singular_values, VTi = np.linalg.svd(S,full_matrices=False)
    R=np.matmul(Ui,VTi)

    neg_det_mask = np.linalg.det(R)<0
    # Adjust the sign of the matrix Ui corresponding to the smallest singular value
    Ui[:, :, -1][neg_det_mask] *= -1
    R[neg_det_mask] = np.matmul(Ui,VTi)[neg_det_mask]
    R= R.transpose(0, 2, 1)
    #assert np.allclose(R,rotation_matrices)
    return R

def Nij(N):
    return np.array([(i,j)for i,Nj in enumerate(N) for j in Nj])



def calculateb(i,j,Pij,R):
    b=np.zeros((len(R),3))

    Rij=R[i]+R[j]

    np.add.at(b,i,(Rij@Pij[:,:,None]).squeeze())
    return b*0.5

def calculateS(i,j,Pij,P_):
    S=np.zeros((len(P_),3,3))

    Pij_=P_[i]-P_[j]
    np.add.at(S, i, Pij[:, :, None] * Pij_[:, None, :]) 

    return S


def mima(x):
    print(np.min(x),np.max(x))
    assert not np.isnan(np.sum(x))



def sparsesolve1(A,b):
    #A = scipy.sparse.csr_matrix(A)
    X =  scipy.sparse.linalg.spsolve(A, b)
    return X


class constrainteqs:
    def __init__(self,L):
        self.L=L
        self.constrained_index=None

    def setpoints(self,ck):
        ck=np.array(ck)
        self.constraintsb=self.LUF@ck
        self.ck=ck
    def setindex(self,idx):
        idx=np.array(idx)
        if not np.array_equal(idx,self.constrained_index):
            self.constrained_index=idx
            self.unconstrained_index=np.full(len(self.L),True)
            self.unconstrained_index[self.constrained_index]=False

            self.LUU=scipy.sparse.csr_matrix(self.L[np.ix_(self.unconstrained_index, self.unconstrained_index)])
            self.LUF=scipy.sparse.csr_matrix(self.L[np.ix_(self.unconstrained_index, self.constrained_index)])

    def setconstraints(self,constraints):
        self.setindex([index for index, c in constraints])
        self.setpoints([c for index, c in constraints])
    def apply(self,b):
        P_=np.zeros(b.shape)
        P_[self.constrained_index]=self.ck
        #mima(P_)
        #mima(b[self.unconstrained_index]-self.constraintsb)
        #mima(self.LUU.toarray())
        #mima(np.diag(self.LUU.toarray()))
        #for i in range(3):
        #    P_[self.unconstrained_index,i], istop, itn, normr = scipy.sparse.linalg.lsmr(self.LUU,(b[self.unconstrained_index]-self.constraintsb)[:,i])[:4]
        P_[self.unconstrained_index]=scipy.sparse.linalg.spsolve(self.LUU,b[self.unconstrained_index]-self.constraintsb)
        #P_[self.unconstrained_index]=np.linalg.solve(self.LUU.toarray(),b[self.unconstrained_index]-self.constraintsb)
        return P_
class constrainteqs3:
    def __init__(self,L):
        self.singleconstrains=[constrainteqs(L),constrainteqs(L),constrainteqs(L)]#constrains for x,y,z



    def setconstraints(self,constraints):
        #print(constraints)
        indices=np.array([index for index, c in constraints])
        c=np.array([c for index, c in constraints])
        #print(indices)
        #print(c)
        for i in range(3):
            constrains=self.singleconstrains[i]
            ci=c[:,i]
            cigood=~np.isnan(ci)
            constrains.setindex(indices[cigood])
            #print(cigood)
            #print(ci)
            constrains.setpoints(ci[cigood])


    def apply(self,b):
        P_i=[self.singleconstrains[i].apply(b[:,i]) for i in range(3)]
        #print(P_i[0])
        return np.vstack(P_i).T

def solvewithconstraints(L, b, constraints):
    #TODO precompute L_constrained and L[np.ix_(unconstrained_index, constrained_index)]
    #only recompute if constrained_index changes
    #also convert to sparse matrices
    constrained_index=np.array([index for index, _ in constraints])
    
    unconstrained_index=np.full(len(L),True)
    unconstrained_index[constrained_index]=False
    #unconstrained_index=~np.isin(np.arange(len(L)), constrained_index)
    
    ck=np.array([c for index, c in constraints])
    
    P_=np.zeros(b.shape)
    P_[constrained_index]=ck
    
    #b_constrained2 = (b-L@P_)[unconstrained_index]
    #b_constrained = b[unconstrained_index]-(L@P_)[unconstrained_index]
    #b_constrained = b[unconstrained_index]-L[unconstrained_index]@P_
    b_constrained=b[unconstrained_index]-L[np.ix_(unconstrained_index, constrained_index)]@ck
    #assert np.allclose(b_constrained,b_constrained2)
    #AUU​ xU​+AUF​ cF​=bU​
    #AUU​ xU​​=bU​-AUF​ cF

    L_constrained = L[np.ix_(unconstrained_index, unconstrained_index)]#AUU

    #P_[unconstrained_index]=np.linalg.solve(L_constrained,b_constrained)
    P_[unconstrained_index]=sparsesolve1(L_constrained,b_constrained)
    return P_

def getmaxbound(mesh):
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    return max(x_range, y_range, z_range)

class arap:
    def __init__(self,mesh,useconstrains3=False):
        self.P=copy.deepcopy(mesh.points)
        self.P_=copy.deepcopy(mesh.points)#guess
        self.N=generateN(mesh)
        W=generateW2(mesh)
        #self.W=W
        #self.W=np.abs(W)
        self.W=np.maximum(W, 0)#TODO not sure how to handle negative cot weights but this seams stable?
        L=generateL(self.W)
        if useconstrains3:
            self.eqsystem=constrainteqs3(L)
        else:
            self.eqsystem=constrainteqs(L)

        i,j=Nij(self.N).T
        self.i=i
        self.j=j
        self.Pij=(self.P[i]-self.P[j])*self.W[i,j,None]
        #Pij_=P_[i]-P_[j]

    def apply(self,P_=None, niter=1):
        if P_ is not None:
            self.P_=P_
        for i in range(niter):
            S=calculateS(self.i,self.j,self.Pij,self.P_)
            R=calculateR(S,self.N,self.W,self.P,self.P_)
            b=calculateb(self.i,self.j,self.Pij,R)
            self.P_=self.eqsystem.apply(b)
        return self.P_
    def setconstraints(self,constrains):
        self.eqsystem.setconstraints(constrains)
