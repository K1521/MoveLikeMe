import pyvista as pv
import numpy as np
import time
import copy
import scipy.sparse

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

def cot(u, v):
    """
    Compute the cotangent of the angle between two vectors u and v.

    Parameters:
    u (array-like): First vector.
    v (array-like): Second vector.

    Returns:
    float: Cotangent of the angle between u and v.
    """
    return np.dot(u, v) / np.linalg.norm(np.cross(u, v))

def generateW(mesh):
    #TODO Maybe use sparse matrices?
    #TODO Vectorise this code by inverting loop order because i think this could be faster by at least an order of manitude
    W=np.zeros((len(mesh.points),len(mesh.points)))
    for triangle in gettriangles(mesh):
        points=mesh.points[triangle]
        for i in range(3):
            a,b,c=np.roll(points,i,axis=0)
            ai,bi,ci=np.roll(triangle,i)
            cotalpha=cot(b-a,c-a)
            W[bi,ci]+=cotalpha
            W[ci,bi]+=cotalpha
    return W/2

def generateW2(mesh):
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
        cotalpha=np.maximum(cotalpha, 0)#TODO not sure how to handle negative cot weights but this seams stable?

        #cosTheta = np.sum(ab*ac,axis=1) / (np.linalg.norm(ab,axis=1) * np.linalg.norm(ac,axis=1))
        #theta = np.arccos(cosTheta)
        #cotalpha = np.cos(theta) / np.sin(theta)#some code i copied from the internet to check if the other code works.worse version

        W[bi,ci]+=cotalpha
        W[ci,bi]+=cotalpha
    W=W/2
    #assert np.allclose(W,generateW(mesh))
    return W

def generateL(W):
    return np.diag(np.sum(W,axis=1))-W

def calculateSalt(N,W,P,P_):
    S=np.zeros((len(N),3,3))
    # for i,Ni in enumerate(N):
    #     for j in Ni:
    #         S[i]+=W[i,j]*np.outer(P[i]-P[j],P_[i]-P_[j])

    for i, Ni in enumerate(N):
        #TODO understand this chatgpt code
        P_diff = P[i] - P[Ni]
        P_prime_diff = P_[i] - P_[Ni]
        S[i] = np.sum(W[i, Ni][:, np.newaxis, np.newaxis] * P_diff[:, :, np.newaxis] * P_prime_diff[:, np.newaxis, :], axis=0)
    

    #print(S)
    return S

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

def calculateR(N,W,P,P_):
    S=calculateS(N,W,P,P_)

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

def calculatebalt(N,W,P,R):
    
    #print(R)
    #I=np.eye(3)
    b=np.zeros((len(N),3))
    # for i,Ni in enumerate(N):
    #     for j in Ni:
    #         b[i]+=0.5*W[i,j]*((R[i]+R[j])@(P[i]-P[j]))
    
    for i, Ni in enumerate(N):
        #TODO understand this chatgpt code
        P_diff = P[i] - P[Ni]  # Shape: (len(Ni), 3)
        R_sum = R[i] + R[Ni]  # Shape: (len(Ni), 3, 3)
        W_ij = W[i, Ni]  # Shape: (len(Ni),)

        # Perform the weighted sum of the transformed differences
        transformed_diff = np.einsum('ijk,ik->ij', R_sum, P_diff)
        b[i] += 0.5 * np.sum(transformed_diff * W_ij[:, np.newaxis], axis=0)
    return b

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

            self.LUU=scipy.sparse.csr_matrix(L[np.ix_(self.unconstrained_index, self.unconstrained_index)])
            self.LUF=scipy.sparse.csr_matrix(L[np.ix_(self.unconstrained_index, self.constrained_index)])

    def setconstraints(self,constraints):
        self.setindex([index for index, c in constraints])
        self.setpoints([c for index, c in constraints])
    def apply(self,b):
        P_=np.zeros(b.shape)
        P_[self.constrained_index]=self.ck
        P_[self.unconstrained_index]=scipy.sparse.linalg.spsolve(self.LUU,b[self.unconstrained_index]-self.constraintsb)
        return P_

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

meshpath = "./resources/meshes/BunnyLowPoly.stl"
#meshpath = "./resources/meshes/bunny.obj"
mesh = pv.read(meshpath)


r =getmaxbound(mesh)


N=generateN(mesh)



W=generateW2(mesh)
#W=np.maximum(W,0)

#print(mesh.extract_all_edges())
#exit()



L=generateL(W)



plotter = pv.Plotter()

plotter.add_mesh(mesh,show_edges=True)
plotter.set_background('black')
plotter.show(interactive_update=True)


import cProfile, pstats, io #TODO profile
from pstats import SortKey

P=copy.deepcopy(mesh.points)
P_=copy.deepcopy(P)#guess

addedspheres=[]
pr = cProfile.Profile(builtins=False)
eqsystem=constrainteqs(L)


eqsystem.setconstraints([(i,P[i]+np.random.uniform(-1,1,3)*r*0.1) for i in [23,62,17,3,21,67]])
R=calculateR(N,W,P,P_)
b=calculateb(N,W,P,R)
P_=eqsystem.apply(b)

for i in range(3000):
    if i%30==0:
        #move the targets to "random" locations (original location+random offset)
        for actor in addedspheres:
            plotter.remove_actor(actor)
        addedspheres=[]
        constrains=[(i,P[i]+np.random.uniform(-1,1,3)*r*0.1) for i in [23,62,17,3,21,67]]
        for i,point in constrains:
            sphere = pv.Sphere(radius=r*0.01, center=point)
            addedspheres.append(plotter.add_mesh(sphere, color='red'))
        eqsystem.setconstraints(constrains)

    t=time.time()#+1/20#max 20 fps

    with pr:
        R=calculateR(N,W,P,P_)
        b=calculateb(N,W,P,R)
        #P_=np.linalg.solve(L,b)
        #P_=solvewithconstraints(L,b,constrains)
        P_=eqsystem.apply(b)
    pstats.Stats(pr).strip_dirs().sort_stats('tottime').print_stats(15)

    mesh.points=P_

    while time.time()<t:
        plotter.update()
        time.sleep(0.01)
    plotter.update()
#print(N)
#print(list(gettriangles(mesh)))
# Display the mesh
