import pyvista as pv
import numpy as np
from time import time
import copy

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
            for i in range(1, num_vertices - 1):
                triangle=face[[0,i,i+1]]
                yield triangle
        index+=num_vertices+1
    
def generateN(mesh):
    N=[[]for i in range(len(mesh.points))]# Neighbours
    for triangle in gettriangles(mesh):
        for i in range(3):
            N[triangle[i]].append(triangle[i-1])
            N[triangle[i-1]].append(triangle[i])
    return N

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
    return W

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
        W[bi,ci]+=cotalpha
        W[ci,bi]+=cotalpha
    return W
def generateL(W):
    return np.diag(np.sum(W,axis=1))-W
def calculateS(N,W,P,P_):
    S=np.zeros((len(N),3,3))
    for i,Ni in enumerate(N):
        for j in Ni:
            S[i]+=W[i,j]*np.outer(P[i]-P[j],P_[i]-P_[j])
    #print(S)
    return S
def calculateR(N,W,P,P_):
    S=calculateS(N,W,P,P_)
    #Ustack, Sstack, Vtstack = np.linalg.svd(S, full_matrices=False)
    
    # for i in range(len(N)):
    #     Ui, Si_singular_values, VTi = np.linalg.svd(S[i],full_matrices=False)
        
    #     # Compute rotation matrix Ri
    #     Ri = np.dot(Ui,VTi).T #np.dot(VTi.T, Ui.T)
    #     if np.linalg.det(Ri) < 0:
    #         Ui[:, -1] *= -1
    #         np.dot(Ui,VTi).T
        
    #     rotation_matrices[i] = Ri
    Ui, Si_singular_values, VTi = np.linalg.svd(S,full_matrices=False)
    R=np.matmul(Ui,VTi)

    neg_det_mask = np.linalg.det(R)<0
    # Adjust the sign of the matrix Ui corresponding to the smallest singular value
    Ui[:, :, -1][neg_det_mask] *= -1
    R[neg_det_mask] = np.matmul(Ui,VTi)[neg_det_mask]
    return R.transpose(0, 2, 1)

def calculateb(N,W,P,R):
    
    print(R)
    b=np.zeros((len(N),3))
    for i,Ni in enumerate(N):
        for j in Ni:
            b[i]+=0.5*W[i,j]*((R[i]+R[j])@(P[i]-P[j]))
    return b

meshpath = "./resources/meshes/BunnyLowPoly.stl"
#meshpath = "./resources/meshes/bunny.obj"
mesh = pv.read(meshpath)



N=generateN(mesh)



W=generateW2(mesh)
#W=generateW2(mesh)-generateW(mesh)
#print(np.min(W),np.max(W))


L=generateL(W)

# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile(builtins=False)
# pr.enable()

# pr.disable()
# pstats.Stats(pr).sort_stats('tottime').print_stats(10)



print(mesh)
#print(list(gettriangles(mesh)))
# Display the mesh
plotter = pv.Plotter()


def getmaxbound(mesh):
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    return max(x_range, y_range, z_range)

r =getmaxbound(mesh) * 0.01

#i=11
points = mesh.points[[103,106]]
#sphere = pv.Sphere(radius=r*0.9, center=mesh.points[i])
#plotter.add_mesh(sphere, color='green')
#print(points)
for point in points:
    sphere = pv.Sphere(radius=r, center=point)
    plotter.add_mesh(sphere, color='red')
plotter.add_mesh(mesh,show_edges=True)
plotter.set_background('black')
plotter.show()