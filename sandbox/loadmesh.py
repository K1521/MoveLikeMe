import pyvista as pv
import numpy as np
from time import time

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
        ba=b-a
        ca=c-a
        cotalpha=np.sum(ba*ca,axis=1) / np.linalg.norm(np.cross(ba, ca,axis=1),axis=1)
        W[bi,ci]+=cotalpha
        W[ci,bi]+=cotalpha
    return W
def generateL(W):
    return np.diag(np.sum(W,axis=1))-W
#meshpath = "./resources/meshes/BunnyLowPoly.stl"
meshpath = "./resources/meshes/bunny.obj"
mesh = pv.read(meshpath)



N=generateN(mesh)

import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile(builtins=False)
pr.enable()

W=generateW2(mesh)
#W=generateW2(mesh)-generateW(mesh)
#print(np.min(W),np.max(W))
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(10)

L=generateL(W)

#print(N)
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

i=11
points = mesh.points[N[i]]
sphere = pv.Sphere(radius=r*0.9, center=mesh.points[i])
plotter.add_mesh(sphere, color='green')
#print(points)
for point in points:
    sphere = pv.Sphere(radius=r, center=point)
    plotter.add_mesh(sphere, color='red')
plotter.add_mesh(mesh,show_edges=True)
plotter.set_background('black')
plotter.show()