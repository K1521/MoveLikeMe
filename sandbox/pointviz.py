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



meshpath = "./resources/meshes/BunnyLowPoly.stl"
meshpath = "./resources/meshes/bunny.obj"
#meshpath = "./resources/meshes/lowpoly_male.obj"
mesh = pv.read(meshpath)
mesh.clean(inplace=True)


N=generateN(mesh)




plotter = pv.Plotter()


def getmaxbound(mesh):
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    return max(x_range, y_range, z_range)

r =getmaxbound(mesh) * 0.01

plotter.add_point_labels(mesh.points,list(map(str,range(len(mesh.points)))))


points = mesh.points[[103,106]]
#i=11
#points = mesh.points[N[i]]
#sphere = pv.Sphere(radius=r*0.9, center=mesh.points[i])
#plotter.add_mesh(sphere, color='green')
#print(points)
for point in points:
    sphere = pv.Sphere(radius=r, center=point)
    plotter.add_mesh(sphere, color='red')
plotter.add_mesh(mesh,show_edges=True)
plotter.set_background('black')
plotter.show()

