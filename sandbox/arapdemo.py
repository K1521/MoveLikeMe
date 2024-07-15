import pyvista as pv
from arap2 import arap
import numpy as np
import time
#import arapjit2
import scipy.sparse.linalg
def getmaxbound(mesh):
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    return max(x_range, y_range, z_range)

def mima(x):
    print(np.min(x),np.max(x))
meshpath = "../resources/meshes/BunnyLowPoly.stl"
meshpath = "../resources/meshes/bunny.obj"
#meshpath="./resources/meshes/lowpoly_male.obj"
mesh = pv.read(meshpath).clean(inplace=True)


r =getmaxbound(mesh)
bunnyarap=arap(mesh)
# print(min(map(len,bunnyarap.N)))
# L=generateL(bunnyarap.W)
# mima(np.diag(L))
# print(np.argmin(np.diag(L)))
#exit()
plotter = pv.Plotter()

plotter.add_mesh(mesh,show_edges=True)
plotter.set_background('black')
plotter.show(interactive_update=True)


import cProfile, pstats #TODO profile




addedspheres=[]
pr = cProfile.Profile(builtins=False)
for i in range(1,3000):
    if i%30==1:
        #move the targets to "random" locations (original location+random offset)
        for actor in addedspheres:
            plotter.remove_actor(actor,render=False)
        addedspheres=[]

        constrains=[(i,bunnyarap.P[i]+np.random.uniform(-1,1,3)*r*0.1+np.array([0,0,np.nan])) for i in [23,62,17,3,21,67]] # 23,62,17,3,21,67
        #constrains.extend([(i,bunnyarap.P[i]+np.random.uniform(-1,1,3)*r*0.1) for i in [1,2]])
        #print(constrains)
        for i,point in constrains:
            if np.any(np.isnan(point)):
                continue
            sphere = pv.Sphere(radius=r*0.01, center=point)
            addedspheres.append(plotter.add_mesh(sphere, color='red',render=False))
        bunnyarap.setconstraints(constrains)

    t=time.time()#+1/20#max 20 fps

    #with pr:
    pr.enable()
    P_=bunnyarap.apply()
    pr.disable()
    #mima(P_)
    
    stats=pstats.Stats(pr)
    print(f"iteration:{i}\navg iter/sec for arap:{i/stats.total_tt}\navg sec/iter for arap:{stats.total_tt/i}")
    stats.strip_dirs().sort_stats('tottime').print_stats(15)

    mesh.points=P_

    while time.time()<t:
        plotter.update()
        time.sleep(0.01)

    plotter.update()
    print(mesh.bounds)
#print(N)
#print(list(gettriangles(mesh)))
# Display the mesh
