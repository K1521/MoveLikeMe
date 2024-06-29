import pyvista as pv
from arap2 import arap
import numpy as np
import time
#import arapjit
#import araptest
def getmaxbound(mesh):
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    return max(x_range, y_range, z_range)

meshpath = "./resources/meshes/BunnyLowPoly.stl"
meshpath = "./resources/meshes/bunny.obj"
mesh = pv.read(meshpath)


r =getmaxbound(mesh)


bunnyarap=arap(mesh)


plotter = pv.Plotter()

plotter.add_mesh(mesh,show_edges=True)
plotter.set_background('black')
plotter.show(interactive_update=True)


import cProfile, pstats, io #TODO profile
from pstats import SortKey



addedspheres=[]
pr = cProfile.Profile(builtins=False)
for i in range(1,3000):
    if i%30==1:
        #move the targets to "random" locations (original location+random offset)
        for actor in addedspheres:
            plotter.remove_actor(actor)
        addedspheres=[]
        constrains=[(i,bunnyarap.P[i]+np.random.uniform(-1,1,3)*r*0.1) for i in [23,62,17,3,21,67]]
        for i,point in constrains:
            sphere = pv.Sphere(radius=r*0.01, center=point)
            addedspheres.append(plotter.add_mesh(sphere, color='red'))
        bunnyarap.setconstraints(constrains)

    t=time.time()#+1/20#max 20 fps

    with pr:
        P_=bunnyarap.apply()
    
    stats=pstats.Stats(pr)
    print(f"iteration:{i}  avg iter/sec for arap:{i/stats.total_tt}")
    stats.strip_dirs().sort_stats('tottime').print_stats(15)

    mesh.points=P_

    while time.time()<t:
        plotter.update()
        time.sleep(0.01)
    plotter.update()
#print(N)
#print(list(gettriangles(mesh)))
# Display the mesh
