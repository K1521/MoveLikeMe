
import pyvista as pv
import arap2

def getmaxbound(mesh):
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    return max(x_range, y_range, z_range)

class Picker:
    def __init__(self,plotter,N,mesh):
        self.plotter = plotter
        self.N=N
        self.spheres=[]
        self.mesh=mesh
        self.r=getmaxbound(mesh)
    def __call__(self, mesh, picker):
        #print(dir(picker))
        idx=picker.GetPointId()
        print(idx)
        #print(mesh)
        #print(idx)
        #print(mesh.points[idx])
        #self.plotter.set_focus(mesh.points[idx])

        for actor in self.spheres:#remove the old red spheres
            self.plotter.remove_actor(actor,render=False)
        self.spheres.clear()
        for i in self.N[idx]:#add the new red spheres
            point=self.mesh.points[i]
            sphere = pv.Sphere(radius=self.r * 0.01, center=point)
            self.spheres.append(self.plotter.add_mesh(sphere, color='red',render=False))
        sphere = pv.Sphere(radius=self.r * 0.01, center=self.mesh.points[idx])
        self.spheres.append(self.plotter.add_mesh(sphere, color='green',render=False))
        self.spheres.append(self.plotter.add_point_labels([self.mesh.points[idx]],[str(idx)],always_visible=True,render=False))
        self.plotter.update()
        

meshpath = "./resources/meshes/BunnyLowPoly.stl"
meshpath = "./resources/meshes/bunny.obj"
#meshpath = "./resources/meshes/lowpoly_male.obj"
mesh = pv.read(meshpath).clean(inplace=True)



pv.set_plot_theme('dark')
plt = pv.Plotter()
plt.add_axes()
plt.show_grid()
plt.enable_point_picking(Picker(plt,arap2.generateN(mesh),mesh), use_picker=True,show_message=False)
plt.add_mesh(mesh,show_edges=True)
plt.show()



