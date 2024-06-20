import pyvista as pv


meshpath = "./resources/meshes/BunnyLowPoly.stl"
#meshpath = "./resources/meshes/bunny.obj"
mesh = pv.read(meshpath)

#vertices = mesh.points
def generateN(mesh):
    faces = mesh.faces
    N=[[]for i in range(len(mesh.points))]# Neighbours

    index = 0
    while index < len(faces):
        num_vertices = faces[index]
        #print(num_vertices)
        face=faces[index+1:index+num_vertices+1]
        if num_vertices==3:
            for i in range(3):
                N[face[i]].append(face[i-1])
                N[face[i-1]].append(face[i])#N[face[i]].append(face[(i+1)%3])
        else:
            # This is a polygon, perform triangulation
            # Using a simple fan triangulation
            for i in range(1, num_vertices - 1):
                triangle=[face[0], face[i], face[i + 1]]
                for i in range(3):
                    N[triangle[i]].append(triangle[i-1])
                    N[triangle[i-1]].append(triangle[i])

        index+=num_vertices+1
    
    return N

N=generateN(mesh)
#print(N)
print(mesh)
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
sphere = pv.Sphere(radius=r, center=mesh.points[i])
plotter.add_mesh(sphere, color='green')
print(points)
for point in points:
    sphere = pv.Sphere(radius=r, center=point)
    plotter.add_mesh(sphere, color='red')
plotter.add_mesh(mesh,show_edges=True)
plotter.set_background('black')
plotter.show()