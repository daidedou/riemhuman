from vtkVisualization import *
import numpy as np
# Construction from
# <http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html>.
# Create 12 vertices of a icosahedron.
t = (1.0 + np.sqrt(5.0)) / 2.0
corners = np.array(
    [
        [-1, +t, +0],
        [+1, +t, +0],
        [-1, -t, +0],
        [+1, -t, +0],
        #
        [+0, -1, +t],
        [+0, +1, +t],
        [+0, -1, -t],
        [+0, +1, -t],
        #
        [+t, +0, -1],
        [+t, +0, +1],
        [-t, +0, -1],
        [-t, +0, +1],
    ]
)

faces = [
    (0, 11, 5),
    (0, 5, 1),
    (0, 1, 7),
    (0, 7, 10),
    (0, 10, 11),
    (1, 5, 9),
    (5, 11, 4),
    (11, 10, 2),
    (10, 7, 6),
    (7, 1, 8),
    (3, 9, 4),
    (3, 4, 2),
    (3, 2, 6),
    (3, 6, 8),
    (3, 8, 9),
    (4, 9, 5),
    (2, 4, 11),
    (6, 2, 10),
    (8, 6, 7),
    (9, 8, 1),
]

surface_actor = VTKSurface(np.array(corners), np.array(faces))
# Interactive plot
renderer = VTKVisualization()
renderer.init() # Needed for interactive plot!
renderer.add_entity(surface_actor)
renderer.show()

# Save snapshot of the plot
renderer = VTKVisualization()
renderer.add_entity(surface_actor)
renderer.write("sphere.png")