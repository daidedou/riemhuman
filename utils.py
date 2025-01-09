import pickle
from surfaces import Surface
import open3d as o3d
import numpy as np
from scipy.io import savemat
import matplotlib


colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'cornsilk', 'crimson', 'cyan', 'darkblue',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey',
    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue'
]

def get_color(i, rgb=False):
    if rgb:
        return matplotlib.colors.to_rgb(colors[i])
    else:
        return colors[i]

def load_pkl(filename):
    tmp = pickle.load(open(filename, 'rb'), encoding="bytes")
    v_template = tmp[b"v_template"]
    faces = tmp[b"f"]  # .astype(np.int32)
    n_v = v_template.shape[0]
    surf_template = Surface(FV=[faces, v_template])
    return surf_template

def load_mesh(filename):
    if ".pkl" in filename:
        return load_pkl(filename)
    else:
        return Surface(filename=filename)

def load_surface(surf_name):
    surf = Surface(filename=surf_name)
    center = surf.surfCenter()
    surf.updateVertices((surf.vertices - center) / surf.volume)
    return surf


def getMeshFromData(mesh, Rho=None, color=None):
    """
    Get Open3d mesh from [V, F] input. Optionnally, you can add colors
    """
    V = mesh[0]
    F = mesh[1]
    # mesh=o3d.geometry.TriangleMesh(o3d.cpu.pybind.utility.Vector3dVector(V),o3d.cpu.pybind.utility.Vector3iVector(F))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))

    if Rho is not None:
        Rho = np.squeeze(Rho)
        col = np.stack((Rho, Rho, Rho))
        mesh.vertex_colors = o3d.utility.Vector3dVector(col.T)

    if color is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(color)
    return mesh

def getDataFromMesh(mesh):
    """
    Get numpy arrays, vertex, colors from open3d mesh object
    """
    V = np.asarray(mesh.vertices, dtype=np.float64) #get vertices of the mesh as a numpy array
    F = np.asarray(mesh.triangles, np.int64) #get faces of the mesh as a numpy array
    color=np.zeros((int(np.size(V)/3),0))
    if mesh.has_vertex_colors():
        color=np.asarray(255*np.asarray(mesh.vertex_colors,dtype=np.float64), dtype=np.int32)
    return V, F, color


def saveData(file_name, extension, V, F, Rho=None, color=None):
    """Save mesh information either as a mat file or ply file.

    Input:
        - file_name: specified path for saving mesh [string]
        - extension: extension for file_name, i.e., "mat" or "ply"
        - V: vertices of the triangulated surface [nVx3 numpy ndarray]
        - F: faces of the triangulated surface [nFx3 numpy ndarray]
        - Rho: weights defined on the vertices of the triangulated surface [nVx1 numpy ndarray, default=None]
        - color: colormap [nVx3 numpy ndarray of RGB triples]

    Output:
        - file_name.mat or file_name.ply file containing mesh information
    """

    # Save as .mat file
    if extension == 'mat':
        if Rho is None:
            savemat(file_name + ".mat", {'V': V, 'F': F + 1})
        else:
            savemat(file_name + ".mat", {'V': V, 'F': F + 1, 'Rho': Rho})

            # Save as .ply file
    else:
        nV = V.shape[0]
        nF = F.shape[0]
        file = open("{}.ply".format(file_name), "w")
        lines = ("ply", "\n", "format ascii 1.0", "\n", "element vertex {}".format(nV), "\n", "property float x", "\n",
                 "property float y", "\n", "property float z", "\n")

        if color is not None:
            lines += ("property uchar red", "\n", "property uchar green", "\n", "property uchar blue", "\n")
            if Rho is not None:
                lines += ("property uchar alpha", "\n")

        lines += ("element face {}".format(nF), "\n", "property list uchar int vertex_index", "\n", "end_header", "\n")

        file.writelines(lines)
        lines = []
        for i in range(0, nV):
            for j in range(0, 3):
                lines.append(str(V[i][j]))
                lines.append(" ")
            if color is not None:
                for j in range(0, 3):
                    lines.append(str(color[i][j]))
                    lines.append(" ")
                if Rho is not None:
                    lines.append(str(Rho[i]))
                    lines.append(" ")

            lines.append("\n")
        for i in range(0, nF):
            l = len(F[i, :])
            lines.append(str(l))
            lines.append(" ")

            for j in range(0, l):
                lines.append(str(F[i, j]))
                lines.append(" ")
            lines.append("\n")

        file.writelines(lines)
        file.close()