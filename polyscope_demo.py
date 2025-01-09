from path_solver import PathSolver
import os
import numpy as np
from surfaces import Surface
import polyscope.imgui as psim
import polyscope as ps
import scipy.interpolate
import torch
import argparse
from utils import get_color, load_mesh

device = "cuda:0"

def register_surface(array, name, triv, index_color, transparency=1.0):
    mesh = ps.register_surface_mesh(name, array, triv)
    mesh.set_color(tuple(int(get_color(index_color)[i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(True)
    mesh.set_transparency(transparency)
    return mesh


def load_pair(wanted_indices, opt, faust_path):
    a = 0.5
    i, j = wanted_indices
    path_0 = os.path.join(faust_path, "tr_reg_{0:03d}.ply".format(i))
    print("loading inputs")
    surf_0 = load_mesh(path_0)
    path_1 = os.path.join(faust_path, "tr_reg_{0:03d}.ply".format(j))
    surf_1 = load_mesh(path_1)


    # compute spatial interp
    inputs_spatial_interp = (1 - a) * surf_0.vertices + a * surf_1.vertices
    # Display
    register_surface(surf_0.vertices, "Input 0", opt['TRIV'], 0, transparency=0.25)
    register_surface(surf_1.vertices, "Input 1", opt['TRIV'], 0, transparency=0.25)
    register_surface(inputs_spatial_interp, "Linear", opt['TRIV'], 5, transparency=0.75)

    return [surf_0.vertices, surf_1.vertices]

def interp(inputs, opt_ui):
    global rec_memory, opt
    rec_memory = None
    print("computation")
    verts_0, verts_1 = torch.from_numpy(inputs[0]).float().cuda(), torch.from_numpy(inputs[1]).float().cuda()
    path, distance = path_solver.compute_path_scipy(opt["a"], opt["b"], opt["c"], verts_0, verts_1, opt["tm"], verbose=True, fast=True)
    full_chemin = path.detach().cpu().numpy()

    time = np.linspace(0, 1, opt["tm"])
    rec_memory = scipy.interpolate.interp1d(time, full_chemin, axis=0)
    # Display
    register_surface(rec_memory(opt_ui["a"]), "Geodesic", opt_ui['TRIV'], 1)


def interp_a(inputs, opt, a):
    global rec_memory
    vertices_interp_i = (1 - a) * inputs[0] + a * inputs[1]
    if rec_memory is not None:
        rec_spatial = rec_memory(a)
        register_surface(rec_spatial, "Geodesic", opt['TRIV'], 1)
    register_surface(vertices_interp_i, "Linear", opt['TRIV'], 5, transparency=0.75)


def get_opts():

    opt_ui = {}

    opt_ui["start"] = True

    opt_ui["index_couple_0"] = 10
    opt_ui["index_couple_1"] = 871

    return opt_ui, opt, opt_infos

changed = True
def callback():
    global opt_ui, opt, opt_infos
    global changed
    global rec_memory
    psim.PushItemWidth(200)

    # TreeNode Latent interpolation
    psim.SetNextItemOpen(True)
    if psim.TreeNode("Latent interpolation"):
        # Choose indices

        _, opt_ui["index_couple_0"] = psim.InputInt("Index 0", opt_ui["index_couple_0"])
        _, opt_ui["index_couple_1"] = psim.InputInt("Index 1", opt_ui["index_couple_1"])

        if opt_ui["index_couple_0"] > 100:
            opt_ui["index_couple_0"] = 100

        if opt_ui["index_couple_1"] > 100:
            opt_ui["index_couple_1"] = 100

        if psim.Button("Random indices"):
            opt_ui["index_couple_0"] = 2
            opt_ui["index_couple_1"] = 7

        if psim.Button("Load pair"):
            ps.remove_all_structures()
            rec_memory = None
            opt_ui["inputs"] = load_pair([opt_ui["index_couple_0"], opt_ui["index_couple_1"]], opt_ui, faust_path)

        psim.TextUnformatted("Metric Parameters")
        _, opt["a"] = psim.InputFloat("a", opt["a"])
        psim.SameLine()
        _, opt["b"] = psim.InputFloat("b", opt["b"])
        _, opt["c"] = psim.InputFloat("c", opt["c"])

        if psim.Button("Compute interp (slow)"):
            psim.TextUnformatted("Computing ...")
            interp(opt_ui["inputs"], opt_ui)
            opt_ui["a"] = 0.5
            psim.TextUnformatted("Done !")


        if "inputs" in opt_ui:
            changed_a, opt_ui["a"] = psim.SliderFloat("Interp a", opt_ui["a"], v_min=0, v_max=1)

            if changed_a:
                interp_a(opt_ui["inputs"], opt_ui, opt_ui["a"])

        psim.TreePop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and show interpolation in an interactive way")
    parser.add_argument('--a', type=float, default=1, help="Coefficient a of the metric")
    parser.add_argument('--b', type=float, default=1, help="Coefficient b of the metric")
    parser.add_argument('--c', type=float, default=0., help="Coefficient c of the metric")
    parser.add_argument('--tm', type=int, default=5, help="Time discretization of the geodesic")
    parser.add_argument("--cache", type=str, default="cache", help="Cache directory")
    parser.add_argument("--faust_path", type=str, default="", help="Faust registered shapes directory")
    parser.add_argument("--base_path", type=str, default="base_shape.npy", help="Deformation basis path")
    parser.add_argument("--n_elem", type=int, default=70, help="Number of basis elements, more will be slower")
    parser.add_argument("--metric", type=str, default="elastic", help="If you want to compare to other metrics")
    args = parser.parse_args()

    faust_path = args.faust_path
    print("Path to faust: ", faust_path)
    cache_path = args.cache

    opt = {"a": args.a, "b": args.b, "c": args.c, "tm": args.tm}

    path_init = os.path.join(faust_path, "tr_reg_000.ply")
    surf = Surface(filename=path_init)
    rec_memory = None
    opt_ui = {"index_couple_0": 2, "index_couple_1": 7, "a": 0.5, "inputs": None, "TRIV": surf.faces}
    path_solver = PathSolver(path_init, args.base_path, args.n_elem, energy_name=args.metric)
    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_ground_plane_height_factor(0)  # adjust the plane height

    ps.set_user_callback(callback)
    ps.show()
