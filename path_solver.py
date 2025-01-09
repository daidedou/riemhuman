import torch
import torch.autograd as ag
import copy
import numpy as np
from scipy.optimize import minimize
from elastic_metric import ElasticMetric
from utils import load_mesh, get_color
from tqdm import tqdm
from surfaces import Surface
import matplotlib.pyplot as plt
from vtkviz.vtkVisualization import *
import polyscope as ps
import os
import argparse

#torch.autograd.set_detect_anomaly(True)


def init_chemin(vertices_init, vertices_fin, tm):
    # Chemin initial
    device = vertices_init.device
    chemin_initial = torch.zeros((tm, vertices_init.shape[0], 3)).to(device)
    for t in range(tm):
        chemin_initial[t, :, :] = vertices_init + (1. * t) / (tm - 1) * (vertices_fin - vertices_init)
    return chemin_initial
class PathSolver:
    def __init__(self, filename, basis_path, n_elem, energy_name="elastic", device="cuda", dtype=torch.float64):
        self.filename = filename
        self.surf_template = load_mesh(filename)
        self.device = device
        self.dtype = dtype
        basis_np = np.load(basis_path)[-n_elem:]
        self.basis = torch.from_numpy(basis_np).to(device, dtype=self.dtype)
        self.verts = torch.from_numpy(self.surf_template.vertices).to(device, dtype=self.dtype)
        self.faces = torch.from_numpy(self.surf_template.faces).to(device).long()

        self.elastic = ElasticMetric(filename, device, dtype=dtype)
        self.energy_name = energy_name

    def get_energy(self, a, b, c, path):
        if self.energy_name == "srnf":
            return self.elastic.get_srnf_energy(path)
        elif self.energy_name == "elastic":
            return self.elastic.get_elastic_energy(a, b, c, path)
        elif self.energy_name == "gauge":
            return self.elastic.get_gauge_invariant_energy(a, b, c, path)
        else:
            raise Exception("Unknown energy name :" + self.energy_name)

    def get_path(self, init_path, X_t, tm, fast=False):
        chemin_exp = copy.deepcopy(init_path.detach())
        device = chemin_exp.device
        if fast:
            ## Reduce the number of parameters by using only one vectors instead of tm-2 vectors
            ## Details in "Gauge Invariant Framework for Shape Analysis of Surfaces", Tumpach et al.
            ## See paragraph 4.3, part 2., we simply set J=1
            tps = torch.linspace(0, 1, tm).to(device, dtype=self.dtype)
            coeffs = torch.sin((0 + 1) * np.pi * tps)[:, np.newaxis] * X_t[np.newaxis, :]
            chemin_exp += torch.einsum("ij, jkl-> ikl", coeffs, self.basis)
        else:
            X_ = X_t.reshape((tm - 2, -1))
            chemin_exp[1:-1, :, :] += torch.einsum("ij, jkl-> ikl", X_, self.basis)
        return chemin_exp

    def compute_path_scipy(self, a, b, c, verts_1, verts_2, tm, verbose=True, fast=False):
        init_path = init_chemin(verts_1, verts_2, tm)
        device = verts_1.device
        E, D = self.get_energy(a, b, c, init_path)
        if verbose:
            print("Initial energy: ", E)
        def objective(X, full_chemin):
            X_t = torch.from_numpy(X).float().to(device)
            X_t.requires_grad = True
            chemin_exp = self.get_path(init_path, X_t, tm, fast=fast)
            E, _ = self.get_energy(a, b, c, chemin_exp)
            first = ag.grad(E, X_t, create_graph=True)[0]
            # Rescale can be needed sometimes
            l = 1e9 * E.item()
            grad = first.detach().cpu().numpy().reshape(-1)
            return l, 1e9*np.double(grad)


        if fast:
            X0 = np.zeros(self.basis.shape[0])
        else:
            X0 = np.zeros((self.basis.shape[0] * (tm - 2)))
        if verbose:
            print("Optimization started")
        res = minimize(objective, X0, jac=True, args=(init_path),
                       method="L-BFGS-B")
        if verbose:
            print("End of optimization")
        with torch.no_grad():
            X_t = torch.from_numpy(res.x).float().to(device)
            chemin_exp = self.get_path(init_path, X_t, tm, fast=fast)
            E, _ = self.get_energy(a, b, c, chemin_exp)
        if verbose:
            print("Final Energy: ", E.item())
        return chemin_exp, D.item()

    def compute_path_torch(self, a, b, c, verts_1, verts_2, tm, n_loop=1000, debug=False, fast=False):
        init_path = init_chemin(verts_1, verts_2, tm)
        device = verts_1.device
        if fast:
            X_t = torch.zeros(self.basis.shape[0]).float().to(device)
        else:
            X_t = torch.zeros((tm-2)*self.basis.shape[0]).float().to(device)
        X_t.requires_grad = True
        optimizer = torch.optim.SGD([X_t], lr=1e-7)
        E, _ = self.get_energy(a, b, c, init_path)
        print(E.item())
        grads_norm = []

        for i in tqdm(range(n_loop), "Optimizing"):
            optimizer.zero_grad()
            chemin_exp = self.get_path(init_path, X_t, tm, fast=fast)
            E, _ = self.get_energy(a, b, c, chemin_exp)
            loss = E
            loss.backward()
            torch.nn.utils.clip_grad_norm_([X_t], 100)
            if debug:
                grads_norm.append(torch.norm(X_t.grad).item())
            optimizer.step()
        print(E.item())
        if debug:
            plt.plot(grads_norm)
            plt.show()
        with torch.no_grad():
            chemin_exp = self.get_path(init_path, X_t, tm, fast=fast)
            _, D = self.elastic.get_elastic_energy(a, b, c, chemin_exp)
        return chemin_exp, D.item()

    def compute_karcher_mean(self, a, b, c, batched_vertices, mean_init, verbose=True):
        print("Computing karcher mean")
        mean_init = batched_vertices.mean(dim=0).detach()
        def objective(X, mean_init):
            mean_exp = copy.deepcopy(mean_init.detach())
            X_t = torch.from_numpy(X).float().to(device)
            X_t.requires_grad = True
            mean_exp = copy.deepcopy(mean_init.detach()) + (X_t[:, None, None] * self.basis).sum(dim=0)
            vals = torch.zeros(batched_vertices.shape[0])
            for i in range(batched_vertices.shape[0]):
                vals[i] = self.elastic.elastic_distance_faces(mean_exp, batched_vertices[i], a, b, c)
            karcher_loss = vals.sum()
            first = ag.grad(karcher_loss, X_t, create_graph=True)[0]
            grad = first.detach().cpu().numpy().reshape(-1)
            return karcher_loss.item(), np.double(grad)

        X0 = np.zeros(self.basis.shape[0])
        if verbose:
            print("Optimization started")
        res = minimize(objective, X0, jac=True, args=(mean_init),
                       method="L-BFGS-B", options={"maxfun": 250})
        if verbose:
            print("End of optimization")
        with torch.no_grad():
            X_t = torch.from_numpy(res.x).float().to(device)
            mean_exp = copy.deepcopy(mean_init.detach())
            mean_exp += (X_t[:, None, None] * self.basis).sum(dim=0)
        return mean_exp




def vizualize_geodesic(chemin, faces):
    renderer = VTKMultipleVizualization(chemin.shape[0])
    actors = []
    for t in range(chemin.shape[0]):
        actors.append(VTKSurface(chemin[t, :, :], faces))
    renderer.add_entities(actors)
    renderer.show()

def vizualize_mean(verts, mean, faces):
    render = ps.init()
    color_0 = tuple(int(get_color(0)[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    for t in range(verts.shape[0]):
        ps.register_surface_mesh(f"Shape {t}", verts[t, :, :], faces, color=color_0, transparency=0.5)
    color_1 = tuple(int(get_color(1)[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    ps.register_surface_mesh("Mean", mean, faces, color=color_1)
    ps.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="base_shape.npy")
    parser.add_argument("--faust_path", type=str, default="")
    parser.add_argument("--a", type=float, default=1.)
    parser.add_argument("--b", type=float, default=1.)
    parser.add_argument("--c", type=float, default=0.)
    parser.add_argument("--n_elem", type=int, default=70, help="Number of basis elements for optimization")
    parser.add_argument("--energy", type=str, default="elastic", help="If you want to use SRNF or more complicated energies")
    parser.add_argument("--karcher", action="store_true", help="Compute Karcher mean")
    parser.add_argument("--classic", action="store_true", help="Classic to reproduce the exact results of paper")

    args = parser.parse_args()
    device = "cuda:0"


    solver = PathSolver(os.path.join(args.faust_path, "tr_reg_000.ply"), args.base_path, args.n_elem,  energy_name=args.energy, device=device)
    a, b, c = args.a, args.b, args.c

    if args.karcher:
        ids_pose = [0, 1, 3, 4, 7]
        verts_list = []
        for id_ in ids_pose:
            surf_ = Surface(filename=os.path.join(args.faust_path, f"tr_reg_{id_:03d}.ply"))
            center = surf_.surfCenter()
            # surf_.updateVertices((surf_.vertices - center) / surf_.volume)
            verts_list.append(torch.from_numpy(surf_.vertices).double().to(device)[None, :])
        surf_ = Surface(filename=os.path.join(args.faust_path, f"tr_reg_{ids_pose[0]:03d}.ply"))
        batched_vertices = torch.cat(verts_list, dim=0)
        mean_exp = solver.compute_karcher_mean(a, b, c, batched_vertices,
                                               torch.from_numpy(surf_.vertices).float().to(device))
        vizualize_mean(batched_vertices.detach().cpu().numpy(), mean_exp.detach().cpu().numpy(), surf_.faces)
    else:
        tm = 8
        surf = Surface(filename=os.path.join(args.faust_path, "tr_reg_002.ply"))
        verts_1 = torch.from_numpy(surf.vertices).float().to(device)
        surf = Surface(filename=os.path.join(args.faust_path, "tr_reg_007.ply"))
        verts_2 = torch.from_numpy(surf.vertices).float().to(device)
        path, distance = solver.compute_path_scipy(a, b, c, verts_1, verts_2, tm, verbose=True, fast=True)
        #path, distance = solver.compute_path_torch(a, b, C, verts_1, verts_2, tm, debug=True, fast=True)
        vizualize_geodesic(path.detach().cpu().numpy(), surf.faces)