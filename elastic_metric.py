import numpy as np
import torch

import os
from surfaces import Surface
import pickle
import torch.nn as nn
from utils import load_mesh
from elastic_utils import multi_dim_inv_for_22mn, Squared_distance

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732
def one_ring(surface):
    res = [[] for i in range(surface.vertices.shape[0])]
    for f in surface.faces:
        for i in range(1, 4):
            index = f[-i]
            for j in range(1, 3):
                if f[-i+j] not in res[index]:
                    res[index].append(f[-i+j])
    for i in range(len(res)):
        res[i] = [i] + res[i]
    return res

def two_ring(one_r, n_vertices):
    res = []
    for i in range(n_vertices):
        temp = [k for k in one_r[i]]
        end = [k for k in one_r[i]]
        for k in temp:
            for j in one_r[k]:
                if j not in end:
                    end.append(j)
        res.append(end)
    return res, None
def get_ring(surf, index=1, device="cpu"):
    one_ring_ngh = one_ring(surf)
    n_v = surf.vertices.shape[0]
    if index == 1:
        ring = one_ring_ngh
    elif index == 2:
        two_ring_ngh, l = two_ring(one_ring_ngh, n_v)
        ring = two_ring_ngh  # two_ring_ngh
    else:
        raise Exception("Ring size not possible")
    m = max([len(r) for r in ring])
    total_m = [m - len(r) for r in ring]
    ring_array = np.array(
        [[ring[k][i] for i in range(m - total_m[k])] + [k for j in range(total_m[k])] for k in range(n_v)],
        dtype=int)
    count_array = torch.from_numpy(np.array([len(r) for r in ring])).to(device)
    return ring_array, count_array, m

def l_vals(verts, faces):
    device = verts.device
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    x12 = v1 - v0
    tan_vec = x12
    l1 = torch.norm(tan_vec, dim=1)
    x13 = v2 - v0
    surfels_face = torch.cross(x12, x13, dim=1)
    normals = surfels_face / torch.norm(surfels_face, dim=1, keepdim=True)
    ortho_vectors = tan_vec / torch.norm(tan_vec, dim=1, keepdim=True)
    second_vectors = torch.cross(normals, ortho_vectors)
    e3u = (x13 * ortho_vectors).sum(dim=1)
    e3v = (x13 * second_vectors).sum(dim=1)
    rotation = torch.stack((ortho_vectors, second_vectors, normals), dim=2)
    rotation_inverse = torch.linalg.inv(rotation)
    return l1.to(device), e3u.to(device), e3v.to(device), rotation_inverse.to(device)




def add(surfeli, faces, surfel_face, i):
    surfeli.scatter_add_(0, faces[:, 0], surfel_face[:, i])
    surfeli.scatter_add_(0, faces[:, 1], surfel_face[:, i])
    surfeli.scatter_add_(0, faces[:, 2], surfel_face[:, i])

def rotate(vertices, faces_packed, ring_var):
    n = vertices.shape[0]
    device = vertices.device
    dtype = vertices.dtype
    face_verts = vertices[faces_packed]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    elem_area = torch.zeros(n).to(device, dtype=dtype)
    x12 = v1 - v0
    x13 = v2 - v0
    surfels_face = torch.cross(x12, x13, dim=1)
    areas_f = torch.norm(surfels_face, dim=1) / 2
    elem_area.scatter_add_(0, faces_packed[:, 0], areas_f / 3)
    elem_area.scatter_add_(0, faces_packed[:, 1], areas_f / 3)
    elem_area.scatter_add_(0, faces_packed[:, 2], areas_f / 3)
    # print(torch.sum(elem_area).item())
    # print(torch.sum(areas_f).item())
    surfelx = torch.zeros(n).to(device, dtype=dtype)
    surfely = torch.zeros(n).to(device, dtype=dtype)
    surfelz = torch.zeros(n).to(device, dtype=dtype)
    add(surfelx, faces_packed, surfels_face, 0)
    add(surfely, faces_packed, surfels_face, 1)
    add(surfelz, faces_packed, surfels_face, 2)
    surfels = torch.stack((surfelx, surfely, surfelz), -1)
    normals = surfels / torch.norm(surfels, dim=1, keepdim=True)
    tan_vec = vertices[ring_var[:, 1], :] - vertices
    tan_vec = tan_vec - torch.einsum('ij,ij->i', tan_vec, normals)[:, None] * normals
    ortho_vectors = tan_vec / torch.norm(tan_vec, dim=1, keepdim=True)
    second_vectors = torch.cross(normals, ortho_vectors)
    rotation = torch.stack((second_vectors, ortho_vectors, normals), dim=2)
    rotation_inverse = torch.inverse(rotation)

    # Calcul de la normale et des courbures principales
    #

    pt = vertices[ring_var, :] - vertices[:, None, :]

    pt_redresse = torch.matmul(rotation_inverse[:, None, :, :], pt[:, :, :, None]).squeeze()
    x = pt_redresse[:, :, 0]
    y = pt_redresse[:, :, 1]
    z = pt_redresse[:, :, 2]
    return x, y, z, elem_area, normals

def compute_B2_ref(x_ref, y_ref, m, count_array):
    n = x_ref.shape[0]
    B = torch.zeros((n, m, 6))

    B[:, :, 0] = x_ref * x_ref
    B[:, :, 1] = y_ref * y_ref
    B[:, :, 2] = x_ref * y_ref
    B[:, :, 3] = x_ref
    B[:, :, 4] = y_ref
    B[:, :, 5] = 1
    B2 = B[:, :, :, None] * B[:, :, None, :]
    B2 = B2.sum(axis=1)
    B2[:, 5, 5] = count_array
    return B2

def surface_quadrique_template(signal, x_ref, y_ref, B2):
    ## Solve for quadric equation of the signal given initial tangent coordinates
    s_B = torch.zeros((B2.shape[0], 6))
    s_B[:, 0] = (signal * x_ref * x_ref).sum(dim=-1)
    s_B[:, 1] = (signal * y_ref * y_ref).sum(dim=-1)
    s_B[:, 2] = (signal * x_ref * y_ref).sum(dim=-1)
    s_B[:, 3] = (signal * x_ref).sum(dim=-1)
    s_B[:, 4] = (signal * y_ref).sum(dim=-1)
    s_B[:, 5] = (signal).sum(dim=-1)
    try:
        A = torch.linalg.solve(B2, s_B[:, :, np.newaxis])
    except RuntimeError as err:
        # TODO: Acessible debugging
        raise("The Quadric computation crashed")
    return A.squeeze()


def area_normals_face(vertices, faces_packed):
    n = vertices.shape[0]
    v0, v1, v2 = vertices[faces_packed[:, 0]], vertices[faces_packed[:, 1]], vertices[faces_packed[:, 2]]
    x12 = v1 - v0
    x13 = v2 - v0
    surfels_face = torch.cross(x12, x13, dim=-1)
    normals = surfels_face / torch.norm(surfels_face, dim=-1, keepdim=True)
    areas_f = torch.norm(surfels_face, dim=-1) / 2
    return areas_f, normals

def area_normals_face_batched(vertices, faces_packed):
    n = vertices.shape[0]
    v0, v1, v2 = vertices[:, faces_packed[:, 0]], vertices[:, faces_packed[:, 1]], vertices[:, faces_packed[:, 2]]
    x12 = v1 - v0
    x13 = v2 - v0
    surfels_face = torch.cross(x12, x13, dim=-1)
    normals = surfels_face / torch.norm(surfels_face, dim=-1, keepdim=True)
    areas_f = torch.norm(surfels_face, dim=-1) / 2
    return areas_f, normals


def robustesse(k, robust=True):
    ## As we compute discrete curvature, it is better to use a robustness "filter"
    # To avoid singularities during optimization.
    if robust:
        c = 0.01
        return 2 * ((k / c) ** 2) / ((k / c) ** 2 + 4)
    else:
        return k**2

class ElasticMetric(nn.Module):

    def __init__(self, filename, device, verbose=False, index_ring=1, dtype=torch.float64):
        if verbose:
            print('computing template quantities')
        self.device = device
        self.dtype = dtype
        self.surf_template = load_mesh(filename)
        self.verts = torch.from_numpy(self.surf_template.vertices).to(device, dtype=self.dtype)
        self.faces = torch.from_numpy(self.surf_template.faces).to(device).long()

        self.areas = torch.from_numpy(np.linalg.norm(self.surf_template.surfel, axis=1)).to(device, dtype=self.dtype)
        self.n_v = self.verts.shape[0]
        self.n_f = self.faces.shape[0]
        av, _ = self.surf_template.computeVertexArea()
        self.elem_vertices = torch.from_numpy(av).to(device, dtype=self.dtype)
        self.elem_faces = areas = torch.from_numpy(np.linalg.norm(self.surf_template.surfel, axis=1)).to(device)


        ## Initial triangle quantities (see paper appendix)
        self.l1, self.e3u, self.e3v, self.rotation = l_vals(self.verts, self.faces)

        #Least square system initialisation (for alternative computation)
        self.ring_array, self.count_array, self.m = get_ring(self.surf_template, index_ring, device=device)
        self.x, self.y, self.z, _, _ = rotate(self.verts, self.faces, self.ring_array)
        self.B2_ref = compute_B2_ref(self.x, self.y, self.m, self.count_array)
        if verbose:
            print("Done")


    def get_metric_face(self, vertices):
        # Uses discrete geometry (see paper appendix) vertices is of size [N_v 3]
        # Note that the operations are all linear for computing dPdu, dPdv, so you might want
        # To use a precomputed matrix as well. This way I find it a bit more understandable compared to the formulas
        # of the article.
        areas_f, normals = area_normals_face(vertices, self.faces)
        face_verts = vertices[self.faces]
        # verts_rot = torch.einsum('ijl, ikl -> ikj', rotation, face_verts)
        p0, p1, p2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
        dPdu = (p1 - p0) / self.l1[:, None]
        dPdv = (p0 - p1) * (self.e3u / (self.e3v * self.l1))[:, None] + (p2 - p0) / self.e3v[:, None]
        eps = 1e-6
        g = torch.zeros((self.faces.shape[0], 2, 2)).to(self.device, dtype=self.dtype)
        g[:, 0, 0] = (dPdu ** 2).sum(dim=1)
        g[:, 0, 0] = g[:, 0, 0]
        g[:, 1, 0] = (dPdu * dPdv).sum(dim=1)
        g[:, 0, 1] = (dPdu * dPdv).sum(dim=1)
        g[:, 1, 1] = (dPdv ** 2).sum(dim=1)
        g[:, 1, 1] = g[:, 1, 1]
        return g, normals, areas_f

    def get_metric_face_batched(self, vertices):
        # Uses discrete geometry, this time vertices is of size [B N_v 3]
        B = vertices.shape[0]
        areas_f, normals = area_normals_face_batched(vertices, self.faces)
        # verts_rot = torch.einsum('ijl, ikl -> ikj', rotation, face_verts)
        p0, p1, p2 = vertices[:, self.faces[:, 0]], vertices[:, self.faces[:, 1]], vertices[:, self.faces[:, 2]]
        dPdu = (p1 - p0) / self.l1[:, None]
        dPdv = (p0 - p1) * (self.e3u / (self.e3v * self.l1))[:, None] + (p2 - p0) / self.e3v[:, None]
        # eps = 1e-6
        g = torch.zeros((B, self.faces.shape[0], 2, 2)).to(vertices.device, dtype=self.dtype)
        g[:, :, 0, 0] = (dPdu ** 2).sum(dim=-1)
        # g[:, 0, 0] = (1 + eps) * g[:, 0, 0]
        g[:, :, 1, 0] = (dPdu * dPdv).sum(dim=-1)
        g[:, :, 0, 1] = (dPdu * dPdv).sum(dim=-1)
        g[:, :, 1, 1] = (dPdv ** 2).sum(dim=-1)
        # g[:, 1, 1] = (1 + eps) * g[:, 1, 1]
        return g, normals, areas_f

    def get_metric_vertices(self, vertices):
        # Uses a quadric definition of the derivative. If you want to use discrete geometry, you would need to define
        # The gradient on the vertices (see diffusionnet implementation for an example). It will be probably faster
        # In the usual monge parameterization, we always have metric = [[1, 0], [0, 1]], we thus need to compute the
        # relative metric to template (fixed) parameterization
        x, y, _, elem_area, normals = rotate(vertices, self.faces, self.ring_array)
        Ax = surface_quadrique_template(x, self.x, self.y, self.B2_ref) #quadric of x coordinate
        Ay = surface_quadrique_template(y, self.x, self.y, self.B2_ref) #quadric of y coordinate
        # The function is :
        # x = Ax(u, v) = Ax[0]*u**2 + Ax[1]*v**2 + Ax[2]*v*u + Ax[3]*x + Ax[4]*y + Ax[5]
        # y = Ay(u, v) = ...
        # z = f(x,y), but gradf(x,y) = 0 (since we rotated the surface, z is a local minima)
        # print(Ax.shape)
        eps = 1e-6
        g = torch.zeros((vertices.shape[0], 2, 2)).to(self.device)
        g[:, 0, 0] = Ax[:, 3] ** 2 + Ay[:, 3] ** 2
        g[:, 0, 0] = (1 + eps) * g[:, 0, 0]
        g[:, 1, 0] = Ax[:, 3] * Ax[:, 4] + Ay[:, 3] * Ay[:, 4]
        g[:, 0, 1] = Ax[:, 3] * Ax[:, 4] + Ay[:, 3] * Ay[:, 4]
        g[:, 1, 1] = Ay[:, 4] ** 2 + Ax[:, 4] ** 2
        g[:, 1, 1] = (1 + eps) * g[:, 1, 1]
        return g, normals, elem_area


    def get_ls_curvature(self, vertices):
        ## Obtaining curvatures in the least square sense
        # For this we use classic monge parameterization.
        # Note that we just have to use the second fundamental form: the first one is the identity!
        x, y, z, elem_area, normals = rotate(vertices, self.faces, self.ring_array)
        B2 = compute_B2_ref(x, y, self.m, self.count_array)
        A = surface_quadrique_template(z, x, y, B2)
        courbure_MOYENNE = A[:, 0] + A[:, 1]
        courbure_GAUSS = 4 * A[:, 0] * A[:, 1] - A[:, 2] ** 2
        k1 = courbure_MOYENNE + torch.sqrt(torch.absolute(courbure_MOYENNE ** 2 - courbure_GAUSS))
        k2 = courbure_MOYENNE - torch.sqrt(torch.absolute(courbure_MOYENNE ** 2 - courbure_GAUSS))
        return k1, k2, normals, elem_area


    def get_elastic_energy(self, a, b, c, path):
        ## Compute elastic energy (paper of a path)
        # Also return D, the distance of the path (in case you need it)
        tm = path.shape[0]
        dgdt = torch.zeros((self.n_f, tm - 1, 2, 2)).to(self.device)
        gt, normals_t, area_t = self.get_metric_face_batched(path)
        dgdt = (gt[1:] - gt[:-1])/(tm-1)


        detg = gt[:, :, 0, 0] * gt[:, :, 1, 1] - gt[:, :, 0, 1] * gt[:, :, 1, 0] + 1e-7
        g_inv = multi_dim_inv_for_22mn(gt, detg)
        dgdeltag = torch.einsum("ijkl, ijlm-> ijkm", g_inv[:-1, :, :, :], dgdt)
        quant_tr = (dgdeltag[:, :, 0, 0] + dgdeltag[:, :, 1, 1])

        deltag0 = dgdt - (0.5 * quant_tr[:, :, None, None]) * gt[:-1, :, :, :]
        mat = torch.einsum("ijkl, ijlm-> ijkm", g_inv[:-1, :, :, :], deltag0)
        mat_sq = torch.einsum("ijkl, ijlm-> ijkm", mat, mat)
        quant = mat_sq[:, :, 0, 0] + mat_sq[:, :, 1, 1]

        dNdt_sq = ((normals_t[1:] - normals_t[:-1])/(tm-1))**2

        integrand = a * quant + b * quant_tr**2 + c * dNdt_sq.sum(dim=2)
        integrand *= torch.sqrt(detg[:-1])

        # We first integrate per shapes, but we could as well do it the other way around
        integrale_shapes = torch.zeros(tm)
        for t in range(tm - 1):
            integrale_shapes[t] = torch.sum(integrand[t] * self.elem_faces)
        tps = torch.linspace(0, 1, tm)
        E = torch.trapz(integrale_shapes, tps)
        D = torch.trapz(torch.sqrt(integrale_shapes), tps)
        return E, D

    def get_srnf_energy(self, path):
        tm = path.shape[0]
        gt, normals_t, area_t = self.get_metric_face_batched(path)
        srnfs = torch.sqrt(area_t)[:, :, None] * normals_t

        detg = gt[:, :, 0, 0] * gt[:, :, 1, 1] - gt[:, :, 0, 1] * gt[:, :, 1, 0] + 1e-7
        dQdt_sq = ((srnfs[1:] - srnfs[:-1])/(tm-1))**2
        integrand = dQdt_sq.sum(dim=2)
        integrand *= torch.sqrt(detg[:-1])# *= area_t[:-1]
        # We first integrate per shapes, but we could as well do it the other way around
        integrale_shapes = torch.zeros(tm)
        for t in range(tm - 1):
            integrale_shapes[t] = torch.sum(integrand[:, t] * self.elem_faces)
        tps = torch.linspace(0, 1, tm)
        E = torch.trapz(integrale_shapes, tps)
        D = torch.trapz(torch.sqrt(integrale_shapes), tps)
        return E, D


    def get_gauge_invariant_energy(self, a, b, c, path, robust=False):
        tm = path.shape[0]
        device = path.device
        elems = torch.zeros((tm-1, self.n_v)).to(device)
        N = torch.zeros((tm, self.n_v, 3)).to(device)
        k1 = torch.zeros((tm-1, self.n_v)).to(device)
        k2 = torch.zeros((tm-1, self.n_v)).to(device)
        for t in range(tm-1):
            # with torch.no_grad():
            k1_out, k2_out, normals, area = self.get_ls_curvature(path[t, :, :])
            elems[t, :] = area
            N[t, :, :] = normals
            k1[t, :] = k1_out
            k2[t, :] = k2_out

        dchem = (path[1:] - path[:-1])/(tm-1)
        _, _, _, _, normals_end = rotate(path[tm-1, :, :], self.faces, self.ring_array)
        N[tm-1, :, :] = normals_end
        dNdt = (N[1:] - N[:-1])/(tm-1)
        dchem_ortho = (dchem * N[:-1, :, :]).sum(dim=-1)
        norme_au_carre_dchem_perp = dchem_ortho ** 2

        integrand = torch.zeros(norme_au_carre_dchem_perp.shape).to(device)
        if a >= 0:
            integrand += 2. * a * norme_au_carre_dchem_perp * robustesse((k1 - k2))
        if b >= 0:
            integrand += 4. * b * norme_au_carre_dchem_perp * (robustesse(k1 + k2))
        if c >= 0:
            integrand += c * torch.norm(dNdt, dim=2) ** 2

        integrale_shapes = torch.zeros(tm)
        for t in range(tm - 1):
            integrale_shapes[t] = torch.sum(integrand[:, t] * self.elem_faces)
        tps = torch.linspace(0, 1, tm)
        E = torch.trapz(integrale_shapes, tps)
        D = torch.trapz(torch.sqrt(integrale_shapes), tps)
        return E, D


    def elastic_distance_faces(self, vertices_1, vertices_2, alpha, lambd, beta, srnf=True):
        # Compute the distance between (g, n) representations. Note that the result will be different from
        # computing geodesic + mesuring distance !!
        g1, normals1, elem1 = self.get_metric_face(vertices_1)
        q1 = normals1
        if srnf:
            q1 = torch.sqrt(elem1)[:, None] * q1
        g2, normals2, elem2 = self.get_metric_face(vertices_2)
        q2 = normals2
        if srnf:
            q2 = torch.sqrt(elem2)[:, None] * q2
        return Squared_distance([g1, q1], [g2, q2], alpha, lambd, beta, self.elem_faces, srnf)


    def elastic_distance_vertices(self, vertices_1, vertices_2, faces_idx, alpha, lambd, beta, srnf=True):
        g1, normals1, elem1 = self.get_metric_vertices(vertices_1)
        q1 = normals1
        if srnf:
            q1 = torch.sqrt(elem1)[:, None] * q1
        g2, normals2, elem2 = self.get_metric_vertices(vertices_2)
        q2 = normals2
        if srnf:
            q2 = torch.sqrt(elem2)[:, None] * q2
        return Squared_distance([g1.double(), q1.double()], [g2.double(), q2.double()], alpha, lambd, beta, self.elem_vertices, srnf)


if __name__ == '__main__':
    path_smpl = "basicmodel_m_lbs_10_207_0_v1.1.0.pkl"
    device = "cuda:0"
    met = ElasticMetric(path_smpl, device, True)
    faust_path = "/home/emerypierson/Documents/datasets/MPI-FAUST/training/registrations"

    tm = 8
    a, b = 1, 1
    C = 0
    n_max = 20

    surf = Surface(filename=os.path.join(faust_path, "tr_reg_000.ply"))
    verts_1 = torch.from_numpy(surf.vertices).double()
    verts_1 = verts_1.to(device)
    surf = Surface(filename=os.path.join(faust_path, "tr_reg_018.ply"))
    verts_2 = torch.from_numpy(surf.vertices).double()
    verts_2 = verts_2.to(device)

    path = torch.cat((verts_1[None, :, :], verts_2[None, :, :]), dim=0).cuda()
    g1, normals1, areas_1 = met.get_metric_face(verts_1)
    g, normals, areas = met.get_metric_face_batched(path)
    print(met.get_elastic_energy(1, 1, 1, path))