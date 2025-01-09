import torch
import numpy as np
from retrieval_utils import print_measures
from sklearn.manifold import MDS, TSNE
import open3d as o3d
from tqdm import tqdm
from surfaces import Surface
import os
import argparse
import utils
from elastic_utils import abc_to_alb
from elastic_metric import ElasticMetric
import matplotlib.pyplot as plt

def folder_loop(folder, metric, a, b, c, cache, extension="ply", srnf=False):
    all_files = sorted([f for f in os.listdir(folder) if ("." + extension) in f])
    N = len(all_files)
    dist_mat = np.zeros((N, N))
    alpha, lam, beta = abc_to_alb(a, b, c)
    surf_list = []
    for i in tqdm(range(N), "computing matrix"):
        filename_i = os.path.join(folder, all_files[i])
        surf_i = Surface(filename=filename_i)
        surf_list.append(surf_i)
        verts_i = torch.from_numpy(surf_i.vertices).float().to(metric.device)
        if not os.path.exists(cache):
            for j in range(i, N):
                filename_j = os.path.join(folder, all_files[j])
                surf_j = Surface(filename=filename_j)
                verts_j = torch.from_numpy(surf_j.vertices).float().to(metric.device)
                dist = metric.elastic_distance_faces(verts_i, verts_j, alpha, lam, beta, srnf=srnf).item()
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist
    if os.path.exists(cache):
        dist_mat = np.load(cache)
    else:
        np.save(cache, dist_mat)
    return dist_mat, surf_list


def get_3D_mds_plot(dist_mat, surf_list, cluster_index, emb="tsne"):

    if emb== "tsne":
        embedding = TSNE(n_components=3, metric="precomputed", init="random")
        scalar = 0.05
    else:
        embedding = MDS(n_components=3, dissimilarity="precomputed")
        scalar = 0.05
    X_transformed = embedding.fit_transform(dist_mat)
    ls=[]
    for id_cluster, cluster in enumerate(cluster_index):
        color = utils.get_color(id_cluster, rgb=True)
        for id_shape in cluster:
            color_array = np.zeros(surf_list[id_shape].vertices.shape)
            color_array += np.array(color)
            mesh = utils.getMeshFromData([surf_list[id_shape].vertices, surf_list[id_shape].faces], color=color_array)
            R = mesh.get_rotation_matrix_from_axis_angle(np.array([0,0,1])*np.pi/4)
            mesh.translate((scalar*X_transformed[id_shape,0],scalar*X_transformed[id_shape,1],scalar*X_transformed[id_shape,2]), relative=False)
            mesh.rotate(R, center=(scalar*X_transformed[id_shape,0],scalar*X_transformed[id_shape,1],scalar*X_transformed[id_shape,2]))
            ls+=[mesh]

    o3d.visualization.draw_geometries(ls)
    full_mesh=ls[0]
    for i in range(1,X_transformed.shape[0]):
        full_mesh+=ls[i]

    V,F,Color = utils.getDataFromMesh(full_mesh)
    if full_mesh.has_vertex_colors():
        utils.saveData("3D_mds","ply",V,F,color=Color)
    else:
        utils.saveData("3D_mds","ply",V,F)

def get_2D_mds_plot(dist_mat, cluster_index,  title=None, emb="tsne"):

    if emb== "tsne":
        embedding = TSNE(n_components=2, metric="precomputed", init="random")
        scalar = 0.05
    else:
        embedding = MDS(n_components=2, dissimilarity="precomputed")
        scalar = 0.05
    X_transformed = embedding.fit_transform(dist_mat)
    ls=[]
    for id_cluster, cluster in enumerate(cluster_index):
        color = utils.get_color(id_cluster, rgb=True)
        plt.scatter(X_transformed[cluster, 0], X_transformed[cluster, 1])
    if title is not None:
        plt.title(title)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and show distances using the elastic distance")
    parser.add_argument('--a', type=float, default=1, help="Coefficient a of the metric")
    parser.add_argument('--b', type=float, default=1, help="Coefficient b of the metric")
    parser.add_argument('--c', type=float, default=0., help="Coefficient c of the metric")
    parser.add_argument("--type_exp", type=str, default="shape", help="Shape, Pose or Manual distances")
    parser.add_argument("--cache", type=str, default="cache", help="Cache directory")
    parser.add_argument("--faust_path", type=str, default="", help="Faust registered shapes directory")
    parser.add_argument("--threed", action='store_true', help='Plot in 3d')
    args = parser.parse_args()

    metric = ElasticMetric(os.path.join(args.faust_path, "tr_reg_000.ply"), "cuda", True)

    cache = os.path.join(args.cache, "distances")
    os.makedirs(cache, exist_ok=True)
    threed = args.threed
    ids_nn = None
    if args.type_exp == "pose":
        a, b, c = 0, 0, 1
        clusters = []
        for i in range(10):
            clusters.append(10*np.arange(10) + i)
        ids_nn = np.array([i%10 for i in range(100)])
        title = f"TSNE plot of FAUST, using {a}, {b}, {c} distance, colored by pose"
    elif args.type_exp == "shape":
        a, b, c = 1, 0.0001, 0
        clusters = []
        for i in range(10):
            clusters.append(np.arange(10) + 10 * i)
        ids_nn = np.array([i // 10 for i in range(100)])
        title = f"TSNE plot of FAUST, using {a}, {b}, {c} distance, colored by shape"
    else:
        a, b, c = args.a, args.b, args.c
        threed = True
        clusters = [np.arange(100)]
    mat_path = os.path.join(cache, f"dist_mat_{a}_{b}_{c}.npy")
    mat, surf_list = folder_loop(args.faust_path, metric, a, b, c, mat_path)

    if ids_nn is not None:
        print_measures(ids_nn, mat)

    if args.threed:
        get_3D_mds_plot(mat, surf_list, clusters)  # , emb="mds")
    else:
        get_2D_mds_plot(mat, clusters, title)  # , emb="mds")