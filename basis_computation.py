import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
from utils import load_mesh
import os
from tqdm import tqdm
import itertools



sids  =['50002', '50004', '50007', '50009', '50020',
        '50021', '50022', '50025', '50026', '50027']

pids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
    'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
    'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
    'one_leg_jump', 'running_on_spot']


files = ["registrations_f.hdf5",
            "registrations_m.hdf5"]

def open_sequence(sid, seq, file):
    sidseq = sid + "_" + seq
    if sidseq not in file:
        print('Sequence %s from subject %s not in file' % (seq, sid))
        return None

    verts = file[sidseq][()].transpose([2, 0, 1])
    faces = file['faces'][()]

    return verts, faces


def load_dfaust_deformations(dfaust_dir):
    deformations = []
    for sid, seq in tqdm(list(itertools.product(sids, pids)), "Loading DFAUST deformations"):
        for fil in files:
            full_fil = os.path.join(dfaust_dir, fil)
            with h5py.File(full_fil, 'r') as f_read:
                output = open_sequence(sid, seq, f_read)
                if output is not None:
                    verts, _ = output
                    verts_full = verts[::10]
                    deformations.append(verts_full[1:] - verts_full[:-1])
    return np.concatenate(deformations, axis=0)


def normalize_mat(M):
    # Normalize matrix according to diagonal
    # No good explanation on this
    norms = np.sqrt(np.diag(M))
    no_nan = norms > 0
    M_norm = np.zeros(M.shape)
    M_no_nan = np.outer(no_nan, no_nan)
    M_norm[M_no_nan] = M[M_no_nan] / (
        (norms[:, np.newaxis] * norms[np.newaxis, :])[M_no_nan])
    M_norm = np.minimum(np.maximum(M_norm, -1), 1)
    return M_norm


def ortho_PCA(deformations, elements_v, threshold=1e-3, vis=False):
    n_v = elements_v.shape[0]
    print("Computing covariance")
    base_elem = (deformations * np.sqrt(elements_v[np.newaxis, np.newaxis, :, :])).squeeze().reshape((len(deformations), n_v * 3))

    ## Covariance matrix, with area-based weighting for inner product
    # As it is a deformation basis, we suppose mean is 0.
    M = np.inner(base_elem, base_elem)
    #M_norm = normalize_mat(M)

    print("Eigendecomposition")
    r = np.linalg.matrix_rank(M, tol=threshold)
    print("Rank of M: {0}".format(r))
    w, v = np.linalg.eigh(M)
    if vis:
        plt.semilogy(w)
        plt.show()

    # Eigenvalues based reweighting. Proved useful to speed up the optimization
    # (also allows to save the eigenvalues somewhere)
    u = np.diag(1 / np.sqrt(w[-r:])) @ v[:, -r:].T

    new_base = []
    for j in range(r):
        t = np.zeros((n_v, 3))
        for k in range(len(deformations)):
            t += u[j, k] * deformations[k]
        new_base.append(t)

    return np.array(new_base)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfaust_path", type=str, default="", help="DFAUST registered directory")
    parser.add_argument("--smpl_path", type=str, default="", help="SMPL pkl directory, can also be a ply file")
    parser.add_argument("--cache", type=str, default="cache", help="Cache directory")
    args = parser.parse_args()

    surf_template = load_mesh(args.smpl_path)
    elements_v, elements_f = surf_template.computeVertexArea()
    basis_path = os.path.join(args.cache, "base_shape.npy")
    deformations = load_dfaust_deformations(args.dfaust_path)
    basis = ortho_PCA(deformations, elements_v)
    np.save(basis_path, basis)