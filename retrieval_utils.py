import numpy as np

def full_tier_mat(dist_mat, positions):
    result = ["" for i in range(dist_mat.shape[0])]
    N = dist_mat.shape[0]
    for i in range(N):
        pose = positions[i]
        tier = (positions==pose).sum()-1
        bests = np.argsort(dist_mat[i, :])
        closest_2_tier = bests[1:tier*2]
        result[i] = closest_2_tier
    return result


def nearest(full_tier, positions):
    N = len(full_tier)
    result = np.zeros(N, dtype=np.float32)
    for i in range(N):
        pose = positions[i]
        nn = full_tier[i][0]
        result[i] = int(positions[nn] == pose)
    return result.mean(), result


def first_tier(full_tier, positions):
    N = len(full_tier)
    result = np.zeros(N)
    for i in range(N):
        pose = positions[i]
        tier = full_tier[i]
        n_class = (positions == pose).sum()-1
        tier = tier[:n_class]
        result[i] = ((positions[tier] == pose).sum()*1.0) / n_class
    return result.mean(), result


def first_goo(dist_mat, positions):
    result = ["" for i in range(dist_mat.shape[0])]
    N = dist_mat.shape[0]
    for i in range(N):
        pose = positions[i]
        same = positions == pose
        tier = (same).sum() - 1
        bests = np.argsort(dist_mat[i, :])
        neighbor_close = bests[np.where(same[bests])[0][1]]
        result[i] = neighbor_close
    return result


def st_goo(dist_mat, positions):
    result = ["" for i in range(dist_mat.shape[0])]
    N = dist_mat.shape[0]
    for i in range(N):
        pose = positions[i]
        same = positions == pose
        tier = (same).sum()
        bests = np.argsort(dist_mat[i, :])
        neighbor_close = bests[1:tier]
        result[i] = neighbor_close
    return result


def second_tier(full_tier, positions):
    N = len(full_tier)
    result = np.zeros(N)
    for i in range(N):
        pose = positions[i]
        tier = full_tier[i]
        n_class = (positions == pose).sum()-1
        result[i] = ((positions[tier] == pose).sum()*1.0) / n_class
    return result.mean(), result

def print_measures(positions, mat):
    my_tier = full_tier_mat(mat, positions)
    NN, NN_array = nearest(my_tier, positions)
    FT, FT_array = first_tier(my_tier, positions)
    ST, ST_array = second_tier(my_tier, positions)
    print("Nearest neighbor: ", NN, "First Tier :",  FT, "Second Tier :", ST)
    score = (NN, FT, ST)
    return score
