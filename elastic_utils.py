import torch
from surfaces import Surface
from tqdm import tqdm
pi = 3.1415927410125732

device = "cuda:0"

def alb_to_abc(alpha, lambd, beta):
    return alpha, alpha * lambd + beta / 16, beta


def abc_to_alb(a, b, c):
    # maps the 3-parameter constants to constants with respect to the new split metric
    if a == 0:
        lambd = 0
    else:
        lambd = (16 * b - c) / (16 * a)
    #print(a, lambd, c)
    return a, lambd, c

def multi_dim_inv_for_22mn(A, detA):  # compute the inverse of the first 2D slice of a 2*2*m*n matrix
    C = torch.zeros(A.size()).to(device)
    C[:, :, 0, 0] = A[:, :, 1, 1]
    C[:, :, 1, 1] = A[:, :, 0, 0]
    C[:, :, 0, 1] = -A[: ,:, 0, 1]
    C[:, :, 1, 0] = -A[:, :,1, 0]
    return C / detA[:, :, None, None]  # C: 2*2*m*n

def batched_dim_inv_for_22(A, detA):  # compute the inverse of the first 2D slice of a 2*2*n matrix
    C = torch.zeros(A.size()).to(device)
    C[:, 0, 0] = A[:, 1, 1]
    C[:, 1, 1] = A[:, 0, 0]
    C[:, 0, 1] = -A[:, 0, 1]
    C[:, 1, 0] = -A[:, 1, 0]
    return C / detA[:, None, None]  # C: 2*2*m*n

# Following functions directly inspired by https://github.com/zhesu1/surfaceRegistration/blob/master/Packages/Funcs.py
def trKsquare(B, A):
    # inputs: B, A: two tensors of size n x 2 x 2...
    # output kappa = tr(K^2) of size n..., where K = inv(B)A

    a1, a2 = A[:, 0, 0], A[:, 1, 1]
    b1, b2 = B[:, 0, 0], B[:, 1, 1]
    a3 = (A[:, 0, 1] + A[:, 1, 0]) / 2.0
    b3 = (B[:, 0, 1] + B[:, 1, 0]) / 2.0

    ##
    ell1 = b1.sqrt()
    ell3 = b3 / ell1
    ell2 = (b2 - ell3 ** 2).sqrt()
    #print(torch.isnan(ell1).sum(), torch.isnan(ell3).sum(), torch.isnan(ell3).sum())
    #print(B[torch.isnan(ell3), :, :].shape)

    w1 = a1 / ell1 ** 2
    mu = ell3 / ell1
    w3 = (a3 - mu * a1) / (ell1 * ell2)
    w2 = (a2 - 2.0 * mu * a3 + mu ** 2 * a1) / (ell2 ** 2)
    #mat = B[torch.isnan(w2), :, :][0]
    #print(mat)
    #print(torch.isnan(w1).sum(), torch.isnan(mu).sum(), torch.isnan(w2).sum(), torch.isnan(w3).sum())
    ##
    eps1 = 1e-12
    eps2 = 1e-12

    eta = (w1 - w2) / (2.0 * w3 + eps2)
    s = torch.where(eta >= 0.0, torch.ones(1, dtype=B.dtype).to(device), -torch.ones(1, dtype=B.dtype).to(device))
    tau = s / (eta.abs() + (1.0 + eta ** 2).sqrt())

    tau = torch.where((w3.abs() - eps2 * (w1 - w2).abs()) >= 0.0, tau, w3 / (w1 - w2 + eps1))
    tau = torch.where((w3.abs() - eps1 * (w1 * w2).sqrt()) >= 0.0, tau, torch.zeros(1, dtype=B.dtype).to(device))

    lambda1 = w1 + w3 * tau
    lambda2 = w2 - w3 * tau
    #print(torch.isnan(lambda1), torch.isnan(lambda2))
    kappa = torch.log(lambda1) ** 2 + torch.log(lambda2) ** 2
    #print(lambda1[torch.isnan(kappa)], lambda2[torch.isnan(lambda2)])
    return kappa


def alb_to_abc(alpha, lambd, beta):
    return alpha, alpha * lambd + beta / 16, beta


def abc_to_alb(a, b, c):
    # maps the 3-parameter constants to constants with respect to the new split metric
    if a == 0:
        lambd = 0
    else:
        lambd = (16 * b - c) / (16 * a)
    #print(a, lambd, c)
    return a, lambd, c


def Squared_distance_abc(gq1, gq2, a, b, c, elem):
    alpha, lambd, beta = abc_to_alb(a, b, c)
    return Squared_distance(gq1, gq2, alpha, lambd, beta, elem)


def Squared_distance(gq1, gq2, alpha, lambd, beta, elem, srnf=True):
    # calculate the square of the distance with respect to SRNFs
    if beta !=0:
        if srnf:
            dist2_q = torch.einsum("ni,ni->n", [gq1[1] - gq2[1], gq1[1] - gq2[1]])
        else:
            n1 = gq1[1]
            n2 = gq2[1]
            dist2_q = torch.arccos(torch.clamp((n1*n2).sum(dim=-1), -1, 1))
    if alpha != 0:
        # calculate the square of the distance with respect to the induced metrics
        g1, g2 = gq1[0], gq2[0]
        inv_g1 = torch.inverse(g1)
        #     set_trace()
        inv_g1_g2 = torch.einsum("...ik,...kj->...ij", [inv_g1, g2])  # mxn*2*2
        # print(torch.isnan(inv_g1_g2).sum())
        # print(torch.isnan(inv_g1_g2).sum())
        trK0square = trKsquare(g1, g2) - (torch.log(torch.det(inv_g1_g2))) ** 2 / 2
        # det_g1 = g1[:, 0, 0] * g1[:, 1, 1] - g1[:, 0, 1] * g1[:, 1, 0] + 1e-7
        # det_g2 = g2[:, 0, 0] * g2[:, 1, 1] - g2[:, 0, 1] * g2[:, 1, 0] + 1e-7  # mxn*2*2
        # trKsquare_temp = trKsquare(g1, g2)
        # trK0square = trKsquare_temp - (torch.log(det_g2 / det_g1)) ** 2 / 2
        # print(torch.log(torch.det(inv_g1_g2[torch.where(torch.isnan(trK0square))])))
        # print(torch.log(torch.det(g2[torch.where(torch.isnan(trK0square))])))
        # print(g1[torch.where(torch.isnan(trK0square))])
        # print(g2[torch.where(torch.isnan(trK0square))])
        # print(inv_g1_g2[torch.where(torch.isnan(trK0square))])
        # print(torch.isnan(trK0square).sum())

        theta = torch.min((trK0square / lambd + 1e-7).sqrt() / 4, torch.tensor([pi], dtype=torch.double).to(device))

        alp, bet = (torch.det(g1) + 1e-7).pow(1 / 4), (torch.det(g2) + 1e-7).pow(1 / 4)
        #alp, bet = (det_g1).pow(1 / 4), (det_g2).pow(1 / 4)
        # print(torch.isnan(alp).sum(), torch.isnan(bet).sum(), torch.isnan(theta).sum())
        dist2_g = 16 * lambd * (alp ** 2 - 2 * alp * bet * torch.cos(theta) + bet ** 2)
    #print(dist2_g, dist2_q)
    if beta!=0 and alpha !=0:
        integrand = beta * dist2_q + alpha * dist2_g
    elif alpha == 0:
        integrand = beta * dist2_q
    else:
        integrand = alpha * dist2_g
    #print(torch.isnan(dist2_g).sum())
    #print(dist2_q.sum())
    #print(torch.isnan(integrand).sum())
    return torch.sum(integrand*elem)


def batch_rodrigues(
    rot_vecs: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    assert len(rot_vecs.shape) == 2, (
        f'Expects an array of size Bx3, but received {rot_vecs.shape}')

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat