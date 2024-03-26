# Sinkhorn-Knopp iterations and friends 
import ot
from ot.backend import get_backend

def sinkhorn_knopp(p, q, u, v, K, numItermax = 1_000, tol = 1e-6, check_convergence = 10, backend = None):
    # Sinkhorn-Knopp iterations for entropy-regularized OT with marginals
    # (p, q), dual variables (u, v) and Gibbs kernel K, returning updated dual variables.
    backend = get_backend(p) if backend is None else backend
    for i in range(numItermax):
        u = p / (K @ v)
        v = q / (K.T @ u)
        if i % check_convergence == 0:
            err = backend.abs((u * (K @ v)) - p).sum()
            if err / p.sum() < tol:
                break
    return u, v

def sinkhorn_knopp_unbalanced(p, q, u, v, K, eps, lambda1, lambda2, numItermax = 1_000, tol = 1e-6, check_convergence = 10, backend = None):
    # Sinkhorn-Knopp iterations for unbalanced entropy-regularized OT with marginals
    # (p, q), dual variables (u, v), Gibbs kernel K
    backend = get_backend(p) if backend is None else backend
    u_, v_ = backend.copy(u), backend.copy(v)
    for i in range(numItermax):
        u = (p / (K @ v))**(lambda1 / (lambda1 + eps))
        v = (q / (K.T @ u))**(lambda2 / (lambda2 + eps))
        if i % check_convergence == 0:
            err = max(backend.abs(u - u_).sum(), backend.abs(v - v_).sum())
            if err / (p.sum() * q.sum())**0.5 < tol:
                break
    return u, v

def sinkhorn(p, q, C, eps, u = None, v = None, **kwargs):
    # solve Sinkhorn problem with marginals (p, q) and cost C,
    # returning (u, K, v)
    nx = get_backend(p, q, C)
    u = nx.ones(p.shape, type_as = p) if u is None else u
    v = nx.ones(q.shape, type_as = q) if v is None else v
    K = nx.exp(-C/eps) * p.reshape(-1, 1) * q.reshape(1, -1)
    u, v = sinkhorn_knopp(p, q, u, v, K, backend = nx, **kwargs)
    return u, K, v

def sinkhorn_unbalanced(p, q, C, eps, lambda1, lambda2, u = None, v = None, **kwargs):
    # solve unbalanced Sinkhorn problem with marginals (p, q) and cost C,
    # returning (u, K, v) 
    nx = get_backend(p, q, C)
    u = nx.ones(p.shape, type_as = p) if u is None else u
    v = nx.ones(q.shape, type_as = q) if v is None else v
    K = nx.exp(-C/eps) * p.reshape(-1, 1) * q.reshape(1, -1)
    u, v = sinkhorn_knopp_unbalanced(p, q, u, v, K, eps, lambda1, lambda2, **kwargs)
    return u, K, v

def emd_ipot(p, q, C, t, u = None, v = None, num_prox_steps = 100, **kwargs):
    # IPOT algorithm following https://arxiv.org/pdf/1802.04307.pdf for marginals (p, q) 
    # with cost C, inverse proximal step size t.
    # initialize dual potentials
    nx = get_backend(p, q, C)
    u = nx.ones(p.shape, type_as = p) if u is None else u
    v = nx.ones(q.shape, type_as = q) if v is None else v
    # initialize coupling
    coupling = nx.outer(p, q)
    # Gibbs kernel
    K = nx.exp(-C/t)
    # At each step, solve a Sinkhorn problem w.r.t. Q=coupling*exp(-C/t)
    Q = K * coupling
    for i in range(num_prox_steps):
        u, v = sinkhorn_knopp(p, q, u, v, Q, **kwargs)
        coupling = nx.diag(u) @ Q @ nx.diag(v)
        Q = K * coupling
    return u, v, coupling

def emd_unbal_ipot(p, q, C, t, lambda1, lambda2, u = None, v = None, num_prox_steps = 100, **kwargs):
    # IPOT algorithm following https://arxiv.org/pdf/1802.04307.pdf for marginals (p, q) 
    # with cost C, inverse proximal step size t.
    # initialize dual potentials
    nx = get_backend(p, q, C)
    u = nx.ones(p.shape, type_as = p) if u is None else u
    v = nx.ones(q.shape, type_as = q) if v is None else v
    # initialize coupling
    coupling = nx.outer(p, q)
    # Gibbs kernel
    K = nx.exp(-C/t)
    # At each step, solve a Sinkhorn problem w.r.t. Q=coupling*exp(-C/t)
    Q = K * coupling
    for i in range(num_prox_steps):
        u, v = sinkhorn_knopp_unbalanced(p, q, u, v, Q, t, lambda1, lambda2, **kwargs)
        coupling = nx.diag(u) @ Q @ nx.diag(v)
        Q = K * coupling
    return u, v, coupling

def emd_partial_ipot(a, b, M, m, nb_dummies = 1, u = None, v = None, **kwargs):
    nx = get_backend(a, b, M)
    b_extension = nx.ones(nb_dummies) * (nx.sum(a) - m) / nb_dummies
    b_extended = nx.concatenate([b, b_extension])
    a_extension = nx.ones(nb_dummies) * (nx.sum(b) - m) / nb_dummies
    a_extended = nx.concatenate([a, a_extension])
    M_extension = nx.ones((nb_dummies, nb_dummies)) * nx.max(M) * 2
    M_extended = nx.concatenate(
        (nx.concatenate((M, nx.zeros((M.shape[0], M_extension.shape[1]))), axis=1),
         nx.concatenate((nx.zeros((M_extension.shape[0], M.shape[1])), M_extension), axis=1)),
        axis=0
    )
    u = nx.ones(a_extended.shape, type_as = a_extended) if u is None else u
    v = nx.ones(b_extended.shape, type_as = b_extended) if v is None else v
    u, v, coupling = emd_ipot(a_extended, b_extended, M_extended, **kwargs)
    return u, v, coupling[:-nb_dummies, :-nb_dummies]

