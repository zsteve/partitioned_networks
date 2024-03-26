import ot
from ot.backend import get_backend
import sk
from util import *

# OT solvers for balanced COOT 
def _solve_balanced(p, q, C, u, v, eps = None, solver = "sinkhorn", pi = None, **kwargs) :
    # Inner loop solver for COOT with marginals (p, q), cost C, intiial dual potentials (u, v) 
    # and previous coupling pi_prev (only for use with IPOT).
    # Returns transport plan and dual potentials 
    nx = get_backend(p, q, C)
    if solver == "sinkhorn":
        # Sinkhorn iterations to solve entropy regularised OT problem 
        # min_T <C, T> + eps * <T, log T> 
        u, K, v = sk.sinkhorn(p, q, C, eps, u = u, v = v, **kwargs)
        return nx.diag(u) @ K @ nx.diag(v), u, v
    elif solver == "emd":
        # Network simplex solution to exact OT problem
        # min_T <C, T>
        return ot.emd(p, q, C, **kwargs), u, v
    elif solver == "ipot":
        # Proximal point method to solve exact OT problem
        # min_T <C, T> 
        # NB. for IPOT, eps takes the role of the prox step
        Q = pi * nx.exp(-C/eps) * p.reshape(-1, 1) * q.reshape(1, -1)
        u, v = sk.sinkhorn_knopp(p, q, u, v, Q, **kwargs)
        return nx.diag(u) @ Q @ nx.diag(v), u, v
    else:
        raise ValueError("Unsupported solver %s" % solver)
        return None, None, None

def _mm_unbalanced(p, q, C, lamda, _lambda, **kwargs):
    # Wrapper for exact unbalanced OT solver 
    return ot.unbalanced.mm_unbalanced(p, q, C, lamda, **kwargs)

def _solve_unbalanced(p, q, C, u, v, eps = None, lamda = None, lamda2 = None, solver = "sinkhorn", pi = None, **kwargs):
    # Solve unbalanced transport problem (possibly with entropic regularisation)
    # min_T <C, T> + lamda*KL(T1 | p) + lamda*KL(T2 | q) + eps*KL(T | pq')
    # Returns transport plan and dual potentials
    lamda2 = lamda if lamda2 is None else lamda2
    nx = get_backend(p, q, C, u, v)
    if solver == "sinkhorn":
        u, K, v = sk.sinkhorn_unbalanced(p, q, C, eps, lamda, lamda2, u = u, v = v, **kwargs)
        return nx.diag(u) @ K @ nx.diag(v), u, v
    elif solver == "mm":
        return _mm_unbalanced(p, q, C, lamda, lamda2, **kwargs), u, v
    elif solver == "ipot":
        # for IPOT, eps takes the role of the prox step
        Q = pi * nx.exp(-C/eps) * p.reshape(-1, 1) * q.reshape(1, -1)
        u, v = sk.sinkhorn_knopp_unbalanced(p, q, u, v, Q, eps, lamda, lamda2, **kwargs)
        return nx.diag(u) @ Q @ nx.diag(v), u, v
    else:
        raise ValueError("Unsupported solver %s" % solver)
        return None, u, v

def _solve_partial(p, q, C, u, v, eps = None, m = 1, solver = "emd_partial", pi = None, **kwargs):
    # Solve partial transport problem
    # min_T <C, T> s.t. sum(T) = m
    if solver == "emd_partial":
        return ot.partial.partial_wasserstein(p, q, C, m = m), u, v
    elif solver == "ipot_partial":
        # TODO: check rigorous
        u, v, pi = sk.emd_partial_ipot(p, q, C, m = m, u = u, v = v, **kwargs)
        return pi, u, v
    else:
        raise ValueError("Unsupported solver %s" % solver)
        return None, u, v
    
def H(x, y):
    # Entropy functional H(x, y) = <x, log(x/y)>
    nx = get_backend(x, y)
    return nx.kl_div(x, y)

def COOT_objective(X1, X2, pi_s, pi_f):
    # returns COOT objective <L, pi_s * pi_f> 
    return dot(eta(X1, X2, pi_f.sum(-1), pi_f.sum(0)) - X1 @ pi_f @ X2.T, pi_s)

"""
    Solve (fused) co-optimal transport problem by alternating minimisation

min_{T0, T1} alpha*<L(X, Y), T0 x T1> + (1-alpha)*(<C0, T0> + <C1, T1>) + eps_0*KL(T0 | p0 q0') + eps_1*KL(T1 | p1 q1')
"""
def COOT(X, Y, p0, q0, p1, q2,
         C0 = None, C1 = None,
         alpha = None,
         u0 = None, v0 = None,
         u1 = None, v1 = None, 
         T0 = None, T1 = None, 
         eps = 0, eps1 = None, solver = "emd",
         iter = 1000, print_iter = 25, rel_tol = 1e-9, abs_tol = 1e-9, verbose = False, **kwargs):
    nx = get_backend(X, Y, p0, q0, p1, q2)
    T0 = nx.outer(p0, q0) if T0 is None else T0
    T1 = nx.outer(p1, q2) if T1 is None else T1
    # allow two separate entropic regularisation levels
    eps0 = eps
    eps1 = eps if eps1 is None else eps1
    obj = 1e6
    is_converged = False 
    # if using sinkhorn or IPOT, retain dual potentials. these are not used for emd
    def init(p, u):
        # helper function for initializing dual potentials
        if u is None:
            return nx.ones(p.shape)
        else:
            return u
    u0, v0 = init(p0, u0), init(q0, v0) if solver != "emd" else (None, None)
    u1, v1 = init(p1, u1), init(q2, v1) if solver != "emd" else (None, None)
    for it in range(iter):
        M0 = eta(X, Y, T1.sum(-1), T1.sum(0)) - X @ T1 @ Y.T
        if alpha is not None:
            M0 = alpha*M0 + (1-alpha)*C0
        T0, u0, v0 = _solve_balanced(p0, q0, M0, u0, v0, eps0, solver = solver, pi = T0, **kwargs)
        M1 = eta(X.T, Y.T, T0.sum(-1), T0.sum(0)) - (X.T) @ T0 @ Y
        if alpha is not None:
            M1 = alpha*M1 + (1-alpha)*C1
        T1, u1, v1 = _solve_balanced(p1, q2, M1, u1, v1, eps1, solver = solver, pi = T1, **kwargs)
        obj_new = COOT_objective(X, Y, T0, T1)
        if (abs(obj_new - obj)/obj < rel_tol*obj) or (abs(obj_new - obj) < abs_tol):
            is_converged = True 
            break
        obj = obj_new 
        if print_iter is not None and it % print_iter == 0:
            print("Iteration %d, obj = %f" % (it, obj, ))
    # calculate objective 
    if not is_converged and verbose:
        print("Warning: COOT not converged")
    return T0, T1, obj

def COOT_unbalanced(X, Y, p0, q0, p1, q1, 
         C0 = None, C1 = None, alpha = None,
         u0 = None, v0 = None, u1 = None, v1 = None, 
         T0 = None, T1 = None,
         eps = 0, lamda = 1,
         solver = "emd", iter = 1000, print_iter = 25, rel_tol = 1e-9, abs_tol = 1e-9, verbose = False, **kwargs):
    nx = get_backend(X, Y, p0, q0, p1, q1)
    T0 = nx.outer(p0, q0) if T0 is None else T0
    T1 = nx.outer(p1, q1) if T1 is None else T1
    # Cs = torch.empty_like(T0)
    # M1 = torch.empty_like(T1)
    obj = 1e6
    is_converged = False 
    # if using sinkhorn or IPOT, retain dual potentials. these are not used for emd
    def init(p, u):
        # helper function for initializing dual potentials
        if u is None:
            return nx.ones(p.shape)
        else:
            return u
    u0, v0 = init(p0, u0), init(q0, v0) if solver != "emd" else (None, None)
    u1, v1 = init(p1, u1), init(q1, v1) if solver != "emd" else (None, None)
    if alpha is not None:
        # rescale X, Y, C0, C1 accordingly
        X = alpha**0.5*X
        Y = alpha**0.5*Y
        C0 = (1-alpha)*C0
        C1 = (1-alpha)*C1
    for it in range(iter):
        M0 = eta(X, Y, T1.sum(-1), T1.sum(0)) - X @ T1 @ Y.T
        M0 += eps*H(T1, nx.outer(p1, q1)) + lamda*H(T1.sum(-1), p1) + lamda*H(T1.sum(0), q1)
        m1 = T1.sum()
        if alpha is not None: # add fused terms
            M0 += m1*C0 + (C1*T1).sum()
        #     M0 = alpha*M0 + (1-alpha)*C0
        T0, u0, v0 = _solve_unbalanced(p0, q0, M0, u0, v0, eps*m1, lamda*m1, solver = solver, pi = T0, **kwargs)
        T0 *= (T1.sum() / T0.sum())**0.5  # rescale
        M1 = eta(X.T, Y.T, T0.sum(-1), T0.sum(0)) - (X.T) @ T0 @ Y
        M1 += eps*H(T0, nx.outer(p0, q0)) + lamda*H(T0.sum(-1), p0) + lamda*H(T0.sum(0), q0)
        m0 = T0.sum()
        if alpha is not None: # add fused terms
            M1 += m0*C1 + (C0*T0).sum()
        #     M1 = alpha*M1 + (1-alpha)*C1
        T1, u1, v1 = _solve_unbalanced(p1, q1, M1, u1, v1, eps*m0, lamda*m0, solver = solver, pi = T1, **kwargs)
        T1 *= (T0.sum() / T1.sum())**0.5  # rescale
        obj_new = COOT_objective(X, Y, T0, T1)
        if (abs(obj_new - obj)/obj < rel_tol*obj) or (abs(obj_new - obj) < abs_tol):
            is_converged = True 
            break
        obj = obj_new 
        if print_iter is not None and it % print_iter == 0:
            print("Iteration %d, obj = %f" % (it, obj, ))
    # calculate objective 
    if not is_converged and verbose:
        print("Warning: COOT not converged")
    return T0, T1, obj

def COOT_partial(X, Y, p0, q0, p1, q1, 
         C0 = None, C1 = None, alpha = None,
         u0 = None, v0 = None, u1 = None, v1 = None, 
         T0 = None, T1 = None,
         eps = 0, m = 1,
                 solver = "emd_partial", iter = 1000, print_iter = 25, rel_tol = 1e-9, abs_tol = 1e-9, verbose = False,
                 nb_dummies = 1, **kwargs):
    nx = get_backend(X, Y, p0, q0, p1, q1)
    T0 = nx.outer(p0, q0) if T0 is None else T0
    T1 = nx.outer(p1, q1) if T1 is None else T1
    # M0 = torch.empty_like(T0)
    # M1 = torch.empty_like(T1)
    obj = 1e6
    is_converged = False 
    # if using sinkhorn or IPOT, retain dual potentials. these are not used for emd
    def init(p, u):
        # helper function for initializing dual potentials
        if u is None:
            return nx.ones(len(p)+nb_dummies, type_as = p)
        else:
            return u
    u0, v0 = init(p0, u0), init(q0, v0) if solver != "emd" else (None, None)
    u1, v1 = init(p1, u1), init(q1, v1) if solver != "emd" else (None, None)
    for it in range(iter):
        M0 = eta(X, Y, T1.sum(-1), T1.sum(0)) - X @ T1 @ Y.T
        if alpha is not None:
            M0 = alpha*M0 + (1-alpha)*C0
        T0, u0, v0 = _solve_partial(p0, q0, M0, u0, v0, eps = eps, m = m, solver = solver, pi = T0, nb_dummies = nb_dummies, **kwargs)
        T0 *= (T1.sum() / T0.sum())**0.5  # rescale
        M1 = eta(X.T, Y.T, T0.sum(-1), T0.sum(0)) - (X.T) @ T0 @ Y
        if alpha is not None:
            M1 = alpha*M1 + (1-alpha)*C1
        T1, u1, v1 = _solve_partial(p1, q1, M1, u1, v1, eps = eps, m = m, solver = solver, pi = T1, nb_dummies = nb_dummies, **kwargs)
        T1 *= (T0.sum() / T1.sum())**0.5  # rescale
        obj_new = COOT_objective(X, Y, T0, T1)
        if (abs(obj_new - obj)/obj < rel_tol*obj) or (abs(obj_new - obj) < abs_tol):
            is_converged = True 
            break
        obj = obj_new 
        if print_iter is not None and it % print_iter == 0:
            print("Iteration %d, obj = %f" % (it, obj, ))
    # calculate objective 
    if not is_converged and verbose:
        print("Warning: COOT not converged")
    return T0, T1, obj

def grad_COOT(X1, X2, pi_s, pi_f):
    # gradient of COOT(X1, X2, pi_s, pi_f)  w.r.t. X1
    nx = get_backend(X1, X2, pi_s, pi_f)
    return nx.diag(pi_s.sum(-1)) @ X1 @ nx.diag(pi_f.sum(-1)) - pi_s @ X2 @ pi_f.T

def COOT_barycenter(Y, w, v, X_all, ws = None, 
                    iter = 50, print_iter = 5, coot_args = {}, coot_fn = COOT, diagonal = False):
    nx = get_backend(Y, w, v)
    ws = nx.ones(len(Y))/len(Y) if ws is None else ws
    for it in range(iter):
        couplings = [coot_fn(Y, x[0], w, x[1], v, x[2], **coot_args) for x in X_all]
        err = (sum([x[-1] for x in couplings]) / len(couplings))
        # Y_new = sum([(Ts @ x @ Tv.T) for ((Ts, Tv, _), (x, _, _)) in zip(couplings, X_all)]) / (torch.outer(w, v) * len(X_all))
        Y_new = sum([_w * (Ts @ x @ Tv.T) / nx.outer(Ts.sum(-1), Tv.sum(-1)) for (_w, (Ts, Tv, _), (x, _, _)) in zip(ws, couplings, X_all)])
        if diagonal: # project onto set of diagonal matrices
            Y_new = nx.diag(nx.diag(Y_new))
        if it % print_iter == 0:
            print(f"Relative change: {(nx.norm(Y - Y_new) / nx.norm(Y)).item()}, Objective: {err.item()}")
        Y = Y_new
    return Y

def fused_GW_COOT(Avv, Bvv, Aee, Bee, Ave, Bve, Aev, Bev, p0, p1, q0, q1, iters = 10, verbose = False, eps_e = 0, eps_v = 0, **kwargs):
    pi_v = p0.reshape(-1, 1) * p1.reshape(1, -1)
    for i in range(iters):
        M_e=(eta(Ave.T, Bve.T, p0, p1) - Ave.T @ pi_v @ Bve) + (eta(Aev, Bev, p0, p1) - Aev @ pi_v @ Bev.T)
        if eps_e > 0:
            pi_e, log_e = ot.gromov.entropic_fused_gromov_wasserstein(M_e, Aee, Bee, q0, q1, log = True, verbose = False, epsilon = eps_e, **kwargs)
        else:
            pi_e, log_e = ot.fused_gromov_wasserstein(M_e, Aee, Bee, q0, q1, log = True, verbose = False, **kwargs)
        M_v=(eta(Ave, Bve, q0, q1) - Ave @ pi_e @ Bve.T) + (eta(Aev.T, Bev.T, q0, q1) - Aev.T @ pi_e @ Bev)
        if eps_v > 0:
            pi_v, log_v = ot.gromov.entropic_fused_gromov_wasserstein(M_v, Avv, Bvv, p0, p1, log = True, verbose = False, epsilon = eps_v, **kwargs)
        else:
            pi_v, log_v = ot.fused_gromov_wasserstein(M_v, Avv, Bvv, p0, p1, log = True, verbose = False, **kwargs)
        if verbose:
            print(log_e['fgw_dist'], log_v['fgw_dist'])
    return pi_v, pi_e

def partitioned_OT_objective(X00, Y00, X11, Y11, X01, Y01, X10, Y10, T0, T1):
    # returns COOT objective <L, pi_s * pi_f>
    return 0.5*(COOT_objective(X00, Y00, T0, T0) + COOT_objective(X11, Y11, T1, T1) + COOT_objective(X01, Y01, T0, T1) + COOT_objective(X10, Y10, T1, T0))

def partitioned_OT(X00, Y00, X11, Y11,
                   X01, Y01,
                   p0, q0, p1, q1,
                   X10 = None, Y10 = None,
                   C0 = None, C1 = None,
                   iters = 50, print_iter = 5, rel_tol = 1e-9, abs_tol = 1e-9, verbose = False, eps0 = 0, eps1 = 0, solver = "sinkhorn", solver_args = {}, solver_args_1 = None, 
                   solve_emd_0 = False, solve_emd_1 = False):
    T0 = p0.reshape(-1, 1) * q0.reshape(1, -1)
    T1 = p1.reshape(-1, 1) * q1.reshape(1, -1)
    X10 = X01.T if X10 is None else X10
    Y10 = Y01.T if Y10 is None else Y10
    nx = get_backend(X00, Y00, X11, Y11, X01, Y01, p0, q0, p1, q1)
    C0 = 0 if C0 is None else C0
    C1 = 0 if C1 is None else C1
    u0, v0 = nx.ones(p0.shape, type_as = p0), nx.ones(q0.shape, type_as = q0)
    u1, v1 = nx.ones(p1.shape, type_as = p1), nx.ones(q1.shape, type_as = q1)
    obj = 1e6
    solver_args_1 = solver_args if solver_args_1 is None else solver_args_1
    is_converged = False
    for it in range(iters):
        if solver == "exact":
            M0=(eta(X01, Y01, p1, q1) - X01 @ T1 @ Y01.T)/2 + (eta(X10.T, Y10.T, p1, q1) - X10.T @ T1 @ Y10)/2 + C0
            if solve_emd_0:
                T0 = ot.emd(p0, q0, M0, **solver_args)
            else:
                T0, _ = ot.fused_gromov_wasserstein(M0, X00, Y00, p0, q0, log = True, verbose = False, **solver_args)
            M1=(eta(X01.T, Y01.T, p0, q0) - X01.T @ T0 @ Y01)/2 + (eta(X10, Y10, p0, q0) - X10 @ T0 @ Y10.T)/2 + C1
            if solve_emd_1:
                T1 = ot.emd(p1, q1, M1, **solver_args_1)
            else:
                T1, _ = ot.fused_gromov_wasserstein(M1, X11, Y11, p1, q1, log = True, verbose = False, **solver_args_1)
        elif (solver == "sinkhorn") or (solver == "ipot"):
            # either IPOT or Sinkhorn
            M0=(eta(X01, Y01, p1, q1) - X01 @ T1 @ Y01.T)/2 + (eta(X10.T, Y10.T, p1, q1) - X10.T @ T1 @ Y10)/2 + C0
            M0 += eta(X00.T, Y00.T, p0, q0) - X00.T @ T0 @ Y00
            M1=(eta(X01.T, Y01.T, p0, q0) - X01.T @ T0 @ Y01)/2 + (eta(X10, Y10, p0, q0) - X10 @ T0 @ Y10.T)/2 + C1
            M1 += eta(X11.T, Y11.T, p1, q1) - X11.T @ T1 @ Y11
            T0, u0, v0 = _solve_balanced(p0, q0, M0, u0, v0, eps = eps0, solver = solver, pi = T0, **solver_args)
            T1, u1, v1 = _solve_balanced(p1, q1, M1, u1, v1, eps = eps1, solver = solver, pi = T1, **solver_args)
            # Bregman gradient descent
            # K0 = nx.exp(-M0/eps0) * p0[:, None] * q0[None, :]
            # K1 = nx.exp(-M1/eps1) * p1[:, None] * q1[None, :]
            # u0, v0 = sk.sinkhorn_knopp(p0, q0, u0, v0, K0, **solver_args)
            # u1, v1 = sk.sinkhorn_knopp(p1, q1, u1, v1, K1, **solver_args)
            # T0 = u0[:, None] * K0 * v0[None, :]
            # T1 = u1[:, None] * K1 * v1[None, :]
            # use POT functions
            # T0, log_0 = ot.sinkhorn(p0, q0, M0, reg = eps0, log = True, verbose = False, **sinkhorn_args)
            # T1, log_1 = ot.sinkhorn(p1, q1, M1, reg = eps1, log = True, verbose = False, **sinkhorn_args)
            # T0, log_0 = ot.smooth.smooth_ot_dual(p0, q0, nx.outer(p0, q0)*M0 / (p0.mean()*q0.mean()), reg = eps0 / (p0.mean()*q0.mean()), log = True, verbose = False, **sinkhorn_args)
            # T1, log_1 = ot.smooth.smooth_ot_dual(p1, q1, nx.outer(p1, q1)*M1 / (p1.mean()*q1.mean()), reg = eps1 / (p1.mean()*q1.mean()), log = True, verbose = False, **sinkhorn_args)
        else:
            raise ValueError("Unsupported solver %s" % solver)
            return T0, T1
        # print objective
        obj_new = partitioned_OT_objective(X00, Y00, X11, Y11, X01, Y01, X10, Y10, T0, T1)
        if (abs(obj_new - obj)/obj < rel_tol*obj) or (abs(obj_new - obj) < abs_tol):
            is_converged = True 
            break
        obj = obj_new 
        if print_iter is not None and it % print_iter == 0:
            print("Iteration %d, obj = %f" % (it, obj, ))
    if not is_converged and verbose:
        print("Warning: not converged")
    return T0, T1, obj

def partitioned_OT_unbalanced(X00, Y00, X11, Y11,
                              X01, Y01,
                              p0, q0, p1, q1,
                              X10 = None, Y10 = None,
                              C0 = None, C1 = None,
                              iters = 50, print_iter = 5, rel_tol = 1e-9, abs_tol = 1e-9, verbose = False,
                              eps = 0.01,
                              lamda1 = 1, lamda2 = 1,
                              solver = "sinkhorn", solver_args = {}):
    nx = get_backend(X00, Y00, X11, Y11, X01, Y01, p0, q0, p1, q1)
    # setup transport plans 
    T0 = p0.reshape(-1, 1) * q0.reshape(1, -1)
    T0 /= (p0.sum()*q0.sum())**0.5
    T1 = p1.reshape(-1, 1) * q1.reshape(1, -1)
    T1 /= (p1.sum()*q1.sum())**0.5
    T0_, T1_ = nx.copy(T0), nx.copy(T1)
    # 
    X10 = X01.T if X10 is None else X10
    Y10 = Y01.T if Y10 is None else Y10
    C0 = 0 if C0 is None else C0
    C1 = 0 if C1 is None else C1
    u0, v0 = nx.ones(p0.shape, type_as = p0), nx.ones(q0.shape, type_as = q0)
    u0_, v0_ = nx.copy(u0), nx.copy(v0)
    u1, v1 = nx.ones(p1.shape, type_as = p1), nx.ones(q1.shape, type_as = q1)
    u1_, v1_ = nx.copy(u1), nx.copy(v1)
    obj = 1e6
    is_converged = False
    for it in range(iters):
        M0 = (eta(X00, Y00, T0_.sum(-1), T0_.sum(0)) - X00 @ T0_ @ Y00.T) + (eta(X01, Y01, T1_.sum(-1), T1_.sum(0)) - X01 @ T1_ @ Y01.T)
        M0 += (T0_.sum()*C0 + (C0*T0_).sum())
        M0 += lamda1*(H(T0_.sum(-1), p0) + H(T1_.sum(-1), p1))
        M0 += lamda2*(H(T0_.sum(0), q0) + H(T1_.sum(0), q1)) 
        M0 += eps*(H(T0_, nx.outer(p0, q0)) + H(T1_, nx.outer(p1, q1)))
        M1 = (eta(X11, Y11, T1_.sum(-1), T1_.sum(0)) - X11 @ T1_ @ Y11.T) + (eta(X10, Y10, T0_.sum(-1), T0_.sum(0)) - X10 @ T0_ @ Y10.T)
        M1 += (T1_.sum()*C1 + (C1*T1_).sum())
        M1 += lamda1*(H(T0_.sum(-1), p0) + H(T1_.sum(-1), p1))
        M1 += lamda2*(H(T0_.sum(0), q0) + H(T1_.sum(0), q1)) 
        M1 += eps*(H(T0_, nx.outer(p0, q0)) + H(T1_, nx.outer(p1, q1)))
        m = T0_.sum() + T1_.sum()
        T0, u0, v0 = _solve_unbalanced(p0, q0, M0, u0, v0, eps*m, lamda1*m, lamda2*m, pi = T0, solver = solver)
        T1, u1, v1 = _solve_unbalanced(p1, q1, M1, u1, v1, eps*m, lamda1*m, lamda2*m, pi = T1, solver = solver)
        # u0, K0, v0 = sk.sinkhorn_unbalanced(p0, q0, M0, eps*m, lamda1*m, lamda2*m, u = u0, v = v0)
        # u1, K1, v1 = sk.sinkhorn_unbalanced(p1, q1, M1, eps*m, lamda1*m, lamda2*m, u = u1, v = v1)
        # T0 = K0*u0[:, None]*v0[None, :]
        # T1 = K1*u1[:, None]*v1[None, :]
        # T0 = ot.unbalanced.sinkhorn_unbalanced(p0, q0, M0, reg = eps*m, reg_m = lamda1*m)
        # T1 = ot.unbalanced.sinkhorn_unbalanced(p1, q1, M1, reg = eps*m, reg_m = lamda1*m)
        # rescale T0, T1 to have equal mass
        r = (T1.sum()/T0.sum())**0.5
        T0 *= r
        T1 /= r
        M0_ = (eta(X00.T, Y00.T, T0.sum(-1), T0.sum(0)) - X00.T @ T0 @ Y00) + (eta(X10.T, Y10.T, T1.sum(-1), T1.sum(0)) - X10.T @ T1 @ Y10)
        M0_ += (T0.sum()*C0 + (C0*T0).sum())
        M0_ += lamda1*(H(T0.sum(-1), p0) + H(T1.sum(-1), p1))
        M0_ += lamda2*(H(T0.sum(0), q0) + H(T1.sum(0), q1)) 
        M0_ += eps*(H(T0, nx.outer(p0, q0)) + H(T1, nx.outer(p1, q1)))
        M1_ = (eta(X11.T, Y11.T, T1.sum(-1), T1.sum(0)) - X11.T @ T1 @ Y11) + (eta(X01.T, Y01.T, T0.sum(-1), T0.sum(0)) - X01.T @ T0 @ Y01)
        M1_ += (T1.sum()*C1 + (C1*T1).sum())
        M1_ += lamda1*(H(T0.sum(-1), p0) + H(T1.sum(-1), p1))
        M1_ += lamda2*(H(T0.sum(0), q0) + H(T1.sum(0), q1)) 
        M1_ += eps*(H(T0, nx.outer(p0, q0)) + H(T1, nx.outer(p1, q1)))
        m = T0.sum() + T1.sum()
        T0_, u0_, v0_ = _solve_unbalanced(p0, q0, M0_, u0_, v0_, eps*m, lamda1*m, lamda2*m, pi = T0_, solver = solver)
        T1_, u1_, v1_ = _solve_unbalanced(p1, q1, M1_, u1_, v1_, eps*m, lamda1*m, lamda2*m, pi = T1_, solver = solver)
        # u0_, K0, v0_ = sk.sinkhorn_unbalanced(p0, q0, M0, eps*m, lamda1*m, lamda2*m, u = u0_, v = v0_)
        # u1_, K1, v1_ = sk.sinkhorn_unbalanced(p1, q1, M1, eps*m, lamda1*m, lamda2*m, u = u1_, v = v1_)
        # T0_ = K0*u0_[:, None]*v0_[None, :]
        # T1_ = K1*u1_[:, None]*v1_[None, :]
        # T0_ = ot.unbalanced.sinkhorn_unbalanced(p0, q0, M0, reg = eps*m, reg_m = lamda1*m)
        # T1_ = ot.unbalanced.sinkhorn_unbalanced(p1, q1, M1, reg = eps*m, reg_m = lamda1*m)
        # rescale T0_, T1_ to have equal mass
        r = (T1_.sum()/T0_.sum())**0.5
        T0_ *= r
        T1_ /= r
        # scale (T0, T1), (T0_, T1_) to have equal mass
        r0 = (T0.sum()/T0_.sum())**0.5
        r1 = (T1.sum()/T1_.sum())**0.5
        T0_ *= r0
        T0 /= r0
        T1_ *= r1
        T1 /= r1
        # print objective
        obj_new = partitioned_OT_objective(X00, Y00, X11, Y11, X01, Y01, X10, Y10, T0, T1)
        if (abs(obj_new - obj)/obj < rel_tol*obj) or (abs(obj_new - obj) < abs_tol):
            is_converged = True 
            break
        obj = obj_new 
        if print_iter is not None and it % print_iter == 0:
            print("Iteration %d, obj = %f" % (it, obj, ))
    if not is_converged and verbose:
        print("Warning: not converged")
    return T0, T1, T0_, T1_

def multiscale_partitioned_OT(Xs, Ys,
                              As, Bs,
                              ps, qs,
                              iters = 50, print_iter = 5, rel_tol = 1e-9, abs_tol = 1e-9, verbose = False,
                              eps = 0.01,
                              Ts = None, 
                              solver = "sinkhorn", solver_args = {}):
    k = len(ps)
    nx = get_backend(*Xs, *Ys, *As, *Bs, *ps, *qs)
    us, vs = [nx.ones(p.shape, type_as = p) for p in ps], [nx.ones(q.shape, type_as = q) for q in qs]
    Ts = [nx.outer(p, q) for (p, q) in zip(ps, qs)] if Ts is None else Ts 
    eps = nx.ones(k)*eps if np.isscalar(eps) else eps
    obj = 1e6
    is_converged = False
    def _obj():
        return sum([COOT_objective(Xs[i], Ys[i], Ts[i], Ts[i+1]) for i in range(k-1)])+sum([COOT_objective(As[i], Bs[i], Ts[i], Ts[i]) for i in range(k)])/2
    for it in range(iters):
        Ms = []
        for i in range(k):
            if i == 0:
                M = eta(Xs[i], Ys[i], Ts[i+1].sum(-1), Ts[i+1].sum(0)) - Xs[i] @ Ts[i+1] @ Ys[i].T
            elif i == k-1:
                M = eta(Xs[i-1].T, Ys[i-1].T, Ts[i-1].sum(-1), Ts[i-1].sum(0)) - Xs[i-1].T @ Ts[i-1] @ Ys[i-1]
            else:
                M = eta(Xs[i], Ys[i], Ts[i+1].sum(-1), Ts[i+1].sum(0)) - Xs[i] @ Ts[i+1] @ Ys[i].T 
                M += eta(Xs[i-1].T, Ys[i-1].T, Ts[i-1].sum(-1), Ts[i-1].sum(0)) - Xs[i-1].T @ Ts[i-1] @ Ys[i-1]
            # Gromov-Wasserstein part
            M += eta(As[i], Bs[i], Ts[i].sum(-1), Ts[i].sum(0)) - As[i] @ Ts[i] @ Bs[i].T
            Ms += [M, ]
        for i in range(k):
            if solver == "sinkhorn":
                Ts[i], us[i], vs[i] = _solve_balanced(ps[i], qs[i], Ms[i], us[i], vs[i], eps = eps[i], pi = Ts[i], solver = solver, **solver_args)
            elif solver == "ipot":
                Ts[i], _, _ = _solve_balanced(ps[i], qs[i], Ms[i], us[i], vs[i], eps = eps[i], pi = Ts[i], solver = solver, **solver_args)
            else:
                raise ValueError("Unsupported solver %s" % solver)
                return Ts
        if print_iter is not None and it % print_iter == 0:
            obj_new = _obj()
            if (abs(obj_new - obj)/obj < rel_tol*obj) or (abs(obj_new - obj) < abs_tol):
                is_converged = True 
                break
            obj = obj_new 
            print("Iteration %d, obj = %f" % (it, obj, ))
    if not is_converged and verbose:
        print("Warning: not converged")
    return Ts

def multiscale_partitioned_OT_cyclic_block(Xs, Ys,
                              As, Bs,
                              ps, qs,
                              iters = 50, print_iter = 5, rel_tol = 1e-9, abs_tol = 1e-9, verbose = False,
                              eps = 0.01,
                              Ts = None, 
                              solver = "sinkhorn", solver_args = {}):
    k = len(ps)
    nx = get_backend(*Xs, *Ys, *As, *Bs, *ps, *qs)
    us, vs = [nx.ones(p.shape, type_as = p) for p in ps], [nx.ones(q.shape, type_as = q) for q in qs]
    Ts = [nx.outer(p, q) for (p, q) in zip(ps, qs)] if Ts is None else Ts 
    eps = nx.ones(k)*eps if np.isscalar(eps) else eps
    obj = 1e6
    is_converged = False
    def _obj():
        return sum([COOT_objective(Xs[i], Ys[i], Ts[i], Ts[i+1]) for i in range(k-1)])+sum([COOT_objective(As[i], Bs[i], Ts[i], Ts[i]) for i in range(k)])/2
    for it in range(iters):
        def get_M(i):
            if i == 0:
                M = eta(Xs[i], Ys[i], Ts[i+1].sum(-1), Ts[i+1].sum(0)) - Xs[i] @ Ts[i+1] @ Ys[i].T
            elif i == k-1:
                M = eta(Xs[i-1].T, Ys[i-1].T, Ts[i-1].sum(-1), Ts[i-1].sum(0)) - Xs[i-1].T @ Ts[i-1] @ Ys[i-1]
            else:
                M = eta(Xs[i], Ys[i], Ts[i+1].sum(-1), Ts[i+1].sum(0)) - Xs[i] @ Ts[i+1] @ Ys[i].T 
                M += eta(Xs[i-1].T, Ys[i-1].T, Ts[i-1].sum(-1), Ts[i-1].sum(0)) - Xs[i-1].T @ Ts[i-1] @ Ys[i-1]
            return M
        for i in list(range(k)) + list(range(k)[::-1])[1:-1]:
            M = get_M(i)
            Ts[i], _ = ot.gromov.entropic_fused_gromov_wasserstein(M, As[i], Bs[i], ps[i], qs[i], epsilon = eps[i], log = True, verbose = False, **solver_args)
            if print_iter is not None and it % print_iter == 0:
                obj_new = _obj()
                if (abs(obj_new - obj)/obj < rel_tol*obj) or (abs(obj_new - obj) < abs_tol):
                    is_converged = True 
                    break
                obj = obj_new 
                print("Iteration %d, block %d, obj = %f" % (it, i, obj, ))
    if not is_converged and verbose:
        print("Warning: not converged")
    return Ts

def multiscale_partitioned_OT_unbalanced(Xs, Ys,
                                         As, Bs,
                                         ps, qs,
                                         iters = 50, print_iter = 5, rel_tol = 1e-9, abs_tol = 1e-9, verbose = False,
                                         eps = 0.01, lamda1 = 1, lamda2 = 1, 
                                         Ts = None, 
                                         solver = "sinkhorn", solver_args = {}):
    k = len(ps)
    nx = get_backend(*Xs, *Ys, *As, *Bs, *ps, *qs)
    us, vs = [nx.ones(p.shape, type_as = p) for p in ps], [nx.ones(q.shape, type_as = q) for q in qs]
    us_, vs_ = [nx.ones(p.shape, type_as = p) for p in ps], [nx.ones(q.shape, type_as = q) for q in qs]
    Ts = [nx.outer(p, q) for (p, q) in zip(ps, qs)] if Ts is None else Ts 
    Ts_ = [nx.outer(p, q) for (p, q) in zip(ps, qs)] if Ts is None else Ts.copy()
    eps = nx.ones(k)*eps if np.isscalar(eps) else eps
    obj = 1e6
    is_converged = False
    def _obj():
        return sum([COOT_objective(Xs[i], Ys[i], Ts[i], Ts_[i+1]) for i in range(k-1)])/2 +\
            sum([COOT_objective(Xs[i], Ys[i], Ts_[i], Ts[i+1]) for i in range(k-1)])/2 +\
            sum([COOT_objective(As[i], Bs[i], Ts[i], Ts_[i]) for i in range(k)])/2
    for it in range(iters):
        # update T
        Ms = []
        m = sum([x.sum() for x in Ts_])
        b = sum([eps[i]*H(T, nx.outer(p, q)) + lamda1*H(T.sum(-1), p) + lamda2*H(T.sum(0), q) for (i, (T, p, q)) in enumerate(zip(Ts_, ps, qs))])
        for i in range(k): # setup cost matrices
            if i == 0:
                M = (eta(Xs[i], Ys[i], Ts_[i+1].sum(-1), Ts_[i+1].sum(0)) - Xs[i] @ Ts_[i+1] @ Ys[i].T) / 2
            elif i == k-1:
                M = (eta(Xs[i-1].T, Ys[i-1].T, Ts_[i-1].sum(-1), Ts_[i-1].sum(0)) - Xs[i-1].T @ Ts_[i-1] @ Ys[i-1]) / 2
            else:
                M = (eta(Xs[i], Ys[i], Ts_[i+1].sum(-1), Ts_[i+1].sum(0)) - Xs[i] @ Ts_[i+1] @ Ys[i].T) / 2
                M += (eta(Xs[i-1].T, Ys[i-1].T, Ts_[i-1].sum(-1), Ts_[i-1].sum(0)) - Xs[i-1].T @ Ts_[i-1] @ Ys[i-1]) / 2
            M += (eta(As[i], Bs[i], Ts_[i].sum(-1), Ts_[i].sum(0)) - As[i] @ Ts_[i] @ Bs[i].T) / 2
            Ms += [M, ]
        for i in range(k): # solve in T
            Ts[i], us[i], vs[i] = _solve_unbalanced(ps[i], qs[i], Ms[i] + b, us[i], vs[i], eps[i]*m, lamda1*m, lamda2*m, pi = Ts[i], solver = solver)
        r = nx.exp(nx.mean([nx.log(x.sum()) for x in Ts]))
        for i in range(k): # scale all T to have same mass
            Ts[i] *= r / Ts[i].sum()
        # update T
        Ms_ = []
        m_ = sum([x.sum() for x in Ts])
        b_ = sum([eps[i]*H(T, nx.outer(p, q)) + lamda1*H(T.sum(-1), p) + lamda2*H(T.sum(0), q) for (i, (T, p, q)) in enumerate(zip(Ts, ps, qs))])
        for i in range(k): # setup cost matrices
            if i == 0:
                M = (eta(Xs[i], Ys[i], Ts[i+1].sum(-1), Ts[i+1].sum(0)) - Xs[i] @ Ts[i+1] @ Ys[i].T) / 2
            elif i == k-1:
                M = (eta(Xs[i-1].T, Ys[i-1].T, Ts[i-1].sum(-1), Ts[i-1].sum(0)) - Xs[i-1].T @ Ts[i-1] @ Ys[i-1]) / 2
            else:
                M = (eta(Xs[i], Ys[i], Ts[i+1].sum(-1), Ts[i+1].sum(0)) - Xs[i] @ Ts[i+1] @ Ys[i].T) / 2
                M += (eta(Xs[i-1].T, Ys[i-1].T, Ts[i-1].sum(-1), Ts[i-1].sum(0)) - Xs[i-1].T @ Ts[i-1] @ Ys[i-1]) / 2
            M += (eta(As[i], Bs[i], Ts[i].sum(-1), Ts[i].sum(0)) - As[i] @ Ts[i] @ Bs[i].T) / 2
            Ms_ += [M, ]
        for i in range(k): # solve in T
            Ts_[i], us_[i], vs_[i] = _solve_unbalanced(ps[i], qs[i], Ms_[i] + b_, us_[i], vs_[i], eps[i]*m_, lamda1*m_, lamda2*m_, pi = Ts_[i], solver = solver)
        r_ = nx.exp(nx.mean([nx.log(x.sum()) for x in Ts_]))
        for i in range(k): # scale all T_ to have same mass
            Ts_[i] *= r_ / Ts_[i].sum()
        # scale all T, T_ to have same mass
        for i in range(k):
            Ts[i] *= (r*r_)**0.5 / Ts[i].sum()
            Ts_[i] *= (r*r_)**0.5 / Ts_[i].sum()
        # print objective
        obj_new = _obj()
        if (abs(obj_new - obj)/obj < rel_tol*obj) or (abs(obj_new - obj) < abs_tol):
            is_converged = True 
            break
        obj = obj_new 
        if print_iter is not None and it % print_iter == 0:
            print("Iteration %d, obj = %f" % (it, obj, ))
    if not is_converged and verbose:
        print("Warning: not converged")
    return Ts, Ts_

def _solve_sr_exact(p, C):
    # min <C, T> under row-sum constraints on T
    min_ = C.min(-1)
    T = (C == min_.reshape(-1, 1))*1.0
    T *= (p / T.sum(-1)).reshape(-1, 1)
    return T 

def _solve_sr_ipot(p, C, reg, T_prev):
    # min <C, T> + reg*H(T, T_prev) under row-sum constraints on T
    nx = get_backend(p, C)
    K = T_prev * nx.exp(-C/reg)
    return (p / K.sum(-1))[:, None] * K

def _solve_sr_sinkhorn(p, C, reg):
    # min <C, T> + reg*H(T) under row-sum constraints on T
    nx = get_backend(p, C)
    K = nx.exp(-C/reg)
    return (p / K.sum(-1))[:, None] * K

def _solve_sr(p, C, solver = "emd", reg = None, T_prev = None):
    if solver == "emd":
        return _solve_sr_exact(p, C)
    elif solver == "sinkhorn":
        return _solve_sr_sinkhorn(p, C, reg)
    elif solver == "ipot":
        return _solve_sr_ipot(p, C, reg, T_prev)
    else:
        raise ValueError("Unsupported solver %s" % solver)

def sinkhorn_knopp_semiunbalanced(p, q, u, v, K, eps, lamda, numItermax = 1_000):
    # Sinkhorn-Knopp iterations for unbalanced entropy-regularized OT with marginals
    # (p, q), dual variables (u, v), Gibbs kernel K
    for i in range(numItermax):
        u = p / (K @ v)
        v = (q / (K.T @ u))**(lamda / (lamda + eps))
    return u, v
    
def _solve_sr_unbalanced(p, q, C, u, v, reg = 1, lamda = 1, solver = "ipot", T_prev = None, **kwargs):
    nx = get_backend(p, q, C, u, v)
    if solver == "ipot":
        # for IPOT, eps takes the role of the prox step
        Q =  T_prev * nx.exp(-C/reg) * p.reshape(-1, 1) * q.reshape(1, -1)
        u, v = sinkhorn_knopp_semiunbalanced(p, q, u, v, Q, reg, lamda, **kwargs)
        return nx.diag(u) @ Q @ nx.diag(v), u, v
    else:
        raise ValueError("Unsupported solver %s" % solver)
        return None, u, v
    
def srCOOT(X, p, q, Y, iter = 100, pi_s0 = None, pi_f0 = None,
           u_s0 = None, v_s0 = None,
           u_f0 = None, v_f0 = None, 
           solver = "emd", reg = 0, lamda = 0, print_iter = 10):
    nx = get_backend(X, p, q, Y)
    def init(p, u):
        # helper function for initializing dual potentials
        if u is None:
            return nx.ones(p.shape)
        else:
            return u
    p_ = ot.utils.unif(Y.shape[0], type_as = Y)
    q_ = ot.utils.unif(Y.shape[1], type_as = Y)
    pi_s = nx.outer(p, p_) if pi_s0 is None else pi_s0
    pi_f = nx.outer(q, q_) if pi_f0 is None else pi_f0
    u_s, v_s = init(p, u_s0), init(p_, v_s0) if solver != "emd" else (None, None)
    u_f, v_f = init(q, u_f0), init(q_, v_f0) if solver != "emd" else (None, None)
    for i in range(iter):
        obj = COOT_objective(X, Y, pi_s, pi_f)
        if (print_iter is not None) and (i % print_iter == 0):
            print(f"Iteration {i}, obj = {obj}")
        Cs = eta(X, Y, pi_f.sum(-1), pi_f.sum(0)) - X @ pi_f @ Y.T
        if lamda > 0:
            pi_s, u_s, v_s = _solve_sr_unbalanced(p, p_, Cs, u_s, v_s, solver = solver, reg = reg, T_prev = pi_s, lamda = lamda)
        else:
            pi_s = _solve_sr(p, Cs, solver = solver, reg = reg, T_prev = pi_s)
        Cf = eta(X.T, Y.T, pi_s.sum(-1), pi_s.sum(0)) - (X.T) @ pi_s @ Y
        if lamda > 0:
            pi_f, u_f, v_f = _solve_sr_unbalanced(q, q_, Cf, u_f, v_f, solver = solver, reg = reg, T_prev = pi_f, lamda = lamda)
        else:
            pi_f = _solve_sr(q, Cf, solver = solver, reg = reg, T_prev = pi_f)
    return pi_s, pi_f, obj

