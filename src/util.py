import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../tools/HyperCOT"))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../tools/hypergraph_cocluster"))
import numpy as np
import ot 
from ot.backend import get_backend
import torch
import hypernetx as hnx
import hypercot
from hypercot import get_hgraph_dual, convert_to_line_graph, get_v, get_omega

# general math functions
def dot(x, y):
    return (x * y).sum()

def outersum(x, y):
    return x.reshape(-1, 1) + y.reshape(1, -1)

def eta(x1, x2, p, q):
    return outersum(x1**2 @ p, x2**2 @ q)/2

def indicator(n, k):
    x = nx.zeros(n)
    x[k] = 1
    return x

import scipy as sp

def hg_heat_kernel(A, t):
    M = np.block([[np.zeros((A.shape[0], A.shape[0])), A], 
              [A.T, np.zeros((A.shape[1], A.shape[1]))]])
    d = M.sum(-1) + 1e-6
    # L = np.eye(M.shape[0]) - (d**-0.5).reshape(1, -1)*(d**-0.5).reshape(-1, 1)*M
    L = np.diag(d) - M
    M = sp.linalg.expm(-t*L)[range(A.shape[0]), :][:, A.shape[0]:]
    # M /= (M.max(-1).reshape(-1, 1) + 1e-6)
    return M / M.max()

def hg_heat_kernel_full(A, t):
    M = np.block([[np.zeros((A.shape[0], A.shape[0])), A], 
              [A.T, np.zeros((A.shape[1], A.shape[1]))]])
    d = M.sum(-1) + 1e-6
    # L = np.eye(M.shape[0]) - (d**-0.5).reshape(1, -1)*(d**-0.5).reshape(-1, 1)*M
    L = np.diag(d) - M
    K = sp.linalg.expm(-t*L)
    K_VE = K[range(A.shape[0]), A.shape[0]:]
    K_EV = K[A.shape[0]:, range(A.shape[0])]
    K_VV = K[:A.shape[0], :A.shape[0]]
    K_EE = K[A.shape[0]:, A.shape[0]:]
    return K_VV, K_EE, K_VE, K_EV


import scipy as sp
from algs import clique_expansion as cx 
from algs import star_expansion as sx 

def hg_heat_kernel2(A, t):
    R = A.T
    hyperedge_weights = sx.comp_hyperedge_weights(R)
    W = cx.comp_W(R, hyperedge_weights)
    P_VE = sx.comp_P_VE(W)
    P_EV = sx.comp_P_EV(R)
    P = sx.comp_P(P_VE, P_EV)
    P_alpha = sx.comp_P_alpha(P)
    pi = sx.comp_pi(P_alpha)
    L = np.eye(P.shape[0]) - P
    K = sp.linalg.expm(-t*L)
    M = K[range(P_VE.shape[0]), P_EV.shape[1]:] + K[P_VE.shape[0]:, range(P_EV.shape[1])].T
    return M / M.max()

def hg_heat_kernel2_full(A, t):
    R = A.T
    hyperedge_weights = sx.comp_hyperedge_weights(R)
    W = cx.comp_W(R, hyperedge_weights)
    P_VE = sx.comp_P_VE(W)
    P_EV = sx.comp_P_EV(R)
    P = sx.comp_P(P_VE, P_EV)
    P_alpha = sx.comp_P_alpha(P)
    pi = sx.comp_pi(P_alpha)
    L = np.eye(P.shape[0]) - P
    K = sp.linalg.expm(-t*L)
    K_VE = K[range(P_VE.shape[0]), P_EV.shape[1]:]
    K_EV = K[P_VE.shape[0]:, range(P_EV.shape[1])]
    K_VV = K[:P_VE.shape[0], :P_VE.shape[0]]
    K_EE = K[P_VE.shape[0]:, P_VE.shape[0]:]
    return K_VV, K_EE, K_VE, K_EV

import projection_simplex

def projection_simplex_rows(X):
    return np.vstack([projection_simplex.projection_simplex_sort(x) for x in X])

# blockmodel-like hypergraph
def hg_blockmodel_sample(k, # number of blocks
                  n, # number of vertices
                  m, # number of hyperedges
                  d, # hyperedge size
                  p_block, return_block_ids = False):
    blockid = np.hstack([np.full(n // k, i) for i in range(k)])
    blockid = np.append(blockid, [blockid[-1], ]*(n % k))
    A = np.zeros((n, m))
    i = 0
    blocks = []
    while i < m:
        if np.random.rand() < p_block:
            # in block sampling
            block = np.random.choice(np.unique(blockid))
            edge_idx = np.random.choice(np.where(blockid == block)[0], d)
            A[edge_idx, i] = 1
            i += 1
            blocks += [block, ]
        else:
            # out-of-block sampling
            edge_idx = np.random.choice(n, d)
            if len(np.unique(blockid[edge_idx])) > 1:
                A[edge_idx, i] = 1
                i += 1
            blocks += [-1, ]
    if return_block_ids:
        return A, blocks
    else:
        return A

def incidence_to_hnx(A):
    hg = hnx.Hypergraph()
    hg.add_edges_from([hnx.Entity(str(i), set(np.where(x.flatten() > 0)[0])) for (i, x) in enumerate(np.array(A))])
    return hg

def hnx_to_incidence(g):
    A = np.zeros(g.shape)
    for e in g.edges():
        A[:, int(e.uid)-1][np.array([int(k) for k in e.elements])-1] = 1
    return A 

def hnx_to_measure_hypernetwork(h):
    h_dual = get_hgraph_dual(h)
    l = convert_to_line_graph(h.incidence_dict)
    v = get_v(h.incidence_dict, h_dual.incidence_dict)
    w = hnx_to_incidence(h)
    return w, np.full((w.shape[0], ), 1/w.shape[0]), v
