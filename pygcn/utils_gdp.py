import numpy as np
import networkx as nx
from scipy import sparse as sp
import copy
import warnings
import json

import torch


def degree_power(A, k):
    r"""
    Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing
    normalised Laplacian.
    :param A: rank 2 array or sparse matrix.
    :param k: exponent to which elevate the degree matrix.
    :return: if A is a dense array, a dense array; if A is sparse, a sparse
    matrix in DIA format.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def normalized_adjacency(A, symmetric=True):
    r"""
    Normalizes the given adjacency matrix using the degree matrix as either
    \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
    :param A: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)


def gcn_filter(A, symmetric=True):
    r"""
    Computes the graph filter described in
    [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907).
    :param A: array or sparse matrix with rank 2 or 3;
    :param symmetric: boolean, whether to normalize the matrix as
    \(\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\) or as \(\D^{-1}\A\);
    :return: array or sparse matrix with rank 2 or 3, same as A;
    """
    out = copy.deepcopy(A)
    if isinstance(A, list) or (isinstance(A, np.ndarray) and A.ndim == 3):
        for i in range(len(A)):
            out[i] = A[i]
            out[i][np.diag_indices_from(out[i])] += 1
            out[i] = normalized_adjacency(out[i], symmetric=symmetric)
    else:
        if hasattr(out, "tocsr"):
            out = out.tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[np.diag_indices_from(out)] += 1
        out = normalized_adjacency(out, symmetric=symmetric)

    if sp.issparse(out):
        out.sort_indices()
    return out



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_labels(path):
    return np.load(path + "/labels.npy")


def load_folded_dataset(path):
    with open(path + "/graph.json", 'r') as f:
        graph_json = json.load(f)
    graph = nx.json_graph.node_link_graph(graph_json)
    adjacency_mat = nx.adjacency_matrix(graph)
    fltr = gcn_filter(adjacency_mat).astype('f4')
    fltr = sp.coo_matrix(fltr)
    fltr = sparse_mx_to_torch_sparse_tensor(fltr)

    features = np.load(path + "/feats.npy")
    train_mask = np.load(path + "/train_mask.npy")
    valid_mask = np.load(path + "/valid_mask.npy")

    features = torch.FloatTensor(features)
    train_mask = torch.BoolTensor(train_mask)
    valid_mask = torch.BoolTensor(valid_mask)
    return  fltr, features, train_mask, valid_mask


def load_embedding_from_txt(file_name):
    names = []
    embeddings = []
    with open(file_name, 'r') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            splitted = line.split()
            names.append(splitted[0])
            embeddings.append([float(value) for value in splitted[1:]])
    print(len(names)," nodes loaded.")
    return names, embeddings

