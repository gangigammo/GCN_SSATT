import numpy as np
import scipy.sparse as sp
import torch
import math

import pickle as pkl
import networkx as nx
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

import random

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # print(sparse_mx.shape)(2708, 2708) 2708 * 2708ではない
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # print(indices.shape) #torch.Size([2, 13264])
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum += 1e-15
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sample_labels(labels, k, sampling='random'):
    n_class = labels.shape[1]
    mask = np.zeros(labels.shape[0])
    for i in range(n_class):
        ids = np.nonzero(labels[:, i])[0]
        if k > len(ids):
            print('Sample size larger than number of labelled examples for class ' + str(i))
            return
        if sampling == 'random':
            sampled = np.random.choice(ids, k, replace=False)
            mask[sampled] = 1
        if sampling == 'degree':
            sampled = np.argsort(D[ids])[-k:]
            mask[sampled] = 1
    return np.array(mask, dtype=np.bool)


def split_data(labels, n_train, n_test=1000, n_val=500, sampling='random'):
    n = labels.shape[0]
    label_mask = list(np.sum(labels, axis=1) == 1)
    train_mask = sample_labels(labels, n_train, sampling=sampling) #(2708,) boolean 9割false
    test_val_mask = label_mask * np.invert(train_mask)
    test_val_idx = np.nonzero(test_val_mask == True)[0]
    test_idx = np.random.choice(test_val_idx, n_test, replace=False) #replaceによって同じデータを二回取らない
    val_idx = list(set(test_val_idx) - set(test_idx))
    val_idx = np.random.choice(val_idx, n_val, replace=False)
    test_mask = sample_mask(test_idx, n)
    val_mask = sample_mask(val_idx, n)
    train_idx = list(set(np.nonzero(train_mask)[0]))
    return train_idx, test_idx, val_idx

def rampup(epoch, scaled_unsup_weight_max, exp=5.0, rampup_length=80):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p * p * exp) * scaled_unsup_weight_max
    else:
        return 1.0 * scaled_unsup_weight_max

def get_scaled_unsup_weight_max(num_labels, X_train_shape, unsup_weight_max=100.0):
    return unsup_weight_max * 1.0 * num_labels / X_train_shape




def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def previous_load_data(path="./data/cora/", dataset="cora" ,n_train = 140):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print(adj)
    # print(features)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    #idx_train = range(140)
    idx_train = range(n_train)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def new_load_data(path="./data/cora/", dataset="cora" ,n_train = 140):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    label_num = len(labels[0])

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    selected_label = np.array([])
    for i in range(label_num):
        z = np.where(labels == i)
        z = list(z)
        z = z[0]
        sampled = np.random.choice(z, n_train, replace=False) # 重複なし
        selected_label = np.append(selected_label, sampled)

    #print(selected_label)
    #idx_train = range(n_train)
    #idx_val = range(200, 500)
    #idx_test = range(500, 1500)

    selected_label = torch.LongTensor(selected_label)
    remain_labels = np.arange(len(adj[0]))
    idx_train = selected_label

    remain_labels = np.delete(remain_labels, idx_train)
    np.set_printoptions(edgeitems=100)
    random.shuffle(remain_labels)
    # print(remain_labels)

    idx_val = remain_labels[:300]
    idx_test = remain_labels[300:1300]
    # print(idx_val)
    # print(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_struc(path="./data/cora/", dataset="cora"):
    path_w = 'data/cora_struc.txt'

    f = open(path_w)
    areas = f.read().splitlines()
    f.close()
    # print(type(areas))
    l = len(areas)
    arr = [[0] * 5] * l
    # print(arr)
    for i in range(l):
        score = areas[i].split(" ")
        # print(len(score))
        for j in range(len(score)):
            # print(score[j])
            # print(float(score[j]))
            arr[i][j] = float(score[j])

    arr = np.array(arr)
    arr = torch.from_numpy(arr)
    return arr



############################################planetoid#######################################################
class Data(object):
    def __init__(self, adj, edge_list, features, labels, train_mask, val_mask, test_mask):
        self.adj = adj
        self.edge_list = edge_list
        self.features = features
        self.labels = labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_features = features.size(1)
        self.num_classes = int(torch.max(labels)) + 1

    def to(self, device):
        self.adj = self.adj.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)

def load_planetoid(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("data/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.Tensor(out))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1))
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    edge_list = adj_list_from_dict(graph)
    edge_list = add_self_loops(edge_list, features.size(0))
    adj = normalize_adj_pla(edge_list)

    train_mask = index_to_mask(train_idx, labels.shape[0])
    val_mask = index_to_mask(val_idx, labels.shape[0])
    test_mask = index_to_mask(test_idx, labels.shape[0])

    data = Data(adj, edge_list, features, labels, train_mask, val_mask, test_mask)

    return adj, features, labels, train_idx, test_idx, val_idx

def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def add_self_loops(edge_list, size):
    i = torch.arange(size).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def normalize_adj_pla(edge_list):
    row, col = edge_list
    deg = torch.bincount(row)
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]
    norm_adj = torch.sparse.FloatTensor(edge_list, v)
    return norm_adj

def preprocess_features_pla(features):
    rowsum = features.sum(dim=1, keepdim=True)
    rowsum[rowsum == 0] = 1
    features = features / rowsum
    return features