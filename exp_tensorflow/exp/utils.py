import numpy as np
import scipy.sparse as sp
import math

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

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

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
    train_mask = sample_labels(labels, n_train, sampling=sampling)
    test_val_mask = label_mask * np.invert(train_mask)
    test_val_idx = np.nonzero(test_val_mask == True)[0]
    test_idx = np.random.choice(test_val_idx, n_test, replace=False)
    val_idx = list(set(test_val_idx) - set(test_idx))
    val_idx = np.random.choice(val_idx, n_val, replace=False)
    test_mask = sample_mask(test_idx, n)
    val_mask = sample_mask(val_idx, n)
    train_idx = list(set(np.nonzero(train_mask)[0]))
    y_train = (labels.T * train_mask).T
    y_test = (labels.T * test_mask).T
    y_val = (labels.T * val_mask).T
    return train_mask, test_mask, val_mask, y_train, y_test, y_val

def rampup(epoch, scaled_unsup_weight_max, exp=5.0, rampup_length=80):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p * p * exp) * scaled_unsup_weight_max
    else:
        return 1.0 * scaled_unsup_weight_max

def get_scaled_unsup_weight_max(num_labels, X_train_shape, unsup_weight_max=100.0):
    return unsup_weight_max * 1.0 * num_labels / X_train_shape