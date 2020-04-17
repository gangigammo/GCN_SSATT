from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models import GCN, GCN_SP
import sys

def run_model(adj, features, labels, lr=0.01, L1_reg=0.00, L2_reg=0.00,
              epochs=200, dataset='cora', dropout=0.3,
              hidden1=32, weight_decay=5e-4, n_train=20, n_test=1000, n_val=500,
              if_cuda=False, seed=42, fastmode=False, struc_features='all', alpha=0.5):

    np.random.seed(seed)
    torch.manual_seed(seed)
    if if_cuda:
        torch.cuda.manual_seed(seed)

    idx_train, idx_test, idx_val = split_data(labels, n_train, n_test, n_val)
    print('n_train : {}, n_test: {}, n_val:{}'.format(np.count_nonzero(idx_train), np.count_nonzero(idx_test),
                                                      np.count_nonzero(idx_val)))

    struc_feat = np.load('../data/' + dataset + '_' + struc_features + '.npy')

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    struc_feat = torch.FloatTensor(struc_feat)

    # Model and optimizer

    model = GCN_SP(nfeat=features.shape[1],
                   nhid=hidden1,
                   nclass=labels.max().item() + 1,
                   dropout=dropout,
                   nstruc=struc_feat.shape[1])
    optimizer = optim.Adam(model.parameters(),
                           lr=lr, weight_decay=weight_decay)

    if if_cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output_supervised, output_unsupervised = model(features, adj)
        loss_train = F.nll_loss(output_supervised[idx_train], labels[idx_train]) \
                     + alpha * F.mse_loss(output_unsupervised, struc_feat)
        acc_train = accuracy(output_supervised[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output_supervised, output_unsupervised = model(features, adj)

        loss_val = (1 - alpha) * F.nll_loss(output_supervised[idx_val], labels[idx_val]) \
            # + alpha * F.mse_loss(output_unsupervised, struc_feat)
        acc_val = accuracy(output_supervised[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def test():
        model.eval()
        output_supervised, output_unsupervised = model(features, adj)
        loss_test = (1 - alpha) * F.nll_loss(output_supervised[idx_test],
                                                  labels[idx_test]) + alpha * F.mse_loss(output_unsupervised,
                                                                                              struc_feat)
        acc_test = accuracy(output_supervised[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    # Train model
    t_total = time.time()
    for epoch in range(epochs):
        train(epoch)
    print("Optimization Finished!")
    t_end = time.time()
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('time per epochs: {:.4f}s'.format(t_end - t_total))

    # Testing
    test()

if __name__ == '__main__':
    dataset = sys.argv[1]
    # Load data
    print('Loading ' + dataset + ' dataset...')
    adj, features, labels = tuple(pkl.load(open('../data/' + dataset + '_data.pkl', 'rb')))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = preprocess_features(features)
    run_model(adj, features, labels)

