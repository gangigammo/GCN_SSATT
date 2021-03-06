from __future__ import division
from __future__ import print_function

import os
import glob
import random
import time
import argparse
import numpy as np
import pickle as pkl
from scipy.stats import zscore
from statistics import mean, median,variance,stdev


import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import *

from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--random',action='store_true', default=False,help='Use random seed.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=32,help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',help='Dataset')
parser.add_argument('--n_train', type=int, default=10,help='Number of training examples per class')
parser.add_argument('--n_test', type=int, default=1000,help='Size of test set')
parser.add_argument('--n_val', type=int, default=500,help='Size of validation set')
parser.add_argument('--struc_features', type=str, default='all',help='Which structural features to use')
parser.add_argument('--alpha', type=float, default=0.5,help='Trade-off parameter alpha')
parser.add_argument('--model', type=str, default='GCN_SP',help='Model string')
####################### additional #############################
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--relu_a', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--times', type=int,default=100,help='Run x times')
parser.add_argument('--patience', type=int, default=100, help='early_stopping')
parser.add_argument('--split', type=int, default=10, help='split_num')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available() #use_cuda

alpha = args.alpha


if not args.random:
    np.random.seed(args.seed)  # decide seed of numpy
    torch.manual_seed(args.seed)  # decide seed of torch
if args.cuda:
    torch.cuda.manual_seed(args.seed)

count = args.times
t_total = time.time()
arr = []
acc_arr_all = np.zeros((count,args.split))
for i in range(count):

    adj, features, labelall = tuple(pkl.load(open('../data/' + args.dataset + '_data.pkl','rb')))  # (2708,2708),(2708,1433),(2708,7)

    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = preprocess_features(features)  # Row-normalize

    n_train = args.n_train
    n_val = args.n_val
    n_test = args.n_test

    idx_train, idx_test, idx_val = split_data(labelall, n_train, n_test, n_val)

    n_labels = labelall.shape[1]
    n_feats = features.shape[0]

    struc_feat = np.load('../data/' + args.dataset + '_' + args.struc_features + '.npy')  # (2708,5)
    struc_feat = zscore(struc_feat, axis=0)  # 標準化
    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to_dense()
    struc_feat = torch.FloatTensor(struc_feat)

    struc = struc_feat.flatten()
    struc = struc.numpy()
    sorted_feat = np.sort(struc)
    arg_feat = np.argsort(struc)
    pos_feat = np.argsort(arg_feat)

    acc_pos = np.zeros(args.split)
    acc_neg = np.zeros(args.split)
    acc_ss = np.zeros(args.split)

    if i == 0:
        print('n_train : {}, n_test: {}, n_val:{}'.format(np.count_nonzero(idx_train), np.count_nonzero(idx_test),
                                                          np.count_nonzero(idx_val)))
        print('training data : {:.4g}%'.format(np.count_nonzero(idx_train) / adj.shape[0] * 100))
    labels = torch.LongTensor(np.where(labelall)[1])

    model = GCN_Attention(nfeat=features.shape[1],
                          nhid=args.hidden1,
                          nclass=labels.max().item() + 1,
                          dropout=args.dropout,
                          nstruc=struc_feat.shape[1])


    model_atten = Model_Attention()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    optimizer_atten = optim.Adam(model_atten.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
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
        output_supervised, output_unsupervised = model(features, adj,model_atten.get_att())
        su_loss_train = F.nll_loss(output_supervised[idx_train], labels[idx_train])
        uns_loss_train = F.mse_loss(output_unsupervised, struc_feat)

        loss_train = (1 - alpha) * su_loss_train + alpha * uns_loss_train
        loss_train.backward(retain_graph=True)
        optimizer.step()

        # model_atten.train()
        # optimizer_atten.zero_grad()
        # output_att = model_atten(output_unsupervised)
        # loss_att = F.mse_loss(output_att, struc_feat)
        # loss_att.backward(retain_graph=True)
        # optimizer_atten.step()
        #print(model_atten.get_att())


        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output_supervised, output_unsupervised = model(features, adj, model_atten.get_att())

        loss_val = (1 - alpha) * F.nll_loss(output_supervised[idx_val], labels[idx_val]) \
            + alpha * F.mse_loss(output_unsupervised, struc_feat)
        #acc_val = accuracy(output_supervised[idx_val], labels[idx_val])

        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'su_loss_train: {:.4f}'.format(su_loss_train.item()),
        #       'uns_loss_train: {:.4f}'.format(uns_loss_train.item()))

        return loss_val.data.item()

    def test():
        # テスト時はmodel.eval()をしないと勝手にパラメータ(dropout等)が更新されるので、model.eval()は絶対不可避。
        model.eval()
        #model_atten.get_att()

        output_supervised, output_unsupervised = model(features, adj, model_atten.get_att())
        loss_test = (1 - alpha) * F.nll_loss(output_supervised[idx_test], labels[idx_test]) + alpha * F.mse_loss(
            output_unsupervised, struc_feat)
        acc_test = accuracy(output_supervised[idx_test], labels[idx_test]) * 100

        acc_arr = np.zeros(args.split)
        nnodes = struc_feat.shape[0]
        for j in range(args.split):
            l_pre = [x for x in idx_test if (j * nnodes / args.split <= pos_feat[x] and pos_feat[x] < (j+1) * nnodes / args.split)]
            acc_arr[j] = accuracy(output_supervised[l_pre], labels[l_pre]) * 100
        acc_arr_all[i,:] = acc_arr[:]
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test

    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train(epoch))

        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    #print("Optimization Finished!")
    #print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    #print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    arr.append(test().item())

m = mean(arr)
stdev = stdev(arr)
print('total accuracy {:.4f} ± {:.4f}'.format(m,stdev))
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print(acc_arr_all.mean(axis=0))
file = open('log.txt','a')
file.write('total accuracy {:.4f} ± {:.4f}\n'.format(m,stdev))
file.close()