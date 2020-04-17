# from __future__ import division
# from __future__ import print_function
#
# import random
# import time
# import argparse
# import numpy as np
# import pickle as pkl
# from scipy.stats import zscore
#
# import torch
# import torch.nn.functional as F
# import torch.optim as optim
#
# from utils import *
# from models import GCN_SP, GCN_SP_three, GCN_SP_three_before, SpGAT, GAT
#
# # Training settings
# parser = argparse.ArgumentParser()
# parser.add_argument('--no_cuda', action='store_true', default=False,
#                     help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
# parser.add_argument('--random',action='store_true', default=False,
#                     help='Use random seed.')
# parser.add_argument('--seed', type=int, default=1, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=200,
#                     help='Number of epochs to train.')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='Initial learning rate.')
# parser.add_argument('--weight_decay', type=float, default=5e-4,
#                     help='Weight decay (L2 loss on parameters).')
# parser.add_argument('--hidden1', type=int, default=32,
#                     help='Number of hidden units.')
# parser.add_argument('--hidden2', type=int, default=16,
#                     help='Number of hidden units.')
# parser.add_argument('--dropout', type=float, default=0.5,
#                     help='Dropout rate (1 - keep probability).')
# parser.add_argument('--dataset', type=str, default='cora',
#                     help='Dataset')
# parser.add_argument('--n_train', type=int, default=10,
#                     help='Number of training examples per class')
# parser.add_argument('--n_test', type=int, default=1000,
#                     help='Size of test set')
# parser.add_argument('--n_val', type=int, default=500,
#                     help='Size of validation set')
# parser.add_argument('--struc_features', type=str, default='all',
#                     help='Which structural features to use')
# parser.add_argument('--alpha', type=float, default=0.5,
#                     help='Trade-off parameter alpha')
# parser.add_argument('--model', type=str, default='GCN_SP',
#                     help='Model string')
# parser.add_argument('--times', type=int,default=1,
#                     help='Run x times')
#
# #######################
# parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
# parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
# parser.add_argument('--relu_a', type=float, default=0.2, help='Alpha for the leaky_relu.')
#
#
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available() #use_cuda
#
# alpha = args.alpha
#
# if not args.random:
#     np.random.seed(args.seed)  # decide seed of numpy
#     torch.manual_seed(args.seed) # decide seed of torch
#
# if args.cuda:
#     torch.cuda.manual_seed(args.seed) #
#
# # Load data
# adj, features, labels = tuple(pkl.load(open('../data/' + args.dataset + '_data.pkl', 'rb'))) #(2708,2708),(2708,1433),(2708,7) all labels are identified.
#
# adj = normalize(adj + sp.eye(adj.shape[0])) #normarize
# features = preprocess_features(features) #Row-normalize
# n_train = args.n_train
# n_val = args.n_val
# n_test = args.n_test
# idx_train, idx_test, idx_val = split_data(labels, n_train, n_test, n_val)
# print('n_train : {}, n_test: {}, n_val:{}'.format(np.count_nonzero(idx_train), np.count_nonzero(idx_test), np.count_nonzero(idx_val)))
# print('training data : {:.4g}%'.format(np.count_nonzero(idx_train) / adj.shape[0] * 100))
#
# n_labels = labels.shape[1]
# n_feats = features.shape[0]
#
# struc_feat = np.load('../data/' + args.dataset + '_' + args.struc_features + '.npy') #(2708,5)
# struc_feat = zscore(struc_feat, axis=0) #標準化
# features = torch.FloatTensor(np.array(features.todense()))
# labels = torch.LongTensor(np.where(labels)[1])
# adj = sparse_mx_to_torch_sparse_tensor(adj)
# struc_feat = torch.FloatTensor(struc_feat)
#
#
# # Model and optimizer
# if args.model == 'GCN_SP':
#     model = GCN_SP(nfeat=features.shape[1],
#                    nhid=args.hidden1,
#                    nclass=labels.max().item() + 1,
#                    dropout=args.dropout,
#                    nstruc=struc_feat.shape[1]) # 1 or 5
#
# elif args.model == 'GCN_SP_three':
#     model = GCN_SP_three(nfeat=features.shape[1],
#                    nhid1=args.hidden1,
#                    nhid2 =args.hidden2,
#                    nclass=labels.max().item() + 1,
#                    dropout=args.dropout,
#                    nstruc=struc_feat.shape[1])
#
# elif args.model == 'GCN_SP_three_before':
#     model = GCN_SP_three_before(nfeat=features.shape[1],
#                    nhid1=args.hidden1,
#                    nhid2 =args.hidden2,
#                    nclass=labels.max().item() + 1,
#                    dropout=args.dropout,
#                    nstruc=struc_feat.shape[1])
#
# elif args.model == 'GAT':
#     model = GAT(nfeat=features.shape[1],
#                    nhid=args.hidden1,
#                    nclass=labels.max().item() + 1,
#                    dropout=args.dropout,
#                    nstruc=struc_feat.shape[1],
#                    nheads=args.nb_heads,
#                    relu_a=args.relu_a)
#
# elif args.model == 'GAT_SP':
#     model = SpGAT(nfeat=features.shape[1],
#                     nhid=args.hidden1,
#                     nclass=labels.max().item() + 1,
#                     dropout=args.dropout,
#                     nstruc=struc_feat.shape[1],
#                     nheads=args.nb_heads,
#                     relu_a=args.relu_a)
#
# else:
#     raise ValueError('Invalid argument for model: ' + str(args.model))
#
# optimizer = optim.Adam(model.parameters(),
#                        lr=args.lr, weight_decay=args.weight_decay)
#
# if args.cuda:
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()
#
# def train(epoch):
#     t = time.time()
#
#     #学習時の振る舞いに
#     model.train()
#
#     optimizer.zero_grad() #バックプロパゲーションの初期値をリセット
#     output_supervised, output_unsupervised = model(features, adj) #順伝播
#
#     #モデル側の返り値が既にsoftmaxであるのでnll_loss関数を使用してクロスエントロピーを出している。
#     #F.mse_loss = 平均二乗誤差
#     loss_train = (1-alpha) * F.nll_loss(output_supervised[idx_train], labels[idx_train]) \
#                  + alpha * F.mse_loss(output_unsupervised, struc_feat)
#
#     acc_train = accuracy(output_supervised[idx_train], labels[idx_train])
#
#     #勾配の計算
#     loss_train.backward()
#
#     #パラメータの更新
#     optimizer.step()
#
#     if not args.fastmode:
#         # Evaluate validation set performance separately,
#         # deactivates dropout during validation run.
#         model.eval()
#         output_supervised, output_unsupervised = model(features, adj)
#
#
#     loss_val = (1-alpha) * F.nll_loss(output_supervised[idx_val], labels[idx_val]) \
#            + alpha * F.mse_loss(output_unsupervised, struc_feat)
#     acc_val = accuracy(output_supervised[idx_val], labels[idx_val])
#
#     print('Epoch: {:04d}'.format(epoch+1),
#           'loss_train: {:.4f}'.format(loss_train.item()),
#           'acc_train: {:.4f}'.format(acc_train.item()),
#           'loss_val: {:.4f}'.format(loss_val.item()),
#           'acc_val: {:.4f}'.format(acc_val.item()),
#           'time: {:.4f}s'.format(time.time() - t))
#
# def test():
#     #テスト時はmodel.eval()をしないと勝手にパラメータ(dropout等)が更新されるので、model.eval()は絶対不可避。
#     model.eval()
#
#     output_supervised, output_unsupervised = model(features, adj)
#     loss_test = (1-alpha) * F.nll_loss(output_supervised[idx_test], labels[idx_test]) + alpha * F.mse_loss(output_unsupervised, struc_feat)
#     acc_test = accuracy(output_supervised[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))
#
# # Train model
# t_total = time.time()
# for epoch in range(args.epochs):
#     train(epoch)
# print("Optimization Finished!")
# t_end = time.time()
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
#
# # Testing
# test()
