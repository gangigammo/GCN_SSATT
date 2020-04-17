from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pickle as pkl
from scipy.stats import zscore
import networkx as nx
from scipy import sparse as sp

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import configure, log_value

from utils import *
from models import GCN, GCN_SP, GCN_SP_three, GCN_SP_three_before

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset')
parser.add_argument('--n_train', type=int, default=20,
                    help='Number of training examples per class')
parser.add_argument('--n_test', type=int, default=300,
                    help='Size of test set data')
parser.add_argument('--n_val', type=int, default=100,
                    help='Size of validation set data per class')
parser.add_argument('--struc_features', type=str, default='all',
                    help='Type of using structural features')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Parametor of weighting for structure preserving')
parser.add_argument('--weight_type', type=str, default='static',
                    help='Weight type')
parser.add_argument('--model', type=str, default='GCN_SP',
                    help='model')
parser.add_argument('--graph_size', type=int, default=5000,
                    help='graph_size')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.weight_type == 'temporal':
    alpha = 0
else:
    alpha = args.alpha

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
graph = nx.barabasi_albert_graph(args.graph_size, 5)
adj = nx.adjacency_matrix(graph).astype(np.int64)
features = np.eye(args.graph_size)
features = sp.lil_matrix(features)
labels = np.random.randint(1, 5, (args.graph_size, 1))
labels = np.eye(args.graph_size)[labels]
n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
idx_train, idx_test, idx_val = split_data(labels, n_train, n_test, n_val)
print('n_train : {}, n_test: {}, n_val:{}'.format(np.count_nonzero(idx_train), np.count_nonzero(idx_test), np.count_nonzero(idx_val)))

n_labels = labels.shape[1]
n_feats = features.shape[0]

struc_feat = np.random.rand(args.graph_size,5)

features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])
adj = sparse_mx_to_torch_sparse_tensor(adj)
struc_feat = torch.FloatTensor(struc_feat)

# Model and optimizer

if args.model == 'GCN_SP':
    model = GCN_SP(nfeat=features.shape[1],
                   nhid=args.hidden1,
                   nclass=labels.max().item() + 1,
                   dropout=args.dropout,
                   nstruc=struc_feat.shape[1])
elif args.model == 'GCN_SP_three':

    model = GCN_SP_three(nfeat=features.shape[1],
                   nhid1=args.hidden1,
                   nhid2 = args.hidden2,
                   nclass=labels.max().item() + 1,
                   dropout=args.dropout,
                   nstruc=struc_feat.shape[1])

elif args.model == 'GCN_SP_three_before':

    model = GCN_SP_three_before(nfeat=features.shape[1],
                   nhid1=args.hidden1,
                   nhid2 = args.hidden2,
                   nclass=labels.max().item() + 1,
                   dropout=args.dropout,
                   nstruc=struc_feat.shape[1])

elif args.model == 'GCN':
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden1,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)

else:
    raise ValueError('Invalid argument for model: ' + str(args.model))


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

configure("runs/run-3GCN-cite")

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    if args.model == 'GCN':
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

    else:
        output_supervised, output_unsupervised = model(features, adj)

        loss_train = F.nll_loss(output_supervised[idx_train], labels[idx_train]) \
                     + alpha * F.mse_loss(output_unsupervised, struc_feat)

        acc_train = accuracy(output_supervised[idx_train], labels[idx_train])

    loss_train.backward()

    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        if args.model == 'GCN':
            output = model(features, adj)
        else:
            output_supervised, output_unsupervised = model(features, adj)

    if args.model =='GCN':
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
    else:
        loss_val = (1-alpha) * F.nll_loss(output_supervised[idx_val], labels[idx_val]) \
           #+ alpha * F.mse_loss(output_unsupervised, struc_feat)
        acc_val = accuracy(output_supervised[idx_val], labels[idx_val])

    '''
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    '''

    log_value('acc', acc_val.item(), epoch)
    log_value('loss', loss_train.item(), epoch)


def test():
    model.eval()
    output_supervised, output_unsupervised = model(features, adj)
    loss_test = (1-alpha) * F.nll_loss(output_supervised[idx_test], labels[idx_test]) + alpha * F.mse_loss(output_unsupervised, struc_feat)
    acc_test = accuracy(output_supervised[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
t_end = time.time()
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print('time per epoch: {:.4f}s'.format((t_end - t_total)/args.epochs))


# Testing
test()