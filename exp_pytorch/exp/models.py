import torch.nn as nn
import torch.nn.functional as F
from layers import *
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj): #GCN1 -> relu -> dropout -> GCN2 -> softmax
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GCN_SP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc):
        super(GCN_SP, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.encoder = nn.Linear(nhid, nstruc)
        self.dropout = dropout

    def forward(self, x, adj): #same to GCN without y
        x = F.dropout(x,self.dropout,training=self.training) #11/27
        h = F.relu(self.gc1(x, adj))
        x = F.dropout(h, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        y = F.dropout(h, self.dropout, training=self.training)
        y = self.encoder(y)
        return F.log_softmax(x, dim=1), y #GCN1 -> relu -> (dropout) -> encoder


class GCN_SP_three(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout, nstruc):
        super(GCN_SP_three, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, nclass)
        self.encoder = nn.Linear(nhid2, nstruc)
        self.dropout = dropout

    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(self.gc2(h, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        out1 = self.gc3(h, adj)
        out2 = self.encoder(h)
        return F.log_softmax(out1, dim=1), out2

class GCN_SP_three_before(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout, nstruc):
        super(GCN_SP_three_before, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.gc3 = GraphConvolution(nhid2, nclass)
        self.encoder = nn.Linear(nhid1, nstruc)
        self.dropout = dropout

    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        out1 = F.relu(self.gc2(h, adj))
        out1 = F.dropout(out1, self.dropout, training=self.training)
        out1 = self.gc3(out1, adj)
        out2 = self.encoder(h)
        return F.log_softmax(out1, dim=1), out2

class GCN_SS(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc,relu_a): # add nstruc
        super(GCN_SS, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.encoder = nn.Linear(nhid, nstruc)
        self.dropout = dropout

        self.W = nn.Linear(nstruc,1)


    def forward(self, x, adj, struc_feat, idx, labels):
        x = F.dropout(x,self.dropout,training=self.training)
        h = F.relu(self.gc1(x, adj))
        x = F.dropout(h, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        y = F.dropout(h, self.dropout, training=self.training)
        y = self.encoder(y)
        out = x
        x = F.log_softmax(x, dim=1)
        #nor_sum = torch.abs(self.W.weight).sum() + 1e-15
        #print(y.shape)
        #print(self.W.weight.shape)
        y = y * self.W.weight.trace()
        # print(struc_feat)
        # print(torch.ones_like(self.W.weight))

        z = F.mse_loss(y, struc_feat)
        return F.nll_loss(x[idx], labels[idx]), z, out










###################################
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc, relu_a, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=relu_a, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=relu_a, concat=False)

        #self.encoder = nn.Linear(nhid,nstruc)
        self.encoder = nn.Linear(nhid * nheads, nstruc) # y = x A^T + b

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(h, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        y = F.dropout(h, self.dropout, training=self.training) # y or h
        y = self.encoder(h)
        return F.log_softmax(x, dim=1), y

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc, relu_a, nheads):
        """Dense version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=relu_a, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=relu_a, concat=False)

        self.encoder = nn.Linear(nhid * nheads, nstruc)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(h, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        y = F.dropout(h, self.dropout, training=self.training)
        y = self.encoder(y)
        return F.log_softmax(x, dim=1), y

class SpGAT_1_1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc, relu_a, nheads):
        """Dense version of GAT."""
        super(SpGAT_1_1, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=relu_a, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=relu_a, concat=False)

        self.gc = GraphConvolution(nhid * nheads, nhid * nheads)

        self.encoder = nn.Linear(nhid * nheads, nstruc)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        h1 = F.dropout(x, self.dropout, training=self.training)
        x = self.gc(h1, adj)
        h2 = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(h2, adj))
        y = self.encoder(h1)
        z = self.encoder(h2)
        return F.log_softmax(x, dim=1), y, z

class SpGAT_2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc, relu_a, nheads):
        """Dense version of GAT."""
        super(SpGAT_2, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=relu_a, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=relu_a, concat=False)

        self.encoder1 = nn.Linear(nhid * nheads, nstruc)
        self.encoder2 = nn.Linear(nclass,nstruc)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(h, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        z = self.encoder2(x)
        x = F.elu(x)
        y = F.dropout(h, self.dropout, training=self.training)
        y = self.encoder1(y)
        return F.log_softmax(x, dim=1), y, z

class SpGAT_3_1(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc, relu_a, nheads):
        """Dense version of GAT."""
        super(SpGAT_3_1, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=relu_a, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=relu_a, concat=False)

        self.gc = GraphConvolution(nhid * nheads, nhid * nheads)

        self.encoder = nn.Linear(nhid * nheads, nstruc)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        h1 = F.dropout(x, self.dropout, training=self.training)
        x = self.gc(h1, adj)
        h2 = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(h2, adj))
        y = self.encoder(h1)
        return F.log_softmax(x, dim=1), y

class SpGAT_3_2(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc, relu_a, nheads):
        """Dense version of GAT."""
        super(SpGAT_3_2, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=relu_a, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=relu_a, concat=False)
        self.gc = GraphConvolution(nhid * nheads, nhid * nheads)
        self.encoder = nn.Linear(nhid * nheads, nstruc)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        h1 = F.dropout(x, self.dropout, training=self.training)
        x = self.gc(h1, adj)
        h2 = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(h2, adj))
        y = self.encoder(h2)
        return F.log_softmax(x, dim=1), y

class SpGAT_3_3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc, relu_a, nheads):
        """Dense version of GAT."""
        super(SpGAT_3_3, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=relu_a, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=relu_a, concat=False)
        self.gc = GraphConvolution(nhid * nheads, nhid * nheads)
        self.encoder = nn.Linear(nclass, nstruc)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        h1 = F.dropout(x, self.dropout, training=self.training)
        x = self.gc(h1, adj)
        h2 = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(h2, adj))
        y = self.encoder(x)
        return F.log_softmax(x, dim=1), y

class GAT_SS(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc, relu_a, nheads):
        """Dense version of GAT."""
        super(GAT_SS, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=relu_a, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=relu_a, concat=False)

        self.encoder = nn.Linear(nhid * nheads, nstruc)

        self.out_att_ss = SpGraphAttentionLayer(nstruc, 1, dropout=dropout, alpha=relu_a, concat=False)

    def forward(self, x, adj,struc_feat,idx,labels):
        x = F.dropout(x, self.dropout, training=self.training)
        h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(h, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        y = F.dropout(h, self.dropout, training=self.training)
        y = self.encoder(y)
        #return F.log_softmax(x, dim=1), y
        out = x
        x = F.log_softmax(x, dim=1)

        y = F.elu(self.out_att_ss(y, adj))
        struc_feat = F.elu(self.out_att_ss(struc_feat, adj))

        z = F.mse_loss(y, struc_feat)
        return F.nll_loss(x[idx], labels[idx]), z, out
###########################################################2 models#######################################################
class GCN_Attention(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc):
        super(GCN_Attention, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.encoder = nn.Linear(nhid, nstruc)
        self.dropout = dropout

    def forward(self, x, adj, att):
        x = F.dropout(x,self.dropout,training=self.training)
        h = F.relu(self.gc1(x, adj))
        x = F.dropout(h, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        y = F.dropout(h, self.dropout, training=self.training)
        y = self.encoder(y)
        y = torch.sigmoid(y) #tried but failed
        y = y * att           #multiply glorot weight
        return F.log_softmax(x, dim=1), y

class Model_Attention(nn.Module):
    def __init__(self):
        super(Model_Attention, self).__init__()
        self.att = Parameter(torch.FloatTensor(1,5))
        glorot(self.att)

    def get_att(self):
        return self.att

    def forward(self,h):
        x = h * self.att
        return x #loss



# class GCN_attentionSS(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, nstruc):
#         super(GCN_attentionSS, self).__init__()
#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.encoder = nn.Linear(nhid, nstruc)
#         self.dropout = dropout
#         self.out_att = SpGraphAttentionLayer(nstruc, nstruc, dropout=dropout, alpha=0.2, concat=False)
#
#     def forward(self, x, adj,struc_feat):
#         x = F.dropout(x,self.dropout,training=self.training)
#         h = F.relu(self.gc1(x, adj))
#         x = F.dropout(h, self.dropout, training=self.training)
#         x = self.gc2(x, adj)
#         y = F.dropout(h, self.dropout, training=self.training)
#         y = self.encoder(y)
#         struc_feat = F.elu(self.out_att(struc_feat, adj))
#         return F.log_softmax(x, dim=1), y,struc_feat

class GCN_subatt(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc):
        super(GCN_subatt, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.encoder = nn.Linear(nhid, nstruc)
        self.dropout = dropout
        self.att = Parameter(torch.FloatTensor(nhid,nstruc))
        glorot(self.att)

    def forward(self, x, adj):  # same to GCN without y
        x = F.dropout(x, self.dropout, training=self.training)
        h = F.relu(self.gc1(x, adj))
        x = F.dropout(h, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        y = F.dropout(h, self.dropout, training=self.training)
        y = self.encoder(y)
        al = torch.mm(h,self.att)
        al = F.softmax(al,dim=1)#(nnode * nstruc)
        #print(al)
        return F.log_softmax(x, dim=1), y,al  # GCN1 -> relu -> (dropout) -> encoder


class GCN_subatt_test(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc):
        super(GCN_subatt_test, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.encoder = nn.Linear(nhid, nstruc)
        self.dropout = dropout
        self.att = Parameter(torch.FloatTensor(nhid,nstruc))
        glorot(self.att)

    def forward(self, x, adj):  # same to GCN without y
        x = F.dropout(x, self.dropout, training=self.training)
        h = F.relu(self.gc1(x, adj))
        x = F.dropout(h, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        y = F.dropout(h, self.dropout, training=self.training)
        y = self.encoder(y)
        al = torch.mm(h,self.att)
        nnode = al.size()[0]
        nstruc = al.size()[1]
        al = torch.flatten(al)
        al = F.softmax(al,dim=0)#(nnode * nstruc)
        al = al.reshape(nnode,nstruc)
        #print(al)
        return F.log_softmax(x, dim=1), y,al  # GCN1 -> relu -> (dropout) -> encoder




class GAT_subatt(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nstruc, relu_a, nheads):
        """Dense version of GAT."""
        super(GAT_subatt, self).__init__()
        self.dropout = dropout
        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=relu_a, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=relu_a, concat=False)
        self.encoder = nn.Linear(nhid * nheads, nstruc)

        self.att = Parameter(torch.FloatTensor(nhid * nheads, nstruc))
        glorot(self.att)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(h, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        y = F.dropout(h, self.dropout, training=self.training)
        y = self.encoder(y)
        al = torch.mm(h,self.att)
        al = F.softmax(al,dim=1)
        return F.log_softmax(x, dim=1), y ,al