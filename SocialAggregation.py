import torch as t
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from Attention import Attention
import scipy.sparse as sp
import dgl
from dgl.ops import edge_softmax

class SocialAgg(nn.Module):
    def __init__(self, trustMat, hide_dim, device, act):
        super(SocialAgg, self).__init__()
        self.userNum = trustMat.shape[0]
        self.uu_mat = trustMat
        self.hide_dim = hide_dim
        self.shape = t.Size(trustMat.shape)
        self.act = act

        row_idxs, col_idxs = self.uu_mat.nonzero()

        self.uu_g = dgl.graph(data=(row_idxs, col_idxs),
                              idtype=t.int32,
                              num_nodes=self.userNum,
                              device=device)

        self.uu_g.add_self_loop()

        self.row_idxs = t.LongTensor(row_idxs).cuda()
        self.col_idxs = t.LongTensor(col_idxs).cuda()
        self.idxs = t.from_numpy(np.vstack((row_idxs, col_idxs)).astype(np.int64)).cuda()

        self.att = Attention(self.hide_dim)
        self.w = nn.Linear(self.hide_dim, self.hide_dim)
    
    
    def forward(self, user_feat, hi):
        trust_emb = user_feat[self.row_idxs]

        trustee_emb = hi[self.col_idxs]

        weight = self.att(trust_emb, trustee_emb).view(-1, 1)

        # value = edge_softmax(self.uu_g, weight, norm_by='src').view(-1)
        value = edge_softmax(self.uu_g, weight).view(-1)
        
        A = t.sparse.FloatTensor(self.idxs, value, self.shape).detach()
        A = A.transpose(0, 1)
        
        
        if self.act is None:
            hs = self.w(t.spmm(A, hi))
        else:
            hs = self.act(self.w(t.spmm(A, hi)))
        return hs
