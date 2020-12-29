import torch as t
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from Attention import Attention
import scipy.sparse as sp
import dgl
import dgl.function as fn
from dgl.ops import edge_softmax

class ItemAgg(nn.Module):
    def __init__(self, trainMat, hide_dim, device, act=None):
        super(ItemAgg, self).__init__()
        self.userNum, self.itemNum = trainMat.shape
        self.vu_mat = trainMat.T.tocoo().tocsr()
        self.hide_dim = hide_dim
        self.shape = t.Size(self.vu_mat.shape)
        self.act = act

        #item_idx, user_idx
        row_idxs, col_idxs = self.vu_mat.nonzero()
        self.vu_g = dgl.graph(data=(row_idxs + self.userNum, col_idxs),
                              idtype=t.int32,
                              num_nodes=self.userNum+self.itemNum,
                              device=device)


        self.row_idxs = t.LongTensor(row_idxs).cuda()
        self.col_idxs = t.LongTensor(col_idxs).cuda()
        self.rating = t.from_numpy(self.vu_mat.data).long().cuda()
        self.idxs = t.from_numpy(np.vstack((row_idxs, col_idxs)).astype(np.int64)).cuda()

        # self.w_r1 = nn.Linear(self.hide_dim * 2, self.hide_dim)
        self.gv = nn.Sequential(
                    nn.Linear(self.hide_dim * 2, self.hide_dim),
                    nn.ReLU(),
                    nn.Linear(self.hide_dim, self.hide_dim),
                    nn.ReLU()
                    )
        self.att = Attention(self.hide_dim)
        self.w = nn.Linear(self.hide_dim, self.hide_dim)
    
    
    def forward(self, user_feat, item_feat, rating_feat):
        r_emb = rating_feat[self.rating]
        i_emb = item_feat[self.row_idxs]
        u_emb = user_feat[self.col_idxs]
        
        # original peper formula (2)
        x = t.cat([i_emb, r_emb], dim=1)
        x_ia = self.gv(x)

        weight = self.att(x_ia, u_emb).view(-1, 1)
        value = edge_softmax(self.vu_g, weight)

        self.vu_g.edata['h'] = x_ia * value

        self.vu_g.update_all(message_func=fn.copy_edge(edge='h', out='m'), \
            reduce_func=fn.sum(msg='m', out='n_f'))

        h = self.vu_g.ndata['n_f'][:self.userNum]
        
        if self.act is None:
            hi = self.w(h)
        else:
            hi = self.act(self.w(h))

        return hi
