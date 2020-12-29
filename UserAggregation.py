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

class UserAgg(nn.Module):
    def __init__(self, trainMat, hide_dim, device, act=None):
        super(UserAgg, self).__init__()
        self.userNum, self.itemNum = trainMat.shape
        self.uv_mat = trainMat
        self.hide_dim = hide_dim
        self.shape = t.Size(self.uv_mat.shape)
        self.act = act

        #item_idx, user_idx
        row_idxs, col_idxs = self.uv_mat.nonzero()
        self.uv_g = dgl.graph(data=(row_idxs, col_idxs + self.userNum),
                              idtype=t.int32,
                              num_nodes=self.userNum+self.itemNum,
                              device=device)


        self.row_idxs = t.LongTensor(row_idxs).cuda()
        self.col_idxs = t.LongTensor(col_idxs).cuda()
        self.rating = t.from_numpy(self.uv_mat.data).long().cuda()
        self.idxs = t.from_numpy(np.vstack((row_idxs, col_idxs)).astype(np.int64)).cuda()

        # self.w_r1 = nn.Linear(self.hide_dim * 2, self.hide_dim)
        self.gu = nn.Sequential(
                    nn.Linear(self.hide_dim * 2, self.hide_dim),
                    nn.ReLU(),
                    nn.Linear(self.hide_dim, self.hide_dim),
                    nn.ReLU()
                    )
        self.att = Attention(self.hide_dim)
        self.w = nn.Linear(self.hide_dim, self.hide_dim)
    
    
    def forward(self, user_feat, item_feat, rating_feat):
        r_emb = rating_feat[self.rating]
        u_emb = user_feat[self.row_idxs]
        i_emb = item_feat[self.col_idxs]
        
        # original peper formula (15)
        x = t.cat([u_emb, r_emb], dim=1)
        f_jt = self.gu(x)

        # f_jt = F.relu(self.w_r1(t.cat([u_emb, r_emb], dim=1)))
        weight = self.att(f_jt, i_emb).view(-1, 1)
        value = edge_softmax(self.uv_g, weight)

        self.uv_g.edata['h'] = f_jt * value

        self.uv_g.update_all(message_func=fn.copy_edge(edge='h', out='m'), \
            reduce_func=fn.sum(msg='m', out='n_f'))

        z = self.uv_g.ndata['n_f'][self.userNum:]
        
        if self.act is None:
            z = self.w(z)
        else:
            z = self.act(self.w(z))
        return z
