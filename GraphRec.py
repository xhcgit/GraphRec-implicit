import torch as t
import numpy as np
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from ItemAggregation import ItemAgg
from SocialAggregation import SocialAgg
from UserAggregation import UserAgg

class GraphRec(nn.Module):
    def __init__(self, args, hide_dim, trainMat, trustMat, device):
        super(GraphRec, self).__init__()
        self.args = args
        self.trainMat = trainMat
        self.trustMat = trustMat

        if args.act == 'relu':
            self.act = nn.ReLU()
        elif args.act == 'leakyrelu':
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = None
        
            

        self.userNum, self.itemNum = trainMat.shape
        self.hide_dim = hide_dim
        self.ratingClass = np.unique(trainMat.data).size
        initializer = nn.init.xavier_uniform_
        #init embedding
        self.userEmbedding = nn.Parameter(initializer(t.empty(self.userNum, hide_dim)))
        self.itemEmbedding = nn.Parameter(initializer(t.empty(self.itemNum, hide_dim)))
        self.ratingEmbedding = nn.Parameter(initializer(t.empty(self.ratingClass+1, hide_dim)))

        self.itemAgg = ItemAgg(self.trainMat, self.hide_dim, device, self.act)
        self.socialAgg = SocialAgg(self.trustMat, self.hide_dim, device, self.act)
        self.W2 = nn.Linear(self.hide_dim * 2, self.hide_dim)
        self.userAgg = UserAgg(self.trainMat, self.hide_dim, device, self.act)

        # self.w_ur1 = nn.Linear(self.hide_dim, self.hide_dim)
        # self.w_ur2 = nn.Linear(self.hide_dim, self.hide_dim)
        # self.w_vr1 = nn.Linear(self.hide_dim, self.hide_dim)
        # self.w_vr2 = nn.Linear(self.hide_dim, self.hide_dim)

        # self.w_uv1 = nn.Linear(self.hide_dim * 2, self.hide_dim)
        # self.w_uv2 = nn.Linear(self.hide_dim, 16)
        # self.w_uv3 = nn.Linear(16, 1)

        # self.bn1 = nn.BatchNorm1d(self.hide_dim, momentum=0.5)
        # self.bn2 = nn.BatchNorm1d(self.hide_dim, momentum=0.5)
        # self.bn3 = nn.BatchNorm1d(self.hide_dim, momentum=0.5)
        # self.bn4 = nn.BatchNorm1d(16, momentum=0.5)


    
    def forward(self):
        hI = self.itemAgg(self.userEmbedding, self.itemEmbedding, self.ratingEmbedding)
        hS = self.socialAgg(self.userEmbedding, hI)
        #original paper formula 12-14
        h = self.W2(t.cat([hI, hS], dim=1))
        if self.act is not None:
            h = self.act(h)

        z = self.userAgg(self.userEmbedding, self.itemEmbedding, self.ratingEmbedding)

        return h, z
        # copy from source code
        # x_u = self.act(self.bn1(self.w_ur1(user_emb)))
        # x_u = F.dropout(x_u, training=self.training)
        # x_u = self.w_ur2(x_u)

        # x_i = self.act(self.bn2(self.w_vr1(item_emb)))
        # x_i = F.dropout(x_i, training=self.training)
        # x_i = self.w_vr2(x_i)

        # x_ui = torch.cat((x_u, x_i), 1)
        # x = F.relu(self.bn3(self.w_uv1(x_ui)))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.bn4(self.w_uv2(x)))
        # x = F.dropout(x, training=self.training)
        # scores_i = self.w_uv3(x)
        # if isTest:
        #     return scores_i.squeeze()

        # x_uj = torch.cat((x_u, x_j), 1)
        # x2 = F.relu(self.bn3(self.w_uv1(x_uj)))
        # x2 = F.dropout(x2, training=self.training)
        # x2 = F.relu(self.bn4(self.w_uv2(x2)))
        # x2 = F.dropout(x2, training=self.training)
        # scores_j = self.w_uv3(x2)

        # return scores_i.squeeze(), scores_j.squeeze()


