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
        self.raitngClass = np.unique(trainMat.data).size
        initializer = nn.init.xavier_uniform_
        #init embedding
        self.userEmbedding = nn.Parameter(initializer(t.empty(self.userNum, hide_dim)))
        self.itemEmbedding = nn.Parameter(initializer(t.empty(self.itemNum, hide_dim)))
        self.ratingEmbedding = nn.Parameter(initializer(t.empty(self.raitngClass+1, hide_dim)))

        self.itemAgg = ItemAgg(self.trainMat, self.hide_dim, device, self.act)
        self.socialAgg = SocialAgg(self.trustMat, self.hide_dim, device, self.act)
        self.W2 = nn.Linear(self.hide_dim * 2, self.hide_dim)
        self.userAgg = UserAgg(self.trainMat, self.hide_dim, device, self.act)

    
    def forward(self):
        hI = self.itemAgg(self.userEmbedding, self.itemEmbedding, self.ratingEmbedding)
        hS = self.socialAgg(self.userEmbedding, hI)
        #original paper formula 12-14
        if self.act is None:
            self.W2(t.cat([hI, hS], dim=1))
        else:
            h = self.act(self.W2(t.cat([hI, hS], dim=1)))

        z = self.userAgg(self.userEmbedding, self.itemEmbedding, self.ratingEmbedding)
        return h, z
