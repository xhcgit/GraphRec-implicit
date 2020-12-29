import torch as t
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)


    def forward(self, src_emb, dst_emb):
        x = t.cat((src_emb, dst_emb), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        scores = self.att3(x)
        return scores


# class Attention(nn.Module):
#     def __init__(self, embedding_dims):
#         super(Attention, self).__init__()
#         self.embed_dim = embedding_dims
#         self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
#         self.att2 = nn.Linear(self.embed_dim, 1)

#     def forward(self, src_emb, dst_emb):
#         x = t.cat((src_emb, dst_emb), 1)
#         x = F.relu(self.att1(x))
#         scores = self.att2(x)
#         return scores


