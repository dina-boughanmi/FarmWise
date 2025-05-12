import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch import nn

class GCNNet(nn.Module):
    def __init__(self, in_dim, hid_dim=32, out_dim=16, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
