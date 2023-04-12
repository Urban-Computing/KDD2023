import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out 

class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots
        self.islastlayer = islastlayer
        self.isbias = isbias
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        
        x_feature = x[:, 1:]
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T 
        x_treat_basis = self.spb.forward(x_treat).to(device) 
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)
        
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2)

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)
        return out

def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, t, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(t, self.num_grid)
        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out


class TyphoonCausal(nn.Module):
    def __init__(self, args, adj_mobility_list):
        super(TyphoonCausal, self).__init__()

        self.x_dim = args["x_dim"]  
        self.h_dim = args["h_dim"]
        self.z_dim = args["z_dim"]
        self.y_dim = args["y_dim"]
        self.n_layers_gcn = args["n_layers_gcn"]
        self.dropout = args["dropout"]
        self.w_dim = args["w_dim"] #dim=1
        self.g_dim = args["g_dim"]
        self.y_hist_dim = args["y_hist_dim"]
        self.history = args["history"]
        self.skip_connect = True

        self.phi_x = nn.Sequential(nn.Linear(x_dim, self.h_dim).to(device), nn.ReLU().to(device))
        self.gc = [GraphEncoder(self.h_dim, self.g_dim).to(device) for t in range(len(adj_mobility_list))]

        if self.skip_connect:
            self.fuse = nn.Sequential(nn.Linear(self.h_dim + self.h_dim + self.g_dim, self.z_dim).to(device), nn.ReLU().to(device))  
        else:
            self.fuse = nn.Sequential(nn.Linear(self.g_dim + self.h_dim, self.z_dim).to(device), nn.ReLU().to(device))  
     
        # memory unit
        if self.history:
            self.rnn = nn.GRUCell(self.z_dim + self.w_dim + self.y_hist_dim, self.h_dim).to(device) 
        else:
            self.rnn = nn.GRUCell(self.z_dim + self.w_dim , self.h_dim).to(device) 
    
        # causal output
        self.num_grid = args["grid"]
        cfg = args["cfg"] # cfg example: [(self.z_dim, self.z_dim, 1, 'relu'), (self.z_dim, 1, 1, 'sigmoid')]
        self.degree = args["degree"]
        self.knots = args["knots"]
      
        
        self.density_hidden_dim = self.z_dim
        self.density_estimator_head = Density_Block(self.num_grid, self.density_hidden_dim, isbias=1)
        
        
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)

        self.outcome_estimator = nn.Sequential(*blocks)
        
        
       
    def forward(self, X_list, edge_index_list, W_list, Y_hist_list, hidden_in, edge_weight_list):

        all_y = []
        all_conditional_density = []

        num_timestep = len(X_list)
        num_node = X_list[-1].size(0)

        if hidden_in is None:
            h = Variable(torch.zeros(num_node, self.h_dim))
        else:
            h = Variable(hidden_in)

        h = h.to(device)

        for t in range(num_timestep):  # time step
            w_t = W_list[t]
            x_t = X_list[t]
            edge_index = edge_index_list[t]
            edge_weight = edge_weight_list[t]
            y_hist = Y_hist_list[t]
            phi_x_t = self.phi_x(x_t)

            # graph
            rep = F.relu(self.gc[t](phi_x_t, edge_index, edge_weight))
            rep = F.dropout(rep, self.dropout, training=self.training)

            if self.skip_connect:
                z_t = self.fuse(torch.cat([h, rep, phi_x_t], 1))
            else:
                z_t = self.fuse(torch.cat([h, rep], 1))
            
        
            # RNN
            if self.history:
                h = self.rnn(torch.cat([z_t, w_t.view(-1, self.w_dim), y_hist], 1), h)
            else:
                h = self.rnn(torch.cat([z_t, w_t.view(-1, self.w_dim)], 1), h)
                
            t_hidden = torch.cat([z_t, w_t], 1)
            g_t = self.density_estimator_head(w_t, z_t)
            y_t = self.outcome_estimator(t_hidden)
            all_y.append(y_t)
            all_conditional_density.append(g_t)
            
        return all_y, all_conditional_density, h
    


class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv = GCN(in_channels, out_channels)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()
        self.gc = GCNConv(nfeat, nhid)

    def forward(self, x, edge_index, edge_weight):
        x = self.gc(x, edge_index, edge_weight.sigmoid())
        return x