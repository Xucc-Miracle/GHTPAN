import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3):
      
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.channel_attention = ChannelAttention(num_channels = out_channels)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, X):
       
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        out = self.channel_attention(out)
        out = out.permute(0, 2, 3, 1)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, num_channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, num_channels, 1, 1)
        return x * y.expand_as(x)


class STGCNBlock(nn.Module):
 
    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes,A_hat):
        
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
    
       
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)


class STGCN(nn.Module):
 

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output,A_hat):
     
        super(STGCN, self).__init__()
        self.A_hat = A_hat
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes,A_hat=A_hat)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes,A_hat=A_hat)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.pred = MultiHeadSelfAttention(128,53*12,12,12)

    def forward(self,X,A_hat):
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.pred(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim_in, dim_k, dim_v, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=True)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=True)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=True)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  
        dv = self.dim_v // nh  
        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  
        dist = torch.softmax(dist, dim=-1)  
        att = torch.matmul(dist, v)  
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  
        return att
