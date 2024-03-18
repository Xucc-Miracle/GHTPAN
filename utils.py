import os
import csv
import warnings

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn


files = {
    'pems03': ['PEMS03/PEMS03.npz', 'PEMS03/PEMS03.csv'],
    'pems04': ['PEMS04/PEMS04.npz', 'PEMS04/PEMS04.csv'],
    'pems08': ['PEMS08/PEMS08.npz', 'PEMS08/PEMS08.csv']
}

Tensor = torch.Tensor
def scaled_laplacian(num_node,node_embeddings, is_eval=False):
    node_num = num_node
    learned_graph = torch.mm(node_embeddings, node_embeddings.transpose(0, 1))
    norm = torch.norm(node_embeddings, p=2, dim=1, keepdim=True)
    norm = torch.mm(norm, norm.transpose(0, 1))
    learned_graph = learned_graph / norm
    learned_graph = (learned_graph + 1) / 2.
    learned_graph = torch.stack([learned_graph, 1 - learned_graph], dim=-1)
    adj = learned_graph
    adj = adj[:, :, 0].clone().reshape(node_num, -1)
    mask = torch.eye(node_num, node_num).bool()
    adj.masked_fill_(mask, 0)

    W = adj
    n = W.shape[0]
    d = torch.sum(W, axis=1)
    L = -W
    L[range(len(L)), range(len(L))] = d

    try:
        lambda_max = (L.max() - L.min())
    except Exception as e:
        print("eig error!!: {}".format(e))
        lambda_max = 1.0
    tilde = (2 * L / lambda_max - torch.eye(n))

    return adj, tilde


def read_data(args):
    filename = args.filename
    file = files[filename]
    filepath = "./data/"
    data = np.load(filepath + file[0])['data']
    data_cp = data
    num_node = data.shape[1]
    mean_value = np.mean(data, axis=(0, 1)).reshape(1, 1, -1)
    std_value = np.std(data, axis=(0, 1)).reshape(1, 1, -1)
    data = (data - mean_value) / std_value
    mean_value = mean_value.reshape(-1)[0]
    std_value = std_value.reshape(-1)[0]
    dist_matrix = np.load(f'./data/{filename}_spatial_distance.npy')

    std = np.std(dist_matrix[dist_matrix != float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
    sp_matrix[sp_matrix < args.thres2] = 0
    node_embeddings = nn.Parameter(torch.randn(args.num_nodes, args.embed_dim))
    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0)/2/num_node}')

    return torch.from_numpy(data_cp.astype(np.float32)), mean_value, std_value,  sp_matrix

def get_normalized_adj(A):
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)

    return torch.from_numpy(A_reg.astype(np.float32))



def generate_graph_seq2seq_io_data( data, x_offsets, y_offsets):

    num_samples, num_nodes = data.shape
    data = data[:, :]

    x, y = [], []
    min_t = abs(min(x_offsets)) 
    max_t = abs(num_samples - abs(max(y_offsets)))  

    for t in tqdm(range(min_t, max_t)):
        x_t = data[t + x_offsets, ...]
        x.append(x_t)
    x = np.stack(x, axis=0)  
    return x

def std_mean(data):
    seq_length_x = 12
    seq_length_y = 12
    y_start=1
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(y_start, (seq_length_y + 1), 1)
    data = generate_graph_seq2seq_io_data(data=data, x_offsets=x_offsets, y_offsets=y_offsets)
    mean_value = np.mean(data)
    std_value = np.std(data)
    print(mean_value, std_value)
    data = (data - mean_value) / std_value
    del data
    return mean_value,std_value

class MyDataset(Dataset):

    def __init__(self, data, split_start, split_end, his_length, pred_length,mean,std):
        split_start = int(split_start)
        split_end = int(split_end)
        self.mean=mean
        self.std=std

        self.data = data[split_start: split_end]
        self.his_length = his_length
        self.pred_length = pred_length

    def __getitem__(self, index):
        x = self.data[index: index + self.his_length].permute(1, 0, 2)
        x=(x - self.mean) / self.std
        y = self.data[index + self.his_length: index + self.his_length + self.pred_length][:, :, 0].permute(1, 0)
        return torch.Tensor(x), torch.Tensor(y)
    def __len__(self):
        return self.data.shape[0] - self.his_length - self.pred_length + 1


def generate_dataset(data, args):

    batch_size = args.batch_size
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    his_length = args.his_length
    pred_length = args.pred_length

    train_mean, train_std = std_mean(data[0: int(data.shape[0] * train_ratio),...,-1])
    train_dataset = MyDataset(data, 0, data.shape[0] * train_ratio, his_length, pred_length,train_mean,train_std)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #-------------------------normal version--------------------------#

    val_mean, val_std = std_mean(data[int(data.shape[0]*train_ratio): int(data.shape[0]*(train_ratio+valid_ratio)),...,-1])
    valid_dataset = MyDataset(data, data.shape[0]*train_ratio, data.shape[0]*(train_ratio+valid_ratio), his_length, pred_length,val_mean, val_std)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


    test_mean, test_std = std_mean(data[int(data.shape[0] * (train_ratio + valid_ratio)):data.shape[0],...,-1])
    test_dataset = MyDataset(data, data.shape[0]*(train_ratio+valid_ratio), data.shape[0], his_length, pred_length,test_mean, test_std)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader, train_mean, train_std, val_mean, val_std, test_mean, test_std
