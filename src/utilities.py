import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import torch.optim as optim
from clippedAdam import Adam
import matplotlib.pyplot as plt

device = 'cpu'
    
def GSP(target, bid, n_ad, n_org):
    bs, n_bidder, n_item = target.shape
    allocation_matrix = np.zeros(target.shape)
    payment_matrix = np.zeros(target.shape)
    for i in range(bs):
        for k in range(n_item):
            # 取出第k列
            column = target[i, :, k]
            
            # 找到第k大的元素及其索引
            sorted_indices = np.argsort(column.numpy())[::-1]
            kth_largest_idx = sorted_indices[k]
            kth_plus_one_largest_idx = sorted_indices[k+1] if k+1 < len(column) else kth_largest_idx
            
            # 将第k列的第k+1大的数据赋给第k大的数，其他位置设置为0
            allocation_matrix[i, kth_largest_idx, k] = 1
            payment_matrix[i, kth_largest_idx, k] = bid[i, kth_plus_one_largest_idx, k]
    payment_matrix[:,n_ad:,:]=0
        
    
    return allocation_matrix, payment_matrix

def cal_revenue(pay):
    """ Given payment (pay), computes revenue
            Input params:
                pay: [num_batches, num_agents]
            Output params:
                revenue: scalar
        """
    return np.mean(np.sum(np.sum(pay, axis=-1), axis=-1))

def cal_volume(alloc, volume):
    winning_volume = torch.mean(torch.sum(torch.sum(np.multiply(alloc, volume), dim=-1), dim=-1))
    return winning_volume

def mechanism(batch_indices, data, alpha, n_ad, n_org):
    # print(data.shape)
    bid = torch.tensor(data[batch_indices][:,:,:,0]).to(device)
    volume = torch.tensor(data[batch_indices][:,:,:,1]).to(device)

    target = bid + alpha * volume

    
    alloc, payment = GSP(target, bid, n_ad, n_org)
    
    # # print("target: ", target[0:5])
    print("alloc: ", alloc[0:5])
    # print("volume: ", volume[0:5])
    # print("bid: ", bid[0:5])
    # print("payment: ", payment[0:5])
    rvn = cal_revenue(payment)
    # print("rvn: ", rvn)
    winning_volume = cal_volume(alloc, volume)
    # print("winning_volume: ", winning_volume)

    return alloc, rvn, winning_volume