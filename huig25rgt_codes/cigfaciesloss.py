import os
import math
import random
import numpy as np
import random
import torch
from collections import deque
import torch.nn as nn
import torch.nn.functional as F

def compute_CIGLoss(input, indexs, ciglabel_dir=None, seeds_num = 500,loss_type="L1"):
    if ciglabel_dir == None:
        print("ciglabel dir is not difined")
    bs, c, h, w = input.size()
    sum_loss = 0
    for bi in range(bs):
        input_bs = input[bi,0,:,:]
#         input_bs = ((input_bs-torch.mean(input_bs))/torch.max(input_bs))*10 # min and max norm
        # load all seeds data and corresponding local path label -- dir:{"0",[num,2], "1":[mum,2], ...} 
        paths_all = np.load(ciglabel_dir+str(indexs[bi].tolist())+".npy",allow_pickle=True).item() 
        pn = len(paths_all)
        
#         # only random pick seeds_num paths
#         random_path = np.linspace(0,pn-1,pn).astype(np.int32)
#         np.random.shuffle(random_path)
#         # randomly select seeds_num(500) paths
#         for pi in range(seeds_num):
#             paths_pixels = torch.tensor(paths_all[str(random_path[pi])],dtype=torch.int32,device=input.device) # one path [num,2] (x,y)
            
        # all the seeds' paths
        for pi in range(pn):
            paths_pixels = torch.tensor(paths_all[str(pi)],dtype=torch.int32,device=input.device) # one path [num,2] (x,y)
            
            Path_pixel_num = len(paths_pixels)
            local_input = input_bs[paths_pixels[:,0],paths_pixels[:,1]] # pred rgt in PATH seeds
            path_mean_val = torch.mean(local_input) # mean value in PATH
            local_label = torch.ones_like(local_input) * path_mean_val # label rgt in PATH seeds
#             lora weight, voild becoming fat in special location during self-supervised learning

            # compute the loss
            if loss_type == "L1":
                local_sub_pred = torch.abs(local_label - local_input)
                local_loss = torch.sum(local_sub_pred)/Path_pixel_num
            elif loss_type == "L2":
                local_sub_pred = torch.abs(local_label - local_input)
                local_sub_pred = torch.pow(local_sub_pred,2)
                local_loss = torch.sum(local_sub_pred)/Path_pixel_num
            else:
                print("Wrong! Loss type is not defined")
            sum_loss = sum_loss + local_loss
    return sum_loss/bs
            
# Hui Gao   
class CIGLoss(nn.Module): 
    """
    This self-supervised loss function is defined by Hui Gao (2024.10.23)
    Parameters:
    width: the half-width of local-data(centered by seed) when calculating loss
    loss_type: the loss of predicted local-rgt(centered by seed) [consistent with cigfacies data]
    """
    def __init__(self, seeds_num = 500, loss_type="L1", ciglabel_dir = None):
        super(CIGLoss, self).__init__()
        self.name = 'CIGLoss'
        self.loss_type = loss_type
        self.ciglabel_dir = ciglabel_dir
        self.seeds_num = seeds_num
        
    def forward(self, input, indexs):
        """
        1. Calculating the seeds based cigfacies data
        2. Pick the local Path for each seed in local windows
        3. Calculating the loss for the predicted RGT whether consistent in the localPath
        Inputs:
        input: Predicted RGT data [batch_size,channel,h,w]
        seeds: Selected seeds [batch_size,channel,num,2]
        seeds_paths: Corresponding local path [batch_size,channel,num,2*width+1,2*width+1]
        """
        loss = compute_CIGLoss(input, indexs, ciglabel_dir= self.ciglabel_dir,
                               seeds_num =self.seeds_num, loss_type=self.loss_type)

        return loss
    
    
def calculate_normal_loss(input, normal, eps=1e-20):
    """
    normal loss
    eps_tensor: avoid division by zero
    """
    bs, _, h, w = input.size()
#     [g1,g2] = torch.gradient(input,dim=(2,3))
#     g1 = g1 * 1e16 
#     g2 = g2 * 1e16
#     eps_tensor = torch.ones_like(g1)*eps
#     gs = 1/torch.max(((g1**2+g2**2)**0.5), eps_tensor) 
#     g1 = g1*gs
#     g2 = g2*gs
#     loss = 1 - (normal[:,:,:,:,0]*g1 + normal[:,:,:,:,1]*g2)
# #     loss = loss * torch.pow(linearity,4)
#     loss = torch.sum(loss)/bs/h/w

    # 1-cos_similarity loss
    gx = torch.zeros_like(normal) 
    [gx[:,:,:,:,0],gx[:,:,:,:,1]] = torch.gradient(input,dim=(2,3))
    cosloss = nn.CosineSimilarity(dim=4, eps=eps)
    loss = 1 - cosloss(gx,normal)
    return torch.sum(loss)/bs/h/w

class NormalLoss(nn.Module): 
    """
    This self-supervised loss function is defined by Hui Gao (2024.10.23)
    1. Using the predicted RGT label to calculate gradient
    2. Multiply the normal vector with gradient
    3. Minimum the dot value (loss)
    """
    def __init__(self):
        super(NormalLoss, self).__init__()
        
    def forward(self, input, normal):
        """
        input: (batch_size,channel,x,y)
        shear: (batch_size,channel,x,y,2) two vectors in x&y
        """
        loss = calculate_normal_loss(input, normal)
        return loss
    
    