import os,time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from vispy.color import get_colormap, Colormap, Color
from matplotlib.colors import ListedColormap

from ssim import SSIMLoss as ssim
from ssim import MultiScaleSSIMLoss as mssim
from cigfaciesloss import CIGLoss, NormalLoss
import loralib as lora

def sort_list_IDs(list_IDs):
    list_nums = [int(i.split(".")[0]) for i in list_IDs]
    list_sort = sorted(enumerate(list_nums), key=lambda x:x[1])
    list_index = [i[0] for i in list_sort]
    list_IDs_new = [list_IDs[i] for i in list_index]
    return list_IDs_new

def threshold_predictions(thresholded_preds, thr=0.01):
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def write_cube(data, path):
    data = np.transpose(data,[1,2,0]).astype(np.single)
    data.tofile(path)

def read_cube(input_data_path, size):
    input_cube = np.fromfile(input_data_path, dtype=np.single)
    input_cube = np.reshape(input_cube, size)
    input_cube = input_cube.transpose((2,0,1))
    return input_cube

# 读取目录
def read_path_list(path):
    file_list = os.listdir(path)
    file_name_list = [i.split(".")[0] for i in file_list]
    file_name_list = sorted(enumerate(file_name_list), key=lambda x:x[1]) 
    file_list = [file_list[i] for i in [j[0] for j in file_name_list]]
    return file_list

# 归一化
def min_max_norm(x):
    if torch.is_tensor(x) and torch.max(x) != torch.min(x):
            x = x - torch.min(x)
            x = x / torch.max(x)        
    elif np.max(x) != np.min(x):
            x = x - np.min(x)
            x = x / np.max(x)
    return x
    
# 标准化
def mea_std_norm(x):
    if torch.is_tensor(x) and torch.std(x) != 0:
            x = (x - torch.mean(x)) / torch.std(x)
    elif np.std(x) != 0:
            x = (x - np.mean(x)) / np.std(x)
    return x

class build_dataset_cigfacies(Dataset):
    def __init__(self, samples_list, dataset_path, mode, 
                 input_attr_list=["seis"],input_attr_list2=["cigfacies"],
                 input_attr_list3=["normal"],input_attr_list4=["linearity"],
                 index_attr_list = ["index"],
                 output_attr_list=["rgt"],output_attr_list2=["unconformities"],norm=None):
        self.samples_list = samples_list
        self.dataset_path = dataset_path
        self.input_attr_list = input_attr_list
        self.input_attr_list2 = input_attr_list2
        self.input_attr_list3 = input_attr_list3
        self.input_attr_list4 = input_attr_list4
        self.output_attr_list = output_attr_list
        self.output_attr_list2 = output_attr_list2
        self.index_attr_list = index_attr_list
        self.mode = mode
        
    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_file_name = self.samples_list[idx]
        sample_file_path = os.path.join(self.dataset_path, sample_file_name + ".npy")
        sample_dict = np.load(sample_file_path, allow_pickle=True).item()
        
        sample_output = {}
        
        if self.mode in ['Train', 'Valid']:
            
            # outputs
            for i, output_attr in enumerate(self.output_attr_list):
                tmp = sample_dict[output_attr].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                tmp = (min_max_norm(tmp) * 10)
                sample_output[output_attr] = tmp # rgt
                
            for i, output_attr2 in enumerate(self.output_attr_list2):
                tmp = sample_dict[output_attr2].astype(np.single)
                tmp = tmp[np.newaxis,:,:]
                sample_output[output_attr2] = tmp # unconformities
                
            ##### inputs 
            for i, input_attr in enumerate(self.input_attr_list):
                tmp = np.array(sample_dict[input_attr]).astype(np.single)
                tmp = mea_std_norm(tmp)
                sample_output[input_attr] = tmp # seismic data
                
            for i, input_attr2 in enumerate(self.input_attr_list2):
                tmp = np.array(sample_dict[input_attr2]).astype(np.single)
                tmp = (tmp * 2) - 1 # range -1,1
                sample_output[input_attr2] = tmp # cigfacies data
                
            for i, input_attr3 in enumerate(self.input_attr_list3):
                tmp = np.array(sample_dict[input_attr3]).astype(np.single)
                sample_output[input_attr3] = tmp # normal vector label
                
            for i, input_attr4 in enumerate(self.input_attr_list4):
                tmp = np.array(sample_dict[input_attr4]).astype(np.single)
                sample_output[input_attr4] = tmp # linearity data
            
            for i, index_attr in enumerate(self.index_attr_list):
                tmp = np.array(int(sample_file_name)).astype(np.int64)
                sample_output[index_attr] = tmp # index data
            
        sample_output["sample_file_path"] = sample_file_path
        return  sample_output

# SSIM Loss 
class SSIMLoss(nn.Module):
    def __init__(self, channel, filter_size):
        super(SSIMLoss, self).__init__()
        self.mssim = mssim(channel=channel, filter_size=filter_size)
    def forward(self, output, target):
        loss = (1 - self.mssim(output, target))
        return loss
    
# 曲线光滑函数
def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed    

####################################################################################
# Model training
####################################################################################

def train_valid_GLP_cigfacies(param, model, train_data, valid_data, ciglabel_dir = None,
                              Loss="L2",CIGLoss_type="L1",Training_stage="step-1",
                              input_attrs=["data"],input_attrs2=["cigfacies"],
                              input_attrs3=["normal"],input_attrs4=["linearity"],
                              output_attrs=["label"],output_attrs2=["label"],plot=True):
    
    # parameters
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    
    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)

    # loss
    if Loss == "L1":
        criterion1_1 = nn.L1Loss().to(device)
    elif Loss == "L2":
        criterion1_1 = nn.MSELoss().to(device)
    else:
        print("criterion1_1 is None")
        
    criterion1_2 = SSIMLoss(channel=1,filter_size=5).to(device)
#     criterion2 =  CIGLoss(seed_num=100,seed_dist=10,width=20,loss_type="L1").to(device)
    if CIGLoss_type =="L1":
        criterion2_1 =  CIGLoss(loss_type="L1",ciglabel_dir=ciglabel_dir).to(device)
    elif CIGLoss_type =="L2":
        criterion2_1 =  CIGLoss(loss_type="L2",ciglabel_dir=ciglabel_dir).to(device)
    else:
        print("CIGLoss is None")
    criterion2_2 = NormalLoss().to(device)
    
    # Main cycle
    epoch_loss_train, epoch_loss_valid, epoch_lr = [], [], []
    epoch_loss1_train, epoch_loss2_train, epoch_loss3_train, epoch_loss4_train = [],[],[],[]
    
    best_mse = 1e50
    
    for epoch in range(epochs):
        # Training stage
        model.train()
        loss_train_per_epoch = 0
        loss1_train_per_epoch = 0
        loss2_train_per_epoch = 0
        loss3_train_per_epoch = 0
        loss4_train_per_epoch = 0
        
        for batch_idx, batch_samples in enumerate(train_loader):
            
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
            
            data2 = batch_samples[input_attrs2[0]].to(torch.float32).unsqueeze(1)
            data2 = Variable(data2.to(device)) # cigfacies data [bs,channel,X,Y]
            
            data3 = batch_samples[input_attrs3[0]].to(torch.float32).unsqueeze(1)
            data3 = Variable(data3.to(device)) # normal vector label [bs,channel,X,Y,2] (u1,u2)
            
            target = batch_samples[output_attrs[0]].to(torch.float32)
            target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            
            target2 = batch_samples[output_attrs2[0]].to(torch.float32)
            target2 = Variable(target2.to(device)) # unconfirmities label [bs,channel,X,Y]
            
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]

            optimizer.zero_grad()
            
            ### training
#             data  = torch.cat((data,data2), dim=1)
            data  = torch.cat((data,data), dim=1)
            target_i = model(data)
            
            # use synthetic data to train the model (only use labeled supervison)
            if Training_stage == "step-1":
                # Training strategy-1
                loss1_1 = 0.2*criterion1_1(target_i, target)
                loss1_2 = 0.8*criterion1_2(target_i, target)
                loss = loss1_1 + loss1_2
                loss1_train_per_epoch += loss1_1.item()
                loss2_train_per_epoch += loss1_2.item()
                
            # introducing geologically-informed unsupervised losses(CIGLoss and NormLoss) to train the pre-trained model 
            elif Training_stage == "step-2":
                # Training strategy-2
                loss1_1 = 0.2*criterion1_1(target_i, target)
                loss1_2 = 0.8*criterion1_2(target_i, target)
                if CIGLoss_type =="L1":
                    loss2_1 = 0.1*criterion2_1(target_i, indexs)
                elif CIGLoss_type =="L2":
                    loss2_1 = criterion2_1(target_i, indexs)
                else:
                    print("CIGLoss is not defined")
                loss2_2 = 10*criterion2_2(target_i, data3)
                loss = loss1_1 + loss1_2 + loss2_1 + loss2_2
                loss1_train_per_epoch += loss1_1.item()
                loss2_train_per_epoch += loss1_2.item()
                loss3_train_per_epoch += loss2_1.item()
                loss4_train_per_epoch += loss2_2.item() 
            else:
                print("stage is not defined")
            
            loss.backward()
            optimizer.step() 
            loss_train_per_epoch += loss.item()
        
        # Validation stage
        model.eval()
        loss_valid_per_epoch = 0
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
                data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]

                data2 = batch_samples[input_attrs2[0]].to(torch.float32).unsqueeze(1)
                data2 = Variable(data2.to(device)) # cigfacies data [bs,channel,X,Y]

                data3 = batch_samples[input_attrs3[0]].to(torch.float32).unsqueeze(1)
                data3 = Variable(data3.to(device)) # normal vector label [bs,channel,X,Y,2] (u1,u2)

                target = batch_samples[output_attrs[0]].to(torch.float32)
                target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]

                target2 = batch_samples[output_attrs2[0]].to(torch.float32)
                target2 = Variable(target2.to(device)) # unconfirmities label [bs,channel,X,Y]

                indexs = batch_samples["index"].to(torch.int32)
                indexs = Variable(indexs.to(device)) # index [bs]
                
                ### Validation
#                 data  = torch.cat((data,data2), dim=1)
                data  = torch.cat((data,data), dim=1)
                target_i = model(data)
                
                if Training_stage == "step-1":
                    # Training strategy-1
                    loss1_1 = 0.2*criterion1_1(target_i, target)
                    loss1_2 = 0.8*criterion1_2(target_i, target)
                    loss = loss1_1 + loss1_2
                elif Training_stage == "step-2":
                    # Training strategy-2
                    loss1_1 = 0.2*criterion1_1(target_i, target)
                    loss1_2 = 0.8*criterion1_2(target_i, target)
                    if CIGLoss_type =="L1":
                        loss2_1 = 0.1*criterion2_1(target_i, indexs)
                    elif CIGLoss_type =="L2":
                        loss2_1 = criterion2_1(target_i, indexs)
                    else:
                        print("CIGLoss is not defined")
                    loss2_2 = 10*criterion2_2(target_i, data3)
                    loss = loss1_1 + loss1_2 + loss2_1 + loss2_2 
                else:
                    print("stage is not defined")
                
                loss_valid_per_epoch += loss.item()
                
        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_valid_per_epoch = loss_valid_per_epoch / len(valid_loader)

        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_valid.append(loss_valid_per_epoch)
        
        epoch_lr.append(optimizer.param_groups[0]['lr'])        
        loss1_train_per_epoch = loss1_train_per_epoch / len(train_loader)
        loss2_train_per_epoch = loss2_train_per_epoch / len(train_loader)
        loss3_train_per_epoch = loss3_train_per_epoch / len(train_loader)
        loss4_train_per_epoch = loss4_train_per_epoch / len(train_loader)
        epoch_loss1_train.append(loss1_train_per_epoch)
        epoch_loss2_train.append(loss2_train_per_epoch)
        epoch_loss3_train.append(loss3_train_per_epoch)
        epoch_loss4_train.append(loss4_train_per_epoch)
        
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        # 保存最优模型(step-1 and step-2)
        if loss_valid_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_valid_per_epoch

        scheduler.step(loss_train_per_epoch)
        
        # show loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, Training Loss:{:.6f} <{:.4f}+{:.4f}+{:.4f}+{:.4f}> Validation Loss:{:.6f} Learning rate: {:.6f}'.format(epoch, loss_train_per_epoch, loss1_train_per_epoch, loss2_train_per_epoch, loss3_train_per_epoch, loss4_train_per_epoch, loss_valid_per_epoch, epoch_lr[epoch]))
            
    # Training loss curves
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.plot(x, smooth(epoch_loss_valid, 0.6), label='Validation loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
        
    if Training_stage == "step-1":        
        logs = {"epoch_loss_train":epoch_loss_train,
                "epoch_loss_valid":epoch_loss_valid,
                "epoch_lr":epoch_lr,
                "epoch_loss1_train":epoch_loss1_train,
                "epoch_loss2_train":epoch_loss2_train}
    elif Training_stage == "step-2":
        logs = {"epoch_loss_train":epoch_loss_train,
                "epoch_loss_valid":epoch_loss_valid,
                "epoch_lr":epoch_lr,
                "epoch_loss1_train":epoch_loss1_train,
                "epoch_loss2_train":epoch_loss2_train,
                "epoch_loss3_train":epoch_loss3_train,
                "epoch_loss4_train":epoch_loss4_train}
    else:
        print("stage is not defined")
        
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    print("log has saved")
    return model

# training stage-3 : introducing the field datasets for LoRA fine-tuning
def train_GLP_cigfacies_lorawithsyn(param, model, train_data,syn_data, 
                                    ciglabel_dir = None, synciglabel_dir = None,ratio=[0.1,10,20],
                                    input_attrs=["data"],input_attrs2=["cigfacies"],
                                    input_attrs3=["normal"],input_attrs4=["linearity"],
                                    output_attrs=["label"],output_attrs2=["label"],plot=True):
    
    # parameters
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    lr_patience = param['lr_patience']
    lr_factor = param['lr_factor']
    optimizer_type = param['optimizer_type']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_path = param['checkpoint_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    syn_train_loader = DataLoader(dataset=syn_data, batch_size=batch_size, shuffle=True, drop_last=True)
    
    if optimizer_type == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=lr_patience, factor=lr_factor)

    criterion1_1 = nn.MSELoss().to(device)
    criterion1_2 = SSIMLoss(channel=1,filter_size=5).to(device)
    criterion2_1 =  CIGLoss(loss_type="L1",ciglabel_dir=ciglabel_dir).to(device)
    criterion2_1_syn =  CIGLoss(loss_type="L1",ciglabel_dir=synciglabel_dir).to(device)
    criterion2_2 = NormalLoss().to(device)
    
    # Main cycle
    epoch_loss_train, epoch_lr = [], []
    epoch_loss1_train, epoch_loss2_train, epoch_loss3_train, epoch_loss4_train = [],[],[],[]
    epoch_loss_syn, epoch_loss_all = [],[]
    
    best_mse = 1e50
    
    for epoch in range(epochs):
        
        # Training stage
        model.train()
        loss_train_per_epoch = 0
        loss_syn_per_epoch = 0
        loss1_train_per_epoch = 0
        loss2_train_per_epoch = 0
        loss3_train_per_epoch = 0
        loss4_train_per_epoch = 0
        
        for batch_idx, batch_samples in enumerate(train_loader):
            
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
            
            data2 = batch_samples[input_attrs2[0]].to(torch.float32).unsqueeze(1)
            data2 = Variable(data2.to(device)) # cigfacies data [bs,channel,X,Y]
            
            data3 = batch_samples[input_attrs3[0]].to(torch.float32).unsqueeze(1)
            data3 = Variable(data3.to(device)) # normal vector label [bs,channel,X,Y,2] (u1,u2)
            
#             data4 = batch_samples[input_attrs4[0]].to(torch.float32).unsqueeze(1)
#             data4 = Variable(data4.to(device)) # linearity [bs,channel,X,Y] 
            
#             target = batch_samples[output_attrs[0]].to(torch.float32)
#             target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            
#             target2 = batch_samples[output_attrs2[0]].to(torch.float32)
#             target2 = Variable(target2.to(device)) # unconfirmities label [bs,channel,X,Y]
            
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]
            
            optimizer.zero_grad()
            ### training
            data  = torch.cat((data,data2), dim=1)
            target_i = model(data)
            # min and max norm
            target_i = ( (target_i-torch.min(target_i))/torch.max(target_i) ) *10
                
            # use field data for lora fine-tuning
            # Training strategy-3 
            loss2_1 = ratio[0]*criterion2_1(target_i, indexs)
            loss2_2 = ratio[1]*criterion2_2(target_i, data3)
            loss = loss2_1 + loss2_2 
            loss3_train_per_epoch += loss2_1.item()
            loss4_train_per_epoch += loss2_2.item() 
            
            loss.backward()
            optimizer.step()
            loss_train_per_epoch += loss.item()
        
        # synthetic data
        model.train()
        
        for batch_idx, batch_samples in enumerate(syn_train_loader):
            
            data = batch_samples[input_attrs[0]].to(torch.float32).unsqueeze(1)
            data = Variable(data.to(device)) # seismic data [bs,channel,X,Y]
            
            data2 = batch_samples[input_attrs2[0]].to(torch.float32).unsqueeze(1)
            data2 = Variable(data2.to(device)) # cigfacies data [bs,channel,X,Y]
            
            data3 = batch_samples[input_attrs3[0]].to(torch.float32).unsqueeze(1)
            data3 = Variable(data3.to(device)) # normal vector label [bs,channel,X,Y,2] (u1,u2)
            
#             data4 = batch_samples[input_attrs4[0]].to(torch.float32).unsqueeze(1)
#             data4 = Variable(data4.to(device)) # linearity [bs,channel,X,Y] 
            
            target = batch_samples["rgt"].to(torch.float32)
            target = Variable(target.to(device)) # rgt label [bs,channel,X,Y]
            
            indexs = batch_samples["index"].to(torch.int32)
            indexs = Variable(indexs.to(device)) # index [bs]
            
            optimizer.zero_grad()
            ### training
            data  = torch.cat((data,data2), dim=1)
            target_i = model(data)
            # min and max norm
            target_i = ( (target_i-torch.min(target_i))/torch.max(target_i) ) *10
                
            # use field data for lora fine-tuning
            # Training strategy-3
            loss1_1 = 0.2*criterion1_1(target_i, target)
            loss1_2 = 0.8*criterion1_2(target_i, target)
            loss2_1 = ratio[0]*criterion2_1_syn(target_i, indexs)
            loss2_2 = ratio[1]*criterion2_2(target_i, data3)
            loss2 = ratio[2]*(loss1_1 + loss1_2) + loss2_1 + loss2_2
#             loss2 = ratio[2]*(loss1_1 + loss1_2)
            
            loss2.backward() 
            optimizer.step()
            loss_syn_per_epoch += loss2.item()
        
        
#         loss_all_per_epoch = (loss_train_per_epoch + loss_syn_per_epoch) / (len(train_loader)+len(syn_train_loader))
        
        loss_train_per_epoch = loss_train_per_epoch / len(train_loader)
        loss_syn_per_epoch = loss_syn_per_epoch / len(syn_train_loader)
        loss_all_per_epoch = (loss_train_per_epoch + loss_syn_per_epoch)/2
        
        epoch_loss_train.append(loss_train_per_epoch)
        epoch_loss_syn.append(loss_syn_per_epoch)
        epoch_loss_all.append(loss_all_per_epoch)
        
        epoch_lr.append(optimizer.param_groups[0]['lr'])        
        loss1_train_per_epoch = loss1_train_per_epoch / len(train_loader)
        loss2_train_per_epoch = loss2_train_per_epoch / len(train_loader)
        loss3_train_per_epoch = loss3_train_per_epoch / len(train_loader)
        loss4_train_per_epoch = loss4_train_per_epoch / len(train_loader)
        epoch_loss1_train.append(loss1_train_per_epoch)
        epoch_loss2_train.append(loss2_train_per_epoch)
        epoch_loss3_train.append(loss3_train_per_epoch)
        epoch_loss4_train.append(loss4_train_per_epoch)
        
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': lora.lora_state_dict(model), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)

        # 保存最优模型(step-1 and step-2)
        if loss_train_per_epoch < best_mse:
            state = {'epoch': epoch, 'state_dict': lora.lora_state_dict(model), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_path, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_mse = loss_train_per_epoch

#         scheduler.step(loss_train_per_epoch)
        scheduler.step(loss_all_per_epoch)
        
        # show loss
        if epoch % disp_inter == 0: 
            print('Epoch:{}, Training Loss:{:.5f} <{:.4f}+{:.4f}>, Synthetic Loss:{:.5f}, Learning rate: {:.6f} --> <{:.5f}>'.format(epoch, loss_train_per_epoch,loss3_train_per_epoch,loss4_train_per_epoch,loss_syn_per_epoch, epoch_lr[epoch],loss_all_per_epoch))
            
    # Training loss curves
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(epoch_loss_train, 0.6), label='Training loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.set_title(f'Training curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title(f'Learning rate curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
        
    logs = {"epoch_loss_train":epoch_loss_train,
            "epoch_lr":epoch_lr,
            "epoch_loss3_train":epoch_loss3_train,
            "epoch_loss4_train":epoch_loss4_train,
            "epoch_syn_train":epoch_loss_syn,
            "epoch_all_train":epoch_loss_all,}  
    np.save(os.path.join(checkpoint_path, 'logs.npy'), logs)
    print("log has saved")
    return model
    
####################
#functions

def getStrataColors(alpha=1,FillExceptMin=False):
    """
    alpha: transparency of the color
    FillExceptMin : whether to fill the color at the minimum value
    """
    rgba = np.full((256,4),0).astype(np.float32)
    for i in range(256):
        if (i<8):
            rgba[i] = [1.0,0.0,0.0,alpha]
        elif (i<16):
            rgba[i] = [1.0,0.5019608,0.0,alpha]
        elif (i<24):
            rgba[i] = [1.0,1.0,0.0,alpha]
        elif (i<32):
            rgba[i] = [0.0,1,0.0,alpha]
        elif (i<40):
            rgba[i] = [0.0,0.5019608,0.0,alpha]
        elif (i<48):
            rgba[i] = [0.0,0.2509804,0.0,alpha]
        elif (i<56):
            rgba[i] = [0,1.0,1.0,alpha]
        elif (i<64):
            rgba[i] = [0.0,0.5019608,1.0,alpha]
        elif (i<72):
            rgba[i] = [0.0,0.0,1.0,alpha]
        elif (i<80):
            rgba[i] = [0.0,0.0,0.627451,alpha]
        elif (i<88):
            rgba[i] = [0.0,0.5019608,0.7529412,alpha]
        elif (i<96):
            rgba[i] = [1.0,0.5019608,0.5019608,alpha]
        elif (i<104):
            rgba[i] = [0.5019608,0.5019608,1.0,alpha]
        elif (i<112):
            rgba[i] = [0.5019608,0.0,1.0,alpha]
        elif (i<120):
            rgba[i] = [0.5019608,0,0.5019608,alpha]
        elif (i<128):
            rgba[i] = [1.0,0.5019608,1.0,alpha]
        elif (i<136):
            rgba[i] = [1.0,0.0,1.0,alpha]
        elif (i<144):
            rgba[i] = [0.5019608,0.2509804,0,alpha]
        elif (i<152):
            rgba[i] = [0.5019608,0.5019608,0.5019608,alpha]
        elif (i<160):
            rgba[i] = [0.7529412,0.7529412,0.7529412,alpha]
        elif (i<168):
            rgba[i] = [0.2509804,0,0.2509804,alpha]
        elif (i<176):
            rgba[i] = [0.90588236,0.7294118,0.19607843,alpha]
        elif (i<184):
            rgba[i] = [0.44313726,0.58431375,0.58431375,alpha]
        elif (i<192):
            rgba[i] = [0.5254902,0.42352942,0.4862745,alpha]
        elif (i<200):
            rgba[i] = [0.7176471,0.54509807,0.44313726,alpha]
        elif (i<208):
            rgba[i] = [0.5019608,0.5019608,0,alpha]
        elif (i<216):
            rgba[i] = [0.7529412,0.7294118,0.8784314,alpha]
        elif (i<224):
            rgba[i] = [0.61960787,0.85882354,0.9882353,alpha]
        elif (i<232):
            rgba[i] = [0.7372549,0.25882354,0.24705882,alpha]
        elif (i<240):
            rgba[i] = [0.8862745,0.8509804,0.627451,alpha]
        elif (i<248):
            rgba[i] = [0.60784316,0.9411765,0.7490196,alpha]
        elif (i<256):
            rgba[i] = [0.62352943,0.79607844,0.105882354,alpha]
    if FillExceptMin is True:
        rgba[0, -1] = 0
    stratamap = ListedColormap(rgba)
    
    return stratamap


# Colormap
def setAlpha(cmap='jet',alpha=1, FillExceptMin=False):
    """
    cmap: colorbar
    alpha: transparency of the color
    FillExceptMin : whether to fill the color at the minimum value
    """
    original_cmap = get_colormap(cmap)
    samples = np.linspace(0, 1, 512) # 128 color samples
    rgba = np.array([original_cmap.map(x) for x in samples]).squeeze()
    rgba[:, -1] = alpha
    if FillExceptMin is True:
        rgba[0, -1] = 0
    cmap_new = ListedColormap(rgba)
    return cmap_new