import torch
import torch.nn as nn
import numpy as np
# from mmcv.runner import load_checkpoint
from mmengine.runner import load_checkpoint #huig new-version
from GLP.mit import mit_b4
from GLP.mit_lora import mit_b4_lora
from torch.autograd import Variable
import torch.nn.functional as F

class GLPDepth(nn.Module):
    def __init__(self, max_depth=10.0, is_train=False, use_lora=False):
        super().__init__()
        self.max_depth = max_depth
        
#         self.Conv_1x1 = nn.Conv2d(2,3,kernel_size=1,stride=1,padding=0) # huig
        if use_lora:
            self.encoder = mit_b4_lora()
        else:
            self.encoder = mit_b4()
            
        if is_train:            
            ckpt_path = './GLP/weights/mit_b4.pth'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './code/models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        channels_in = [512, 320, 128]
        channels_out = 64
            
        self.decoder = Decoder(channels_in, channels_out)
    
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
#         x = torch.cat((x,x,x), dim=1) # huig 假三通道 因为用到了预训练模型
#         x = self.Conv_1x1(x) # 2通道数据 --> 3通道 huig
        conv1, conv2, conv3, conv4 = self.encoder(x)
        out,out1,out2,out3,out4 = self.decoder(conv1, conv2, conv3, conv4)
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth

#         return {'pred_d': out_depth}
        return out_depth

class GLPDepth_merge(nn.Module):
    def __init__(self, max_depth=10.0, is_train=False, use_lora=False ):
        super().__init__()
        self.max_depth = max_depth
        
        self.Conv_1x1 = nn.Conv2d(2,2,kernel_size=1,stride=1,padding=0) # huig
        
        if use_lora:
            self.encoder = mit_b4_lora()
        else:
            self.encoder = mit_b4()
            
        if is_train:            
            ckpt_path = './GLP/weights/mit_b4.pth'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './code/models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        channels_in = [512, 320, 128]
        channels_out = 64
            
        self.decoder = Decoder(channels_in, channels_out)
    
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
#         x = torch.cat((x,x,x), dim=1) # huig 假三通道 因为用到了预训练模型
        x = self.Conv_1x1(x) # seis与cigfacies merge huig
        conv1, conv2, conv3, conv4 = self.encoder(x)
        out,out1,out2,out3,out4 = self.decoder(conv1, conv2, conv3, conv4)
        out_depth = self.last_layer_depth(out)
        out_depth = torch.sigmoid(out_depth) * self.max_depth
        return out_depth

class rgt2fl(nn.Module):
    def __init__(self,in_fl_c):
        super().__init__()
        self.in_fl_c = in_fl_c
        self.conv1 = nn.Conv2d(in_fl_c, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        
        output = self.conv3(x2)
        
        return output

class GLPDepth_add_gradient(nn.Module):
    def __init__(self, max_depth=10.0, is_train=False,relu=False,rgt2gr2rgt=False,lkfl = False):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = mit_b4()
        if is_train:            
            ckpt_path = './models_glp/weights/mit_b4.pth'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        channels_in = [512, 320, 128]
        channels_out = 64
        self.relu = relu
        self.lkfl = lkfl
        self.rgt2gr2rgt = rgt2gr2rgt    
        self.decoder = Decoder(channels_in, channels_out)
        self.ReLu_gradient = nn.ReLU(inplace=False)    
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x): 
#         x = torch.cat((x,x,x), dim=1) # 假三通道 因为用到了预训练模型
#         print(torch.max(x),torch.min(x))
        conv1, conv2, conv3, conv4 = self.encoder(x)
        out,out1,out2,out3,out4 = self.decoder(conv1, conv2, conv3, conv4)
        out_gradient = self.last_layer_depth(out)
        out_gradient = torch.sigmoid(out_gradient) * self.max_depth
        return out_gradient
#######################  #######################  #######################  #######################        
        # rgt2grad2relu2rgt
#         out_gradient = torch.sigmoid(out_gradient) * self.max_depth
#         out_gradient[:,:,1:] = out_gradient[:,:,1:] - out_gradient[:,:,:-1]
#         out_gradient_relu = self.ReLu_gradient(out_gradient)
#         for i in range(1,x.shape[2]):
#             out_gradient_relu[:,:,i] = out_gradient_relu[:,:,i]+out_gradient_relu[:,:,i-1]
#         return out_gradient_relu
        
#         out_gradient = torch.sigmoid(out_gradient) * self.max_depth
#         return out_gradient
        
#         if self.rgt2gr2rgt == True:
#             out_gradient = torch.sigmoid(out_gradient) * self.max_depth
#             out_gradient[:,:,1:] = out_gradient[:,:,1:] - out_gradient[:,:,:-1]
        
#         if self.relu == True:
#             out_uz = out_gradient.clone()
#             out_gradient_relu = self.ReLu_gradient(out_gradient)
#             out_uz_rl = out_gradient_relu.clone()
#         else:
#             out_gradient_relu = out_gradient

#         for i in range(1,x.shape[2]):
#             out_gradient_relu[:,:,i] = out_gradient_relu[:,:,i]+out_gradient_relu[:,:,i-1]

#         if self.lkfl:
#             return out_gradient_relu,conv1,conv2,conv3,conv4,out1,out2,out3,out4,out_gradient
#         return out_gradient_relu
#         if self.relu == True:
#             return out_gradient_relu,out_uz,out_uz_rl
#         else:
#             return out_gradient_relu    
    
    
    
class GLPDepth_add_meanhr(nn.Module):
    def __init__(self,batch_size=0,device=None,max_depth=10.0, is_train=False):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = mit_b4()
        if is_train:            
            ckpt_path = './models_glp/weights/mit_b4.pth'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        channels_in = [512, 320, 128]
        channels_out = 64

        self.decoder = Decoder_add_meanhr(channels_in, channels_out)
    
#         self.last_layer_depth = nn.Sequential(
#             nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
        self.last_layer_depth = Last_layer_depth_add_meanhr(channels_out, channels_out)
        
        self.ReLu_gradient = nn.ReLU(inplace=False)
    def forward(self, x,mask,mx_pred):                
        conv1, conv2, conv3, conv4 = self.encoder(x)
        out = self.decoder(conv1, conv2, conv3, conv4,mask,mx_pred)
        out_depth = self.last_layer_depth(out,mask,mx_pred)
        out_depth = torch.sigmoid(out_depth) * self.max_depth                

        return {'pred_d': out_depth}

class GLPDepth_add_meanhr_gradient(nn.Module):
    def __init__(self,batch_size=0,device=None,max_depth=10.0, is_train=False):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = mit_b4()
        if is_train:            
            ckpt_path = './models_glp/weights/mit_b4.pth'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        channels_in = [512, 320, 128]
        channels_out = 64

        self.decoder = Decoder_add_meanhr(channels_in, channels_out)
#         self.decoder = Decoder(channels_in, channels_out)
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
#         self.last_layer_depth = Last_layer_depth_add_meanhr(channels_out, channels_out)
        
        self.ReLu_gradient = nn.ReLU(inplace=False)
    def forward(self, x,mask,mx_len):                
        conv1, conv2, conv3, conv4 = self.encoder(x)
#         out,out1,out2,out3,out4 = self.decoder(conv1, conv2, conv3, conv4,mask,mx_len)
        out = self.decoder(conv1, conv2, conv3, conv4,mask,mx_len)

#         b,c,f,h,w = mask.size()
# #         print(c)
#         mask = mask.expand(b,64,f,h,w)
#         for i in range(f):
#             mxt = mx_len[:,0,4,i]
# #             print(f'mxtshape={mxt.shape}')
#             mxt = mxt.expand(b,64)
#             mask_rs = mask[:,:,i,::1,::1]
#             mask_o = mask_rs[:,:,:,:]*out
# #             print(f'mask_o={mask_o.shape}')
# #             print(f'mxt={mxt.shape}')
# #             print(f'torch.sum(mask_o,axis=(2,3))={torch.sum(mask_o,axis=(2,3)).shape}')
#             mask_mean = torch.sum(mask_o,axis=(2,3)).squeeze()/mxt
# #             print(f'mask_mean={mask_mean.shape}')
# #             print(f'mask_rs={mask_rs.shape}')
# #             print(f'b*mask.shape[1]={b*mask.shape[1]}')
# #             print(f'diag={(torch.diag(mask_mean.reshape(b*mask.shape[1]))@(mask_rs.reshape(b*mask.shape[1],h*w))).shape}')
# # #             mask_ti = mask_mean*mask_rs
#             mask_ti = (torch.diag(mask_mean.reshape(b*mask.shape[1]))@mask_rs.reshape(b*mask.shape[1],h*w)) \
#                    .reshape(mask_rs.shape)
#             out = out*(1-mask_rs)+mask_ti        

        out_depth = self.last_layer_depth(out)
        
#         b,c,f,h,w = mask.size()
# #         print(c)
# #         mask = mask.expand(b,64,f,h,w)
#         for i in range(f):
#             mxt = mx_len[:,0,4,i]
# #             print(f'mxtshape={mxt.shape}')
# #             mxt = mxt.expand(b,64)
#             mask_rs = mask[:,:,i,::1,::1]
#             mask_o = mask_rs[:,:,:,:]*out_depth
# #             print(f'mask_o={mask_o.shape}')
# #             print(f'mxt={mxt.shape}')
# #             print(f'torch.sum(mask_o,axis=(2,3))={torch.sum(mask_o,axis=(2,3)).shape}')
#             mask_mean = torch.sum(mask_o,axis=(2,3)).squeeze()/mxt
# #             print(f'mask_mean={mask_mean.shape}')
# #             print(f'mask_rs={mask_rs.shape}')
# #             print(f'b*mask.shape[1]={b*mask.shape[1]}')
# #             print(f'diag={(torch.diag(mask_mean.reshape(b*mask.shape[1]))@(mask_rs.reshape(b*mask.shape[1],h*w))).shape}')
# # #             mask_ti = mask_mean*mask_rs
#             mask_ti = (torch.diag(mask_mean.reshape(b*mask.shape[1]))@mask_rs.reshape(b*mask.shape[1],h*w)) \
#                    .reshape(mask_rs.shape)
#             out_depth = out_depth*(1-mask_rs)+mask_ti  

#             out_gradient[:,:,i] = out_gradient[:,:,i]+out_gradient[:,:,i-1] 

        out_gradient = torch.cumsum(out_depth,dim=2)     
              

        return out_gradient
    
    
class GLPDepth_add_meanhr(nn.Module):
    def __init__(self,batch_size=0,device=None,max_depth=10.0, is_train=False):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = mit_b4()
        if is_train:            
            ckpt_path = './models_glp/weights/mit_b4.pth'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        channels_in = [512, 320, 128]
        channels_out = 64

        self.decoder = Decoder_add_meanhr(channels_in, channels_out)
    
#         self.last_layer_depth = nn.Sequential(
#             nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))
        self.last_layer_depth = Last_layer_depth_add_meanhr(channels_out, channels_out)
        
        self.ReLu_gradient = nn.ReLU(inplace=False)
    def forward(self, x,mask,mx_pred):                
        conv1, conv2, conv3, conv4 = self.encoder(x)
        out = self.decoder(conv1, conv2, conv3, conv4,mask,mx_pred)
        out_depth = self.last_layer_depth(out,mask,mx_pred)
        out_depth = torch.sigmoid(out_depth) * self.max_depth
        
        
#         out_gradient = self.last_layer_depth(out)
#         out_gradient = self.ReLu_gradient(out_gradient)
# #         print(f'out_gradient.shape={out_gradient.shape}')
# #         ux_integradient = self.ux_integradient
# #         ux_integradient = torch.zeros(out_gradient.shape,dtype = torch.float32)
# #         ux_integradient[:,:,0] = out_gradient[:,:,0]
#         for i in range(1,128):
# #             ux_integradient[:,:,i] = torch.sum(out_gradient[:,:,1:i+1,:],axis=2)+out_gradient[:,:,0,:]
#             out_gradient[:,:,i] = out_gradient[:,:,i]+out_gradient[:,:,i-1]
            
            
# #         out_depth = torch.sigmoid(out_gradient) * self.max_depth
# #         print(f'out_depth.max() = {out_gradient.max()}')
# #         print(f'out_depth.min() = {out_gradient.min()}')

        return {'pred_d': out_depth}




class GLPDepth_add_gradient_peg(nn.Module):
    def __init__(self, max_depth=10.0, is_train=False,relu=True,rgt2gr2rgt=False,lkfl = False):
        super().__init__()
        self.max_depth = max_depth

        self.encoder = mit_b4_peg()
        if is_train:            
            ckpt_path = './models_glp/weights/mit_b4.pth'
            try:
                load_checkpoint(self.encoder, ckpt_path, logger=None)
            except:
                import gdown
                print("Download pre-trained encoder weights...")
                id = '1BUtU42moYrOFbsMCE-LTTkUE-mrWnfG2'
                url = 'https://drive.google.com/uc?id=' + id
                output = './models/weights/mit_b4.pth'
                gdown.download(url, output, quiet=False)

        channels_in = [512, 320, 128]
        channels_out = 64
        self.relu = relu
        self.lkfl = lkfl
        self.rgt2gr2rgt = rgt2gr2rgt    
        self.decoder = Decoder(channels_in, channels_out)
        self.ReLu_gradient = nn.ReLU(inplace=False)    
        self.last_layer_depth = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):                
        conv1, conv2, conv3, conv4 = self.encoder(x)
#         print(f'conv1shape{conv1.shape}')
#         print(f'conv2shape{conv2.shape}')
#         print(f'conv3shape{conv3.shape}')
#         print(f'conv4shape{conv4.shape}')
        out,out1,out2,out3,out4 = self.decoder(conv1, conv2, conv3, conv4)
        
        out_gradient = self.last_layer_depth(out)
        if self.rgt2gr2rgt == True:
            out_gradient = torch.sigmoid(out_gradient) * self.max_depth
            out_gradient[:,:,1:] = out_gradient[:,:,1:] - out_gradient[:,:,:-1]
        
        if self.relu == True:
            out_uz = out_gradient.clone()
            out_gradient_relu = self.ReLu_gradient(out_gradient)
            out_uz_rl = out_gradient_relu.clone()
        else:
            out_gradient_relu = out_gradient

        for i in range(1,x.shape[2]):
            out_gradient_relu[:,:,i] = out_gradient_relu[:,:,i]+out_gradient_relu[:,:,i-1]
        if self.lkfl:

            return out_gradient_relu,conv1,conv2,conv3,conv4,out1,out2,out3,out4,out_gradient
        if self.relu == True:
            return out_gradient_relu,out_uz,out_uz_rl
        else:
            return out_gradient_relu


class Last_layer_depth_add_meanhr(nn.Module):
    
    def __init__(self, in_channels, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1)
        self.Relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1)
    def forward(self,out,mask,mx_len):
        b,c,f,h,w = mask.size()
        out = self.conv1(out)
        for i in range(f):
            mask_rs = mask[:,:,i,::1,::1]

            mask_o = mask_rs[:,:,:,:]*out

            mask_mean = ((torch.sum(mask_o,axis = (2,3)).transpose(1,0))/mx_len[:,0,4,i])

            mask_ti = mask_mean*(mask_rs.permute(2,3,1,0))
            out = out*(1-mask_rs)+mask_ti.permute(3,2,0,1)
            
        out = self.Relu(out)
        out = self.conv2(out)
        
        return out

class Decoder_add_meanhr(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

#     def forward(self, x_1, x_2, x_3, x_4):
#         x_4_ = self.bot_conv(x_4)
#         out = self.up(x_4_)
        
#         x_3_ = self.skip_conv1(x_3)
#         out = self.fusion1(x_3_, out)
#         out = self.up(out)

#         x_2_ = self.skip_conv2(x_2)
#         out = self.fusion2(x_2_, out)
#         out = self.up(out)

#         out = self.fusion3(x_1, out)
#         out = self.up(out)
#         out = self.up(out)

#         return out
    def forward(self, x_1, x_2, x_3, x_4,mask,mx_len):
        x_4_ = self.bot_conv(x_4)
        out = self.up1(x_4_)
#         print(f'mask.size()={mask.size()}')
        b,c,f,h,w = mask.size()
        mask = mask.expand(b,64,f,h,w)
#         for i in range(f):
            
#             mask_rs = mask[:,:,i,::16,::16]
#             mask_o = mask_rs[:,:,:,:]*out
#             mask_mean = ((torch.sum(mask_o,axis = (2,3)).transpose(1,0))/mx_len[:,0,0,i])
#             mask_ti = mask_mean*(mask_rs.permute(2,3,1,0))
#             out = out*(1-mask_rs)+mask_ti.permute(3,2,0,1)

        

        x_3_ = self.skip_conv1(x_3)
        out = self.fusion1(x_3_, out)
        out = self.up2(out)
#         for i in range(f):
#             mask_rs = mask[:,:,i,::8,::8]
#             mask_o = mask_rs[:,:,:,:]*out
#             mask_mean = ((torch.sum(mask_o,axis = (2,3)).transpose(1,0))/mx_len[:,0,1,i])
#             mask_ti = mask_mean*(mask_rs.permute(2,3,1,0))
#             out = out*(1-mask_rs)+mask_ti.permute(3,2,0,1)
        
        x_2_ = self.skip_conv2(x_2)
        out = self.fusion2(x_2_, out)
        out = self.up3(out)
#         for i in range(f):
#             mask_rs = mask[:,:,i,::4,::4]
#             mask_o = mask_rs[:,:,:,:]*out
#             mask_mean = ((torch.sum(mask_o,axis = (2,3)).transpose(1,0))/mx_len[:,0,2,i])
#             mask_ti = mask_mean*(mask_rs.permute(2,3,1,0))
#             out = out*(1-mask_rs)+mask_ti.permute(3,2,0,1)
        
        out = self.fusion3(x_1, out)
        out = self.up4(out)
        
#         for i in range(f):
#             mask_rs = mask[:,:,i,::2,::2]
#             mask_o = mask_rs[:,:,:,:]*out
#             mask_mean = ((torch.sum(mask_o,axis = (2,3)).transpose(1,0))/mx_len[:,0,3,i])
#             mask_ti = mask_mean*(mask_rs.permute(2,3,1,0))
#             out = out*(1-mask_rs)+mask_ti.permute(3,2,0,1)
        b,c,f,h,w = mask.size()
    #         print(c)
        mask = mask.expand(b,64,f,h,w)
        for i in range(f):
            mxt = mx_len[:,0:1,3,i]
    #             print(f'mxtshape={mxt.shape}')
            mxt = mxt.expand(b,64)
            mask_rs = mask[:,:,i,::2,::2]
            _,_,hrs,wrs = mask_rs.size()
            mask_o = mask_rs[:,:,:,:]*out
    #             print(f'mask_o={mask_o.shape}')
    #             print(f'mxt={mxt.shape}')
    #             print(f'torch.sum(mask_o,axis=(2,3))={torch.sum(mask_o,axis=(2,3)).shape}')
            mask_mean = torch.sum(mask_o,axis=(2,3)).squeeze()/mxt
    #             print(f'mask_mean={mask_mean.shape}')
    #             print(f'mask_rs={mask_rs.shape}')
    #             print(f'b*mask.shape[1]={b*mask.shape[1]}')
    #             print(f'diag={(torch.diag(mask_mean.reshape(b*mask.shape[1]))@(mask_rs.reshape(b*mask.shape[1],h*w))).shape}')
    # #             mask_ti = mask_mean*mask_rs
            mask_ti = (torch.diag(mask_mean.reshape(b*mask.shape[1]))@mask_rs.reshape(b*mask.shape[1],hrs*wrs)) \
                   .reshape(mask_rs.shape)
            out = out*(1-mask_rs)+mask_ti         
        out = self.up5(out)
        
#         for i in range(f):
#             mask_rs = mask[:,:,i,::1,::1]
#             mask_o = mask_rs[:,:,:,:]*out
#             mask_mean = ((torch.sum(mask_o,axis = (2,3)).transpose(1,0))/mx_len[:,0,4,i])
#             mask_ti = mask_mean*(mask_rs.permute(2,3,1,0))
#             out = out*(1-mask_rs)+mask_ti.permute(3,2,0,1)
        
        return out
    
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.bot_conv = nn.Conv2d(
            in_channels=in_channels[0], out_channels=out_channels, kernel_size=1)
        self.skip_conv1 = nn.Conv2d(
            in_channels=in_channels[1], out_channels=out_channels, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(
            in_channels=in_channels[2], out_channels=out_channels, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.fusion1 = SelectiveFeatureFusion(out_channels)
        self.fusion2 = SelectiveFeatureFusion(out_channels)
        self.fusion3 = SelectiveFeatureFusion(out_channels)

    def forward(self, x_1, x_2, x_3, x_4):
        x_4_ = self.bot_conv(x_4)
#         out1 = self.up(x_4_)
        out1 = interp_method(x_4_,tuple(x_3.shape[-2:]))
#         print(f'out1shape={out1.shape}')

        x_3_ = self.skip_conv1(x_3)
        out2 = self.fusion1(x_3_, out1)
#         out2 = self.up(out2)
        out2 = interp_method(out2,tuple(x_2.shape[-2:]))
#         print(f'out2shape={out2.shape}')
        x_2_ = self.skip_conv2(x_2)
        out3 = self.fusion2(x_2_, out2)
#         out3 = self.up(out3)
        out3 = interp_method(out3,tuple(x_1.shape[-2:]))
#         print(f'out3shape={out3.shape}')
        out4 = self.fusion3(x_1, out3)
        out4 = self.up(out4)
        out = self.up(out4)
#         print(f'out4shape={out.shape}')
        return out,out1,out2,out3,out4

class SelectiveFeatureFusion(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=int(in_channel*2),
                      out_channels=in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, 
                      out_channels=int(in_channel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_channel / 2)),
            nn.ReLU())

        self.conv3 = nn.Conv2d(in_channels=int(in_channel / 2), 
                               out_channels=2, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_local, x_global):
        x = torch.cat((x_local, x_global), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        attn = self.sigmoid(x)

        out = x_local * attn[:, 0, :, :].unsqueeze(1) + \
              x_global * attn[:, 1, :, :].unsqueeze(1)

        return out

def interp_method(x,size):
    x = F.interpolate(x,size,mode='bilinear',align_corners=True) 
    return x    
    
def grad2d(x,dim):
    if dim == 0:
        x_cat = torch.cat((x,x[:,:,-1:,:],),dim+2)
        x_grad = x_cat[:,:,1:] - x_cat[:,:,:-1]
    if dim == 1:
        x_cat = torch.cat((x,x[:,:,:,-1:],),dim+2)
        x_grad = x_cat[:,:,:,1:] - x_cat[:,:,:,:-1]
    
    return x_grad
        
        