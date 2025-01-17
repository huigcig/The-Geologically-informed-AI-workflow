import torch.nn as nn
from . import model
import torch
import torch.nn.functional as F
from unets import *
class models(nn.Module):
    def __init__(self,max_depth,is_train,in_fl_c,fl_model =None,
                 unet_nf = (8,128,256),relu=True,rgt2gr2rgt=True,lkfl=False,uxrg=False):
        super(models, self).__init__()
        self.fl_model = fl_model
        self.hr = model.GLPDepth_add_gradient(max_depth,is_train,relu,rgt2gr2rgt,lkfl)
        self.uxrg = uxrg
        if in_fl_c ==4:
            print('use ux uxx uy uyy')
        if fl_model =='unet':
            print('use unet')
            self.fl = U_Net_simple(in_fl_c,1)
        elif fl_model =='unet_cat':
            print('use unet_cat')
            self.fl = U_Net_simple_cat(in_fl_c,1,unet_nf)
        else:
            self.fl = model.rgt2fl(in_fl_c)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,epoch):
        if epoch <400:
            
            rgt,conv1,conv2,conv3,conv4,out1,out2,out3,out4,out_gradient = self.hr(x)

            ux = model.grad2d(rgt,dim=0)
            ux_bn =self.sigmoid(self.bn2(ux))
            uxx = torch.abs(model.grad2d(ux,dim=0))
            uxx_bn = self.sigmoid(self.bn1(uxx))

            uy = model.grad2d(rgt,dim=1)
            uy_bn =self.sigmoid(self.bn2(uy))
            uyy = torch.abs(model.grad2d(uy,dim=1))    
            uyy_bn =self.sigmoid(self.bn2(uyy))
            uxxyy = torch.cat((uxx_bn,uyy_bn),1)
            if self.uxrg ==False:
                uxxyy=uxxyy.detach()
            with torch.no_grad():
                if self.fl_model =='unet_cat':
                    fl,x3_cat,x3,d3,d2,d1 = self.fl(uxxyy,conv1,conv2)
                else:
                    fl = self.fl(uxxyy)
        elif 400<epoch<600:
            with torch.no_grad():
                rgt,conv1,conv2,conv3,conv4,out1,out2,out3,out4,out_gradient = self.hr(x)

            ux = model.grad2d(rgt,dim=0)
            ux_bn =self.sigmoid(self.bn2(ux))
            uxx = torch.abs(model.grad2d(ux,dim=0))
            uxx_bn = self.sigmoid(self.bn1(uxx))

            uy = model.grad2d(rgt,dim=1)
            uy_bn =self.sigmoid(self.bn2(uy))
            uyy = torch.abs(model.grad2d(uy,dim=1))    
            uyy_bn =self.sigmoid(self.bn2(uyy))
            uxxyy = torch.cat((uxx_bn,uyy_bn),1)
            if self.uxrg ==False:
                uxxyy=uxxyy.detach()
            
            if self.fl_model =='unet_cat':
                fl,x3_cat,x3,d3,d2,d1 = self.fl(uxxyy,conv1,conv2)
            else:
                fl = self.fl(uxxyy)
        else:

            rgt,conv1,conv2,conv3,conv4,out1,out2,out3,out4,out_gradient = self.hr(x)

            ux = model.grad2d(rgt,dim=0)
            ux_bn =self.sigmoid(self.bn2(ux))
            uxx = torch.abs(model.grad2d(ux,dim=0))
            uxx_bn = self.sigmoid(self.bn1(uxx))

            uy = model.grad2d(rgt,dim=1)
            uy_bn =self.sigmoid(self.bn2(uy))
            uyy = torch.abs(model.grad2d(uy,dim=1))    
            uyy_bn =self.sigmoid(self.bn2(uyy))
            uxxyy = torch.cat((uxx_bn,uyy_bn),1)
            if self.uxrg ==False:
                uxxyy=uxxyy.detach()
            
            if self.fl_model =='unet_cat':
                fl,x3_cat,x3,d3,d2,d1 = self.fl(uxxyy,conv1,conv2)
            else:
                fl = self.fl(uxxyy)           
#         return rgt,fl,ux_bn,uxx_bn,uy_bn,uyy_bn,conv1,conv2,conv3,conv4,out1,out2,out3,out4,out_gradient,x3_cat,x3,d3,d2,d1
        return rgt,fl