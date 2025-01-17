import os
import math
import random
import numpy as np
import random
import torch
import draw,utils
import matplotlib.pyplot as plt
from scipy import interpolate

def min_max_norm(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return(x)

def Vis_Rgt_Curves(data, curves_num=10):
    rgt_min,rgt_max = np.min(data), np.max(data)
    xx = np.linspace(rgt_max,rgt_min,curves_num)
    rgt_line = np.zeros_like(data)
    for j in range(data.shape[1]):
        for ni in range(curves_num):
            xi_arg = np.argmin(np.abs(data[:,j]-xx[ni]))
            xi_1,xi_2 = np.max((0,xi_arg-2)), np.min((data.shape[0],xi_arg+1))
            if xi_2<=(data.shape[0]-1): # volid the bottom horizons lines
                rgt_line[xi_1:xi_2, j] = xx[ni]
    return rgt_line

def min_max_norm(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return(x)

def rgt_newsampling(rgt, layer_num=1001, min_width=15, print_mid_results=False):
    """
    Input: 
    rgt: [depth, crossline]
    layer_num: the number of dense sampling for the layer in old rgt
    min_width: the rule for reselect stratigraphic layers (the width between two layers)
    Output:
    rgt_layers: [layer_num, crossline]
    rgt_layers_new: [new_num, crossline]
    rgt_new: [depth, crossline]
    """
    # 1. resampling the high-resolution new stratigraphic layers [1001,crossline]
    layer_num = layer_num
    rgt = min_max_norm(rgt)*10
    vi_all = np.linspace(np.min(rgt),np.max(rgt),layer_num)[::-1]
    rgt_layers = np.full((rgt.shape[1],vi_all.shape[0]),np.nan)
    x = np.linspace(0,rgt.shape[0]-1,rgt.shape[0])

    for i in range(rgt_layers.shape[0]):
        ri = rgt[:,i]
        mini,maxi = np.min(ri),np.max(ri)
        for j in range(rgt_layers.shape[1]):
            if mini<=vi_all[j]<=maxi:
                func = interpolate.interp1d(ri,x,kind="linear")
                rgt_layers[i,j] = func(vi_all[j])
    
    # 2. build new stratigraphic layers with new rule(>minwidth)
#     min_width = min_width
    rgt_layers_new = []
    old_vi = []
    for j in range(rgt_layers.shape[1]):
        if j==0:
            bottom_layer = rgt_layers[:,j].copy()
            rgt_layers_new.append(bottom_layer)
        else:
            top_layer = rgt_layers[:,j].copy()
            if np.nanmax(bottom_layer-top_layer) >= min_width:
                bottom_layer = top_layer.copy()
                rgt_layers_new.append(top_layer)
                old_vi.append(vi_all[j])
                
    rgt_layers_new = np.array(rgt_layers_new)
    old_vi = np.array(old_vi)
    
    # 3. inverse-interpolation with new layers to obtain new_rgt with Equally-sampling 
    new_vi = np.linspace(len(rgt_layers_new)-1,0,len(rgt_layers_new))
    
    rgt_new = np.zeros_like(rgt)
    x_new = np.linspace(0,rgt_new.shape[0]-1,rgt_new.shape[0])
    for i in range(rgt.shape[1]):
        mask = np.isnan(rgt_layers_new[:,i]) # remove nan
        xi = rgt_layers_new[:,i][~mask]
        yi = new_vi[~mask]
        func = interpolate.interp1d(xi,yi,kind="linear",fill_value="extrapolate")
        rgt_new[:,i] = func(x_new)
    rgt_new = min_max_norm(rgt_new)*10
    
    if print_mid_results:
        return rgt_new, rgt_layers, rgt_layers_new, old_vi
    else:
        return rgt_new
    
    
def build_stratigraphic_wheeler_thickness(rgt, time_num=100, res=None):
    strat_layer = np.full((time_num+1,rgt.shape[1]),np.nan)
    ti = np.linspace(np.min(rgt),np.max(rgt),time_num+1)
    for j in range(rgt.shape[1]):
        ri = rgt[1:,j]
        xi = np.linspace(0,ri.shape[0]-1,ri.shape[0])
        mini,maxi = np.min(ri),np.max(ri)
        mask = (ti>=mini)&(ti<=maxi)
        func = interpolate.interp1d(ri,xi,kind="linear")
        strat_layer[:,j][mask] = func(ti[mask])
    wheeler_strat = np.diff(strat_layer,axis=0)
    wheeler_strat[wheeler_strat<0] = 0
    if res is not None:
        wheeler_strat[wheeler_strat<res] = np.nan
    return wheeler_strat

def build_flatten_seis(rgt, seis, time_num =100, wheeler_mask=None, res=None):
    flatten_seis = np.full((time_num,rgt.shape[1]),np.nan)
    ti = np.linspace(np.min(rgt),np.max(rgt),time_num)
    for j in range(rgt.shape[1]):
        ri = rgt[1:,j]
        si = seis[1:,j]
        mini,maxi = np.min(ri),np.max(ri)
        mask = (ti>=mini)&(ti<=maxi)
        func = interpolate.interp1d(ri,si,kind="nearest")
        flatten_seis[:,j][mask] = func(ti[mask])
    if (wheeler_mask is not None) & (res is not None) :
#         flatten_seis[wheeler_mask<res] = np.nan
        flatten_seis[wheeler_mask!=wheeler_mask] = np.nan
    return flatten_seis


def build_unconformties(line_points,rgt,):
    surfs = []
    ti = np.linspace(np.min(rgt),np.max(rgt),rgt.shape[0])
    num = len(line_points)
    for ni in range(num):
        pi = line_points[ni]
        x1 = np.linspace(pi[0],pi[1],int(pi[1]-pi[0]+1))
        y1 = np.ones_like(x1)*pi[2]
        t1 = ti[pi[2]]
        surf1 = np.full(x1.shape,np.nan)
        yi = np.linspace(0,rgt.shape[0]-1,rgt.shape[0])
        for j in range(x1.shape[0]):
            slice_j = j + pi[0]
            mini,maxi = np.min(rgt[:,slice_j]),np.max(rgt[:,slice_j])
            if mini<=t1<=maxi:
                func = interpolate.interp1d(rgt[:,slice_j],yi,kind="linear")
                surf1[j] = func(t1)
        surfs.append([x1,y1,surf1])
    return surfs

def build_unconf_volume(rgt,wheeler_strat,w_scale):
    unconf = np.zeros_like(rgt)
    xj = np.linspace(0,rgt.shape[0]-1,rgt.shape[0])
    T = np.linspace(np.min(rgt),np.max(rgt),rgt.shape[0])
    for j in range(wheeler_strat.shape[1]):
        tj = rgt[:,j]
        func = interpolate.interp1d(tj,xj,kind="linear")
        wj_un = ((np.where(wheeler_strat[:,j]!=wheeler_strat[:,j])[0]) / w_scale).astype(np.int16)    
        tj_un = T[wj_un]
        mini,maxi = np.min(rgt[:,j]),np.max(rgt[:,j])
        for t1 in tj_un:
            if mini<=t1<=maxi:
                unconf[int(func(t1)),j] = 1
    return unconf


