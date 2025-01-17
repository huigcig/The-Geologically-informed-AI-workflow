import torch
import torch.nn.functional as F
import numpy as np 

def _fspecial_gaussian(size, channel, sigma):
    coords = torch.tensor([(x - (size - 1.) / 2.) for x in range(size)])
    coords = -coords ** 2 / (2. * sigma ** 2)
    grid = coords.view(1, -1) + coords.view(-1, 1)
    grid = grid.view(1, -1)
    grid = grid.softmax(-1)
    kernel = grid.view(1, 1, size, size)
    kernel = kernel.expand(channel, 1, size, size).contiguous()
    return kernel

# zfbi
def _fspecial_gaussian3d(size, channel, sigma):
    coords = torch.tensor([(x - (size - 1.) / 2.) for x in range(size)])
    coords = -coords ** 2 / (2. * sigma ** 2)
    grid = coords.view(1, -1, 1) + coords.view(-1, 1, 1) + coords.view(1, 1, -1)
    grid = grid.view(1, -1)
    grid = grid.softmax(-1)
    kernel = grid.view(1, 1, size, size, size)
    kernel = kernel.expand(channel, 1, size, size, size).contiguous()
    return kernel

def _ssim(output, target, max_val, k1, k2, channel, kernel):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mu1 = F.conv2d(output, kernel, groups=channel)
    mu2 = F.conv2d(target, kernel, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(output * output, kernel, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = F.conv2d(output * target, kernel, groups=channel) - mu1_mu2

    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim, v1 / v2

# zfbi
def _ssim3d(input, target, max_val, k1, k2, channel, kernel):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    
    mu1 = F.conv3d(input, kernel, groups=channel)
    mu2 = F.conv3d(target, kernel, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(input * input, kernel, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = F.conv3d(input * target, kernel, groups=channel) - mu1_mu2

    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim, v1 / v2

# 22/11/16, zfbi
def _masked_ssim3d(input, target, mask, max_val, k1, k2, channel, kernel):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    
    mu1 = F.conv3d(input, kernel, groups=channel)
    mu2 = F.conv3d(target, kernel, groups=channel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(input * input, kernel, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(target * target, kernel, groups=channel) - mu2_sq
    sigma12 = F.conv3d(input * target, kernel, groups=channel) - mu1_mu2

    mk = F.interpolate(mask, size=sigma12.shape[2:], 
                       mode='trilinear', align_corners=True) 
    
    v1 = 2 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
    return ssim, v1 / v2, mk

def ssim_loss(input, target, max_val, filter_size=7, k1=0.01, k2=0.03,
              sigma=1.5, kernel=None, size_average=None, reduce=None, reduction='mean'):

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, target.dim(-2), target.dim(-1))
    elif dim == 3:
        input = input.expand(1, input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, target.dim(-3), target.dim(-2), target.dim(-1))
    elif dim != 4:
        raise ValueError('Expected 2, 3, or 4 dimensions (got {})'.format(dim))

    _, channel, _, _ = input.size()

    if kernel is None:
        kernel = _fspecial_gaussian(filter_size, channel, sigma)
    kernel = kernel.to(device=input.device)

    ret, _ = _ssim(input, target, max_val, k1, k2, channel, kernel)

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def ms_ssim_loss(input, target, max_val, filter_size=7, k1=0.01, k2=0.03,
                 sigma=1.5, kernel=None, weights=None, size_average=None, reduce=None, reduction='mean'):

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, target.dim(-2), target.dim(-1))
    elif dim == 3:
        input = input.expand(1, input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, target.dim(-3), target.dim(-2), target.dim(-1))
    elif dim != 4:
        raise ValueError('Expected 2, 3, or 4 dimensions (got {})'.format(dim))

    _, channel, _, _ = input.size()

    if kernel is None:
        kernel = _fspecial_gaussian(filter_size, channel, sigma)
    kernel = kernel.to(device=input.device)
    
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] 
    weights = torch.tensor(weights, device=input.device)
    weights = weights.unsqueeze(-1).unsqueeze(-1)
    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim(input, target, max_val, k1, k2, channel, kernel)
        ssim = ssim.mean((2, 3))
        cs = cs.mean((2, 3))
        mssim.append(ssim)
        mcs.append(cs)

        input = F.avg_pool2d(input, (2, 2))
        target = F.avg_pool2d(target, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    # Normalize 
    mssim = (mssim + 1) / 2
    mcs = (mcs + 1) / 2
    p1 = mcs ** weights
    p2 = mssim ** weights
    
    ret = torch.prod(p1[:-1], 0) * p2[-1]

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

# zfbi
def ssim_loss3d(input, target, max_val, filter_size=7, k1=0.01, k2=0.03,
              sigma=1.5, kernel=None, size_average=None, reduce=None, reduction='mean'):

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, 1, input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, 1, target.dim(-2), target.dim(-1))
    elif dim == 3:
        input = input.expand(1, 1, input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, target.dim(-3), target.dim(-2), target.dim(-1))
    elif dim == 4:
        input = input.expand(1, input.dim(-4), input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, target.dim(-4), target.dim(-3), target.dim(-2), target.dim(-1))        
    elif dim != 5:
        raise ValueError('Expected 2, 3, 4, or 5 dimensions (got {})'.format(dim))

    _, channel, _, _, _ = input.size()

    if kernel is None:
        kernel = _fspecial_gaussian3d(filter_size, channel, sigma)
    kernel = kernel.to(device=input.device)
    
    # I modify ssim here to ignore average term [zfbi]
    ret, _ = _ssim3d(input, target, max_val, k1, k2, channel, kernel)
    # _, ret = _ssim3d(input, target, max_val, k1, k2, channel, kernel)
    
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def ms_ssim_loss3d(input, target, max_val, filter_size=7, k1=0.01, k2=0.03,
                 sigma=1.5, kernel=None, weights=None, size_average=None, reduce=None, reduction='mean'):

    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, 1, input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, 1, target.dim(-2), target.dim(-1))
    elif dim == 3:
        input = input.expand(1, 1, input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, target.dim(-3), target.dim(-2), target.dim(-1))
    elif dim == 4:
        input = input.expand(1, input.dim(-4), input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, target.dim(-4), target.dim(-3), target.dim(-2), target.dim(-1))        
    elif dim != 5:
        raise ValueError('Expected 2, 3, 4, or 5 dimensions (got {})'.format(dim))

    _, channel, _, _, _ = input.size()

    if kernel is None:
        kernel = _fspecial_gaussian3d(filter_size, channel, sigma)
    kernel = kernel.to(device=input.device)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] 
    weights = torch.tensor(weights, device=input.device)
    weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _ssim3d(input, target, max_val, k1, k2, channel, kernel)
        
        ssim = ssim.mean((2, 3, 4))
        cs = cs.mean((2, 3, 4))
        
        mssim.append(ssim)
        mcs.append(cs)

        input = F.avg_pool3d(input, (2, 2, 2))
        target = F.avg_pool3d(target, (2, 2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    # Normalize 
    mssim = (mssim + 1) / 2
    mcs = (mcs + 1) / 2
    p1 = mcs ** weights
    p2 = mssim ** weights
    
    # I modify ssim here to ignore average term [zfbi]
    ret = torch.prod(p1[:-1], 0) * p2[-1]
    # ret = torch.prod(p1[:-1], 0)

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def masked_ms_ssim_loss3d(input, target, mask, max_val, filter_size=7, k1=0.01, k2=0.03,
                 sigma=1.5, kernel=None, weights=None, size_average=None, reduce=None, reduction='mean'):
    if input.size() != target.size():
        raise ValueError('Expected input size ({}) to match target size ({}).'
                         .format(input.size(0), target.size(0)))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    dim = input.dim()
    if dim == 2:
        input = input.expand(1, 1, 1, input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, 1, target.dim(-2), target.dim(-1))
    elif dim == 3:
        input = input.expand(1, 1, input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, 1, target.dim(-3), target.dim(-2), target.dim(-1))
    elif dim == 4:
        input = input.expand(1, input.dim(-4), input.dim(-3), input.dim(-2), input.dim(-1))
        target = target.expand(1, target.dim(-4), target.dim(-3), target.dim(-2), target.dim(-1))        
    elif dim != 5:
        raise ValueError('Expected 2, 3, 4, or 5 dimensions (got {})'.format(dim))

    _, channel, _, _, _ = input.size()

    if kernel is None:
        kernel = _fspecial_gaussian3d(filter_size, channel, sigma)
    kernel = kernel.to(device=input.device)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] 
    weights = torch.tensor(weights, device=input.device)
    weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs, mk = _masked_ssim3d(input, target, mask, max_val, k1, k2, channel, kernel)
        
        ssim = (ssim * mk).sum((2, 3, 4)) / mk.sum((2, 3, 4))
        cs = (cs * mk).sum((2, 3, 4)) / mk.sum((2, 3, 4))        

        mssim.append(ssim)
        mcs.append(cs)

        input = F.avg_pool3d(input, (2, 2, 2))
        target = F.avg_pool3d(target, (2, 2, 2))
        mask = F.avg_pool3d(mask, (2, 2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    # Normalize 
    mssim = (mssim + 1) / 2
    mcs = (mcs + 1) / 2
    p1 = mcs ** weights
    p2 = mssim ** weights
    
    # I modify ssim here to ignore average term [zfbi]
    ret = torch.prod(p1[:-1], 0) * p2[-1]
    # ret = torch.prod(p1[:-1], 0)

    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

class _Loss(torch.nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class SSIMLoss(_Loss):

    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, channel=3, filter_size=7, k1=0.01, k2=0.03, sigma=1.5, size_average=None, reduce=None, reduction='mean'):
        super(SSIMLoss, self).__init__(size_average, reduce, reduction)
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.kernel = _fspecial_gaussian(filter_size, channel, sigma)

    def forward(self, input, target, max_val=1.):
        return ssim_loss(input, target, max_val=max_val, filter_size=self.filter_size, k1=self.k1, k2=self.k2,
                           sigma=self.sigma, reduction=self.reduction, kernel=self.kernel)

class MultiScaleSSIMLoss(_Loss):

    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, channel=3, filter_size=7, k1=0.01, k2=0.03, sigma=1.5, size_average=None, reduce=None, reduction='mean'):
        super(MultiScaleSSIMLoss, self).__init__(size_average, reduce, reduction)
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.kernel = _fspecial_gaussian(filter_size, channel, sigma)

    def forward(self, input, target, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], max_val=1.):
        return ms_ssim_loss(input, target, max_val=max_val, k1=self.k1, k2=self.k2, sigma=self.sigma, kernel=self.kernel,
                              weights=weights, filter_size=self.filter_size, reduction=self.reduction)
# zfbi
class SSIMLoss3d(_Loss):

    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, channel=3, filter_size=7, k1=0.01, k2=0.03, sigma=1.5, size_average=None, reduce=None, reduction='mean'):
        super(SSIMLoss3d, self).__init__(size_average, reduce, reduction)
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.kernel = _fspecial_gaussian3d(filter_size, channel, sigma)

    def forward(self, input, target, max_val=1.):
        return ssim_loss3d(input, target, max_val=max_val, filter_size=self.filter_size, k1=self.k1, k2=self.k2,
                           sigma=self.sigma, reduction=self.reduction, kernel=self.kernel)
    
# zfbi
class MultiScaleSSIMLoss3d(_Loss):
    
    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, channel=3, filter_size=7, k1=0.01, k2=0.03, sigma=1.5, size_average=None, reduce=None, reduction='mean'):
        super(MultiScaleSSIMLoss3d, self).__init__(size_average, reduce, reduction)
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.kernel = _fspecial_gaussian3d(filter_size, channel, sigma)

    def forward(self, input, target, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], max_val=1.):
        return ms_ssim_loss3d(input, target, max_val=max_val, k1=self.k1, k2=self.k2, sigma=self.sigma, kernel=self.kernel,
                              weights=weights, filter_size=self.filter_size, reduction=self.reduction)
    
# zfbi 22/11/16
class MaskedMultiScaleSSIMLoss3d(_Loss):
    
    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, channel=3, filter_size=7, k1=0.01, k2=0.03, sigma=1.5, size_average=None, reduce=None, reduction='mean'):
        super(MaskedMultiScaleSSIMLoss3d, self).__init__(size_average, reduce, reduction)
        self.filter_size = filter_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma
        self.kernel = _fspecial_gaussian3d(filter_size, channel, sigma)

    def forward(self, input, target, mask=None, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], max_val=1.):
        if mask is not None:
            return masked_ms_ssim_loss3d(input, target, mask, max_val=max_val, k1=self.k1, k2=self.k2, sigma=self.sigma, kernel=self.kernel,
                                         weights=weights, filter_size=self.filter_size, reduction=self.reduction) 
        else:
            return ms_ssim_loss3d(input, target, max_val=max_val, k1=self.k1, k2=self.k2, sigma=self.sigma, kernel=self.kernel,
                                  weights=weights, filter_size=self.filter_size, reduction=self.reduction)             