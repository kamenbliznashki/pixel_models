"""
PixelCNN++ implementation following https://github.com/openai/pixel-cnn/

References:
    1. Salimans, PixelCNN++ 2017
    2. van den Oord, Pixel Recurrent Neural Networks 2016a
    3. van den Oord, Conditional Image Generation with PixelCNN Decoders, 2016c
    4. Reed 2016 http://www.scottreed.info/files/iclr2017.pdf
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


# --------------------
# Helper functions
# --------------------

def down_shift(x):
#    B, C, H, W = x.shape
#    return torch.cat([torch.zeros([B, C, 1, W], device=x.device), x[:,:,:H-1,:]], 2)
    return F.pad(x, (0,0,1,0))[:,:,:-1,:]

def right_shift(x):
#    B, C, H, W = x.shape
#    return torch.cat([torch.zeros([B, C, H, 1], device=x.device), x[:,:,:,:W-1]], 3)
    return F.pad(x, (1,0))[:,:,:,:-1]

def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))

# --------------------
# Model components
# --------------------

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)

class DownShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad H above and W on each side
        Hk, Wk = self.kernel_size
        x = F.pad(x, ((Wk-1)//2, (Wk-1)//2, Hk-1, 0))
        return super().forward(x)

class DownRightShiftedConv2d(Conv2d):
    def forward(self, x):
        # pad above and on left (ie shift input down and right)
        Hk, Wk = self.kernel_size
        x = F.pad(x, (Wk-1, 0, Hk-1, 0))
        return super().forward(x)

class DownShiftedConvTranspose2d(ConvTranspose2d):
    def forward(self, x):
        x = super().forward(x)
        _, _, Hout, Wout = x.shape
        Hk, Wk = self.kernel_size
        Hs, Ws = self.stride
#        return x[:, :, :Hout - Hk + 1, (Wk-1)//2: Wout - (Wk-1)//2]
        return x[:, :, :Hout-Hk+Hs, (Wk)//2: Wout]  # see pytorch doc for ConvTranspose output

class DownRightShiftedConvTranspose2d(ConvTranspose2d):
    def forward(self, x):
        x = super().forward(x)
        _, _, Hout, Wout = x.shape
        Hk, Wk = self.kernel_size
        Hs, Ws = self.stride
#        return x[:, :, :Hout+1-Hk, :Wout+1-Wk]  # see pytorch doc for ConvTranspose output
        return x[:, :, :Hout-Hk+Hs, :Wout-Wk+Ws]  # see pytorch doc for ConvTranspose output

class GatedResidualLayer(nn.Module):
    def __init__(self, conv, n_channels, kernel_size, drop_rate=0, shortcut_channels=None, n_cond_classes=None, relu_fn=concat_elu):
        super().__init__()
        self.relu_fn = relu_fn

        self.c1 = conv(2*n_channels, n_channels, kernel_size)
        if shortcut_channels:
            self.c1c = Conv2d(2*shortcut_channels, n_channels, kernel_size=1)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.c2 = conv(2*n_channels, 2*n_channels, kernel_size)
        if n_cond_classes:
            self.proj_h = nn.Linear(n_cond_classes, 2*n_channels)

    def forward(self, x, a=None, h=None):
        c1 = self.c1(self.relu_fn(x))
        if a is not None:  # shortcut connection if auxiliary input 'a' is given
            c1 = c1 + self.c1c(self.relu_fn(a))
        c1 = self.relu_fn(c1)
        if hasattr(self, 'dropout'):
            c1 = self.dropout(c1)
        c2 = self.c2(c1)
        if h is not None:
            c2 += self.proj_h(h)[:,:,None,None]
        a, b = c2.chunk(2,1)
        out = x + a * torch.sigmoid(b)
        return out

# --------------------
# PixelCNN
# --------------------

class PixelCNNpp(nn.Module):
    def __init__(self, image_dims=(3,28,28), n_channels=128, n_res_layers=5, n_logistic_mix=10, n_cond_classes=None, drop_rate=0.5):
        super().__init__()

        # input layers for `up` and `up and to the left` pixels
        self.u_input  = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,3))
        self.ul_input_d = DownShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(1,3))
        self.ul_input_dr = DownRightShiftedConv2d(image_dims[0]+1, n_channels, kernel_size=(2,1))

        # up pass
        self.up_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConv2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, None, n_cond_classes) for _ in range(n_res_layers)]])

        self.up_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConv2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)]])

        # down pass
        self.down_u_modules = nn.ModuleList([
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,3), stride=(2,2)),
            *[GatedResidualLayer(DownShiftedConv2d, n_channels, (2,3), drop_rate, n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        self.down_ul_modules = nn.ModuleList([
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)],
            DownRightShiftedConvTranspose2d(n_channels, n_channels, kernel_size=(2,2), stride=(2,2)),
            *[GatedResidualLayer(DownRightShiftedConv2d, n_channels, (2,2), drop_rate, 2*n_channels, n_cond_classes) for _ in range(n_res_layers+1)]])

        # output logistic mix params
        #   each component has 3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
        self.output_conv = Conv2d(n_channels, (3*image_dims[0]+1)*n_logistic_mix, kernel_size=1)

    def forward(self, x, h=None):
        # add channel of ones to distinguish image from padding later on
        x = F.pad(x, (0,0,0,0,0,1), value=1)

        # input layer
        u_list  = [down_shift(self.u_input(x))]
        ul_list = [down_shift(self.ul_input_d(x)) + right_shift(self.ul_input_dr(x))]

        # up pass
        for u_module, ul_module in zip(self.up_u_modules, self.up_ul_modules):
            u_list  += [u_module(u_list[-1], h=h) if isinstance(u_module, GatedResidualLayer) else u_module(u_list[-1])]
            ul_list += [ul_module(ul_list[-1], u_list[-1], h)] if isinstance(ul_module, GatedResidualLayer) else [ul_module(ul_list[-1])]

        # down pass
        u = u_list.pop()
        ul = ul_list.pop()
        for u_module, ul_module in zip(self.down_u_modules, self.down_ul_modules):
            u  = u_module(u, u_list.pop(), h) if isinstance(u_module, GatedResidualLayer) else u_module(u)
            ul = ul_module(u, torch.cat([u, ul_list.pop()],1), h) if isinstance(ul_module, GatedResidualLayer) else ul_module(ul)

        return self.output_conv(F.elu(ul))

# --------------------
# Loss functions
# --------------------

def discretized_mix_logistic_loss(l, x, n_bits):
    """ log likelihood for mixture of discretized logistics
    Args
        l -- model output tensor of shape (B, 10*n_mix, H, W), where for each n_mix there are
                3 params for means, 3 params for coefficients, 3 params for logscales, 1 param for logits
        x -- data tensor of shape (B, C, H, W) with values in model space [-1, 1]
    """
    # shapes
    B, C, H, W = x.shape
    n_mix = l.shape[1] // (1 + 3*C)

    # unpack params of mixture of logistics
    logits = l[:, :n_mix, :, :]                         # (B, n_mix, H, W)
    l = l[:, n_mix:, :, :].reshape(B, 3*n_mix, C, H, W)
    means, logscales, coeffs = l.split(n_mix, 1)        # (B, n_mix, C, H, W)
    logscales = logscales.clamp(min=-7)
    coeffs = coeffs.tanh()

    # adjust means of channels based on preceding subpixel (cf PixelCNN++ eq 3)
    x  = x.unsqueeze(1).expand_as(means)
    if C!=1:
        m1 = means[:, :, 0, :, :]
        m2 = means[:, :, 1, :, :] + coeffs[:, :, 0, :, :] * x[:, :, 0, :, :]
        m3 = means[:, :, 2, :, :] + coeffs[:, :, 1, :, :] * x[:, :, 0, :, :] + coeffs[:, :, 2, :, :] * x[:, :, 1, :, :]
        means = torch.stack([m1, m2, m3], 2)  # out (B, n_mix, C, H, W)

    # log prob components
    scales = torch.exp(-logscales)
    plus = scales * (x - means + 1/(2**n_bits-1))
    minus = scales * (x - means - 1/(2**n_bits-1))

    # partition the logistic pdf and cdf for x in [<-0.999, mid, >0.999]
    # 1. x<-0.999 ie edge case of 0 before scaling
    cdf_minus = torch.sigmoid(minus)
    log_one_minus_cdf_minus = - F.softplus(minus)
    # 2. x>0.999 ie edge case of 255 before scaling
    cdf_plus = torch.sigmoid(plus)
    log_cdf_plus = plus - F.softplus(plus)
    # 3. x in [-.999, .999] is log(cdf_plus - cdf_minus)

    # compute log probs:
    # 1. for x < -0.999, return log_cdf_plus
    # 2. for x > 0.999,  return log_one_minus_cdf_minus
    # 3. x otherwise,    return cdf_plus - cdf_minus
    log_probs = torch.where(x < -0.999, log_cdf_plus,
                            torch.where(x > 0.999, log_one_minus_cdf_minus,
                                        torch.log((cdf_plus - cdf_minus).clamp(min=1e-12))))
    log_probs = log_probs.sum(2) + F.log_softmax(logits, 1) # log_probs sum over channels (cf eq 3), softmax over n_mix components (cf eq 1)

    # marginalize over n_mix components and return negative log likelihood per data point
    return - log_probs.logsumexp(1).sum([1,2])  # out (B,)

loss_fn = discretized_mix_logistic_loss

# --------------------
# Sampling and generation functions
# --------------------

def sample_from_discretized_mix_logistic(l, image_dims):
    # shapes
    B, _, H, W = l.shape
    C = image_dims[0]#3
    n_mix = l.shape[1] // (1 + 3*C)

    # unpack params of mixture of logistics
    logits = l[:, :n_mix, :, :]
    l = l[:, n_mix:, :, :].reshape(B, 3*n_mix, C, H, W)
    means, logscales, coeffs = l.split(n_mix, 1)  # each out (B, n_mix, C, H, W)
    logscales = logscales.clamp(min=-7)
    coeffs = coeffs.tanh()

    # sample mixture indicator
    argmax = torch.argmax(logits - torch.log(-torch.log(torch.rand_like(logits).uniform_(1e-5, 1 - 1e-5))), dim=1)
    sel = torch.eye(n_mix, device=logits.device)[argmax]
    sel = sel.permute(0,3,1,2).unsqueeze(2)  # (B, n_mix, 1, H, W)

    # select mixture components
    means = means.mul(sel).sum(1)
    logscales = logscales.mul(sel).sum(1)
    coeffs = coeffs.mul(sel).sum(1)

    # sample from logistic using inverse transform sampling
    u = torch.rand_like(means).uniform_(1e-5, 1 - 1e-5)
    x = means + logscales.exp() * (torch.log(u) - torch.log1p(-u))  # logits = inverse logistic

    if C==1:
        return x.clamp(-1,1)
    else:
        x0 = torch.clamp(x[:,0,:,:], -1, 1)
        x1 = torch.clamp(x[:,1,:,:] + coeffs[:,0,:,:] * x0, -1, 1)
        x2 = torch.clamp(x[:,2,:,:] + coeffs[:,1,:,:] * x0 + coeffs[:,2,:,:] * x1, -1, 1)
        return torch.stack([x0, x1, x2], 1)  # out (B, C, H, W)

def generate_fn(model, n_samples, image_dims, device, h=None):
    out = torch.zeros(n_samples, *image_dims, device=device)
    with tqdm(total=(image_dims[1]*image_dims[2]), desc='Generating {} images'.format(n_samples)) as pbar:
        for yi in range(image_dims[1]):
            for xi in range(image_dims[2]):
                l = model(out, h)
                out[:,:,yi,xi] = sample_from_discretized_mix_logistic(l, image_dims)[:,:,yi,xi]
                pbar.update()
    return out
