"""
PixelCNN implementation

References:
    1. van den Oord, Pixel Recurrent Neural Networks 2016a
    2. van den Oord, Conditional Image Generation with PixelCNN Decoders, 2016c
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


# --------------------
# Model components
# --------------------

def pixelcnn_gate(x):
    a, b = x.chunk(2,1)
    return torch.tanh(a) * torch.sigmoid(b)

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask_type=None, mask_n_channels=None, gated=False, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.constant_(self.bias, 0.)

        # set up mask -- cf PixelRNN paper Figure 2 Right: masks A and B
        self.mask_type = mask_type
        self.mask_n_channels = mask_n_channels
        center_row = self.kernel_size[0] // 2
        center_col = self.kernel_size[1] // 2

        mask = torch.ones_like(self.weight)         # shape (out_channels, in_channels, kernel_height, kernel_width)

        # mask out 1/ rows below the middle and 2/ center row pixels right of middle
        if center_row == 0:                         # case when kernel_size = (1,k) in horizontal stack
            mask[:, :, :, center_col+1:] = 0
        elif center_col == 0:                       # case when kernel_size = (k,1)
            mask[:, :, center_row+1:, :] = 0
        else:                                       # case when kernel_size = (k,k)
            mask[:, :, center_row+1:, :] = 0
            mask[:, :, center_row, center_col+1:] = 0

        # mask out center pixel in future channels -- mask A current channel is 0; mask B current channel is 1
        for i in range(mask_n_channels):
            for j in range(mask_n_channels):
                if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                    mask[j::mask_n_channels, i::mask_n_channels, center_row, center_col] = 0

        # mask out center row (vertical stack in a Gated Residual Layer); cf Conditional image generation with PixelCNN Decoders
        if mask_type == 'vstack':
            mask[:, :, center_row, :] = 0

        if gated:
            # pixelcnn gate splits the input in two along the channel dim;
            # ensure that both chunks receive the same mask by replicating the first half of the mask over the second
            mask = mask.chunk(2,0)[0].repeat(2,1,1,1)

        # final mask
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

    def __repr__(self):
        s = super().__repr__()
        return s[:-1] + ', mask_type={}, mask_n_channels={}'.format(self.mask_type, self.mask_n_channels) + s[-1]


class GatedResidualLayer(nn.Module):
    """ Figure 2 in Conditional image generation with PixelCNN Decoders """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, mask_n_channels, n_cond_classes, norm_layer):
        super().__init__()
        self.residual = (in_channels==out_channels)
        self.norm_layer = norm_layer

        self.v   = MaskedConv2d(in_channels, 2*out_channels, kernel_size, padding=kernel_size//2,
                                mask_type='vstack', mask_n_channels=mask_n_channels, gated=True)
        self.h   = MaskedConv2d(in_channels, 2*out_channels, (1, kernel_size), padding=(0, kernel_size//2),
                                mask_type=mask_type, mask_n_channels=mask_n_channels, gated=True)
        self.v2h = MaskedConv2d(2*out_channels, 2*out_channels, kernel_size=1,
                                mask_type=mask_type, mask_n_channels=mask_n_channels, gated=True)
        self.h2h = MaskedConv2d(out_channels, out_channels, kernel_size=1,
                                mask_type=mask_type, mask_n_channels=mask_n_channels, gated=False)

        if n_cond_classes:
            self.proj_h = nn.Linear(n_cond_classes, 2*out_channels)

        if self.norm_layer:
            self.norm_layer_v = nn.BatchNorm2d(out_channels)
            self.norm_layer_h = nn.BatchNorm2d(out_channels)

    def forward(self, x_v, x_h, h=None):
        # projection of h if included for conditional generation (cf paper section 2.3 -- added before the pixelcnn_gate)
        proj_y = self.proj_h(h)[:,:,None,None] if h is not None else 0

        # vertical stack
        x_v_out = self.v(x_v)
        x_v2h = self.v2h(x_v_out) + proj_y
        x_v_out = pixelcnn_gate(x_v_out)

        # horizontal stack
        x_h_out = self.h(x_h) + x_v2h + proj_y
        x_h_out = pixelcnn_gate(x_h_out)
        x_h_out = self.h2h(x_h_out)

        # residual connection
        if self.residual:
            x_h_out = x_h_out + x_h

        # normalization
        if self.norm_layer:
            x_v_out = self.norm_layer_v(x_v_out)
            x_h_out = self.norm_layer_h(x_h_out)

        return x_v_out, x_h_out

    def extra_repr(self):
        return 'residual={}, norm_layer={}'.format(self.residual, self.norm_layer)


# --------------------
# PixelCNN
# --------------------

class PixelCNN(nn.Module):
    def __init__(self, image_dims, n_bits, n_channels, n_out_conv_channels, kernel_size, n_res_layers, n_cond_classes, norm_layer=True):
        super().__init__()
        C, H, W = image_dims

        self.input_conv = MaskedConv2d(C, 2*n_channels, kernel_size=7, padding=3, mask_type='a', mask_n_channels=C, gated=True)
        self.res_layers = nn.ModuleList([
            GatedResidualLayer(n_channels, n_channels, kernel_size, 'b', C, n_cond_classes, norm_layer)
            for _ in range(n_res_layers)])
        self.conv_out1 = MaskedConv2d(n_channels, 2*n_out_conv_channels, kernel_size=1, mask_type='b', mask_n_channels=C, gated=True)
        self.conv_out2 = MaskedConv2d(n_out_conv_channels, 2*n_out_conv_channels, kernel_size=1, mask_type='b', mask_n_channels=C, gated=True)
        self.output = MaskedConv2d(n_out_conv_channels, C * 2**n_bits, kernel_size=1, mask_type='b', mask_n_channels=C)

        if n_cond_classes:
            self.proj_h = nn.Linear(n_cond_classes, 2*n_channels)

    def forward(self, x, h=None):
        B, C, H, W = x.shape

        x = pixelcnn_gate(self.input_conv(x) + (self.proj_h(h)[:,:,None,None] if h is not None else 0.))
        x_v, x_h = x, x

        for l in self.res_layers:
            x_v, x_h = l(x_v, x_h)

        out = pixelcnn_gate(self.conv_out1(x_h))
        out = pixelcnn_gate(self.conv_out2(out))
        out = self.output(out)

        return out.reshape(B, -1, C, H, W)

# --------------------
# Loss and generation functions
# --------------------

def loss_fn(logits, targets):
    """
    Args
        logits -- model output of shape (B, 2**n_bits, C, H, W)
        targets -- data tensor of shape (B, C, H, W) with values in pixel space [0, 2**n_bits)
    """
    return F.cross_entropy(logits, targets, reduction='none').sum([1,2,3])

def generate_fn(model, n_samples, image_dims, device, preprocess_fn, n_bits, h=None):
    out = torch.zeros(n_samples, *image_dims, device=device)
    with tqdm(total=(image_dims[0]*image_dims[1]*image_dims[2]), desc='Generating {} images'.format(n_samples)) as pbar:
        for yi in range(image_dims[1]):
            for xi in range(image_dims[2]):
                for ci in range(image_dims[0]):
                    logits = model(out, h)
                    probs = F.softmax(logits, dim=1)
                    sample = torch.multinomial(probs[:,:,ci,yi,xi], num_samples=1).squeeze()
                    out[:,ci,yi,xi] = preprocess_fn(sample, n_bits)
                    pbar.update()
    return out
