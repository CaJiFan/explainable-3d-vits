import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Reduce

# helpers
def get_activation_func(activation):
    if activation == 'silu':
        return nn.SiLU()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'leaky':
        return nn.LeakyReLU()

def conv_1x1x1_bn(inp, oup, activation='silu'):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        get_activation_func(activation)
    )

def conv_nxnxn_bn(inp, oup, kernel_size=3, stride=1, activation='silu'):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        get_activation_func(activation)
    )

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., activation='silu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            get_activation_func(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., activation='silu'):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout, activation))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/abs/1801.04381
    """
    
    def __init__(self, inp, oup, stride=1, expansion=4, activation='silu'):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.activation = get_activation_func(activation)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm3d(hidden_dim),
                self.activation,
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                self.activation,
                # dw
                nn.Conv3d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm3d(hidden_dim),
                self.activation,
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out
    
class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0., activation='silu'):
        super().__init__()
        self.pd, self.ph, self.pw = patch_size

        self.conv1 = conv_nxnxn_bn(channel, channel, kernel_size, activation=activation)
        self.conv2 = conv_1x1x1_bn(channel, dim, activation=activation)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout, activation)

        self.conv3 = conv_1x1x1_bn(dim, channel, activation=activation)
        self.conv4 = conv_nxnxn_bn(2 * channel, channel, kernel_size, activation=activation)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

         # Global representations
        _, _, d, h, w = x.shape
        x = rearrange(x, 'b c (d pd) (h ph) (w pw) -> b (pd ph pw) (d h w) c ', ph=self.ph, pw=self.pw, pd=self.pd)
        x = self.transformer(x)        
        x = rearrange(x, 'b (pd ph pw) (d h w) c -> b c (d pd) (h ph) (w pw)', c=x.shape[1], d=d//self.pd, h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw, pd=self.pd)
        
        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class MobileViT(nn.Module):
    def __init__(
        self,
        in_channels,
        volume_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2, 2),
        depths=(2, 4, 3), 
        activation='silu'
    ):
        super().__init__()
#         assert len(dims) == 3, 'dims must be a tuple of 3'
#         assert len(depths) == 3, 'depths must be a tuple of 3'

#         ih, iw, id = volume_size
#         ph, pw, pd = patch_size
#         assert ih % ph == 0 and iw % pw == 0 and id % pd == 0

        init_dim, *_, last_dim = channels

        self.conv1 = conv_nxnxn_bn(in_channels, init_dim, stride=2, activation=activation)

        self.stem = nn.ModuleList([])
        self.stem.append(MV2Block(channels[0], channels[1], 1, expansion, activation))
        self.stem.append(MV2Block(channels[1], channels[2], 2, expansion, activation))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion, activation))
        self.stem.append(MV2Block(channels[2], channels[3], 1, expansion, activation))

        self.trunk = nn.ModuleList([])
        self.trunk.append(nn.ModuleList([
            MV2Block(channels[3], channels[4], 2, expansion, activation),
            MobileViTBlock(
                dims[0], depths[0], channels[5], kernel_size, patch_size, int(dims[0] * 2), activation=activation
            )
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[5], channels[6], 2, expansion, activation),
            MobileViTBlock(
                dims[1], depths[1], channels[7], kernel_size, patch_size, int(dims[1] * 4), activation=activation
            )
        ]))

        self.trunk.append(nn.ModuleList([
            MV2Block(channels[7], channels[8], 2, expansion, activation),
            MobileViTBlock(
                dims[2], depths[2], channels[9], kernel_size, patch_size, int(dims[2] * 4), activation=activation
            )
        ]))

        self.to_logits = nn.Sequential(
            conv_1x1x1_bn(channels[-2], last_dim, activation=activation),
            Reduce('b c d h w -> b c', 'mean'),
            nn.Linear(channels[-1], num_classes, bias=False)
        )

    def forward(self, x):
        x = self.conv1(x)
        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.to_logits(x)
