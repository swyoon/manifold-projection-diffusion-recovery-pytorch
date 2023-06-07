import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
import math
from models.modules import FCNet


def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = timesteps.type(dtype=torch.float)[:, None] * emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.pad(emb, [0, 1], value=0.0)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def process_single_t(x, t):
    if isinstance(t, int) or len(t.shape) == 0:
        t = torch.ones([x.shape[0]], dtype=torch.long, device=x.device) * t
    return t

class FCNet_temb(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_dim=128, t_emb_dim=32, activation="relu", spec_norm=False,
    ):
        super().__init__()
        self.net1 = FCNet(
            in_dim=in_dim,
            out_dim=hidden_dim,
            l_hidden=[],
            activation=activation,
            out_activation="linear",
            use_spectral_norm=spec_norm
        )
        self.net2 = FCNet(
            in_dim=t_emb_dim,
            out_dim=hidden_dim,
            l_hidden=(hidden_dim, hidden_dim),
            activation=activation,
            out_activation="linear",
            use_spectral_norm=spec_norm
        )
        self.net3 = FCNet(
            in_dim=2 * hidden_dim,
            out_dim=out_dim,
            l_hidden=(hidden_dim, hidden_dim),
            activation=activation,
            out_activation="linear",
        )
        self.t_emb_dim = t_emb_dim

    def forward(self, x, t):
        if len(x.shape) == 4:
            x = x.view(x.shape[0], -1)
        x_ = self.net1(x)
        t = process_single_t(x, t)
        t_emb = get_timestep_embedding(t, self.t_emb_dim).to(x.device)
        t_emb = self.net2(t_emb)
        x_ = torch.cat([x_, t_emb], dim=1)
        return self.net3(x_)


"""
Network architecture used in the paper.
Converted into PyTorch.
"""

nonlinearity = F.leaky_relu

class normalize(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, spec_norm=False, use_scale=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.spec_norm = spec_norm
        self.use_scale = use_scale
        if self.spec_norm:
            self.conv = P.spectral_norm(self.conv)
        if self.use_scale:
            self.register_parameter('g', nn.Parameter(torch.ones(out_channels, dtype=torch.float)))
            
    def forward(self, inputs):
        z = self.conv(inputs)
        if self.use_scale:
            z = z * self.g[:,None,None]
        return z
        


class Resnet_Block(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, conv_shortcut=True, spec_norm=True, use_scale=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.temb_ch = temb_ch
        self.conv_shortcut = conv_shortcut
        self.spec_norm = spec_norm
        self.use_scale = use_scale
        
        self.dense = nn.Linear(in_features=temb_ch ,out_features=out_ch)
        # self.conv2d_1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), padding=1)
        # self.conv2d_2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3,3), padding=1)
        # self.conv2d_shortcut = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), padding=1)
        self.conv2d_1 = Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, use_scale=False,
                               spec_norm=spec_norm)
        self.conv2d_2 = Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, use_scale=True,
                              spec_norm=spec_norm)
        self.conv2d_shortcut = Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, use_scale=False,
                                     spec_norm=spec_norm)
        
        self.normalize_1 = normalize()
        self.normalize_2 = normalize()
        
        if spec_norm:
            self.dense = P.spectral_norm(self.dense)
            # self.conv2d_1 = P.spectral_norm(self.conv2d_1)
            # self.conv2d_2 = P.spectral_norm(self.conv2d_2)
            # self.conv2d_shortcut = P.spectral_norm(self.conv2d_shortcut)

    
    def forward(self, inputs, temb=None, dropout=0.):
        x = inputs
        h = inputs
        _, C, _, _ = inputs.shape
        
        h = nonlinearity(self.normalize_1(h))
        h = self.conv2d_1(h)
        
        if temb is not None:
            h += self.dense(nonlinearity(temb))[:,:,None,None]  # spatial dimension broadcasting
            
        h = nonlinearity(self.normalize_2(h))
        h = F.dropout(h, p=dropout)
        h = self.conv2d_2(h)
        
        if C != self.out_ch:
            x = self.conv2d_shortcut(x)
            
        assert x.shape == h.shape, f'{x.shape} {h.shape}'
        return x + h
        

class WideResNet_temb2(nn.Module):
    """
    Pytorch conversion of the network used in the paper
    https://github.com/ruiqigao/recovery_likelihood/blob/main/network.py
    """
    def __init__(
        self, *,in_channels=3, ch=128, img_sz=32, num_res_blocks=2, attn_resolutions=(16,)
    ):
        super().__init__()
        self.ch = ch
        self.img_sz = img_sz
        if self.img_sz == 32:
            self.ch_mult = (1, 2, 2, 2)
        elif self.img_sz == 128:
            self.ch_mult = (1, 2, 2, 2, 4, 4)
        elif self.img_sz == 64:
            self.ch_mult = (1, 2, 2, 2, 4)
        elif self.img_sz == 256:
            self.ch_mult = (1, 1, 2, 2, 2, 4, 4,)
        elif self.img_sz == 28:
            self.ch_mult = (1, 2, 2)
        else:
            raise NotImplementedError
            
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_resolutions = len(self.ch_mult)
        self.resamp_with_conv = False  # downsampling does not involve conv
        self.use_attention = False  # no attention used
        self.spec_norm = True  # use spectral norm
        self.final_act = 'relu'
        self.in_channels = in_channels
        
        self.build()

    def build(self):
        # timestep embedding
        self.temb_dense_0 = P.spectral_norm(nn.Linear(in_features=self.ch, 
                                                      out_features=self.ch * 4))
        self.temb_dense_1 = P.spectral_norm(nn.Linear(in_features=self.ch * 4, 
                                                      out_features=self.ch * 4))
        self.temb_dense_2 = nn.Linear(in_features=self.ch * 4, 
                                      out_features=self.ch * self.ch_mult[-1])

        S = self.img_sz  # image size
        self.res_levels = nn.ModuleList()
        self.attn_s = dict()
        self.downsample_s = nn.ModuleList()

        self.conv2d_in = P.spectral_norm(nn.Conv2d(in_channels=self.in_channels, out_channels=self.ch, kernel_size=3))
        prev_ch = self.ch
        for i_level in range(self.num_resolutions):
            res_s = nn.ModuleList()
            # if self.use_attention and S in self.attn_resolutions:
            #     self.attn_s[str(S)] = []
            for i_block in range(self.num_res_blocks):
                res_s.append(
                    Resnet_Block(
                        in_ch=prev_ch,
                        out_ch=self.ch * self.ch_mult[i_level],
                        temb_ch=self.ch * 4
                    )
                )
                prev_ch = self.ch * self.ch_mult[i_level]
                # if self.use_attention and S in self.attn_resolutions:
                #     self.attn_s[str(S)].append(
                #         attn_block(name="down_{}_attn_{}".format(i_level, i_block))
                #     )
            self.res_levels.append(res_s)

            if i_level != self.num_resolutions - 1:
                self.downsample_s.append(
                    nn.AvgPool2d(2)
                    # downsample(
                    #     with_conv=self.resamp_with_conv,
                    # )
                )
                S = S // 2

        # end
        self.normalize_out = normalize()
        # self.fc_out = nn.Linear(in_features= num_units=1, spec_norm=False)

    def forward(self, inputs, t, dropout=0.):
        x = inputs
        B, _, S, _ = x.shape
        # assert x.dtype == tf.float32 and x.shape[2] == S
        if isinstance(t, int) or len(t.shape) == 0:
            t = torch.ones([B], dtype=torch.long, device=inputs.device) * t

        # Timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb_dense_0(temb)
        temb = self.temb_dense_1(nonlinearity(temb))
        assert temb.shape == (B, self.ch * 4)

        # downsample
        h = self.conv2d_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.res_levels[i_level][i_block](h, temb=temb, dropout=dropout)

                if self.use_attention:
                    if h.shape[1] in self.attn_resolutions:
                        h = self.attn_s[str(h.shape[1])][i_block](h)

            if i_level != self.num_resolutions - 1:
                h = self.downsample_s[i_level](h)

        # end
        if self.final_act == "relu":
            h = F.relu(h)
        elif self.final_act == "swish":
            h = F.swish(h)
        elif self.final_act == "lrelu":
            h = F.leaky_relu(x, negative_slope=0.2)
        else:
            raise NotImplementedError
        h = torch.sum(h, dim=[2,3])
        temb_final = self.temb_dense_2(nonlinearity(temb))
        h = torch.sum(h * temb_final, dim=1)

        return h
