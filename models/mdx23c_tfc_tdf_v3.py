import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from utils.model_utils import prefer_target_instrument

class STFT:
    def __init__(self, config):
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = config.dim_f
        
        # CMX (Channel Multiplexing) configuration for memory efficiency
        self.use_cmx = getattr(config, 'use_cmx', False)
        self.cmx_reduction = getattr(config, 'cmx_reduction', 2)

    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape([*batch_dims, c * 2, -1, x.shape[-1]])
        return x[..., :self.dim_f, :]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims, 2, -1])
        return x


def get_norm(norm_type):
    def norm(c, norm_type):
        if norm_type == 'BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type == 'InstanceNorm':
            return nn.InstanceNorm2d(c, affine=True)
        elif 'GroupNorm' in norm_type:
            g = int(norm_type.replace('GroupNorm', ''))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()

    return partial(norm, norm_type=norm_type)


def get_act(act_type):
    if act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type[:3] == 'elu':
        alpha = float(act_type.replace('elu', ''))
        return nn.ELU(alpha)
    else:
        raise Exception


class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                norm(in_c),
                act,
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c),
                act,
                nn.Linear(f, f // bn, bias=False),
                norm(c),
                act,
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c),
                act,
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


class TFC_TDF_net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        norm = get_norm(norm_type=config.model.norm)
        act = get_act(act_type=config.model.act)

        self.num_target_instruments = len(prefer_target_instrument(config))
        self.num_subbands = config.model.num_subbands
        
        # CMX (Channel Multiplexing) configuration for memory efficiency
        # Inspired by MARS paper (arXiv:2509.26007)
        self.use_cmx = getattr(config.audio, 'use_cmx', False)
        self.cmx_reduction = getattr(config.audio, 'cmx_reduction', 2)

        dim_c = self.num_subbands * config.audio.num_channels * 2
        
        # Adjust channels if CMX is enabled (spatial reduction increases channels)
        if self.use_cmx:
            dim_c = dim_c * (self.cmx_reduction ** 2)
        
        n = config.model.num_scales
        scale = config.model.scale
        l = config.model.num_blocks_per_scale
        c = config.model.num_channels
        g = config.model.growth
        bn = config.model.bottleneck_factor
        
        # Adjust frequency dimension for CMX spatial reduction
        f = config.audio.dim_f // self.num_subbands
        if self.use_cmx:
            f = f // self.cmx_reduction

        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.encoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(c, c, l, f, bn, norm, act)
            block.downscale = Downscale(c, c + g, scale, norm, act)
            f = f // scale[1]
            c += g
            self.encoder_blocks.append(block)

        self.bottleneck_block = TFC_TDF(c, c, l, f, bn, norm, act)

        self.decoder_blocks = nn.ModuleList()
        for i in range(n):
            block = nn.Module()
            block.upscale = Upscale(c, c - g, scale, norm, act)
            f = f * scale[1]
            c -= g
            block.tfc_tdf = TFC_TDF(2 * c, c, l, f, bn, norm, act)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

        self.stft = STFT(config.audio)

    def cac2cws(self, x):
        """Enhanced with Channel Multiplexing (CMX) for memory efficiency.
        
        CMX reduces spatial dimensions by redistributing values across channels
        in a chessboard-like pattern, reducing memory by up to 75% while
        preserving all frequency information (inspired by MARS paper).
        """
        k = self.num_subbands
        b, c, f, t = x.shape
        
        # Original subband transformation
        x = x.reshape(b, c, k, f // k, t)
        x = x.reshape(b, c * k, f // k, t)
        
        # Apply Channel Multiplexing if enabled
        if self.use_cmx:
            return self._apply_cmx(x)
        
        return x
    
    def _apply_cmx(self, x):
        """Apply Channel Multiplexing transformation.
        
        Converts [B, C, F, T] -> [B, C*rf*rf, F//rf, T//rf]
        where rf is the reduction factor (default: 2)
        """
        B, C, F, T = x.shape
        rf = self.cmx_reduction
        
        # Ensure dimensions are divisible by reduction factor
        F_pad = (rf - F % rf) % rf
        T_pad = (rf - T % rf) % rf
        
        if F_pad > 0 or T_pad > 0:
            x = F.pad(x, (0, T_pad, 0, F_pad))
            _, _, F, T = x.shape
        
        # Chessboard-like reshaping: redistribute spatial info to channels
        x = x.view(B, C, F // rf, rf, T // rf, rf)
        x = x.permute(0, 1, 3, 5, 2, 4)  # [B, C, rf, rf, F//rf, T//rf]
        x = x.contiguous().view(B, C * rf * rf, F // rf, T // rf)
        
        return x

    def cws2cac(self, x):
        """Enhanced with Channel Multiplexing (CMX) reversal.
        
        Restores original spatial dimensions from channel-multiplexed format.
        """
        # Reverse Channel Multiplexing if enabled
        if self.use_cmx:
            x = self._reverse_cmx(x)
        
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c // k, k, f, t)
        x = x.reshape(b, c // k, f * k, t)
        return x
    
    def _reverse_cmx(self, x):
        """Reverse Channel Multiplexing transformation.
        
        Converts [B, C*rf*rf, F//rf, T//rf] -> [B, C, F, T]
        """
        B, C_mult, F_red, T_red = x.shape
        rf = self.cmx_reduction
        
        # Calculate original channel count
        C_orig = C_mult // (rf * rf)
        
        # Reverse the chessboard reshaping
        x = x.view(B, C_orig, rf, rf, F_red, T_red)
        x = x.permute(0, 1, 4, 2, 5, 3)  # [B, C, F_red, rf, T_red, rf]
        x = x.contiguous().view(B, C_orig, F_red * rf, T_red * rf)
        
        # Remove padding if needed (stored during forward pass)
        if hasattr(self, 'original_freq_dim') and hasattr(self, 'original_time_dim'):
            x = x[:, :, :self.original_freq_dim, :self.original_time_dim]
        
        return x

    def forward(self, x):

        x = self.stft(x)
        
        # Store original dimensions for CMX reversal
        if self.use_cmx:
            self.original_freq_dim = x.shape[2]
            self.original_time_dim = x.shape[3]

        mix = x = self.cac2cws(x)

        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1, -2)

        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)

        x = self.bottleneck_block(x)

        for block in self.decoder_blocks:
            x = block.upscale(x)
            x = torch.cat([x, encoder_outputs.pop()], 1)
            x = block.tfc_tdf(x)

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = self.cws2cac(x)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)

        x = self.stft.inverse(x)

        return x
