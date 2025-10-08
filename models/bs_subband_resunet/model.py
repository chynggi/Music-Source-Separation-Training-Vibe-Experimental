"""ByteDance Subband ResUNet integration for MSST.

This module reproduces the original ByteDance `ResUNet143_Subbandtime`
architecture so that official checkpoints can be loaded without key
remapping. A light wrapper adapts the expected input/output interface to
MSST while leaving the module hierarchy intact for state-dict
compatibility.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torchlibrosa.stft import ISTFT, STFT, magphase


# ---------------------------------------------------------------------------
# Helper initialisation routines (mirrors ByteDance implementation)
# ---------------------------------------------------------------------------


def init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.zero_()


def init_bn(bn: nn.Module) -> None:
    bn.bias.data.zero_()
    bn.weight.data.fill_(1.0)
    bn.running_mean.data.zero_()
    bn.running_var.data.fill_(1.0)


class Base:
    """Utility mixin providing STFT helpers (copied from ByteDance code)."""

    def spectrogram(self, input_tensor: Tensor, eps: float = 0.0) -> Tensor:
        real, imag = self.stft(input_tensor)
        return torch.clamp(real ** 2 + imag ** 2, eps, np.inf).sqrt()

    def spectrogram_phase(self, input_tensor: Tensor, eps: float = 0.0) -> Tuple[Tensor, Tensor, Tensor]:
        real, imag = self.stft(input_tensor)
        mag = torch.clamp(real ** 2 + imag ** 2, eps, np.inf).sqrt()
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(self, input_tensor: Tensor, eps: float = 1e-10) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, channels, segment_samples = input_tensor.shape
        reshaped = input_tensor.reshape(batch_size * channels, segment_samples)
        mag, cos, sin = self.spectrogram_phase(reshaped, eps=eps)
        _, _, time_steps, freq_bins = mag.shape
        mag = mag.reshape(batch_size, channels, time_steps, freq_bins)
        cos = cos.reshape(batch_size, channels, time_steps, freq_bins)
        sin = sin.reshape(batch_size, channels, time_steps, freq_bins)
        return mag, cos, sin


# ---------------------------------------------------------------------------
# PQMF filter coefficients (hard-coded to avoid runtime downloads)
# ---------------------------------------------------------------------------

_F_COEFFS_4_64 = torch.tensor(
    [
        [
            1.2016000e-04,
            2.1540001e-04,
            -7.2747999e-04,
            -9.9940000e-04,
            1.8986000e-04,
            6.3080003e-04,
            2.0138499e-03,
            2.0926401e-03,
            1.2170201e-03,
            3.3690000e-05,
            -1.1006720e-02,
            -1.2622080e-02,
            2.0831801e-03,
            1.2587880e-02,
            2.7822190e-02,
            2.6332231e-02,
            8.9065400e-03,
            -2.4143090e-02,
            -7.9117242e-02,
            -8.4420151e-02,
            -3.0127900e-03,
            6.6062481e-02,
            1.3925081e-01,
            1.1543633e-01,
            9.8405901e-03,
            -1.4687460e-01,
            -2.7471939e-01,
            -2.1870518e-01,
            6.0749108e-02,
            4.4897437e-01,
            8.6596245e-01,
            1.1074084e00,
            1.0593343e00,
            7.8901540e-01,
            3.7227273e-01,
            6.6088198e-03,
            -1.8432947e-01,
            -1.9918205e-01,
            -7.2726035e-02,
            5.6349292e-02,
            1.0554641e-01,
            9.2877132e-02,
            2.1509019e-02,
            -3.9335142e-02,
            -5.4853340e-02,
            -3.6801010e-02,
            -2.1953851e-02,
            3.4432523e-02,
            3.7022226e-02,
            4.1652988e-02,
            -2.2946733e-02,
            -1.0056511e-02,
            -1.3509110e-02,
            -1.9815340e-02,
            9.9413000e-03,
            1.2168000e-03,
            5.1465499e-03,
            6.9869302e-03,
            -4.5345400e-03,
            1.7000200e-03,
            -9.1366002e-04,
            -1.9887600e-03,
            8.8683998e-04,
            -9.7281999e-04,
        ],
        [
            -6.0213996e-04,
            -1.2343300e-03,
            4.0440303e-03,
            7.8320205e-03,
            -2.5253301e-03,
            -6.1796002e-03,
            -7.8350097e-03,
            -4.4878898e-03,
            -1.3449250e-02,
            -1.4807191e-02,
            7.2426667e-02,
            1.0333088e-01,
            -1.6189521e-02,
            -9.4837051e-02,
            -9.2517057e-02,
            -2.5319794e-02,
            -1.3537130e-01,
            -1.6771699e-02,
            4.3128762e-01,
            5.8174688e-01,
            -7.9391703e-02,
            -6.3318264e-01,
            -3.3203592e-01,
            5.2846943e-02,
            -2.0959438e-01,
            -6.2277729e-02,
            1.0115482e00,
            1.1899328e00,
            -5.9871402e-01,
            -1.8848170e00,
            -6.2045610e-01,
            1.2526729e00,
            1.0561498e00,
            -3.3577168e-01,
            -5.4399959e-01,
            1.7063093e-01,
            1.6379479e-01,
            -2.7068179e-01,
            -2.0175444e-01,
            1.3065769e-01,
            1.5291695e-01,
            -9.6974895e-03,
            -4.5590999e-04,
            2.0754248e-02,
            -5.6368876e-02,
            -2.9587961e-02,
            2.4557671e-02,
            1.0229872e-02,
            2.1193271e-02,
            2.2099333e-02,
            -1.0587248e-02,
            3.9778899e-03,
            -1.3833099e-02,
            -5.0008799e-03,
            9.2323291e-03,
            -7.5579099e-03,
            2.2850701e-03,
            1.7640100e-03,
            -1.4589801e-03,
            1.6064501e-03,
            -5.2895001e-04,
            3.6682998e-04,
            9.2860003e-05,
            -4.1076000e-04,
        ],
        [
            1.4906999e-03,
            1.5903999e-03,
            -3.4547301e-03,
            -6.1278001e-03,
            6.5463002e-04,
            6.8364200e-03,
            -6.9242798e-03,
            -3.0780000e-05,
            2.9528849e-02,
            1.1028320e-02,
            -2.8197610e-02,
            -6.3299253e-02,
            3.2237889e-02,
            7.5474777e-02,
            -1.2175762e-01,
            1.5396972e-02,
            1.4482770e-01,
            6.9702837e-02,
            -1.1661034e-01,
            -2.9735026e-01,
            3.9252260e-01,
            2.7420175e-01,
            -6.6034969e-01,
            1.2630118e-01,
            4.8068094e-01,
            -4.1543750e-02,
            -3.1812439e-01,
            -5.7511899e-01,
            1.0670862e00,
            5.1732633e-01,
            -1.9447103e00,
            7.5450902e-01,
            1.5247788e00,
            -1.6046674e00,
            -1.5692398e-01,
            9.2026078e-01,
            -3.4920573e-01,
            1.2959373e-01,
            -2.7688542e-01,
            -8.1930958e-02,
            6.7261783e-01,
            -4.3659216e-01,
            -2.3071670e-01,
            3.9404744e-01,
            -1.3999663e-01,
            8.7253501e-03,
            1.1648208e-02,
            -8.0371189e-02,
            1.2547614e-01,
            -5.5095432e-02,
            -5.1854544e-02,
            7.2334620e-02,
            -2.0820010e-02,
            2.2462000e-03,
            7.8393498e-03,
            -1.4471670e-02,
            8.9179602e-03,
            -3.2918199e-03,
            -3.5268299e-03,
            5.8727297e-03,
            -1.4697800e-03,
            3.4132000e-04,
            8.0511996e-04,
            -1.0137400e-03,
        ],
        [
            2.2073001e-04,
            4.9844000e-04,
            6.0270001e-05,
            -5.0322999e-04,
            -1.1756300e-03,
            8.8553001e-04,
            -1.1624200e-03,
            -1.3839900e-03,
            4.9094598e-03,
            1.6615300e-03,
            1.0116170e-02,
            -4.8197500e-03,
            -9.6880998e-04,
            9.8483292e-03,
            -1.9691970e-02,
            1.0180251e-02,
            -1.1642600e-02,
            2.5523767e-02,
            3.4923143e-02,
            -5.7524444e-02,
            9.4718896e-03,
            2.8706033e-02,
            -6.6657611e-02,
            5.7507887e-02,
            4.0627376e-02,
            -1.5222552e-01,
            1.9542965e-01,
            -8.0581849e-02,
            -2.2963196e-01,
            6.1597300e-01,
            -9.3835059e-01,
            1.0798270e00,
            -9.5289395e-01,
            6.3098538e-01,
            -2.4887024e-01,
            -8.4369909e-02,
            2.0859116e-01,
            -1.8158396e-01,
            8.1349421e-02,
            8.1363052e-02,
            -1.2765093e-01,
            9.7520683e-02,
            -4.8372481e-02,
            -4.7082931e-02,
            7.5083477e-02,
            -4.5695953e-02,
            1.0232290e-02,
            2.3043959e-02,
            -4.6575242e-02,
            -1.5129360e-02,
            1.2239219e-02,
            -2.0576501e-03,
            2.2086079e-02,
            5.8653298e-03,
            -1.2352610e-02,
            7.0312502e-03,
            -5.1710600e-03,
            -4.5549902e-03,
            4.2506898e-03,
            -2.2150700e-03,
            1.8095300e-03,
            7.6354002e-04,
            -1.0199800e-03,
            1.6767400e-03,
        ],
    ],
    dtype=torch.float32,
)

_H_COEFFS_4_64 = torch.tensor(
    [
        [
            3.4589999e-05,
            -5.0830001e-05,
            -1.6240999e-04,
            1.1040000e-04,
            -2.6799000e-04,
            2.5979000e-04,
            5.7608002e-04,
            -2.0667000e-04,
            9.4400003e-05,
            3.0540000e-04,
            -2.4475300e-03,
            1.5857501e-03,
            -1.0230999e-04,
            -2.0640101e-03,
            6.6916998e-04,
            -8.1722001e-04,
            -7.6083998e-04,
            6.9974304e-03,
            1.7102700e-03,
            -4.7839599e-03,
            -4.8016202e-03,
            1.9135699e-03,
            1.3610280e-02,
            1.3087579e-02,
            -2.0252499e-03,
            -2.1588920e-02,
            -3.7504160e-02,
            -2.7188018e-02,
            2.3818721e-02,
            1.0357373e-01,
            1.8805431e-01,
            2.4401928e-01,
            2.4945712e-01,
            2.0157565e-01,
            1.1827825e-01,
            3.0963564e-02,
            -2.9818310e-02,
            -5.1137618e-02,
            -3.6476778e-02,
            -9.5113497e-03,
            1.1598000e-02,
            2.1515790e-02,
            9.7505795e-03,
            -1.1109000e-04,
            -2.9707702e-03,
            -7.1890796e-03,
            -2.0887900e-03,
            -3.7783301e-03,
            8.6880998e-04,
            5.5230002e-05,
            2.1323399e-03,
            1.5631000e-03,
            -7.0050999e-04,
            6.8088004e-04,
            -5.3789001e-04,
            5.8063999e-04,
            3.4533001e-04,
            -4.4852000e-04,
            -7.4570004e-05,
            -4.0008002e-04,
            -2.1575000e-04,
            2.3771999e-04,
            4.1340002e-05,
            2.5205000e-04,
        ],
        [
            1.1590000e-05,
            3.3700000e-05,
            -1.0164999e-04,
            -9.5229997e-05,
            2.2840000e-05,
            1.3512000e-04,
            1.9132000e-04,
            -5.2360002e-05,
            -3.0717000e-04,
            -2.3700999e-04,
            4.6867000e-04,
            1.5332400e-03,
            -7.2446004e-04,
            -2.2755299e-03,
            -1.0810799e-03,
            1.6757900e-03,
            3.5836502e-03,
            1.6235000e-03,
            -4.8013303e-03,
            -1.2851831e-02,
            7.6923000e-03,
            1.5713950e-02,
            1.1636975e-02,
            -1.1014560e-02,
            -2.4850450e-02,
            2.3220001e-05,
            3.4618694e-02,
            2.3489420e-02,
            -3.4319491e-02,
            -7.1182358e-02,
            -1.1724000e-02,
            1.0758500e-01,
            1.1687053e-01,
            -4.3740445e-02,
            -1.7266071e-01,
            -8.9201018e-02,
            9.6078431e-02,
            1.3703905e-01,
            2.4063627e-02,
            -5.7279918e-02,
            -3.8627480e-02,
            -1.5944671e-02,
            -1.8383760e-02,
            3.9579603e-03,
            2.8517745e-02,
            2.1278611e-02,
            -6.4676003e-03,
            -1.3184219e-02,
            -9.5117995e-04,
            -4.1034700e-03,
            -1.6863599e-03,
            3.0396201e-03,
            1.2067000e-04,
            6.1845499e-03,
            -3.3893000e-04,
            1.9054700e-03,
            7.3018001e-04,
            -1.6362099e-03,
            1.8764000e-04,
            -1.0183200e-03,
            -4.8502998e-04,
            6.0494001e-04,
            2.2940001e-04,
            7.3201998e-04,
        ],
        [
            2.3000001e-07,
            -6.7330001e-05,
            5.8239998e-05,
            8.1980003e-05,
            -1.5407999e-04,
            -5.7820002e-05,
            2.7134001e-04,
            1.3499999e-06,
            -4.3606998e-04,
            1.1236600e-03,
            -8.7101999e-04,
            -1.1893600e-03,
            2.4603900e-03,
            -1.6755000e-04,
            -3.4926499e-03,
            2.1748000e-03,
            3.7574699e-03,
            -7.4296398e-03,
            7.7237804e-03,
            2.4588600e-03,
            -1.2358471e-02,
            -1.2744500e-03,
            1.4187099e-02,
            1.4350399e-03,
            -3.4985110e-02,
            3.8571871e-02,
            1.0053199e-02,
            -7.8366348e-02,
            8.1464528e-02,
            1.3556240e-02,
            -1.3283467e-01,
            1.0929701e-01,
            5.6927735e-02,
            -1.5735020e-01,
            5.6374734e-02,
            9.6444988e-02,
            -9.0649658e-02,
            -2.4442958e-02,
            5.6858515e-02,
            1.6216500e-03,
            -2.5199495e-02,
            2.2443001e-04,
            1.5751212e-02,
            -6.8592901e-03,
            1.2368499e-03,
            8.2321700e-03,
            -1.4693339e-02,
            3.4759199e-03,
            6.9051401e-03,
            -4.4752201e-03,
            -1.8905799e-03,
            -3.7801899e-03,
            -1.2830699e-03,
            -6.4880001e-05,
            3.6246999e-04,
            -1.5706999e-04,
            2.5987000e-04,
            -1.6799999e-06,
            8.1650003e-05,
            -3.2516999e-04,
            6.2199999e-06,
            1.8467000e-04,
            -4.5100001e-06,
            1.5370999e-04,
        ],
        [
            -1.0465000e-04,
            1.0313000e-04,
            7.4550003e-05,
            -4.6000000e-05,
            -8.1600001e-06,
            3.4490000e-05,
            -4.4137999e-04,
            2.6126000e-04,
            1.0769200e-03,
            -1.8234300e-03,
            -3.5026999e-04,
            1.4506600e-03,
            -1.8041600e-03,
            3.0250500e-03,
            3.7747799e-03,
            -7.8941700e-03,
            5.0851000e-04,
            5.3935903e-03,
            -1.0815910e-02,
            1.7413520e-02,
            -6.0220898e-03,
            -1.1377129e-02,
            2.9177152e-02,
            -2.8974622e-02,
            1.2170702e-02,
            2.0280403e-02,
            -4.7753444e-02,
            5.1206052e-02,
            -1.1763050e-02,
            -6.5598199e-02,
            1.5958942e-01,
            -2.3232079e-01,
            2.5981686e-01,
            -2.3161231e-01,
            1.5466312e-01,
            -6.5897993e-02,
            -9.1985798e-03,
            4.7071043e-02,
            -4.3501672e-02,
            2.1268048e-02,
            5.5377497e-03,
            -2.1299613e-02,
            1.5165438e-02,
            -3.6892599e-03,
            -8.2693602e-03,
            1.2360390e-02,
            -8.1207099e-03,
            -3.4388401e-03,
            3.3812800e-03,
            -1.1271879e-02,
            2.1100300e-03,
            -3.8725799e-03,
            -3.1986200e-03,
            4.3504801e-03,
            -4.0603002e-04,
            3.5478602e-03,
            1.3082800e-03,
            -1.5785200e-03,
            -2.4498001e-04,
            -1.5965800e-03,
            -2.6116999e-04,
            4.4277999e-04,
            6.4470000e-05,
            4.2515001e-04,
        ],
    ],
    dtype=torch.float32,
)


class PQMF(nn.Module):
    """PQMF with frozen coefficients (identical to ByteDance implementation)."""

    def __init__(self, n_subbands: int = 4, taps: int = 64) -> None:
        super().__init__()
        if (n_subbands, taps) != (4, 64):
            raise ValueError("Only 4 subbands with 64 taps are supported.")

        self.N = n_subbands
        self.M = taps
        self.pad_samples = taps

        self.ana_pad = nn.ConstantPad1d((taps - n_subbands, 0), 0)
        self.ana_conv_filter = nn.Conv1d(1, out_channels=n_subbands, kernel_size=taps, stride=n_subbands, bias=False)

        ana_weights = _F_COEFFS_4_64.clone() / float(n_subbands)
        ana_weights = torch.flip(ana_weights, dims=[1]).view(n_subbands, 1, taps)
        with torch.no_grad():
            self.ana_conv_filter.weight.copy_(ana_weights)

        self.syn_pad = nn.ConstantPad1d((0, taps // n_subbands - 1), 0)
        self.syn_conv_filter = nn.Conv1d(
            n_subbands,
            out_channels=n_subbands,
            kernel_size=taps // n_subbands,
            stride=1,
            bias=False,
        )

        syn_weights = _H_COEFFS_4_64.clone()
        syn_weights = syn_weights.view(n_subbands, taps // n_subbands, n_subbands)
        syn_weights = syn_weights.transpose(0, 1) * float(n_subbands)
        syn_weights = syn_weights.flip(0).transpose(0, 2).contiguous()
        with torch.no_grad():
            self.syn_conv_filter.weight.copy_(syn_weights)

        for param in self.parameters():
            param.requires_grad = False

    def __analysis_channel(self, inputs: Tensor) -> Tensor:
        return self.ana_conv_filter(self.ana_pad(inputs))

    def __synthesis_channel(self, inputs: Tensor) -> Tensor:
        ret = self.syn_conv_filter(self.syn_pad(inputs)).permute(0, 2, 1)
        return torch.reshape(ret, (ret.shape[0], 1, -1))

    def analysis(self, inputs: Tensor) -> Tensor:
        padded = F.pad(inputs, (0, self.pad_samples))
        ret = []
        for ch in range(padded.size(1)):
            ret.append(self.__analysis_channel(padded[:, ch : ch + 1, :]))
        return torch.cat(ret, dim=1)

    def synthesis(self, data: Tensor) -> Tensor:
        ret = []
        for ch in range(0, data.size(1), self.N):
            ret.append(self.__synthesis_channel(data[:, ch : ch + self.N, :]))
        ret = torch.cat(ret, dim=1)
        return ret[..., : -self.pad_samples]

    def forward(self, inputs: Tensor) -> Tensor:
        return self.analysis(inputs)


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], momentum: float) -> None:
        super().__init__()
        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.shortcut = None
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self) -> None:
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)
        if self.is_shortcut:
            init_layer(self.shortcut)  # type: ignore[arg-type]

    def forward(self, input_tensor: Tensor) -> Tensor:
        x = self.conv1(F.leaky_relu_(self.bn1(input_tensor), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))
        if self.is_shortcut and self.shortcut is not None:
            return self.shortcut(input_tensor) + x
        return input_tensor + x


class EncoderBlockRes4B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        downsample: Tuple[int, int],
        momentum: float,
    ) -> None:
        super().__init__()
        self.conv_block1 = ConvBlockRes(in_channels, out_channels, kernel_size, momentum)
        self.conv_block2 = ConvBlockRes(out_channels, out_channels, kernel_size, momentum)
        self.conv_block3 = ConvBlockRes(out_channels, out_channels, kernel_size, momentum)
        self.conv_block4 = ConvBlockRes(out_channels, out_channels, kernel_size, momentum)
        self.downsample = downsample

    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        encoder = self.conv_block1(input_tensor)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes4B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        upsample: Tuple[int, int],
        momentum: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample,
            stride=upsample,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.conv_block2 = ConvBlockRes(out_channels * 2, out_channels, kernel_size, momentum)
        self.conv_block3 = ConvBlockRes(out_channels, out_channels, kernel_size, momentum)
        self.conv_block4 = ConvBlockRes(out_channels, out_channels, kernel_size, momentum)
        self.conv_block5 = ConvBlockRes(out_channels, out_channels, kernel_size, momentum)
        self.init_weights()

    def init_weights(self) -> None:
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(self, input_tensor: Tensor, concat_tensor: Tensor) -> Tensor:
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class ResUNet143_Subbandtime(nn.Module, Base):
    def __init__(self, input_channels: int, output_channels: int, target_sources_num: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.target_sources_num = target_sources_num

        window_size = 512  # 2048 // 4
        hop_size = 110  # 441 // 4
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.subbands_num = 4
        self.K = 4  # outputs: |M|, cos∠M, sin∠M, Q
        self.time_downsample_ratio = 2 ** 5

        self.pqmf = PQMF(n_subbands=self.subbands_num, taps=64)

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes4B(
            in_channels=self.input_channels * self.subbands_num,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
        )
        self.encoder_block2 = EncoderBlockRes4B(32, 64, (3, 3), (2, 2), momentum)
        self.encoder_block3 = EncoderBlockRes4B(64, 128, (3, 3), (2, 2), momentum)
        self.encoder_block4 = EncoderBlockRes4B(128, 256, (3, 3), (2, 2), momentum)
        self.encoder_block5 = EncoderBlockRes4B(256, 384, (3, 3), (2, 2), momentum)
        self.encoder_block6 = EncoderBlockRes4B(384, 384, (3, 3), (1, 2), momentum)

        self.conv_block7a = EncoderBlockRes4B(384, 384, (3, 3), (1, 1), momentum)
        self.conv_block7b = EncoderBlockRes4B(384, 384, (3, 3), (1, 1), momentum)
        self.conv_block7c = EncoderBlockRes4B(384, 384, (3, 3), (1, 1), momentum)
        self.conv_block7d = EncoderBlockRes4B(384, 384, (3, 3), (1, 1), momentum)

        self.decoder_block1 = DecoderBlockRes4B(384, 384, (3, 3), (1, 2), momentum)
        self.decoder_block2 = DecoderBlockRes4B(384, 384, (3, 3), (2, 2), momentum)
        self.decoder_block3 = DecoderBlockRes4B(384, 256, (3, 3), (2, 2), momentum)
        self.decoder_block4 = DecoderBlockRes4B(256, 128, (3, 3), (2, 2), momentum)
        self.decoder_block5 = DecoderBlockRes4B(128, 64, (3, 3), (2, 2), momentum)
        self.decoder_block6 = DecoderBlockRes4B(64, 32, (3, 3), (2, 2), momentum)

        self.after_conv_block1 = EncoderBlockRes4B(32, 32, (3, 3), (1, 1), momentum)
        self.after_conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=self.target_sources_num * self.output_channels * self.K * self.subbands_num,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        self.init_weights()

    def init_weights(self) -> None:
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def feature_maps_to_wav(
        self,
        input_tensor: Tensor,
        sp: Tensor,
        sin_in: Tensor,
        cos_in: Tensor,
        audio_length: int,
    ) -> Tensor:
        batch_size, _, time_steps, freq_bins = input_tensor.shape
        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.output_channels,
            self.K,
            time_steps,
            freq_bins,
        )

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)

        out_cos = cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        out_sin = sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin

        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        wav = self.istft(out_real, out_imag, audio_length)
        wav = wav.reshape(batch_size, self.target_sources_num * self.output_channels, audio_length)
        return wav

    def forward(self, input_dict: Union[Tensor, dict]) -> dict:
        if isinstance(input_dict, Tensor):
            mixtures = input_dict
        else:
            mixtures = input_dict["waveform"]

        subband_x = self.pqmf.analysis(mixtures)
        mag, cos_in, sin_in = self.wav_to_spectrogram_phase(subband_x)

        x = mag.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x[..., 0 : x.shape[-1] - 1]

        x1_pool, x1 = self.encoder_block1(x)
        x2_pool, x2 = self.encoder_block2(x1_pool)
        x3_pool, x3 = self.encoder_block3(x2_pool)
        x4_pool, x4 = self.encoder_block4(x3_pool)
        x5_pool, x5 = self.encoder_block5(x4_pool)
        x6_pool, x6 = self.encoder_block6(x5_pool)

        x_center, _ = self.conv_block7a(x6_pool)
        x_center, _ = self.conv_block7b(x_center)
        x_center, _ = self.conv_block7c(x_center)
        x_center, _ = self.conv_block7d(x_center)

        x7 = self.decoder_block1(x_center, x6)
        x8 = self.decoder_block2(x7, x5)
        x9 = self.decoder_block3(x8, x4)
        x10 = self.decoder_block4(x9, x3)
        x11 = self.decoder_block5(x10, x2)
        x12 = self.decoder_block6(x11, x1)
        x, _ = self.after_conv_block1(x12)

        x = self.after_conv2(x)
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = subband_x.shape[2]
        separated_subband_audio = torch.stack(
            [
                self.feature_maps_to_wav(
                    input_tensor=x[:, j :: self.subbands_num, :, :],
                    sp=mag[:, j :: self.subbands_num, :, :],
                    sin_in=sin_in[:, j :: self.subbands_num, :, :],
                    cos_in=cos_in[:, j :: self.subbands_num, :, :],
                    audio_length=audio_length,
                )
                for j in range(self.subbands_num)
            ],
            dim=2,
        )

        separated_subband_audio = separated_subband_audio.reshape(
            separated_subband_audio.shape[0],
            self.target_sources_num * self.output_channels * self.subbands_num,
            audio_length,
        )
        separated_audio = self.pqmf.synthesis(separated_subband_audio)
        output_dict = {"waveform": separated_audio}
        return output_dict


class BandSplitSubbandResUNet(ResUNet143_Subbandtime):
    """Wrapper exposing the ByteDance model with an MSST-friendly forward."""

    def __init__(self, input_channels: int, output_channels: int, num_sources: int) -> None:
        super().__init__(input_channels=input_channels, output_channels=output_channels, target_sources_num=num_sources)

    def forward(self, mixture: Tensor) -> Tensor:
        output_dict = super().forward({"waveform": mixture})
        waveform = output_dict["waveform"]
        batch = mixture.shape[0]
        return waveform.view(batch, self.target_sources_num, self.output_channels, -1)
