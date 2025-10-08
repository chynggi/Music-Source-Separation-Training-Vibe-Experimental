"""ByteDance-inspired subband ResUNet separator for MSST.

The implementation follows the ResUNet143_Subbandtime architecture released by
ByteDance, adapted to the waveform-to-waveform interface expected by MSST. The
network operates on PQMF subbands, estimates complex masks in the STFT domain,
and reconstructs full-band waveforms via inverse PQMF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torchlibrosa.stft import ISTFT, STFT, magphase

_LEAKY_SLOPE = 0.01
_EPS = 1.0e-10

_F_COEFFS_4_64 = torch.tensor(
    [
        [
            0.00012016,
            0.0002154,
            -0.00072748,
            -0.0009994,
            0.00018986,
            0.0006308,
            0.00201385,
            0.00209264,
            0.00121702,
            0.00003369,
            -0.01100672,
            -0.01262208,
            0.00208318,
            0.01258788,
            0.02782219,
            0.02633223,
            0.00890654,
            -0.02414309,
            -0.07911724,
            -0.08442015,
            -0.00301279,
            0.06606248,
            0.13925081,
            0.11543633,
            0.00984059,
            -0.1468746,
            -0.2747194,
            -0.21870518,
            0.06074911,
            0.44897437,
            0.86596245,
            1.1074084,
            1.0593343,
            0.7890154,
            0.37227273,
            0.00660882,
            -0.18432947,
            -0.19918205,
            -0.07272603,
            0.05634929,
            0.10554641,
            0.09287713,
            0.02150902,
            -0.03933514,
            -0.05485334,
            -0.03680101,
            -0.02195385,
            0.03443252,
            0.03702223,
            0.04165299,
            -0.02294673,
            -0.01005651,
            -0.01350911,
            -0.01981534,
            0.0099413,
            0.0012168,
            0.00514655,
            0.00698693,
            -0.00453454,
            0.00170002,
            -0.00091366,
            -0.00198876,
            0.00088684,
            -0.00097282,
        ],
        [
            -0.00060214,
            -0.00123433,
            0.00404403,
            0.00783202,
            -0.00252533,
            -0.0061796,
            -0.00783501,
            -0.00448789,
            -0.01344925,
            -0.01480719,
            0.07242667,
            0.10333088,
            -0.01618952,
            -0.09483705,
            -0.09251706,
            -0.02531979,
            -0.1353713,
            -0.0167717,
            0.43128762,
            0.5817469,
            -0.0793917,
            -0.63318264,
            -0.33203593,
            0.05284694,
            -0.20959438,
            -0.06227773,
            1.0115482,
            1.1899328,
            -0.5987141,
            -1.884817,
            -0.6204561,
            1.2526729,
            1.0561498,
            -0.33577168,
            -0.5439996,
            0.17063093,
            0.16379479,
            -0.2706818,
            -0.20175444,
            0.13065769,
            0.15291695,
            -0.00969749,
            -0.00045591,
            0.02075425,
            -0.05636888,
            -0.02958796,
            0.02455767,
            0.01022987,
            0.02119327,
            0.02209933,
            -0.01058725,
            0.00397789,
            -0.0138331,
            -0.00500088,
            0.00923233,
            -0.00755791,
            0.00228507,
            0.00176401,
            -0.00145898,
            0.00160645,
            -0.00052895,
            0.00036683,
            0.00009286,
            -0.00041076,
        ],
        [
            0.0014907,
            0.0015904,
            -0.00345473,
            -0.0061278,
            0.00065463,
            0.00683642,
            -0.00692428,
            -0.00003078,
            0.02952885,
            0.01102832,
            -0.02819761,
            -0.06329925,
            0.03223789,
            0.07547478,
            -0.12175762,
            0.01539697,
            0.1448277,
            0.06970284,
            -0.11661034,
            -0.29735026,
            0.3925226,
            0.27420175,
            -0.66034967,
            0.12630118,
            0.48068094,
            -0.04154375,
            -0.3181244,
            -0.575119,
            1.0670862,
            0.51732635,
            -1.9447103,
            0.754509,
            1.5247788,
            -1.6046674,
            -0.15692398,
            0.9202608,
            -0.34920573,
            0.12959373,
            -0.27688542,
            -0.08193096,
            0.6726178,
            -0.43659216,
            -0.2307167,
            0.39404744,
            -0.13999663,
            0.00872535,
            0.01164821,
            -0.08037119,
            0.12547614,
            -0.05509543,
            -0.05185454,
            0.07233462,
            -0.02082001,
            0.0022462,
            0.00783935,
            -0.01447167,
            0.00891796,
            -0.00329182,
            -0.00352683,
            0.00587273,
            -0.00146978,
            0.00034132,
            0.00080512,
            -0.00101374,
        ],
        [
            0.00022073,
            0.00049844,
            0.00006027,
            -0.00050323,
            -0.00117563,
            0.00088553,
            -0.00116242,
            -0.00138399,
            0.00490946,
            0.00166153,
            0.01011617,
            -0.00481975,
            -0.00096881,
            0.00984833,
            -0.01969197,
            0.01018025,
            -0.0116426,
            0.02552377,
            0.03492314,
            -0.05752444,
            0.00947189,
            0.02870603,
            -0.06665761,
            0.05750789,
            0.04062738,
            -0.15222552,
            0.19542965,
            -0.08058185,
            -0.22963196,
            0.615973,
            -0.9383506,
            1.079827,
            -0.952894,
            0.6309854,
            -0.24887024,
            -0.08436991,
            0.20859116,
            -0.18158396,
            0.08134942,
            0.08136305,
            -0.12765093,
            0.09752068,
            -0.04837248,
            -0.04708293,
            0.07508348,
            -0.04569595,
            0.01023229,
            0.02304396,
            -0.04657524,
            -0.01512936,
            0.01223922,
            -0.00205765,
            0.02208608,
            0.00586533,
            -0.01235261,
            0.00703125,
            -0.00517106,
            -0.00455499,
            0.00425069,
            -0.00221507,
            0.00180953,
            0.00076354,
            -0.00101998,
            0.00167674,
        ],
    ],
    dtype=torch.float32,
)

_H_COEFFS_4_64 = torch.tensor(
    [
        [
            0.00003459,
            -0.00005083,
            -0.00016241,
            0.0001104,
            -0.00026799,
            0.00025979,
            0.00057608,
            -0.00020667,
            0.0000944,
            0.0003054,
            -0.00244753,
            0.00158575,
            -0.00010231,
            -0.00206401,
            0.00066917,
            -0.00081722,
            -0.00076084,
            0.00699743,
            0.00171027,
            -0.00478396,
            -0.00480162,
            0.00191357,
            0.01361028,
            0.01308758,
            -0.00202525,
            -0.02158892,
            -0.03750416,
            -0.02718802,
            0.02381872,
            0.10357373,
            0.18805431,
            0.24401927,
            0.24945712,
            0.20157565,
            0.11827825,
            0.03096356,
            -0.02981831,
            -0.05113762,
            -0.03647678,
            -0.00951135,
            0.011598,
            0.02151579,
            0.00975058,
            -0.00011109,
            -0.00297077,
            -0.00718908,
            -0.00208879,
            -0.00377833,
            0.00086881,
            0.00005523,
            0.00213234,
            0.0015631,
            -0.00070051,
            0.00068088,
            -0.00053789,
            0.00058064,
            0.00034533,
            -0.00044852,
            -0.00007457,
            -0.00040008,
            -0.00021575,
            0.00023772,
            0.00004134,
            0.00025205,
        ],
        [
            0.00001159,
            0.0000337,
            -0.00010165,
            -0.00009523,
            0.00002284,
            0.00013512,
            0.00019132,
            -0.00005236,
            -0.00030717,
            -0.00023701,
            0.00046867,
            0.00153324,
            -0.00072446,
            -0.00227553,
            -0.00108108,
            0.00167579,
            0.00358365,
            0.0016235,
            -0.00480133,
            -0.01285183,
            0.0076923,
            0.01571395,
            0.01163697,
            -0.01101456,
            -0.02485045,
            0.00002322,
            0.03461869,
            0.02348942,
            -0.03431949,
            -0.07118236,
            -0.011724,
            0.107585,
            0.11687053,
            -0.04374044,
            -0.17266071,
            -0.08920102,
            0.09607843,
            0.13703905,
            0.02406363,
            -0.05727992,
            -0.03862748,
            -0.01594467,
            -0.01838376,
            0.00395796,
            0.02851775,
            0.02127861,
            -0.0064676,
            -0.01318422,
            -0.00095118,
            -0.00410347,
            -0.00168636,
            0.00303962,
            0.00012067,
            0.00618455,
            -0.00033893,
            0.00190547,
            0.00073018,
            -0.00163621,
            0.00018764,
            -0.00101832,
            -0.00048503,
            0.00060494,
            0.0002294,
            0.00073202,
        ],
        [
            0.00000023,
            -0.00006733,
            0.00005824,
            0.00008198,
            -0.00015408,
            -0.00005782,
            0.00027134,
            0.00000135,
            -0.00043607,
            0.00112366,
            -0.00087102,
            -0.00118936,
            0.00246039,
            -0.00016755,
            -0.00349265,
            0.0021748,
            0.00375747,
            -0.00742964,
            0.00772378,
            0.00245886,
            -0.01235847,
            -0.00127445,
            0.0141871,
            0.00143504,
            -0.03498511,
            0.03857187,
            0.0100532,
            -0.07836635,
            0.08146453,
            0.01355624,
            -0.13283467,
            0.10929701,
            0.05692774,
            -0.1573502,
            0.05637473,
            0.09644499,
            -0.09064966,
            -0.02444296,
            0.05685851,
            0.00162165,
            -0.0251995,
            0.00022443,
            0.01575121,
            -0.00685929,
            0.00123685,
            0.00823217,
            -0.01469334,
            0.00347592,
            0.00690514,
            -0.00447522,
            -0.00189058,
            -0.00378019,
            -0.00128307,
            -0.00006488,
            0.00036247,
            -0.00015707,
            0.00025987,
            -0.00000168,
            0.00008165,
            -0.00032517,
            0.00000622,
            0.00018467,
            -0.00000451,
            0.00015371,
        ],
        [
            -0.00010465,
            0.00010313,
            0.00007455,
            -0.000046,
            -0.00000816,
            0.00003449,
            -0.00044138,
            0.00026126,
            0.00107692,
            -0.00182343,
            -0.00035027,
            0.00145066,
            -0.00180416,
            0.00302505,
            0.00377478,
            -0.00789417,
            0.00050851,
            0.00539359,
            -0.01081591,
            0.01741352,
            -0.00602209,
            -0.01137713,
            0.02917715,
            -0.02897462,
            0.0121707,
            0.02028041,
            -0.04775344,
            0.05120605,
            -0.01176305,
            -0.0655982,
            0.15958942,
            -0.23232079,
            0.25981686,
            -0.23161231,
            0.15466312,
            -0.06589799,
            -0.00919858,
            0.04707104,
            -0.04350167,
            0.02126805,
            0.00553775,
            -0.02129961,
            0.01516544,
            -0.00368926,
            -0.00826936,
            0.01236039,
            -0.00812071,
            -0.00343884,
            0.00338128,
            -0.01127188,
            0.00211003,
            -0.00387258,
            -0.00319862,
            0.00435048,
            -0.00040603,
            0.00354786,
            0.00130828,
            -0.00157852,
            -0.00024498,
            -0.00159658,
            -0.00026117,
            0.00044278,
            0.00006447,
            0.00042515,
        ],
    ],
    dtype=torch.float32,
)


class _ConvBlockRes(nn.Module):
    """Residual 2D convolutional block used across the network."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int]) -> None:
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.shortcut: nn.Module | None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
        else:
            self.shortcut = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn1.bias.data.zero_()
        self.bn1.weight.data.fill_(1.0)
        self.bn2.bias.data.zero_()
        self.bn2.weight.data.fill_(1.0)
        if self.shortcut is not None:
            nn.init.xavier_uniform_(self.shortcut.weight)
            if self.shortcut.bias is not None:
                self.shortcut.bias.data.zero_()

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        out = self.conv1(F.leaky_relu(self.bn1(inputs), negative_slope=_LEAKY_SLOPE))
        out = self.conv2(F.leaky_relu(self.bn2(out), negative_slope=_LEAKY_SLOPE))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return residual + out


class _EncoderBlock(nn.Module):
    """Encoder block composed of four residual units followed by pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        downsample: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.block1 = _ConvBlockRes(in_channels, out_channels, kernel_size)
        self.block2 = _ConvBlockRes(out_channels, out_channels, kernel_size)
        self.block3 = _ConvBlockRes(out_channels, out_channels, kernel_size)
        self.block4 = _ConvBlockRes(out_channels, out_channels, kernel_size)
        self.downsample = downsample

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        pooled = F.avg_pool2d(x, kernel_size=self.downsample)
        return pooled, x


class _DecoderBlock(nn.Module):
    """Decoder block with transposed convolution and four residual units."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        upsample: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample,
            stride=upsample,
            bias=False,
        )
        self.block2 = _ConvBlockRes(out_channels * 2, out_channels, kernel_size)
        self.block3 = _ConvBlockRes(out_channels, out_channels, kernel_size)
        self.block4 = _ConvBlockRes(out_channels, out_channels, kernel_size)
        self.block5 = _ConvBlockRes(out_channels, out_channels, kernel_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.deconv.weight)
        self.bn.bias.data.zero_()
        self.bn.weight.data.fill_(1.0)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        x = self.deconv(F.relu(self.bn(inputs)))
        x = torch.cat([x, skip], dim=1)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


@dataclass
class _SpectrogramInfo:
    magnitude: Tensor
    cosine: Tensor
    sine: Tensor


class _PQMF(nn.Module):
    """Polyphase Quadrature Mirror Filter bank with fixed coefficients."""

    def __init__(self, subbands: int = 4, taps: int = 64) -> None:
        super().__init__()
        if (subbands, taps) != (4, 64):
            raise ValueError("Only 4 subbands with 64 taps are supported.")

        self.subbands = subbands
        self.pad_samples = taps

        analysis = _F_COEFFS_4_64.clone() / float(subbands)
        analysis = torch.flip(analysis, dims=[1]).view(subbands, 1, taps)
        self.analysis_pad = nn.ConstantPad1d((taps - subbands, 0), 0.0)
        self.analysis_filter = nn.Conv1d(1, subbands, kernel_size=taps, stride=subbands, bias=False)
        with torch.no_grad():
            self.analysis_filter.weight.copy_(analysis)

        synthesis = _H_COEFFS_4_64.clone()
        synthesis = synthesis.view(subbands, taps // subbands, subbands)
        synthesis = synthesis.transpose(0, 1) * float(subbands)
        synthesis = synthesis.flip(0).transpose(0, 2).contiguous()
        self.synthesis_pad = nn.ConstantPad1d((0, taps // subbands - 1), 0.0)
        self.synthesis_filter = nn.Conv1d(
            subbands,
            subbands,
            kernel_size=taps // subbands,
            stride=1,
            bias=False,
        )
        with torch.no_grad():
            self.synthesis_filter.weight.copy_(synthesis)

        for parameter in self.parameters():
            parameter.requires_grad = False

    def analysis(self, waveform: Tensor) -> Tensor:
        padded = F.pad(waveform, (0, self.pad_samples))
        outputs = []
        for channel in range(padded.shape[1]):
            channel_wave = padded[:, channel : channel + 1, :]
            outputs.append(self.analysis_filter(self.analysis_pad(channel_wave)))
        return torch.cat(outputs, dim=1)

    def synthesis(self, subband_wave: Tensor) -> Tensor:
        outputs = []
        for channel in range(0, subband_wave.shape[1], self.subbands):
            chunk = subband_wave[:, channel : channel + self.subbands, :]
            synth = self.synthesis_filter(self.synthesis_pad(chunk)).permute(0, 2, 1)
            outputs.append(synth.reshape(synth.shape[0], 1, -1))
        full_band = torch.cat(outputs, dim=1)
        return full_band[..., : -self.pad_samples]

    def forward(self, waveform: Tensor) -> Tensor:
        return self.analysis_filter(self.analysis_pad(waveform))


class _SubbandResUNetCore(nn.Module):
    """Core ResUNet architecture working on PQMF subbands."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_sources: int,
        window_size: int,
        hop_size: int,
        subbands_num: int,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_sources = num_sources
        self.subbands_num = subbands_num
        self.time_downsample_ratio = 2 ** 5
        self.components_per_time_freq = 4

        self.pqmf = _PQMF(subbands=subbands_num, taps=64)

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

        freq_bins = window_size // 2 + 1
        self.batch_norm = nn.BatchNorm2d(freq_bins, momentum=0.01)

        self.enc1 = _EncoderBlock(input_channels * subbands_num, 32, (3, 3), (2, 2))
        self.enc2 = _EncoderBlock(32, 64, (3, 3), (2, 2))
        self.enc3 = _EncoderBlock(64, 128, (3, 3), (2, 2))
        self.enc4 = _EncoderBlock(128, 256, (3, 3), (2, 2))
        self.enc5 = _EncoderBlock(256, 384, (3, 3), (2, 2))
        self.enc6 = _EncoderBlock(384, 384, (3, 3), (1, 2))

        self.mid1 = _EncoderBlock(384, 384, (3, 3), (1, 1))
        self.mid2 = _EncoderBlock(384, 384, (3, 3), (1, 1))
        self.mid3 = _EncoderBlock(384, 384, (3, 3), (1, 1))
        self.mid4 = _EncoderBlock(384, 384, (3, 3), (1, 1))

        self.dec1 = _DecoderBlock(384, 384, (3, 3), (1, 2))
        self.dec2 = _DecoderBlock(384, 384, (3, 3), (2, 2))
        self.dec3 = _DecoderBlock(384, 256, (3, 3), (2, 2))
        self.dec4 = _DecoderBlock(256, 128, (3, 3), (2, 2))
        self.dec5 = _DecoderBlock(128, 64, (3, 3), (2, 2))
        self.dec6 = _DecoderBlock(64, 32, (3, 3), (2, 2))

        self.post = _EncoderBlock(32, 32, (3, 3), (1, 1))

        self.to_mask = nn.Conv2d(
            in_channels=32,
            out_channels=num_sources * output_channels * self.components_per_time_freq * subbands_num,
            kernel_size=(1, 1),
            bias=True,
        )
        nn.init.xavier_uniform_(self.to_mask.weight)
        if self.to_mask.bias is not None:
            self.to_mask.bias.data.zero_()

    def _wave_to_spectrogram(self, waveform: Tensor) -> _SpectrogramInfo:
        mag, cosine, sine = self._wav_to_spectrogram_phase(waveform)
        return _SpectrogramInfo(mag, cosine, sine)

    def _wav_to_spectrogram_phase(self, waveform: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch, channels, samples = waveform.shape
        stacked = waveform.view(batch * channels, samples)
        real, imag = self.stft(stacked)
        mag = torch.clamp(real ** 2 + imag ** 2, _EPS, torch.finfo(real.dtype).max) ** 0.5
        cosine = real / mag
        sine = imag / mag
        mag = mag.view(batch, channels, *mag.shape[-2:])
        cosine = cosine.view(batch, channels, *cosine.shape[-2:])
        sine = sine.view(batch, channels, *sine.shape[-2:])
        return mag, cosine, sine

    def _feature_maps_to_wave(self, features: Tensor, spec: _SpectrogramInfo, length: int) -> Tensor:
        batch = features.shape[0]
        time_steps = features.shape[2]
        freq_bins = features.shape[3]
        k = self.components_per_time_freq

        features = features.view(
            batch,
            self.num_sources,
            self.output_channels,
            self.subbands_num,
            k,
            time_steps,
            freq_bins,
        )

        outputs = []
        for subband in range(self.subbands_num):
            # Each PQMF subband produces an independent complex mask estimate.
            item = features[:, :, :, subband]
            magnitude_mask = torch.sigmoid(item[:, :, :, 0])
            real_mask = torch.tanh(item[:, :, :, 1])
            imag_mask = torch.tanh(item[:, :, :, 2])
            linear_mag = torch.tanh(item[:, :, :, 3])
            _, mask_cos, mask_sin = magphase(real_mask, imag_mask)

            mag = spec.magnitude[:, subband::self.subbands_num]
            cos_in = spec.cosine[:, subband::self.subbands_num]
            sin_in = spec.sine[:, subband::self.subbands_num]

            cos_out = cos_in[:, None] * mask_cos - sin_in[:, None] * mask_sin
            sin_out = sin_in[:, None] * mask_cos + cos_in[:, None] * mask_sin
            out_mag = F.relu(mag[:, None] * magnitude_mask + linear_mag)
            out_real = out_mag * cos_out
            out_imag = out_mag * sin_out

            shape = (
                batch * self.num_sources * self.output_channels,
                1,
                time_steps,
                freq_bins,
            )
            out_real = out_real.view(shape)
            out_imag = out_imag.view(shape)
            waveform = self.istft(out_real, out_imag, length)
            waveform = waveform.view(batch, self.num_sources * self.output_channels, length)
            outputs.append(waveform)

        stacked = torch.stack(outputs, dim=2)
        stacked = stacked.view(
            batch,
            self.num_sources * self.output_channels * self.subbands_num,
            length,
        )
        return self.pqmf.synthesis(stacked)

    def forward(self, mixture: Tensor) -> Tensor:
        subband_wave = self.pqmf.analysis(mixture)
        spec = self._wave_to_spectrogram(subband_wave)

        x = spec.magnitude.transpose(1, 3)
        x = self.batch_norm(x)
        x = x.transpose(1, 3)

        original_time = x.shape[2]
        pad_frames = (
            (original_time + self.time_downsample_ratio - 1) // self.time_downsample_ratio * self.time_downsample_ratio
        ) - original_time
        x = F.pad(x, (0, 0, 0, pad_frames))
        x = x[..., :-1]

        x1_pool, x1 = self.enc1(x)
        x2_pool, x2 = self.enc2(x1_pool)
        x3_pool, x3 = self.enc3(x2_pool)
        x4_pool, x4 = self.enc4(x3_pool)
        x5_pool, x5 = self.enc5(x4_pool)
        x6_pool, x6 = self.enc6(x5_pool)

        mid, _ = self.mid1(x6_pool)
        mid, _ = self.mid2(mid)
        mid, _ = self.mid3(mid)
        mid, _ = self.mid4(mid)

        x7 = self.dec1(mid, x6)
        x8 = self.dec2(x7, x5)
        x9 = self.dec3(x8, x4)
        x10 = self.dec4(x9, x3)
        x11 = self.dec5(x10, x2)
        x12 = self.dec6(x11, x1)
        x, _ = self.post(x12)

        x = self.to_mask(x)
        x = F.pad(x, (0, 1))
        x = x[:, :, :original_time, :]

        length = subband_wave.shape[-1]
        return self._feature_maps_to_wave(x, spec, length)


class BandSplitSubbandResUNet(nn.Module):
    """Waveform separator that wraps the ByteDance subband ResUNet."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_sources: int,
        window_size: int = 512,
        hop_size: int = 110,
        subbands_num: int = 4,
    ) -> None:
        super().__init__()
        self.num_sources = num_sources
        self.output_channels = output_channels
        self.core = _SubbandResUNetCore(
            input_channels=input_channels,
            output_channels=output_channels,
            num_sources=num_sources,
            window_size=window_size,
            hop_size=hop_size,
            subbands_num=subbands_num,
        )

    def forward(self, mixture: Tensor) -> Tensor:
        if mixture.ndim != 3:
            raise ValueError("Input mixture must be shaped as (batch, channels, samples).")
        separated = self.core(mixture)
        batch = mixture.shape[0]
        separated = separated.view(batch, self.num_sources, self.output_channels, -1)
        return separated
