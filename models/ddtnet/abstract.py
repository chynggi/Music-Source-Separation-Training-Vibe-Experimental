import torch
import torch.nn as nn

class DTTNetBase(nn.Module):
    def __init__(self, target_name, dim_f, dim_t, n_fft, hop_length, overlap, audio_ch, **kwargs):
        super().__init__()
        self.target_name = target_name
        self.dim_c_in = audio_ch * 2
        self.dim_c_out = audio_ch * 2
        self.dim_f = dim_f
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.audio_ch = audio_ch
        self.overlap = overlap
        self.chunk_size = hop_length * (self.dim_t - 1)
        
        # Use nn.Parameter for better mixed precision handling
        self.window = nn.Parameter(torch.hann_window(window_length=self.n_fft, periodic=True), requires_grad=False)
        self.freq_pad = nn.Parameter(torch.zeros([1, self.dim_c_out, self.n_bins - self.dim_f, 1]), requires_grad=False)

    def stft(self, x):
        dim_b = x.shape[0]
        x = x.reshape([dim_b * self.audio_ch, -1])  # (B*C, L)
        spec_c = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,  # complex output
        )  # (B*C, n_bins, n_frames)
        spec_ri = torch.view_as_real(spec_c)  # (B*C, n_bins, n_frames, 2)
        spec_ri = spec_ri.permute(0, 3, 1, 2)  # (B*C, 2, n_bins, n_frames)
        spec_ri = spec_ri.reshape(dim_b, self.audio_ch, 2, self.n_bins, -1).reshape(
            dim_b, self.audio_ch * 2, self.n_bins, -1
        )
        return spec_ri[:, :, :self.dim_f]

    def istft(self, x):
        dim_b = x.shape[0]
        x = torch.cat([x, self.freq_pad.repeat([x.shape[0], 1, 1, x.shape[-1]])], -2)
        x = x.reshape(dim_b, self.audio_ch, 2, self.n_bins, -1).reshape(
            dim_b * self.audio_ch, 2, self.n_bins, -1
        )  # (B*C, 2, n_bins, n_frames)
        x = x.permute(0, 2, 3, 1)  # (B*C, n_bins, n_frames, 2)
        spec_c = torch.view_as_complex(x.contiguous())  # complex tensor
        wav = torch.istft(
            spec_c,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
        )  # (B*C, L)
        return wav.reshape(dim_b, self.audio_ch, -1)