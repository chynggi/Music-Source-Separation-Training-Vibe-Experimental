import torch
import typing as tp


def get_fftfreq(sr: int = 44100, n_fft: int = 2048) -> torch.Tensor:
    """Return FFT frequencies for the given sampling rate and FFT size."""
    out = sr * torch.fft.fftfreq(n_fft)[: n_fft // 2 + 1]
    out[-1] = sr // 2
    return out


def get_subband_indices(freqs: torch.Tensor, splits: tp.List[tp.Tuple[int, int]]) -> tp.List[tp.Tuple[int, int]]:
    """Compute start/end indices for each band specified in ``splits``."""
    indices = []
    start_freq, start_index = 0, 0
    for end_freq, step in splits:
        bands = torch.arange(start_freq + step, end_freq + step, step)
        start_freq = end_freq
        for band in bands:
            end_index = freqs[freqs < band].shape[0]
            indices.append((start_index, end_index))
            start_index = end_index
    indices.append((start_index, freqs.shape[0]))
    return indices


def freq2bands(bandsplits: tp.List[tp.Tuple[int, int]], sr: int = 44100, n_fft: int = 2048) -> tp.List[tp.Tuple[int, int]]:
    """Translate frequency band definitions into FFT bin index ranges."""
    freqs = get_fftfreq(sr=sr, n_fft=n_fft)
    band_indices = get_subband_indices(freqs, bandsplits)
    return band_indices
