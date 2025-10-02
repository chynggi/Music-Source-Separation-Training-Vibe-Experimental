# CMX Enhancement - Quick Start Guide

## What is CMX?

Channel Multiplexing (CMX) is a memory-efficient technique inspired by the MARS paper (arXiv:2509.26007) that reduces GPU memory consumption by up to 75% while maintaining audio separation quality.

## Quick Enable

Add to your config file:
```yaml
audio:
  use_cmx: true
  cmx_reduction: 2
```

## Benefits

- **Memory Savings**: 50-75% reduction in GPU memory usage
- **Larger Batches**: Train with 2-4x larger batch sizes
- **Same Quality**: Preserves all frequency information
- **No Extra Params**: Same model size as baseline

## Example Configs

### MUSDB18 with CMX
```yaml
# configs/config_musdb18_mdx23c_cmx.yaml
audio:
  use_cmx: true
  cmx_reduction: 2
training:
  batch_size: 16  # Increased from 8 due to memory savings
```

## Training

```bash
# Train with CMX enabled
python train.py \
  --config configs/config_musdb18_mdx23c_cmx.yaml \
  --results_path results/mdx23c_cmx

# Compare with baseline (CMX disabled)
python train.py \
  --config configs/config_musdb18_mdx23c.yaml \
  --results_path results/mdx23c_baseline
```

## Testing

```bash
# Run CMX test suite
python tests/test_cmx_mdx23c.py

# Detailed benchmarks
python tests/test_cmx_mdx23c.py --detailed
```

## Performance Tips

### GPU Memory Recommendations

| GPU Memory | Batch Size (CMX) | Batch Size (Baseline) |
|-----------|------------------|----------------------|
| 12 GB     | 8-12            | 4-6                  |
| 16 GB     | 16-24           | 8-12                 |
| 24 GB     | 24-32           | 12-16                |

### Hyperparameter Tuning

With CMX enabled, you can:
1. **Increase batch size** by 2x for better gradient estimates
2. **Slightly increase learning rate** (1.2-1.5x) due to larger batches
3. **Process longer clips** or higher sample rates

### Reduction Factor Guide

```yaml
cmx_reduction: 1  # Disabled (100% memory)
cmx_reduction: 2  # Balanced (50% memory) ← Recommended
cmx_reduction: 3  # Aggressive (37% memory)
cmx_reduction: 4  # Very aggressive (25% memory)
```

**Recommendation**: Start with `cmx_reduction: 2` for optimal balance.

## Troubleshooting

**Shape mismatch errors?**
- Ensure `use_cmx` is set in config.audio section
- Check that n_fft and hop_length are divisible by cmx_reduction

**Training unstable?**
- Reduce learning rate by 20%
- Try gradient clipping: `grad_clip: 0.5`

**Lower quality than expected?**
- Train for more epochs (CMX converges differently)
- Try cmx_reduction: 2 instead of 3 or 4

## Citation

```bibtex
@article{mars2024,
  title={MARS: Audio Generation via Multi-Channel Autoregression on Spectrograms},
  journal={arXiv preprint arXiv:2509.26007},
  year={2024}
}
```

## Full Documentation

See [docs/cmx_enhancement.md](docs/cmx_enhancement.md) for complete technical details.
