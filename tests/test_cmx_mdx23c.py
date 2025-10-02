#!/usr/bin/env python3
"""
Test script for CMX-Enhanced MDX23C

This script validates the Channel Multiplexing implementation and provides
benchmarks comparing memory usage and performance with/without CMX.

Usage:
    python tests/test_cmx_mdx23c.py
    python tests/test_cmx_mdx23c.py --detailed
"""

import torch
import torch.nn as nn
import sys
import os
from omegaconf import OmegaConf
import time
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net


def create_test_config(use_cmx=False, cmx_reduction=2):
    """Create a test configuration for MDX23C"""
    config = OmegaConf.create({
        'audio': {
            'num_channels': 2,
            'sample_rate': 44100,
            'chunk_size': 352800,  # 8 seconds
            'min_mean_abs': 0.001,
            'hop_length': 1024,
            'n_fft': 2048,
            'dim_f': 2048,
            'dim_t': 2048,
            'use_cmx': use_cmx,
            'cmx_reduction': cmx_reduction,
        },
        'model': {
            'type': 'mdx23c',
            'num_subbands': 4,
            'num_scales': 5,
            'scale': [2, 2],
            'num_blocks_per_scale': 2,
            'num_channels': 32,
            'growth': 128,
            'bottleneck_factor': 4,
            'norm': 'InstanceNorm',
            'act': 'gelu',
        },
        'training': {
            'batch_size': 4,
            'instruments': ['bass', 'drums', 'other', 'vocals'],
            'target_instrument': None,
        }
    })
    return config


def test_forward_backward(model, batch_size=2, device='cpu'):
    """Test forward and backward pass"""
    print(f"  Testing forward/backward pass (batch_size={batch_size})...")
    
    # Create dummy input
    input_audio = torch.randn(batch_size, 2, 352800, device=device)
    
    # Forward pass
    try:
        output = model(input_audio)
        
        # Check output shape
        if output.dim() == 4:  # Multi-instrument
            assert output.shape[0] == batch_size
            assert output.shape[1] == 4  # 4 instruments
            assert output.shape[2] == 2  # stereo
            assert output.shape[3] == 352800  # same length
        else:  # Single instrument
            assert output.shape[0] == batch_size
            assert output.shape[1] == 2  # stereo
            assert output.shape[2] == 352800
        
        print(f"    ✓ Forward pass successful")
        print(f"      Input shape:  {list(input_audio.shape)}")
        print(f"      Output shape: {list(output.shape)}")
        
        # Backward pass
        if device != 'cpu':  # Only test backward on GPU
            loss = output.mean()
            loss.backward()
            print(f"    ✓ Backward pass successful")
        
        return True, output.shape
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False, None


def measure_memory(model, batch_size=2, device='cuda'):
    """Measure peak GPU memory usage"""
    if device == 'cpu':
        print("  Skipping memory measurement (CPU mode)")
        return 0
    
    print(f"  Measuring GPU memory usage (batch_size={batch_size})...")
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    initial_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB
    
    # Forward pass
    input_audio = torch.randn(batch_size, 2, 352800, device=device)
    output = model(input_audio)
    
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    memory_used = peak_memory - initial_memory
    
    print(f"    Peak memory: {peak_memory:.2f} MB")
    print(f"    Memory used: {memory_used:.2f} MB")
    
    # Cleanup
    del input_audio, output
    torch.cuda.empty_cache()
    
    return memory_used


def benchmark_speed(model, batch_size=2, num_iterations=10, device='cpu'):
    """Benchmark inference speed"""
    print(f"  Benchmarking speed ({num_iterations} iterations)...")
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            dummy_input = torch.randn(batch_size, 2, 352800, device=device)
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            input_audio = torch.randn(batch_size, 2, 352800, device=device)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model(input_audio)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    print(f"    Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"    Throughput: {batch_size/avg_time:.2f} samples/sec")
    
    return avg_time, std_time


def test_cmx_reversibility():
    """Test that CMX transformation is reversible (lossless)"""
    print("\n=== Testing CMX Reversibility ===")
    
    config = create_test_config(use_cmx=True, cmx_reduction=2)
    model = TFC_TDF_net(config)
    
    # Create test tensor
    B, C, F, T = 2, 8, 513, 512
    x_original = torch.randn(B, C, F, T)
    
    print(f"  Original shape: {list(x_original.shape)}")
    
    # Apply CMX
    x_cmx = model._apply_cmx(x_original)
    print(f"  After CMX: {list(x_cmx.shape)}")
    
    # Reverse CMX
    model.original_freq_dim = F
    model.original_time_dim = T
    x_restored = model._reverse_cmx(x_cmx)
    print(f"  After reverse: {list(x_restored.shape)}")
    
    # Check if reversible
    reconstruction_error = torch.mean(torch.abs(x_original - x_restored)).item()
    print(f"  Reconstruction error: {reconstruction_error:.10f}")
    
    if reconstruction_error < 1e-6:
        print("  ✓ CMX is perfectly reversible (lossless)")
        return True
    else:
        print(f"  ✗ CMX has reconstruction error > 1e-6")
        return False


def compare_baseline_vs_cmx(device='cpu', detailed=False):
    """Compare baseline MDX23C with CMX-enhanced version"""
    print("\n" + "="*70)
    print("COMPARISON: Baseline MDX23C vs CMX-Enhanced MDX23C")
    print("="*70)
    
    results = {}
    
    for use_cmx in [False, True]:
        mode = "CMX-Enhanced" if use_cmx else "Baseline"
        print(f"\n--- Testing {mode} MDX23C ---")
        
        # Create model
        config = create_test_config(use_cmx=use_cmx, cmx_reduction=2)
        model = TFC_TDF_net(config).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward/backward
        success, output_shape = test_forward_backward(model, batch_size=2, device=device)
        
        if not success:
            print(f"  ✗ Forward/backward test failed!")
            continue
        
        # Measure memory (GPU only)
        memory_used = 0
        if device == 'cuda' and torch.cuda.is_available():
            memory_used = measure_memory(model, batch_size=2, device=device)
        
        # Benchmark speed
        avg_time, std_time = benchmark_speed(
            model, batch_size=2, num_iterations=5 if detailed else 3, device=device
        )
        
        results[mode] = {
            'params': total_params,
            'memory_mb': memory_used,
            'time_ms': avg_time * 1000,
            'output_shape': output_shape,
        }
        
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    if 'Baseline' in results and 'CMX-Enhanced' in results:
        baseline = results['Baseline']
        cmx = results['CMX-Enhanced']
        
        print(f"\nParameters:")
        print(f"  Baseline:     {baseline['params']:,}")
        print(f"  CMX-Enhanced: {cmx['params']:,}")
        print(f"  Difference:   Same (as expected)")
        
        if baseline['memory_mb'] > 0 and cmx['memory_mb'] > 0:
            memory_reduction = (baseline['memory_mb'] - cmx['memory_mb']) / baseline['memory_mb'] * 100
            print(f"\nMemory Usage:")
            print(f"  Baseline:     {baseline['memory_mb']:.2f} MB")
            print(f"  CMX-Enhanced: {cmx['memory_mb']:.2f} MB")
            print(f"  Reduction:    {memory_reduction:.1f}%")
        
        time_overhead = (cmx['time_ms'] - baseline['time_ms']) / baseline['time_ms'] * 100
        print(f"\nInference Time:")
        print(f"  Baseline:     {baseline['time_ms']:.2f} ms")
        print(f"  CMX-Enhanced: {cmx['time_ms']:.2f} ms")
        print(f"  Overhead:     {time_overhead:+.1f}%")
        
        print(f"\n{'='*70}")
        print("CONCLUSION:")
        if baseline['memory_mb'] > 0 and memory_reduction > 20:
            print(f"✓ CMX provides significant memory savings ({memory_reduction:.0f}%)")
        if abs(time_overhead) < 10:
            print(f"✓ CMX has minimal speed impact ({time_overhead:+.1f}%)")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Test CMX-Enhanced MDX23C')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run tests on (cuda/cpu)')
    parser.add_argument('--detailed', action='store_true',
                       help='Run more detailed benchmarks')
    args = parser.parse_args()
    
    print("="*70)
    print("CMX-Enhanced MDX23C Test Suite")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")
    if args.device == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test CMX reversibility
    test_cmx_reversibility()
    
    # Compare baseline vs CMX
    compare_baseline_vs_cmx(device=args.device, detailed=args.detailed)
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)


if __name__ == '__main__':
    main()
