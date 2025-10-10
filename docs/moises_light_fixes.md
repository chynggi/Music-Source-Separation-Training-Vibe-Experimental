# Moises-Light Loss 문제 수정 내역

## 문제 분석

Loss가 0으로 고정되거나 NaN을 반환하는 주요 원인들:

### 1. ComplexSTFT의 차원 문제
- **문제**: `forward()`에서 `spec[..., : self.freq_bins, :]`로 불필요한 슬라이싱
- `torch.stft`는 이미 올바른 크기의 spectrogram을 반환하므로 추가 슬라이싱이 필요 없음
- **수정**: 동적으로 freq_bins를 계산하도록 변경

### 2. 마스크와 Spectrogram 차원 불일치
- **문제**: Padded spec과 cropped mask 간의 차원 불일치
- **수정**: 마스크 크기를 명시적으로 확인하고 조정

### 3. 극단적인 값으로 인한 NaN 발생
- **문제**: STFT/ISTFT 과정에서 극단적인 값이 NaN을 유발
- **수정**: 
  - Spectrogram에 `torch.clamp(min=-1e4, max=1e4)` 적용
  - 최종 오디오 출력에 `torch.clamp(min=-1.0, max=1.0)` 적용

### 4. 잘못된 Weight Initialization
- **문제**: 기본 PyTorch 초기화가 Moises-Light에 적합하지 않음
- **수정**: Xavier/Glorot 초기화 적용 (gain=0.01)
- Output projection에는 더 작은 gain (0.001) 적용

### 5. Gradient Explosion/Vanishing
- **문제**: 깊은 네트워크에서 gradient가 폭발하거나 소멸
- **수정**: 
  - 각 주요 모듈 출력에 `torch.clamp(min=-10, max=10)` 적용
  - Band split, encoder, transformer 출력 모두 안정화

## 수정 사항 상세

### ComplexSTFT.forward()
```python
# 변경 전
spec = spec[..., : self.freq_bins, :]

# 변경 후
# torch.stft already returns (freq_bins, time), no need to slice
freq_bins = spec.shape[2]
```

### ComplexSTFT.inverse()
```python
# 변경 전
if freq < self.freq_bins:
    pad = self.freq_bins - freq

# 변경 후
expected_freq = self.n_fft // 2 + 1
if freq < expected_freq:
    pad = expected_freq - freq
```

### MoisesLight.forward() - 마스크 적용
```python
# 추가된 안전장치
# 1. 마스크 크기 확인
if mask.shape[-2:] != (original_freq, original_time):
    mask = mask[..., : original_freq, : original_time]

# 2. Spectrogram 클램핑
estimated_spec = torch.clamp(estimated_spec, min=-1e4, max=1e4)

# 3. 오디오 출력 클램핑
audio = torch.clamp(audio, min=-1.0, max=1.0)
```

### Weight Initialization
```python
def _initialize_weights(self) -> None:
    """Initialize weights to prevent gradient issues and NaN values."""
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    # Special initialization for output projection
    nn.init.xavier_uniform_(self.output_proj.weight, gain=0.001)
```

### Gradient 안정화
```python
# Band split 출력 안정화
for module in self.band_splits_enc:
    feats = module(feats)
    feats = torch.clamp(feats, min=-10, max=10)

# Encoder 출력 안정화
for idx, stage in enumerate(self.encoder):
    feats = stage["block"](feats)
    feats = torch.clamp(feats, min=-10, max=10)

# Transformer 출력 안정화
feats = self.transformer(feats)
feats = torch.clamp(feats, min=-10, max=10)
```

## 테스트 권장사항

### 1. 입력 데이터 검증
학습 시작 전에 데이터가 올바른지 확인:
```python
# 학습 데이터 체크
assert torch.isfinite(mixture).all(), "Input contains NaN or Inf"
assert mixture.abs().max() <= 1.0, "Input should be normalized to [-1, 1]"
```

### 2. Gradient Clipping 설정
`config_moises_light.yaml`에서:
```yaml
training:
  grad_clip: 1.0  # null에서 1.0으로 변경 권장
```

### 3. Loss 함수 확인
`bsmamba2_loss` 또는 `multi_resolution_mae` 사용 시:
- Lambda 값이 너무 크지 않은지 확인 (lambda_time: 10.0 권장)
- STFT window sizes가 적절한지 확인

### 4. Learning Rate 조정
초기 학습이 불안정하면:
```yaml
training:
  lr: 1.0e-04  # 5.0e-04에서 감소
  warmup_steps: 1000  # Warmup 추가 권장
```

### 5. Mixed Precision 주의
AMP 사용 시 문제가 있다면:
```yaml
training:
  use_amp: false  # 일시적으로 비활성화하고 테스트
```

## 추가 개선 사항 (선택적)

### 1. Gradient Checkpointing
메모리 사용량 감소를 위해:
```python
# transformer 모듈에 적용
from torch.utils.checkpoint import checkpoint
feats = checkpoint(self.transformer, feats, use_reentrant=False)
```

### 2. EMA (Exponential Moving Average)
Config에 이미 설정되어 있음 (0.999), 이를 활용하면 학습 안정성 향상

### 3. Loss Monitoring
학습 중 각 loss 항목을 개별적으로 모니터링:
- Time domain loss
- Frequency domain loss (각 window size별)
- Total loss

### 4. Validation 강화
Overfitting 방지를 위해:
```yaml
training:
  validation_split: 0.1
  early_stopping_patience: 20
```

## 참고사항

이 수정사항들은 다음을 목표로 합니다:
1. **수치 안정성**: NaN과 Inf 방지
2. **Gradient 안정성**: Explosion/vanishing 방지
3. **차원 일치**: 모든 텐서 연산에서 차원 호환성 보장
4. **초기화 개선**: 더 나은 수렴을 위한 weight 초기화

수정 후에도 문제가 지속되면:
- 배치 크기를 줄여보세요 (8 → 4 또는 2)
- Gradient accumulation steps를 늘려보세요 (4 → 8)
- 더 간단한 loss 함수로 시작해보세요 (l1_loss만 사용)
- Dropout을 줄여보세요 (0.1 → 0.05)
