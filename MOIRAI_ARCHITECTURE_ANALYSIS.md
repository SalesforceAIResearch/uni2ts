# Moirai 모델 아키텍처 분석 (Universal Time Series Forecasting)

## 목차
1. [Transformer 아키텍처](#1-transformer-아키텍처)
2. [Novel Components (논문의 Novelty)](#2-novel-components-논문의-novelty)
3. [서로 다른 Frequency 처리 메커니즘](#3-서로-다른-frequency-처리-메커니즘)

---

## 1. Transformer 아키텍처

### 1.1 전체 구조
Moirai는 Encoder-only Transformer 아키텍처를 사용합니다.

**핵심 구성 요소** (`src/uni2ts/module/transformer.py:91-237`):
```python
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: Optional[int] = None,
        num_groups: Optional[int] = None,  # Grouped Query Attention
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.LayerNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_moe: bool = False,
        use_glu: bool = True,  # Gated Linear Unit
        use_qk_norm: bool = True,  # Query-Key Normalization
        var_attn_bias_layer: Optional[Callable] = None,  # Variate attention bias
        time_attn_bias_layer: Optional[Callable] = None,  # Time attention bias
        var_qk_proj_layer: Optional[Callable] = None,
        time_qk_proj_layer: Optional[Callable] = None,  # Rotary Position Encoding
        ...
    )
```

### 1.2 Grouped Query Attention (GQA)
일반적인 Multi-Head Attention 대신 Grouped Query Attention을 사용합니다.

**구현** (`src/uni2ts/module/attention.py:58-306`):
```python
class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,  # num_groups < num_heads
        ...
    ):
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups

        # Query는 모든 head에 대해, Key/Value는 group에 대해만 projection
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
        self.v_proj = nn.Linear(dim, self.head_dim * num_groups, bias=bias)
```

**특징**:
- Query는 `num_heads`개의 head 사용
- Key/Value는 `num_groups`개의 group만 사용 (메모리 절약)
- 각 group이 여러 query head를 담당

### 1.3 Normalization: RMSNorm
LayerNorm 대신 RMSNorm 사용 (`src/uni2ts/model/moirai/module.py:133`):
```python
self.encoder = TransformerEncoder(
    d_model,
    num_layers,
    ...
    norm_layer=RMSNorm,  # RMSNorm 사용
    ...
)
```

### 1.4 Feed-Forward Network: Gated Linear Unit (GLU)
표준 FFN 대신 GLU 사용:
```python
self.encoder = TransformerEncoder(
    ...
    use_glu=True,  # Gated Linear Unit 활성화
    activation=F.silu,  # SiLU (Swish) activation
    ...
)
```

### 1.5 Query-Key Normalization
Query와 Key에 normalization 적용 (`src/uni2ts/module/attention.py:91-96`):
```python
self.q_norm = (
    norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
)
self.k_norm = (
    norm_layer(self.head_dim) if norm_layer is not None else nn.Identity()
)
```

---

## 2. Novel Components (논문의 Novelty)

### 2.1 ⭐ Multi-InSize Linear Layer
**가장 중요한 Novelty**: 서로 다른 patch size를 처리하기 위한 특별한 linear layer

**구현** (`src/uni2ts/module/ts_embed.py:37-103`):
```python
class MultiInSizeLinear(nn.Module):
    def __init__(
        self,
        in_features_ls: tuple[int, ...],  # 예: (8, 16, 32, 64, 128)
        out_features: int,
        ...
    ):
        super().__init__()
        # 각 patch size에 대해 별도의 weight 생성
        self.weight = nn.Parameter(
            torch.empty(
                (len(in_features_ls), out_features, max(in_features_ls))
            )
        )

    def forward(
        self,
        x: Float[torch.Tensor, "*batch max_feat"],
        in_feat_size: Int[torch.Tensor, "*batch"],  # 실제 patch size
    ) -> Float[torch.Tensor, "*batch out_feat"]:
        out = 0
        for idx, feat_size in enumerate(self.in_features_ls):
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            # patch_size가 feat_size와 일치하는 경우만 계산
            out = out + (
                torch.eq(in_feat_size, feat_size).unsqueeze(-1)
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)
            )
        return out
```

**핵심 아이디어**:
- 각 가능한 patch size (8, 16, 32, 64, 128)에 대해 별도의 weight 학습
- 입력의 실제 patch size에 따라 해당하는 weight만 선택적으로 사용
- 이를 통해 다양한 frequency의 time series를 동일한 모델로 처리

### 2.2 ⭐ Frequency-Aware Patch Size Constraints
Time series의 frequency에 따라 적절한 patch size 범위를 자동 결정

**구현** (`src/uni2ts/transform/patch.py:57-75`):
```python
class DefaultPatchSizeConstraints(PatchSizeConstraints):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    DEFAULT_RANGES = {
        "S": (64, 128),  # 초 단위: 512s = 8.53분, 4096s = 68.26분
        "T": (32, 128),  # 분 단위: 64분 = 1.07시간, 512분 = 8.53시간
        "H": (32, 64),   # 시간 단위: 128시간 = 5.33일
        "D": (16, 32),   # 일 단위
        "B": (16, 32),   # 영업일 단위
        "W": (16, 32),   # 주 단위
        "M": (8, 32),    # 월 단위
        "Q": (1, 8),     # 분기 단위
        "Y": (1, 8),     # 연 단위
        "A": (1, 8),     # 연 단위
    }
```

**작동 방식**:
1. 입력 time series의 frequency 확인 (예: "H" for hourly)
2. 해당 frequency에 맞는 patch size 범위 선택 (예: 32~64)
3. 시계열 길이와 `min_time_patches`를 고려하여 최종 patch size 결정

### 2.3 ⭐ Variate & Time Attention Bias
Variate dimension과 Time dimension에 대해 서로 다른 attention bias/projection 적용

**구현** (`src/uni2ts/model/moirai/module.py:137-143`):
```python
self.encoder = TransformerEncoder(
    ...
    var_attn_bias_layer=partial(BinaryAttentionBias),  # Variate dimension
    time_qk_proj_layer=partial(
        QueryKeyProjection,
        proj_layer=RotaryProjection,  # Time dimension에 RoPE 적용
        kwargs=dict(max_len=max_seq_len),
        partial_factor=(0.0, 0.5),  # head dimension의 50%에만 적용
    ),
    ...
)
```

#### 2.3.1 Binary Attention Bias (Variate)
같은 variate 내의 token 간 attention 강화 (`src/uni2ts/module/position/attn_bias.py:67-87`):
```python
class BinaryAttentionBias(AttentionBias):
    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Int[torch.Tensor, "*batch 1 1 q_len"],
        kv_id: Int[torch.Tensor, "*batch 1 1 kv_len"],
    ) -> Float[torch.Tensor, "*batch #group #hpg q_len kv_len"]:
        # 같은 variate_id를 가진 token에는 다른 bias 적용
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        weight = rearrange(self.emb.weight, "two num_heads -> two num_heads 1 1")
        bias = rearrange(
            ~ind * weight[:1] + ind * weight[1:],  # 같으면 weight[1], 다르면 weight[0]
            "... 1 (group hpg) q_len kv_len -> ... group hpg q_len kv_len",
            ...
        )
        return bias
```

#### 2.3.2 Rotary Position Encoding (Time)
Time dimension에 RoPE 적용 (`src/uni2ts/module/position/attn_projection.py:55-107`):
```python
class RotaryProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,
        num_heads: int,
        num_groups: int,
        max_len: int = 512,
        base: int = 10000,
    ):
        super().__init__(proj_width, num_heads, num_groups)
        # RoPE의 frequency 계산
        self.register_buffer(
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )

    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:
        # position에 따른 rotation 적용
        rot_cos = self.cos[seq_id]
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)
```

### 2.4 ⭐ Auto Patch Size Selection
Inference 시 여러 patch size를 시도하고 validation loss가 가장 낮은 것을 자동 선택

**구현** (`src/uni2ts/model/moirai/forecast.py:255-333`):
```python
def forward(self, ...):
    if self.hparams.patch_size == "auto":
        val_loss = []
        preds = []
        # 모든 가능한 patch size 시도
        for patch_size in self.module.patch_sizes:  # [8, 16, 32, 64, 128]
            # validation loss 계산
            val_loss.append(
                self._val_loss(
                    patch_size=patch_size,
                    target=past_target[..., : self.past_length, :],
                    ...
                )
            )
            # prediction 생성
            distr = self._get_distr(patch_size, ...)
            preds.append(...)

        val_loss = torch.stack(val_loss)
        preds = torch.stack(preds)
        # 가장 낮은 validation loss를 가진 patch size의 prediction 선택
        idx = val_loss.argmin(dim=0)
        return preds[idx, torch.arange(len(idx), device=idx.device)]
```

### 2.5 ⭐ Packed Scaler
서로 다른 scale의 time series를 효율적으로 처리 (`src/uni2ts/model/moirai/module.py:181-186`):
```python
loc, scale = self.scaler(
    target,
    observed_mask * ~prediction_mask.unsqueeze(-1),
    sample_id,  # 어느 sample에 속하는지
    variate_id,  # 어느 variate에 속하는지
)
scaled_target = (target - loc) / scale
```

**특징**:
- 각 (sample, variate) 조합에 대해 독립적으로 standardization 수행
- Batching된 여러 시계열을 한 번에 효율적으로 처리

---

## 3. 서로 다른 Frequency 처리 메커니즘

Moirai가 Universal Time Series Forecasting을 달성하는 핵심 메커니즘입니다.

### 3.1 전체 흐름

```
입력 시계열 (서로 다른 frequency)
    ↓
[1] Frequency 감지 및 Patch Size 결정
    ↓
[2] Patching (시계열 → 패치 시퀀스)
    ↓
[3] Multi-InSize Linear (패치 → 임베딩)
    ↓
[4] Transformer Encoder
    ↓
[5] 예측 생성 및 Un-patching
```

### 3.2 [1] Frequency 감지 및 Patch Size 결정

**코드 위치**: `src/uni2ts/transform/patch.py:78-121`

```python
class GetPatchSize(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        freq = data_entry["freq"]  # 예: "H" (hourly)

        # Frequency에 따른 patch size 범위 가져오기
        constraints = self.patch_size_constraints(freq)  # 예: range(32, 65)

        # 시계열 길이 고려
        target: list[UnivarTimeSeries] = data_entry[self.target_field]
        length = target[0].shape[0]
        patch_size_ceil = length // self.min_time_patches

        # 최종 patch size 후보 결정
        patch_size_candidates = [
            patch_size
            for patch_size in self.patch_sizes  # [8, 16, 32, 64, 128]
            if (patch_size in constraints) and (patch_size <= patch_size_ceil)
        ]

        # 랜덤하게 하나 선택 (학습 시 augmentation 효과)
        data_entry["patch_size"] = np.random.choice(patch_size_candidates)
        return data_entry
```

**예시**:
- Hourly data (freq="H"): patch_size ∈ [32, 64]
- Daily data (freq="D"): patch_size ∈ [16, 32]
- Monthly data (freq="M"): patch_size ∈ [8, 32]

### 3.3 [2] Patching

**코드 위치**: `src/uni2ts/transform/patch.py:124-160`

```python
class Patchify(MapFuncMixin, Transformation):
    def _patchify_arr(
        self, arr: Num[np.ndarray, "var time*patch"], patch_size: int
    ) -> Num[np.ndarray, "var time max_patch"]:
        # 1. Reshape: [var, time*patch] → [var, time, patch]
        arr = rearrange(arr, "... (time patch) -> ... time patch", patch=patch_size)

        # 2. Padding: 모든 patch를 max_patch_size로 맞춤
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        pad_width[-1] = (0, self.max_patch_size - patch_size)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=self.pad_value)

        return arr  # [var, time, max_patch]
```

**예시**:
```python
# 입력: hourly data, 길이 = 512, patch_size = 64
arr.shape  # [1, 512]

# Patching 후
arr.shape  # [1, 8, 128]
# 8 = 512 / 64 (number of patches)
# 128 = max_patch_size (64는 0으로 padding)
```

### 3.4 [3] Multi-InSize Linear

**코드 위치**: `src/uni2ts/model/moirai/module.py:188`

```python
reprs = self.in_proj(scaled_target, patch_size)
```

`self.in_proj`는 `MultiInSizeLinear`로, 각 patch size에 맞는 weight를 선택적으로 사용합니다.

**처리 과정**:
```python
# 입력: [batch, seq_len, max_patch=128], patch_size = 64
# MultiInSizeLinear는 patch_size=64에 해당하는 weight 선택
# 출력: [batch, seq_len, d_model]
```

### 3.5 [4] Transformer Encoder에서의 처리

**코드 위치**: `src/uni2ts/model/moirai/module.py:190-195`

```python
reprs = self.encoder(
    masked_reprs,
    packed_attention_mask(sample_id),
    time_id=time_id,      # 시간 위치 정보
    var_id=variate_id,    # 변수 ID
)
```

**주요 메커니즘**:
1. **Time ID**: RoPE를 통해 시간 순서 정보 인코딩
2. **Variate ID**: Binary Attention Bias로 같은 변수 내 attention 강화
3. **Sample ID**: Packed attention mask로 서로 다른 샘플 간 attention 차단

### 3.6 실제 사용 예시

**시나리오**: 서로 다른 3개의 시계열을 동시에 처리

```python
# 시계열 1: Hourly electricity consumption (freq="H", length=512)
# 시계열 2: Daily sales (freq="D", length=365)
# 시계열 3: Monthly revenue (freq="M", length=36)

# [1] Patch Size 결정
# TS1: patch_size = 64 (512 / 64 = 8 patches)
# TS2: patch_size = 32 (365 / 32 ≈ 11.4 → 12 patches with padding)
# TS3: patch_size = 8 (36 / 8 = 4.5 → 5 patches with padding)

# [2] Patching 후
# TS1: [1, 8, 128]   (64 + 64 padding)
# TS2: [1, 12, 128]  (32 + 96 padding)
# TS3: [1, 5, 128]   (8 + 120 padding)

# [3] Batch로 결합
# batch shape: [3, max_seq=12, 128]
# sample_id: [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2]
# patch_size: [64]*8 + [32]*12 + [8]*5

# [4] Multi-InSize Linear
# 각 token의 patch_size에 맞는 weight 사용하여 embedding 생성

# [5] Transformer 처리
# sample_id를 통해 서로 다른 시계열 간 attention 차단
# time_id로 각 시계열 내부의 시간 순서 인코딩
```

### 3.7 핵심 장점

1. **Unified Architecture**:
   - 하나의 모델로 모든 frequency 처리
   - 별도의 frequency-specific 모델 불필요

2. **Flexible Patch Size**:
   - Frequency에 맞는 적절한 patch size 자동 선택
   - 너무 짧거나 긴 패치 방지

3. **Efficient Batching**:
   - 서로 다른 frequency의 시계열을 한 batch에 처리
   - Packed attention으로 효율적인 학습

4. **Multi-InSize Linear**:
   - 각 patch size에 최적화된 embedding 학습
   - 단순 padding보다 더 나은 representation

---

## 4. 코드 참조

주요 파일 위치:
- Transformer: `src/uni2ts/module/transformer.py`
- Attention: `src/uni2ts/module/attention.py`
- Position Encoding: `src/uni2ts/module/position/`
- Model Module: `src/uni2ts/model/moirai/module.py`
- Forecasting: `src/uni2ts/model/moirai/forecast.py`
- Patching: `src/uni2ts/transform/patch.py`
- Time Series Embedding: `src/uni2ts/module/ts_embed.py`

---

## 5. 참고 논문

- **Moirai**: "Unified Training of Universal Time Series Forecasting Transformers" (ICML 2024)
- **Moirai-MoE**: "Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts" (2024)
