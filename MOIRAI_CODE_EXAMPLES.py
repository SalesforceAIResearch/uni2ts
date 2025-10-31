"""
Moirai 모델의 핵심 컴포넌트 코드 예제

이 파일은 Moirai의 주요 novelty 컴포넌트를 실제 코드와 함께 설명합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
import numpy as np


# =============================================================================
# 1. Multi-InSize Linear: 서로 다른 patch size를 처리하는 핵심 컴포넌트
# =============================================================================

class MultiInSizeLinear(nn.Module):
    """
    서로 다른 patch size를 가진 입력을 처리하기 위한 특수 Linear Layer.

    핵심 아이디어:
    - 각 가능한 patch size (8, 16, 32, 64, 128)에 대해 별도의 weight 학습
    - 입력의 실제 patch size에 따라 해당하는 weight만 선택적으로 사용
    - 이를 통해 다양한 frequency의 time series를 동일한 모델로 처리

    실제 구현: src/uni2ts/module/ts_embed.py:37-103
    """

    def __init__(
        self,
        in_features_ls: tuple[int, ...] = (8, 16, 32, 64, 128),
        out_features: int = 512,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features

        # 각 patch size에 대해 별도의 weight 생성
        # shape: [num_patch_sizes, out_features, max_patch_size]
        self.weight = nn.Parameter(
            torch.empty(
                (len(in_features_ls), out_features, max(in_features_ls))
            )
        )

        if bias:
            # 각 patch size에 대해 별도의 bias
            # shape: [num_patch_sizes, out_features]
            self.bias = nn.Parameter(
                torch.empty((len(in_features_ls), out_features))
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # Mask: 각 patch size에 대해 유효한 weight만 사용
        # 예: patch_size=8이면 처음 8개 weight만 사용, 나머지는 0으로 masking
        mask = []
        for feat_size in in_features_ls:
            mask_i = torch.zeros(1, max(in_features_ls))
            mask_i[:, :feat_size] = 1.0
            mask.append(mask_i)
        self.register_buffer(
            "mask",
            rearrange(
                torch.stack(mask, dim=0),
                "num_feats 1 max_feat -> num_feats 1 max_feat",
            ),
            persistent=False,
        )

    def reset_parameters(self):
        """각 patch size별로 독립적으로 초기화"""
        for idx, feat_size in enumerate(self.in_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :, :feat_size])
            nn.init.zeros_(self.weight[idx, :, feat_size:])
            if self.bias is not None:
                bound = 1 / np.sqrt(feat_size) if feat_size > 0 else 0
                nn.init.uniform_(self.bias[idx], -bound, bound)

    def forward(self, x, in_feat_size):
        """
        Args:
            x: [batch, seq_len, max_patch_size] - padded input
            in_feat_size: [batch, seq_len] - actual patch size for each token

        Returns:
            out: [batch, seq_len, out_features]
        """
        out = 0
        for idx, feat_size in enumerate(self.in_features_ls):
            # 이 patch size에 해당하는 weight (masking 적용)
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0

            # patch_size가 feat_size와 일치하는 token만 계산
            mask = torch.eq(in_feat_size, feat_size).unsqueeze(-1)
            out = out + mask * (einsum(weight, x, "out inp, ... inp -> ... out") + bias)

        return out


def example_multi_in_size_linear():
    """Multi-InSize Linear 사용 예제"""
    batch_size = 2
    seq_len = 10
    max_patch_size = 128
    d_model = 512

    # 모델 생성
    layer = MultiInSizeLinear(
        in_features_ls=(8, 16, 32, 64, 128),
        out_features=d_model,
        bias=True,
    )

    # 입력 데이터 (padding된 형태)
    x = torch.randn(batch_size, seq_len, max_patch_size)

    # 각 token의 실제 patch size
    # 예: 처음 5개는 patch_size=64, 나머지 5개는 patch_size=32
    patch_sizes = torch.tensor([64] * 5 + [32] * 5)
    patch_sizes = patch_sizes.unsqueeze(0).expand(batch_size, -1)

    # Forward pass
    out = layer(x, patch_sizes)
    print(f"Input shape: {x.shape}")
    print(f"Patch sizes: {patch_sizes[0]}")
    print(f"Output shape: {out.shape}")
    # Output: [batch_size, seq_len, d_model]

    return out


# =============================================================================
# 2. Frequency-Aware Patch Size Selection
# =============================================================================

class DefaultPatchSizeConstraints:
    """
    Time series의 frequency에 따라 적절한 patch size 범위를 결정.

    핵심 아이디어:
    - 고빈도 데이터 (초, 분)는 큰 patch size 사용
    - 저빈도 데이터 (월, 분기)는 작은 patch size 사용
    - 이를 통해 각 frequency에서 적절한 수준의 temporal abstraction 달성

    실제 구현: src/uni2ts/transform/patch.py:57-75
    """

    DEFAULT_RANGES = {
        "S": (64, 128),   # 초 단위: 512s = 8.53분, 4096s = 68.26분
        "T": (32, 128),   # 분 단위: 64분 = 1.07시간, 512분 = 8.53시간
        "H": (32, 64),    # 시간 단위: 128시간 = 5.33일
        "D": (16, 32),    # 일 단위
        "B": (16, 32),    # 영업일 단위
        "W": (16, 32),    # 주 단위
        "M": (8, 32),     # 월 단위
        "Q": (1, 8),      # 분기 단위
        "Y": (1, 8),      # 연 단위
        "A": (1, 8),      # 연 단위
    }

    def __call__(self, freq: str) -> range:
        """
        Args:
            freq: pandas frequency string (e.g., "H", "D", "M")

        Returns:
            range of valid patch sizes
        """
        # 실제로는 더 복잡한 parsing 로직 포함
        # 여기서는 간단히 첫 글자만 사용
        freq_key = freq[0] if freq else "D"
        start, stop = self.DEFAULT_RANGES.get(freq_key, (16, 32))
        return range(start, stop + 1)


def get_patch_size(
    target_length: int,
    freq: str,
    available_patch_sizes: tuple[int, ...] = (8, 16, 32, 64, 128),
    min_time_patches: int = 2,
):
    """
    주어진 time series에 대해 적절한 patch size를 결정.

    Args:
        target_length: time series 길이
        freq: frequency string (e.g., "H", "D", "M")
        available_patch_sizes: 모델이 지원하는 patch size list
        min_time_patches: 최소 patch 개수

    Returns:
        selected patch size

    실제 구현: src/uni2ts/transform/patch.py:78-121
    """
    constraints = DefaultPatchSizeConstraints()
    valid_range = constraints(freq)

    # 시계열 길이 고려: 너무 큰 patch size는 사용 불가
    patch_size_ceil = target_length // min_time_patches

    # 최종 patch size 후보
    candidates = [
        ps for ps in available_patch_sizes
        if (ps in valid_range) and (ps <= patch_size_ceil)
    ]

    if not candidates:
        raise ValueError(
            f"No valid patch size for length={target_length}, freq={freq}"
        )

    # 랜덤 선택 (학습 시 augmentation 효과)
    return np.random.choice(candidates)


def example_patch_size_selection():
    """Patch size selection 예제"""
    scenarios = [
        (512, "H"),    # Hourly data, 512 시간
        (365, "D"),    # Daily data, 365 일
        (36, "M"),     # Monthly data, 36 개월
        (1000, "T"),   # Minute data, 1000 분
    ]

    print("Patch Size Selection Examples:")
    print("-" * 60)
    for length, freq in scenarios:
        patch_size = get_patch_size(length, freq)
        num_patches = length // patch_size
        print(f"Length: {length:4d}, Freq: {freq:2s} -> "
              f"Patch Size: {patch_size:3d}, Num Patches: {num_patches:3d}")


# =============================================================================
# 3. Patching Process
# =============================================================================

def patchify(
    arr: np.ndarray,
    patch_size: int,
    max_patch_size: int = 128,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Time series를 patch sequence로 변환.

    Args:
        arr: [var, time] - input time series
        patch_size: actual patch size to use
        max_patch_size: maximum patch size (for padding)
        pad_value: value to use for padding

    Returns:
        patched: [var, num_patches, max_patch_size]

    실제 구현: src/uni2ts/transform/patch.py:151-159
    """
    # 1. Reshape: [var, time] -> [var, num_patches, patch_size]
    arr = rearrange(arr, "... (time patch) -> ... time patch", patch=patch_size)

    # 2. Padding: patch dimension을 max_patch_size로 맞춤
    pad_width = [(0, 0) for _ in range(arr.ndim)]
    pad_width[-1] = (0, max_patch_size - patch_size)
    arr = np.pad(arr, pad_width, mode="constant", constant_values=pad_value)

    return arr


def unpatchify(
    arr: torch.Tensor,
    patch_size: int,
    target_length: int,
) -> torch.Tensor:
    """
    Patch sequence를 time series로 역변환.

    Args:
        arr: [batch, num_patches, max_patch_size] - patched predictions
        patch_size: actual patch size used
        target_length: desired output length

    Returns:
        time_series: [batch, target_length]
    """
    # 1. Extract actual patch (remove padding)
    arr = arr[..., :patch_size]

    # 2. Reshape: [batch, num_patches, patch_size] -> [batch, time]
    arr = rearrange(arr, "... time patch -> ... (time patch)")

    # 3. Truncate to target length
    arr = arr[..., :target_length]

    return arr


def example_patching():
    """Patching/Unpatching 예제"""
    # 입력 time series
    batch_size = 2
    var_dim = 1
    time_len = 512  # hourly data
    patch_size = 64
    max_patch_size = 128

    # 원본 데이터
    ts = np.random.randn(batch_size, var_dim, time_len)
    print(f"Original time series shape: {ts.shape}")

    # Patching
    patched = np.stack([
        patchify(ts[i], patch_size, max_patch_size)
        for i in range(batch_size)
    ])
    print(f"Patched shape: {patched.shape}")
    # [batch_size, var_dim, num_patches=8, max_patch_size=128]

    # Unpatching (역변환)
    patched_torch = torch.from_numpy(patched)
    reconstructed = unpatchify(patched_torch, patch_size, time_len)
    print(f"Reconstructed shape: {reconstructed.shape}")
    # [batch_size, var_dim, time_len=512]


# =============================================================================
# 4. Binary Attention Bias (Variate Dimension)
# =============================================================================

class BinaryAttentionBias(nn.Module):
    """
    같은 variate 내의 token 간 attention을 강화하는 bias.

    핵심 아이디어:
    - 같은 variate_id를 가진 token pair: bias_1 적용
    - 다른 variate_id를 가진 token pair: bias_0 적용
    - 이를 통해 multivariate time series에서 각 변수의 독립성 유지

    실제 구현: src/uni2ts/module/position/attn_bias.py:67-87
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        # 2개의 bias: [같은 variate, 다른 variate]
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=num_heads)

    def forward(
        self,
        query_id: torch.Tensor,  # [batch, q_len]
        kv_id: torch.Tensor,     # [batch, kv_len]
    ) -> torch.Tensor:
        """
        Returns:
            bias: [batch, num_heads, q_len, kv_len]
        """
        # 같은 variate_id인지 확인: [batch, q_len, kv_len]
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))

        # Bias weight: [2, num_heads]
        weight = self.emb.weight

        # 조건부 bias 적용
        # 다른 variate: weight[0], 같은 variate: weight[1]
        bias = (~ind).float().unsqueeze(1) * weight[0] + \
               ind.float().unsqueeze(1) * weight[1]

        return bias  # [batch, num_heads, q_len, kv_len]


def example_binary_attention_bias():
    """Binary Attention Bias 예제"""
    batch_size = 2
    seq_len = 10
    num_heads = 8

    # Variate ID: 2개의 변수
    # [0,0,0,0,0, 1,1,1,1,1]
    variate_id = torch.tensor([0] * 5 + [1] * 5).unsqueeze(0).expand(batch_size, -1)

    # Binary Attention Bias
    bias_layer = BinaryAttentionBias(dim=512, num_heads=num_heads)
    bias = bias_layer(variate_id, variate_id)

    print(f"Variate ID: {variate_id[0]}")
    print(f"Bias shape: {bias.shape}")  # [batch, num_heads, seq_len, seq_len]
    print(f"\nBias matrix (first head, first batch):")
    print(bias[0, 0].detach().numpy())
    # 같은 variate끼리는 높은 값, 다른 variate 간에는 낮은 값


# =============================================================================
# 5. Rotary Position Encoding (Time Dimension)
# =============================================================================

class RotaryProjection(nn.Module):
    """
    Time dimension에 대한 Rotary Position Encoding (RoPE).

    핵심 아이디어:
    - 각 dimension에 대해 서로 다른 frequency로 rotation 적용
    - Position에 따라 query와 key를 rotate
    - 상대적 위치 정보를 내재적으로 인코딩

    실제 구현: src/uni2ts/module/position/attn_projection.py:55-107
    """

    def __init__(
        self,
        proj_width: int,
        max_len: int = 512,
        base: int = 10000,
    ):
        super().__init__()
        assert proj_width % 2 == 0, "proj_width must be even"

        self.proj_width = proj_width

        # Frequency 계산: theta_i = base^(-2i/d)
        theta = 1.0 / (base ** (torch.arange(0, proj_width, 2).float() / proj_width))
        self.register_buffer("theta", theta, persistent=False)

        # Precompute cos/sin for efficiency
        self._init_freq(max_len)

    def _init_freq(self, max_len: int):
        """Precompute rotation matrices"""
        position = torch.arange(max_len, dtype=self.theta.dtype)
        # m_theta: [max_len, proj_width/2]
        m_theta = torch.einsum("i,j->ij", position, self.theta)
        # Duplicate: [max_len, proj_width/2] -> [max_len, proj_width]
        m_theta = repeat(m_theta, "length width -> length (width 2)")

        self.register_buffer("cos", torch.cos(m_theta), persistent=False)
        self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        """Rotate x by 90 degrees in each 2D plane"""
        # [x1, x2, x3, x4, ...] -> [-x2, x1, -x4, x3, ...]
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange(torch.stack([-x2, x1]), "r ... dim -> ... (dim r)")

    def forward(self, x: torch.Tensor, seq_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_heads, seq_len, head_dim]
            seq_id: [batch, seq_len] - time position indices

        Returns:
            rotated: [batch, num_heads, seq_len, head_dim]
        """
        # Get rotation matrices for positions
        rot_cos = self.cos[seq_id]  # [batch, seq_len, proj_width]
        rot_sin = self.sin[seq_id]

        # Apply rotation: x' = x * cos + rotate(x) * sin
        return rot_cos.unsqueeze(1) * x + rot_sin.unsqueeze(1) * self._rotate(x)


def example_rotary_encoding():
    """Rotary Position Encoding 예제"""
    batch_size = 2
    num_heads = 8
    seq_len = 10
    head_dim = 64

    # Query/Key
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Time position indices
    seq_id = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Rotary encoding
    rope = RotaryProjection(proj_width=head_dim, max_len=512)
    query_rotated = rope(query, seq_id)
    key_rotated = rope(key, seq_id)

    print(f"Original query shape: {query.shape}")
    print(f"Rotated query shape: {query_rotated.shape}")

    # Attention scores with RoPE
    # RoPE의 특성: (RoPE(q) @ RoPE(k).T)는 상대적 위치에만 의존
    attn_scores = torch.einsum(
        "bhqd,bhkd->bhqk",
        query_rotated,
        key_rotated,
    ) / np.sqrt(head_dim)

    print(f"Attention scores shape: {attn_scores.shape}")
    # [batch, num_heads, seq_len, seq_len]


# =============================================================================
# 6. Complete Example: Processing Different Frequencies
# =============================================================================

def complete_example():
    """
    서로 다른 frequency를 가진 3개의 시계열을 동시에 처리하는 완전한 예제.
    """
    print("=" * 80)
    print("Complete Example: Processing Multiple Frequencies")
    print("=" * 80)

    # 3개의 시계열
    scenarios = [
        ("TS1", 512, "H", "Hourly electricity consumption"),
        ("TS2", 365, "D", "Daily sales"),
        ("TS3", 36, "M", "Monthly revenue"),
    ]

    max_patch_size = 128
    d_model = 512

    # Multi-InSize Linear layer
    layer = MultiInSizeLinear(
        in_features_ls=(8, 16, 32, 64, 128),
        out_features=d_model,
    )

    all_patches = []
    all_patch_sizes = []

    print("\nStep 1: Patching")
    print("-" * 80)

    for name, length, freq, desc in scenarios:
        # 1. Patch size 결정
        patch_size = get_patch_size(length, freq)
        num_patches = length // patch_size

        print(f"{name}: {desc}")
        print(f"  Length: {length}, Freq: {freq}")
        print(f"  Patch Size: {patch_size}, Num Patches: {num_patches}")

        # 2. Patching
        ts = np.random.randn(1, length)  # [1, time]
        # Pad to multiple of patch_size
        pad_length = -length % patch_size
        if pad_length > 0:
            ts = np.pad(ts, [(0, 0), (0, pad_length)])

        patched = patchify(ts, patch_size, max_patch_size)
        print(f"  Patched shape: {patched.shape}\n")

        all_patches.append(torch.from_numpy(patched).float())
        all_patch_sizes.append(
            torch.full((patched.shape[1],), patch_size, dtype=torch.long)
        )

    print("\nStep 2: Batching")
    print("-" * 80)

    # Pad to same length
    max_seq_len = max(p.shape[1] for p in all_patches)
    batched_patches = []
    for patches in all_patches:
        if patches.shape[1] < max_seq_len:
            pad_len = max_seq_len - patches.shape[1]
            patches = F.pad(patches, (0, 0, 0, pad_len))
        batched_patches.append(patches)

    # Batch: [batch=3, var=1, seq_len, max_patch_size]
    batched = torch.cat(batched_patches, dim=0)
    print(f"Batched patches shape: {batched.shape}")

    # Patch sizes: [batch=3, seq_len]
    batched_patch_sizes = []
    for ps in all_patch_sizes:
        if ps.shape[0] < max_seq_len:
            ps = F.pad(ps, (0, max_seq_len - ps.shape[0]))
        batched_patch_sizes.append(ps)
    batched_patch_sizes = torch.stack(batched_patch_sizes)
    print(f"Batched patch sizes shape: {batched_patch_sizes.shape}")
    print(f"Patch sizes per sample:\n{batched_patch_sizes}")

    print("\nStep 3: Multi-InSize Linear Embedding")
    print("-" * 80)

    # Reshape for layer: [batch, seq_len, max_patch_size]
    x = batched.squeeze(1)  # Remove var dimension for simplicity
    embeddings = layer(x, batched_patch_sizes)

    print(f"Input shape: {x.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Each sample uses different patch-size-specific weights!")

    print("\n" + "=" * 80)
    print("Key Insight:")
    print("  - TS1 (hourly, ps=64): uses weight[3]")
    print("  - TS2 (daily, ps=32): uses weight[2]")
    print("  - TS3 (monthly, ps=8): uses weight[0]")
    print("  - All processed by the SAME model with SHARED transformer!")
    print("=" * 80)


# =============================================================================
# Main: Run all examples
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Moirai 핵심 컴포넌트 코드 예제")
    print("=" * 80 + "\n")

    print("\n[1] Multi-InSize Linear Example")
    print("-" * 80)
    example_multi_in_size_linear()

    print("\n\n[2] Patch Size Selection Example")
    print("-" * 80)
    example_patch_size_selection()

    print("\n\n[3] Patching/Unpatching Example")
    print("-" * 80)
    example_patching()

    print("\n\n[4] Binary Attention Bias Example")
    print("-" * 80)
    example_binary_attention_bias()

    print("\n\n[5] Rotary Position Encoding Example")
    print("-" * 80)
    example_rotary_encoding()

    print("\n\n[6] Complete Example: Processing Multiple Frequencies")
    print("-" * 80)
    complete_example()

    print("\n\n" + "=" * 80)
    print("모든 예제 실행 완료!")
    print("=" * 80)
