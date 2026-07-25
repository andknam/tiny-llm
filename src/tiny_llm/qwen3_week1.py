import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert hidden_size // num_heads == head_dim
        self.head_dim = head_dim

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        self.q_norm = q_norm
        self.k_norm = k_norm

        self.theta = theta
        self.max_seq_len = max_seq_len
        self.rms_norm_eps = rms_norm_eps

        self.rope = RoPE(
            self.head_dim, self.max_seq_len, base=self.theta, traditional=False
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = linear(x, self.wq)  # (B, L, H_q * D)
        k = linear(x, self.wk)  # (B, L, H * D)
        v = linear(x, self.wv)  # (B, L, H * D)

        # sep projected dims into indiv attn heads
        q = mx.reshape(q, (B, L, self.num_heads, self.head_dim))  # (B, L, H_q, D)
        k = mx.reshape(k, (B, L, self.num_kv_heads, self.head_dim))
        v = mx.reshape(v, (B, L, self.num_kv_heads, self.head_dim))

        # normalize ea Q/K head across its head dim
        q = mx.fast.rms_norm(q, self.q_norm, self.rms_norm_eps)
        k = mx.fast.rms_norm(k, self.k_norm, self.rms_norm_eps)

        # apply pos encoding to q and k
        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))

        # grouped attn expects H before seq len
        q = mx.swapaxes(q, 1, 2)  # (B, H_q, L, D)
        k = mx.swapaxes(k, 1, 2)  # (B, H, L, D)
        v = mx.swapaxes(v, 1, 2)  # (B, H, L, D)

        attn = scaled_dot_product_attention_grouped(
            q, k, v, mask=mask
        )  # (B, H_q, L, D)

        # concat query heads into one embedding per token
        attn = mx.swapaxes(attn, 1, 2)  # (B, L, H_q, D)
        attn = mx.reshape(
            attn,
            (B, L, self.num_heads * self.head_dim),
        )

        return linear(attn, self.wo)  # (B, L, E)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        pass

    def __call__(self, x: mx.array) -> mx.array:
        pass


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
