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
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

        assert self.w_gate.shape == (hidden_dim, dim)
        assert self.w_up.shape == (hidden_dim, dim)
        assert self.w_down.shape == (dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        gate_proj = mx.matmul(x, mx.transpose(self.w_gate))
        gate = silu(gate_proj)

        up = mx.matmul(x, mx.transpose(self.w_up))

        hidden = gate * up

        output = mx.matmul(hidden, mx.transpose(self.w_down))

        return output


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
        # model dims
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps

        # attn weights
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

        # q/k norm weights
        self.q_norm = q_norm
        self.k_norm = k_norm

        # RoPE config
        self.max_seq_len = max_seq_len
        self.theta = theta

        # self attn
        self.self_attn = Qwen3MultiHeadAttention(
            self.hidden_size,
            self.num_attention_heads,
            self.num_kv_heads,
            self.head_dim,
            self.wq,
            self.wk,
            self.wv,
            self.wo,
            self.q_norm,
            self.k_norm,
            self.max_seq_len,
            self.theta,
            self.rms_norm_eps,
        )

        # mlp weights
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

        # mlp
        self.mlp = Qwen3MLP(
            self.hidden_size,
            self.intermediate_size,
            self.w_gate,
            self.w_up,
            self.w_down,
        )

        # transformer-layer norm weights
        self.w_input_layernorm = w_input_layernorm
        self.w_post_attention_layernorm = w_post_attention_layernorm

        # layer norm
        self.input_layernorm = RMSNorm(
            self.hidden_size, self.w_input_layernorm, self.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, self.w_post_attention_layernorm, self.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        attention_output = self.self_attn(self.input_layernorm(x), mask)
        hidden = x + attention_output

        mlp_output = self.mlp(self.post_attention_layernorm(hidden))
        output = hidden + mlp_output

        return output


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
