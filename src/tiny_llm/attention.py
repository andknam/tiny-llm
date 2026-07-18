import math
import mlx.core as mx
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = query.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(d_k)

    scores = mx.matmul(query, mx.swapaxes(key, -1, -2)) * scale

    if mask is not None:
        scores = scores + mask

    attention_probs = softmax(scores, axis=-1)
    return mx.matmul(attention_probs, value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        assert hidden_size % num_heads == 0  # must be an int
        self.head_dim = self.hidden_size // self.num_heads

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        query_tensor = linear(query, self.wq)  # (N, L, H * D)
        key_tensor = linear(key, self.wk)
        value_tensor = linear(value, self.wv)

        # split ea token embedding into multiple attn heads
        assert query.shape == key.shape == value.shape
        N, L, _ = query_tensor.shape
        H, D = self.num_heads, self.head_dim

        query_reshaped = mx.reshape(query_tensor, (N, L, H, D)).swapaxes(1, 2)
        key_reshaped = mx.reshape(key_tensor, (N, L, H, D)).swapaxes(1, 2)
        value_reshaped = mx.reshape(value_tensor, (N, L, H, D)).swapaxes(1, 2)

        # calc attn indep for ea head
        attention = scaled_dot_product_attention_simple(
            query_reshaped, key_reshaped, value_reshaped, mask=mask
        )
        # merge attn heads back into single embedding per token
        attention = attention.swapaxes(1, 2).reshape(N, L, H * D)

        # project concatenated heads back to model embedding dim
        return linear(attention, self.wo)  # (N, L, E)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    H_q, L, D = query.shape[-3:]
    batch_shape = query.shape[:-3]

    H, S, _ = key.shape[-3:]

    n_repeats = H_q // H

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # (N, H, n_repeats, L, D)
    # split query heads into groups that share same K/V head (H)
    query_reshaped = mx.reshape(query, (*batch_shape, H, n_repeats, L, D))

    # add broadcast dim so K/V head is reused across its query group
    key_expanded = mx.expand_dims(key, axis=-3)  # (N, H, 1, S, D)
    value_expanded = mx.expand_dims(value, axis=-3)  # (N, H, 1, S, D)

    scores = mx.matmul(query_reshaped, mx.swapaxes(key_expanded, -1, -2)) * scale

    if mask is not None:
        # reshape mask to match grouped query heads
        mask = mx.reshape(mask, (*batch_shape, H, n_repeats, L, S))
        scores = scores + mask

    attention_probs = softmax(scores, axis=-1)
    output = mx.matmul(attention_probs, value_expanded)

    return mx.reshape(
        output,
        (*batch_shape, H_q, L, D),
    )


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
