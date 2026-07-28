import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x_float = x.astype(mx.float32)

        mean_square = mx.mean(
            x_float**2,
            axis=-1,
            keepdims=True,
        )

        normalized = x_float * mx.rsqrt(mean_square + self.eps)
        normalized = normalized.astype(input_dtype)

        return normalized * self.weight
