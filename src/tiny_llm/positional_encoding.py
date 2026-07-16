import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

        assert dims % 2 == 0

        self.sin_freqs = []
        self.cos_freqs = []

        for pos in range(seq_len):
            sin_row = []
            cos_row = []

            # compute angles for ea pair
            for pair_idx in range(dims // 2):
                frequency = 1.0 / (base ** (2 * pair_idx / dims))
                angle = pos * frequency

                sin_row.append(mx.sin(angle))
                cos_row.append(mx.cos(angle))

            self.sin_freqs.append(sin_row)
            self.cos_freqs.append(cos_row)

        # (seq_len, dims // 2)
        self.sin_freqs = mx.array(self.sin_freqs)
        self.cos_freqs = mx.array(self.cos_freqs)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        n, length, num_heads, dims = x.shape
        assert dims == self.dims

        if offset is None:
            sin = self.sin_freqs[:length]
            cos = self.cos_freqs[:length]

        # will be used when impl continuous batching feature
        else:
            sin = self.sin_freqs[offset]
            cos = self.cos_freqs[offset]

        # consecutive pairs (x0, x1), (x2, x3) etc
        if self.traditional:
            # (N, L, H, D) -> (N, L, H, D // 2, 2)
            x_pairs = x.reshape(
                n,
                length,
                num_heads,
                dims // 2,
                2,
            )

            even = x_pairs[..., 0]  # (N, L, H, D // 2)
            odd = x_pairs[..., 1]

            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]

            # rotate every pair
            rotated_even = even * cos - odd * sin
            rotated_odd = even * sin + odd * cos

            rotated_pairs = mx.stack(
                [rotated_even, rotated_odd],
                axis=-1,
            )  # (N, L, H, D // 2, 2)

            # (N, L, H, D)
            return rotated_pairs.reshape(
                n,
                length,
                num_heads,
                dims,
            )

        # (x0, x[D/2]), (x1, x[D/2 + 1]), ...
        else:
            half_dim = dims // 2

            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]

            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]

            rotated_x1 = x1 * cos - x2 * sin
            rotated_x2 = x1 * sin + x2 * cos

            return mx.concatenate(
                [rotated_x1, rotated_x2],
                axis=-1,
            )
