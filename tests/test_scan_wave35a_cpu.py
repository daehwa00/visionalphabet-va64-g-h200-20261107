"""CPU structural and numerical checks for Wave 35A staging and replay."""

import pytest
import torch
from torch import Tensor


def _advance_rows(source: Tensor, a: Tensor, initial: Tensor) -> list[Tensor]:
    state = initial
    result = []
    for value in source:
        state = (
            torch.complex(
                a.real * state.real - a.imag * state.imag,
                a.real * state.imag + a.imag * state.real,
            )
            + value
        )
        result.append(state)
    return result


@pytest.mark.parametrize(
    ("h", "w", "m", "coarse"),
    [
        (56, 56, 160, False),
        (28, 56, 160, True),
        (14, 20, 63, True),
        (7, 19, 63, False),
        (13, 19, 63, False),
        (37, 13, 160, False),
    ],
)
def test_cooperative_column_replay(h: int, w: int, m: int, coarse: bool) -> None:
    rng = torch.Generator().manual_seed(125)
    source = torch.randn(h, w, m, 2, generator=rng, dtype=torch.bfloat16).float().contiguous()
    source = torch.view_as_complex(source)
    if coarse:
        source = source[:, 1::2]
    a = torch.complex(torch.tensor(0.3), torch.tensor(-0.2))
    a = a.expand(source.shape[1], m)
    zero = torch.zeros_like(source[0])
    pos = _advance_rows(source, a, zero)
    neg = list(reversed(_advance_rows(source.flip(0), a.conj(), zero)))
    boundaries = list(range(0, h, 8))
    previous = [zero, *(pos[index - 1] for index in boundaries[1:])]
    for block in reversed(range(len(boundaries))):
        start, end = boundaries[block], min(h, boundaries[block] + 8)
        columns = source.shape[1]
        for col in range(0, columns, 4):
            tile = slice(col, col + 4)
            positive = _advance_rows(source[start:end, tile], a[tile], previous[block][tile])
            current = neg[end][tile] if end < h else zero[tile]
            negative = _advance_rows(source[start:end, tile].flip(0), a[tile].conj(), current)
            for i, row in enumerate(range(start, end)):
                torch.testing.assert_close(positive[i], pos[row][tile], atol=0, rtol=0)
                torch.testing.assert_close(negative[-1 - i], neg[row][tile], atol=0, rtol=0)


def _bf16(value: Tensor) -> Tensor:
    return value.bfloat16().float()


@pytest.mark.parametrize("m", [63, 160])
def test_hidden_split_aosoa_partials(m: int) -> None:
    generator = torch.Generator().manual_seed(125)
    cells, tiles = 11, (m + 31) // 32
    features = _bf16(torch.randn(cells, m, 8, generator=generator))
    wi = _bf16(torch.randn(m, 8, 8, generator=generator) * 0.1)
    bias = _bf16(torch.randn(m, 8, generator=generator) * 0.1)
    wo = _bf16(torch.randn(m, 2, 8, generator=generator) * 0.1)
    dy = _bf16(torch.randn(cells, m, 2, generator=generator))
    layout = torch.zeros(cells * tiles * 8 * 32)
    c, mode, k = torch.arange(cells)[:, None, None], torch.arange(m)[None, :, None], torch.arange(8)
    offsets = ((c * tiles + mode // 32) * 8 + k) * 32 + mode % 32
    layout[offsets] = features
    torch.testing.assert_close(layout[offsets], features, atol=0, rtol=0)

    def backward(p: Tensor, output_grad: Tensor, owned: slice) -> tuple[Tensor, ...]:
        weights = wi[:, owned]
        pre = torch.zeros((*p.shape[:2], weights.shape[1]))
        for k in range(8):
            pre = torch.addcmul(pre, p[..., k, None], weights[..., k])
        z = _bf16(_bf16(pre) + bias[:, owned])
        sig = z.sigmoid()
        act = _bf16(z * sig)
        hidden_grad = torch.einsum("cmo,moh->cmh", output_grad, wo[..., owned])
        dz = _bf16(_bf16(hidden_grad) * (sig * (1 + z * (1 - sig))))
        return (
            torch.einsum("cmh,mhk->cmk", dz, weights),
            torch.einsum("cmh,cmk->mhk", dz, p),
            dz.sum(dim=0),
            torch.einsum("cmo,cmh->moh", output_grad, act),
        )

    expected = backward(features, dy, slice(None))
    groups = []
    dp = torch.zeros_like(features)
    for worker in range(3):
        owned_cells = slice(worker, None, 3)
        warps = [
            backward(layout[offsets[owned_cells]], dy[owned_cells], slice(h, h + 2))
            for h in range(0, 8, 2)
        ]
        dp[owned_cells] = sum(warp[0] for warp in warps)
        groups.append(
            tuple(
                torch.cat([warp[i] for warp in warps], dim=dim)
                for i, dim in ((1, 1), (2, 1), (3, 2))
            )
        )
    torch.testing.assert_close(dp, expected[0], atol=2e-6, rtol=1e-6)
    for actual, reference in zip(map(sum, zip(*groups, strict=True)), expected[1:], strict=True):
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=2e-6)

    # Exercise integer arithmetic around the 32-bit boundary without allocating.
    big_cells = 1024 * 56 * 56
    assert ((big_cells - 1) * 5 * 8 + 4 * 8 + 7) * 32 + 31 == big_cells * 160 * 8 - 1
    assert big_cells * 160 * 8 + 1 > 2**31


@pytest.mark.parametrize(
    ("w", "m", "compact"), [(56, 160, False), (28, 128, True), (19, 64, False)]
)
def test_horizontal_vec2_checkpoint_replay(w: int, m: int, compact: bool) -> None:
    generator = torch.Generator().manual_seed(125)
    source = torch.randn(w, m, 2, generator=generator, dtype=torch.bfloat16).float().contiguous()
    source = torch.view_as_complex(source)
    if compact:
        source[1::2] = 0
    a = torch.complex(torch.tensor(0.4), torch.tensor(0.1))
    expected = _advance_rows(source, a, torch.zeros(m, dtype=source.dtype))
    paired = source.reshape(w, m // 2, 2)
    for begin in reversed(range(0, w, 16)):
        checkpoint = (
            expected[begin - 1].reshape(m // 2, 2) if begin else torch.zeros_like(paired[0])
        )
        replay = _advance_rows(paired[begin : begin + 16], a, checkpoint)
        for row, value in enumerate(replay, begin):
            torch.testing.assert_close(value.flatten(), expected[row], atol=0, rtol=0)
