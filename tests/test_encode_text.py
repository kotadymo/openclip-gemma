from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from openclip_gemma.model import EmbeddingGemmaCLIP


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 2
    eos_token_id = 1
    vocab_size = 32


class FakeBackend(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(
                max_position_embeddings=8,
                vocab_size=FakeTokenizer.vocab_size,
            )
        )
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        self.calls.append(
            {"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone()}
        )
        hidden = input_ids.to(dtype=self.scale.dtype).unsqueeze(-1).expand(-1, -1, 768)
        hidden = hidden.clone()
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden[:, :, 1] = positions.to(dtype=hidden.dtype)
        return SimpleNamespace(last_hidden_state=hidden * self.scale)


@pytest.fixture
def model() -> tuple[EmbeddingGemmaCLIP, FakeBackend]:
    backend = FakeBackend()
    processor = SimpleNamespace(tokenizer=FakeTokenizer())
    return EmbeddingGemmaCLIP(backend, processor).eval(), backend


def test_encode_text_pools_only_non_padding_tokens_and_preserves_order(
    model: tuple[EmbeddingGemmaCLIP, FakeBackend],
) -> None:
    wrapper, backend = model
    tokens = torch.tensor(
        [
            [2, 3, 1, 0],
            [2, 4, 5, 1],
        ],
        dtype=torch.long,
    )

    result = wrapper.encode_text(tokens)

    expected_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    torch.testing.assert_close(backend.calls[0]["input_ids"], tokens)
    torch.testing.assert_close(backend.calls[0]["attention_mask"], expected_mask)
    assert result.shape == (2, 768)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.isfinite(result).all()
    assert not result.requires_grad
    torch.testing.assert_close(result[0], wrapper.encode_text(tokens[:1])[0])
    torch.testing.assert_close(result[1], wrapper.encode_text(tokens[1:])[0])


def test_encode_text_preserves_native_normalization_and_normalize_argument(
    model: tuple[EmbeddingGemmaCLIP, FakeBackend],
) -> None:
    wrapper, _ = model
    tokens = torch.tensor([[2, 3, 4, 1]], dtype=torch.long)

    native = wrapper.encode_text(tokens, normalize=False)
    explicitly_normalized = wrapper.encode_text(tokens, normalize=True)

    torch.testing.assert_close(native, explicitly_normalized)
    torch.testing.assert_close(native.norm(dim=-1), torch.ones(1))


def test_encode_text_accepts_single_and_batched_tokens(
    model: tuple[EmbeddingGemmaCLIP, FakeBackend],
) -> None:
    wrapper, _ = model

    single = wrapper.encode_text(torch.tensor([[2, 3, 1]], dtype=torch.long))
    batch = wrapper.encode_text(
        torch.tensor([[2, 3, 1, 0], [2, 4, 5, 1]], dtype=torch.long)
    )

    assert single.shape == (1, 768)
    assert batch.shape == (2, 768)


@pytest.mark.parametrize(
    ("tokens", "exception", "message"),
    [
        (torch.tensor([2, 3, 1]), ValueError, r"\[B,L\]"),
        (torch.empty((0, 3), dtype=torch.long), ValueError, "at least one"),
        (torch.empty((1, 0), dtype=torch.long), ValueError, "length must be positive"),
        (torch.tensor([[2.0, 3.0, 1.0]]), TypeError, "torch.int64"),
        (torch.tensor([[2, -1, 1]]), ValueError, "range"),
        (torch.tensor([[2, 3, 32]]), ValueError, "range"),
        (torch.tensor([[2, 0, 3, 1]]), ValueError, "only as a right-padding suffix"),
        (torch.tensor([[2, 3, 0]]), ValueError, "official EOS token"),
        (torch.tensor([[3, 4, 1]]), ValueError, "official BOS token"),
        (
            torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 1]]),
            ValueError,
            "supported context limit",
        ),
    ],
)
def test_encode_text_rejects_invalid_token_tensors(
    model: tuple[EmbeddingGemmaCLIP, FakeBackend],
    tokens: torch.Tensor,
    exception: type[Exception],
    message: str,
) -> None:
    wrapper, _ = model

    with pytest.raises(exception, match=message):
        wrapper.encode_text(tokens)


def test_encode_text_rejects_non_tensor_input(
    model: tuple[EmbeddingGemmaCLIP, FakeBackend],
) -> None:
    wrapper, _ = model

    with pytest.raises(TypeError, match="torch.Tensor"):
        wrapper.encode_text(["a cat"])  # type: ignore[arg-type]


def test_encode_text_rejects_device_mismatch(
    model: tuple[EmbeddingGemmaCLIP, FakeBackend],
) -> None:
    wrapper, _ = model
    tokens = torch.empty((1, 3), dtype=torch.long, device="meta")

    with pytest.raises(ValueError, match="model backend is on"):
        wrapper.encode_text(tokens)


def test_encode_text_rejects_non_bool_normalize(
    model: tuple[EmbeddingGemmaCLIP, FakeBackend],
) -> None:
    wrapper, _ = model

    with pytest.raises(TypeError, match="normalize must be a bool"):
        wrapper.encode_text(
            torch.tensor([[2, 3, 1]], dtype=torch.long),
            normalize=1,  # type: ignore[arg-type]
        )


def test_fake_backend_pooling_matches_masked_mean(
    model: tuple[EmbeddingGemmaCLIP, FakeBackend],
) -> None:
    wrapper, _ = model
    tokens = torch.tensor([[2, 3, 1, 0]], dtype=torch.long)
    hidden = tokens.float().unsqueeze(-1).expand(-1, -1, 768).clone()
    hidden[:, :, 1] = torch.arange(4, dtype=torch.float32)
    expected = F.normalize(hidden[:, :3].mean(dim=1), p=2, dim=-1)

    actual = wrapper.encode_text(tokens)

    torch.testing.assert_close(actual, expected)
