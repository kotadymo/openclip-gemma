import inspect
from collections.abc import Callable
from typing import Any

import openclip_gemma


def _signature_contract(
    function: Callable[..., Any],
) -> list[tuple[str, str, object]]:
    return [
        (parameter.name, parameter.kind.name, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


def test_public_api_exports() -> None:
    assert openclip_gemma.__all__ == [
        "EmbeddingGemmaCLIP",
        "create_model",
        "create_model_and_transforms",
        "create_model_from_pretrained",
        "get_tokenizer",
        "list_models",
        "list_pretrained",
    ]
    assert openclip_gemma.EmbeddingGemmaCLIP.__name__ == "EmbeddingGemmaCLIP"
    assert callable(openclip_gemma.create_model)
    assert callable(openclip_gemma.create_model_and_transforms)
    assert callable(openclip_gemma.create_model_from_pretrained)
    assert callable(openclip_gemma.get_tokenizer)
    assert callable(openclip_gemma.list_models)
    assert callable(openclip_gemma.list_pretrained)


def test_factory_signatures_preserve_v040_contract() -> None:
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD.name
    keyword_only = inspect.Parameter.KEYWORD_ONLY.name
    variadic_keyword = inspect.Parameter.VAR_KEYWORD.name
    empty = inspect.Parameter.empty

    factory_arguments = [
        ("model_name", positional, "embeddinggemma-2"),
        ("pretrained", positional, "google/embeddinggemma-2"),
        ("precision", positional, "fp32"),
        ("device", positional, "cpu"),
        ("cache_dir", positional, None),
        ("output_dict", keyword_only, False),
        ("logit_scale_mode", keyword_only, "strict"),
        ("modalities", keyword_only, None),
        ("kwargs", variadic_keyword, empty),
    ]
    assert _signature_contract(openclip_gemma.create_model) == factory_arguments
    assert (
        _signature_contract(openclip_gemma.create_model_and_transforms)
        == factory_arguments
    )
    assert _signature_contract(openclip_gemma.create_model_from_pretrained) == [
        ("model_name", positional, "embeddinggemma-2"),
        ("pretrained", positional, "official"),
        ("precision", positional, "fp32"),
        ("device", positional, "cpu"),
        ("return_transform", keyword_only, True),
        ("cache_dir", keyword_only, None),
        ("output_dict", keyword_only, False),
        ("logit_scale_mode", keyword_only, "strict"),
        ("modalities", keyword_only, None),
        ("kwargs", variadic_keyword, empty),
    ]
    assert _signature_contract(openclip_gemma.get_tokenizer) == [
        ("model_name", positional, "embeddinggemma-2"),
        ("prompt_name", keyword_only, None),
        ("context_length", keyword_only, None),
        ("cache_dir", keyword_only, None),
    ]
    assert _signature_contract(openclip_gemma.list_models) == []
    assert _signature_contract(openclip_gemma.list_pretrained) == [
        ("as_str", positional, False)
    ]


def test_model_signatures_preserve_v040_contract() -> None:
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD.name
    keyword_only = inspect.Parameter.KEYWORD_ONLY.name
    empty = inspect.Parameter.empty
    model = openclip_gemma.EmbeddingGemmaCLIP

    assert _signature_contract(model.encode_image) == [
        ("self", positional, empty),
        ("image", positional, empty),
        ("normalize", positional, False),
        ("truncate_dim", keyword_only, None),
    ]
    assert _signature_contract(model.encode_text) == [
        ("self", positional, empty),
        ("text", positional, empty),
        ("normalize", positional, False),
        ("truncate_dim", keyword_only, None),
    ]
    assert _signature_contract(model.forward) == [
        ("self", positional, empty),
        ("image", positional, None),
        ("text", positional, None),
        ("truncate_dim", keyword_only, None),
    ]
    assert _signature_contract(model.get_logits) == [
        ("self", positional, empty),
        ("image", positional, empty),
        ("text", positional, empty),
        ("truncate_dim", keyword_only, None),
    ]
