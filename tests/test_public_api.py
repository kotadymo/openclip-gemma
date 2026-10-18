import openclip_gemma


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
