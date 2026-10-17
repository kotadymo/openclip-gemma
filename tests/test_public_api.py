import openclip_gemma


def test_public_api_exports() -> None:
    assert openclip_gemma.__all__ == [
        "EmbeddingGemmaCLIP",
        "create_model",
        "create_model_and_transforms",
        "get_tokenizer",
    ]
    assert openclip_gemma.EmbeddingGemmaCLIP.__name__ == "EmbeddingGemmaCLIP"
    assert callable(openclip_gemma.create_model)
    assert callable(openclip_gemma.create_model_and_transforms)
    assert callable(openclip_gemma.get_tokenizer)
