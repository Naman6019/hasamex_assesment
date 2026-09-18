from core.llm_service import HostedModelService, get_hosted_model_service


def test_hosted_selector_requires_all_server_side_settings(monkeypatch):
    for key in ("CLOUD_MODEL_BASE_URL", "CLOUD_MODEL_NAME", "CLOUD_MODEL_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert get_hosted_model_service() is None

    monkeypatch.setenv("CLOUD_MODEL_BASE_URL", "https://provider.example")
    monkeypatch.setenv("CLOUD_MODEL_NAME", "reviewer-model")
    monkeypatch.setenv("CLOUD_MODEL_API_KEY", "test-secret")
    service = get_hosted_model_service()

    assert isinstance(service, HostedModelService)
    assert service.is_configured()
    assert service.base_url == "https://provider.example"
