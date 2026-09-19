from core.llm_service import (
    HostedModelService,
    build_hosted_model_service,
    get_hosted_model_service,
)


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


def test_reviewer_supplied_settings_build_a_service_only_when_complete():
    assert build_hosted_model_service("", "https://provider.example", "sk-x") is None
    assert build_hosted_model_service("m", "", "sk-x") is None
    assert build_hosted_model_service("m", "https://provider.example", "") is None
    assert build_hosted_model_service("   ", "  ", "  ") is None

    service = build_hosted_model_service("gpt-4o-mini", "https://api.openai.com", "sk-test")
    assert isinstance(service, HostedModelService)
    assert service.is_configured()
    assert service.base_url == "https://api.openai.com"


def test_reviewer_supplied_settings_are_trimmed():
    service = build_hosted_model_service("  gpt-4o-mini  ", "  https://api.openai.com  ", "  sk-test  ")
    assert service is not None
    assert service.model == "gpt-4o-mini"
    assert service.base_url == "https://api.openai.com"
    assert service.api_key == "sk-test"
