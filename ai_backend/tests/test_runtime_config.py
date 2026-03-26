from services.runtime_config import parse_cors_settings


def test_parse_cors_settings_disables_credentials_for_wildcard() -> None:
    settings = parse_cors_settings("*", "true")
    assert settings.allow_origins == ["*"]
    assert settings.allow_credentials is False


def test_parse_cors_settings_keeps_credentials_for_specific_origins() -> None:
    settings = parse_cors_settings("http://localhost:3000,http://127.0.0.1:3000", "true")
    assert settings.allow_credentials is True
    assert len(settings.allow_origins) == 2
