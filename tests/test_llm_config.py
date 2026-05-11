import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_client


@pytest.fixture(autouse=True)
def reset_config_cache():
    llm_client.reset_llm_config_cache()
    yield
    llm_client.reset_llm_config_cache()


def write_config(tmp_path: Path, text: str) -> None:
    (tmp_path / "llm_config.yaml").write_text(text, encoding="utf-8")


def use_tmp_config(monkeypatch, tmp_path: Path, text: str) -> None:
    write_config(tmp_path, text)
    monkeypatch.setattr(llm_client, "_REPO_ROOT", tmp_path)
    llm_client.reset_llm_config_cache()


BASE_CONFIG = """
run_profiles:
  all: null
  quick:
    - constructiveness
    - treereview
  aspects_only:
    - constructiveness
    - novelty
  reviewers_only:
    - treereview
    - local_review

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    base_url: ${OPENAI_BASE_URL:https://api.openai.com/v1}
  gemini:
    api_key: ${GOOGLE_API_KEY}
    base_url: https://generativelanguage.googleapis.com/v1beta/openai/
  mimo:
    api_key: ${MIMO_API_KEY}
    base_url: ${MIMO_BASE_URL:https://api.xiaomimimo.com/v1}
  devmate:
    api_key: ${DEVMATE_API_KEY}
    base_url: ${DEVMATE_BASE_URL:https://devmate.example/api:v3}
    disable_ssl_verify: ${DEVMATE_DISABLE_SSL_VERIFY:false}
  openrouter:
    api_key: ${LLM_API_KEY}
    base_url: https://openrouter.ai/api/v1
defaults:
  provider: openai
  model: gpt-4o-mini
  temperature: "0.2"
  max_tokens: "100"
  max_retries: "2"
  retry_delay: "0.5"
  timeout: "12"
provider_defaults:
  openai:
    default_model: gpt-4o-mini
    default_temperature: "1.0"
    default_max_tokens: "4096"
  mimo:
    default_model: mimo-v2.5-pro
    default_temperature: "0"
    default_max_tokens: "2048"
  devmate:
    default_model: gemini-3-flash-preview
    default_temperature: "0"
    default_max_tokens: "4096"
aspects:
  constructiveness:
    enabled: true
    provider: mimo
    model: openai/gpt-oss-20b
    temperature: "0"
    max_tokens: "4000"
  novelty:
    enabled: false
    provider: mimo
    model: openai/gpt-oss-20b
    temperature: "0"
    max_tokens: "4000"
  flaw_identification:
    enabled: true
    provider: gemini
    temperature: "0.4"
    step1:
      model: gemini-step1
      max_tokens: "777"
    step2:
      model: gemini-step2
reviewers:
  local_review:
    enabled: true
    type: local
    temperature: "0.7"
    batch_size: "8"
    disable_ssl_verify: "true"
  treereview:
    enabled: true
    type: api
    provider: gemini
    model: gemini-2.0-flash
    max_tokens: "32768"
  deepreview:
    enabled: false
    type: local
    model_size: "14B"
"""


# ── Existing tests (adapted for enabled flags) ────────────────────────────

def test_env_interpolation_preserves_url_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("DEVMATE_BASE_URL", raising=False)
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    cfg = llm_client.load_llm_config()

    assert cfg["providers"]["devmate"]["base_url"] == "https://devmate.example/api:v3"


def test_cache_reset_reloads_env(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://first.example/v1")
    assert llm_client.load_llm_config()["providers"]["openai"]["base_url"] == "https://first.example/v1"

    monkeypatch.setenv("OPENAI_BASE_URL", "https://second.example/v1")
    assert llm_client.load_llm_config()["providers"]["openai"]["base_url"] == "https://first.example/v1"

    llm_client.reset_llm_config_cache()
    assert llm_client.load_llm_config()["providers"]["openai"]["base_url"] == "https://second.example/v1"


def test_typed_coercion_and_aspect_step_merge(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    cfg = llm_client.get_aspect_config("flaw_identification", step="step1")
    reviewer = llm_client.get_reviewer_config("local_review")

    assert cfg["provider"] == "gemini"
    assert cfg["temperature"] == 0.4
    assert cfg["max_tokens"] == 777
    assert cfg["max_retries"] == 2
    assert cfg["retry_delay"] == 0.5
    assert cfg["timeout"] == 12.0
    # enabled should NOT leak into the config dict
    assert "enabled" not in cfg
    assert reviewer["temperature"] == 0.7
    assert reviewer["batch_size"] == 8
    assert reviewer["disable_ssl_verify"] is True
    assert "enabled" not in reviewer


def test_unknown_aspect_step_and_local_reviewer_rejection(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    with pytest.raises(KeyError):
        llm_client.get_aspect_config("missing")
    with pytest.raises(KeyError):
        llm_client.get_aspect_config("flaw_identification", step="missing")
    with pytest.raises(ValueError, match="type 'local'"):
        llm_client.PRISMLLMClient.for_reviewer("local_review")


def test_ai_config_routes_aspects_from_yaml_and_legacy_provider_shape(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)
    ai_config = importlib.reload(importlib.import_module("ai_config"))

    calls = []

    def fake_for_aspect(cls, name, step=None, **overrides):
        calls.append(("aspect", name, step, overrides))
        return calls[-1]

    def fake_for_reviewer(cls, name, **overrides):
        calls.append(("reviewer", name, None, overrides))
        return calls[-1]

    monkeypatch.setattr(llm_client.PRISMLLMClient, "for_aspect", classmethod(fake_for_aspect))
    monkeypatch.setattr(llm_client.PRISMLLMClient, "for_reviewer", classmethod(fake_for_reviewer))

    assert ai_config.get_llm_client("constructiveness") == ("aspect", "constructiveness", None, {})
    assert ai_config.get_llm_client("treereview") == ("reviewer", "treereview", None, {})

    provider_cfg = ai_config.get_provider_config("mimo")
    assert set(provider_cfg) == {
        "api_key",
        "base_url",
        "default_model",
        "default_temperature",
        "default_max_tokens",
    }
    assert provider_cfg["default_model"] == "mimo-v2.5-pro"
    assert provider_cfg["default_max_tokens"] == 2048
    assert ai_config.get_provider_config("gemini-devmate")["default_model"] == "gemini-3-flash-preview"


def test_validate_env_is_yaml_driven_and_ignores_local_reviewers(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)
    ai_config = importlib.reload(importlib.import_module("ai_config"))
    monkeypatch.setattr(ai_config, "MIMO_API_KEY", "")
    monkeypatch.setattr(ai_config, "GOOGLE_API_KEY", "gemini-key")
    monkeypatch.setattr(ai_config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(ai_config, "DEVMATE_API_KEY", "")
    monkeypatch.setattr(ai_config, "LLM_API_KEY", "")
    monkeypatch.setattr(ai_config, "AZURE_OPENAI_API_KEY", "")

    with pytest.raises(EnvironmentError, match="MIMO_API_KEY"):
        ai_config.validate_env()

    monkeypatch.setattr(ai_config, "MIMO_API_KEY", "mimo-key")
    ai_config.validate_env()


# ── New tests: enabled/disabled ───────────────────────────────────────────

def test_is_enabled(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    assert llm_client.is_enabled("aspects", "constructiveness") is True
    assert llm_client.is_enabled("aspects", "novelty") is False
    assert llm_client.is_enabled("aspects", "flaw_identification") is True
    assert llm_client.is_enabled("reviewers", "local_review") is True
    assert llm_client.is_enabled("reviewers", "treereview") is True
    assert llm_client.is_enabled("reviewers", "deepreview") is False
    # Non-existent items return False
    assert llm_client.is_enabled("aspects", "nonexistent") is False


def test_list_all_vs_list_enabled(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    all_aspects = llm_client.list_all("aspects")
    enabled_aspects = llm_client.list_enabled("aspects")
    disabled_aspects = llm_client.list_disabled("aspects")

    assert set(all_aspects) == {"constructiveness", "flaw_identification", "novelty"}
    assert set(enabled_aspects) == {"constructiveness", "flaw_identification"}
    assert set(disabled_aspects) == {"novelty"}

    all_reviewers = llm_client.list_all("reviewers")
    enabled_reviewers = llm_client.list_enabled("reviewers")
    disabled_reviewers = llm_client.list_disabled("reviewers")

    assert set(all_reviewers) == {"deepreview", "local_review", "treereview"}
    assert set(enabled_reviewers) == {"local_review", "treereview"}
    assert set(disabled_reviewers) == {"deepreview"}


# ── New tests: profiles ──────────────────────────────────────────────────

def test_get_profile_names(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    names = llm_client.get_profile_names()
    assert set(names) == {"all", "quick", "aspects_only", "reviewers_only"}


def test_get_profile_items(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    assert llm_client.get_profile_items("all") is None  # no filter
    assert llm_client.get_profile_items("quick") == ["constructiveness", "treereview"]
    assert llm_client.get_profile_items("aspects_only") == ["constructiveness", "novelty"]

    with pytest.raises(KeyError, match="Unknown profile"):
        llm_client.get_profile_items("nonexistent")


# ── New tests: resolve_items ─────────────────────────────────────────────

def test_resolve_items_defaults_to_enabled(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    # No filters → only enabled items
    aspects = llm_client.resolve_items("aspects")
    assert set(aspects) == {"constructiveness", "flaw_identification"}

    reviewers = llm_client.resolve_items("reviewers")
    assert set(reviewers) == {"local_review", "treereview"}


def test_resolve_items_with_explicit_only(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    # --only overrides enabled flags (can include disabled items)
    aspects = llm_client.resolve_items("aspects", only=["novelty", "constructiveness"])
    assert set(aspects) == {"constructiveness", "novelty"}

    # Items from other section silently ignored
    aspects = llm_client.resolve_items("aspects", only=["novelty", "treereview"])
    assert set(aspects) == {"novelty"}


def test_resolve_items_with_skip(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    aspects = llm_client.resolve_items("aspects", skip=["flaw_identification"])
    assert set(aspects) == {"constructiveness"}


def test_resolve_items_with_profile(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    # Profile "quick" = [constructiveness, treereview]
    # For aspects section → only constructiveness is in both quick AND enabled
    aspects = llm_client.resolve_items("aspects", profile="quick")
    assert aspects == ["constructiveness"]

    # For reviewers section → only treereview is in both quick AND enabled
    reviewers = llm_client.resolve_items("reviewers", profile="quick")
    assert reviewers == ["treereview"]


def test_resolve_items_profile_respects_enabled(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    # "aspects_only" profile = [constructiveness, novelty]
    # But novelty is disabled → only constructiveness returned
    aspects = llm_client.resolve_items("aspects", profile="aspects_only")
    assert aspects == ["constructiveness"]

    # "all" profile (null filter) → just uses enabled flags
    aspects = llm_client.resolve_items("aspects", profile="all")
    assert set(aspects) == {"constructiveness", "flaw_identification"}


def test_resolve_items_profile_with_skip(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    # "aspects_only" = [constructiveness, novelty], skip constructiveness
    aspects = llm_client.resolve_items("aspects", profile="aspects_only", skip=["constructiveness"])
    assert aspects == []  # novelty is disabled, constructiveness skipped


def test_resolve_items_only_overrides_profile(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    # --only takes priority over profile
    aspects = llm_client.resolve_items(
        "aspects", only=["flaw_identification"], profile="aspects_only"
    )
    assert aspects == ["flaw_identification"]


def test_resolve_items_unknown_profile_raises(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    with pytest.raises(KeyError, match="Unknown profile"):
        llm_client.resolve_items("aspects", profile="nonexistent")


# ── New tests: get_aspect_config strips enabled ──────────────────────────

def test_aspect_config_never_contains_enabled(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    for name in llm_client.list_all("aspects"):
        cfg = llm_client.get_aspect_config(name)
        assert "enabled" not in cfg, f"enabled leaked into aspect config: {name}"


def test_reviewer_config_never_contains_enabled(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    for name in llm_client.list_all("reviewers"):
        cfg = llm_client.get_reviewer_config(name)
        assert "enabled" not in cfg, f"enabled leaked into reviewer config: {name}"


# ── Edge cases ───────────────────────────────────────────────────────────

def test_resolve_items_empty_when_all_disabled(monkeypatch, tmp_path):
    config = """
providers:
  openai:
    api_key: test
    base_url: https://api.openai.com/v1
defaults:
  provider: openai
  model: gpt-4o
aspects:
  a1:
    enabled: false
    provider: openai
    model: gpt-4o
  a2:
    enabled: false
    provider: openai
    model: gpt-4o
reviewers:
  r1:
    enabled: false
    type: local
"""
    use_tmp_config(monkeypatch, tmp_path, config)

    assert llm_client.resolve_items("aspects") == []
    assert llm_client.resolve_items("reviewers") == []

    # --only still works even when disabled
    assert llm_client.resolve_items("aspects", only=["a1"]) == ["a1"]


def test_resolve_items_empty_section(monkeypatch, tmp_path):
    config = """
providers:
  openai:
    api_key: test
    base_url: https://api.openai.com/v1
defaults:
  provider: openai
  model: gpt-4o
aspects: {}
reviewers: {}
"""
    use_tmp_config(monkeypatch, tmp_path, config)

    assert llm_client.resolve_items("aspects") == []
    assert llm_client.list_enabled("aspects") == []
