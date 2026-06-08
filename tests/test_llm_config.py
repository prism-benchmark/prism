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
    monkeypatch.delenv("MIMO_BASE_URL", raising=False)
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)

    cfg = llm_client.load_llm_config()

    assert cfg["providers"]["mimo"]["base_url"] == "https://api.xiaomimimo.com/v1"


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


def test_validate_env_is_yaml_driven_and_ignores_local_reviewers(monkeypatch, tmp_path):
    use_tmp_config(monkeypatch, tmp_path, BASE_CONFIG)
    ai_config = importlib.reload(importlib.import_module("ai_config"))
    monkeypatch.setattr(ai_config, "MIMO_API_KEY", "")
    monkeypatch.setattr(ai_config, "GOOGLE_API_KEY", "gemini-key")
    monkeypatch.setattr(ai_config, "OPENAI_API_KEY", "")
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


def test_token_bucket():
    bucket = llm_client.TokenBucket(rate=10, capacity=100)
    assert bucket.tokens == 100

    # Consume within capacity
    wait_time = bucket.consume(30)
    assert wait_time == 0.0
    assert bucket.tokens == 70

    # Consume more than capacity
    wait_time = bucket.consume(80)
    assert wait_time > 0.0
    # tokens becomes negative (since we consumed more than available, rate-limiting wait time is returned)
    assert abs(wait_time - 1.0) < 1e-5


def test_is_rate_limit_error():
    class DummyException(Exception):
        pass

    exc1 = DummyException("rate limit exceeded")
    assert llm_client.PRISMLLMClient._is_rate_limit_error(exc1) is True

    exc2 = DummyException("Resource Exhausted")
    assert llm_client.PRISMLLMClient._is_rate_limit_error(exc2) is True

    exc3 = DummyException("Something went wrong")
    assert llm_client.PRISMLLMClient._is_rate_limit_error(exc3) is False

    # Status code 429
    exc4 = DummyException()
    exc4.status_code = 429
    assert llm_client.PRISMLLMClient._is_rate_limit_error(exc4) is True


def test_prism_llm_client_rate_limiter_and_refund(monkeypatch):
    rpm_bucket = llm_client.TokenBucket(rate=0, capacity=60)
    tpm_bucket = llm_client.TokenBucket(rate=0, capacity=1000)

    client = llm_client.PRISMLLMClient(provider="openai", model="gpt-4o-mini", api_key="mock-api-key", rate_limit_rpm=60, rate_limit_tpm=1000)
    monkeypatch.setattr(client, "_get_rate_limiters", lambda: (rpm_bucket, tpm_bucket))

    # Mock the actual LLM call method to succeed
    monkeypatch.setattr(client, "_call_openai_compat", lambda sys, usr, temp, max_tok, model, json_mode, **kwargs: "hello")

    client.max_tokens = 10

    res = client.generate_text("system", "hello")
    assert res == "hello"
    # TPM tokens initially: 1000.
    # Reserved: prompt (len("system" + "hello")//4 = 11//4 = 2) + max_tokens (10) = 12.
    # Actual: prompt (2) + len("hello")//4 (5//4 = 1) = 3.
    # Refund: 12 - 3 = 9.
    # Expected remaining tokens: 1000 - 12 + 9 = 997.
    assert tpm_bucket.tokens == 997

    # Mock the actual LLM call to fail, checking that ALL reserved tokens are refunded
    def failing_call(*args, **kwargs):
        raise ValueError("failing")

    monkeypatch.setattr(client, "_call_openai_compat", failing_call)
    with pytest.raises(ValueError):
        client.generate_text("system", "hello")
    # TPM tokens before call: 997. Should still be 997 after failure due to full refund.
    assert tpm_bucket.tokens == 997


def test_prism_llm_client_exponential_backoff(monkeypatch):
    import time
    client = llm_client.PRISMLLMClient(provider="openai", model="gpt-4o-mini", api_key="mock-api-key", max_retries=2, retry_delay=1.0)

    # Mock TokenBucket to return 0 wait times
    monkeypatch.setattr(client, "_get_rate_limiters", lambda: (None, None))

    # Dummy class for rate limit exception
    class MockRateLimitError(Exception):
        pass

    sleep_calls = []
    monkeypatch.setattr(time, "sleep", lambda t: sleep_calls.append(t))

    # Mock actual API call to fail with rate limit error
    call_count = 0
    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise MockRateLimitError("429 Rate Limit Exceeded")

    monkeypatch.setattr(client._client.chat.completions, "create", mock_create)

    # Verify rate limit check helper returns True
    monkeypatch.setattr(client, "_is_rate_limit_error", lambda exc: isinstance(exc, MockRateLimitError))

    with pytest.raises(MockRateLimitError):
        client.generate_text("system", "hello")

    # max_retries = 2, so attempts = 3.
    # It should call mock_call 3 times, and sleep twice (after 1st and 2nd attempts).
    assert call_count == 3
    assert len(sleep_calls) == 2
    # backoff formula: (2 ** attempt) * self.retry_delay + random.uniform(0.5, 1.5)
    # retry_delay = 1.0.
    # attempt 0: (2 ** 0) * 1.0 + jitter -> 1.0 + jitter
    # attempt 1: (2 ** 1) * 1.0 + jitter -> 2.0 + jitter
    assert 1.5 <= sleep_calls[0] <= 2.5
    assert 2.5 <= sleep_calls[1] <= 3.5


