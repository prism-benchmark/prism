"""
llm_client.py — Unified LLM Client & Config for the entire PRISM project.

Single entry point for ALL LLM config and API calls. Supports:
  - OpenAI, Gemini, Azure, Mimo, Devmate, OpenRouter
  - Per-aspect and per-reviewer config from llm_config.yaml
  - Run profiles (--profile quick, --profile aspects, etc.)
  - Enable/disable toggles on every component
  - Subset selection (--only, --skip)

Usage:
    from llm_client import PRISMLLMClient, get_aspect_config, list_enabled

    # Use default config (from llm_config.yaml)
    client = PRISMLLMClient.for_aspect("constructiveness")
    result = client.generate_text("You are a judge.", "Score this review...")

    # List what's available
    aspects = list_enabled("aspects")       # only enabled ones
    reviewers = list_enabled("reviewers")   # only enabled ones

    # Filter by profile
    from llm_client import apply_profile
    apply_profile("quick")  # sets filter context

    # Direct instantiation
    client = PRISMLLMClient(provider="openai", model="gpt-4o-mini", api_key="sk-...")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("prism.llm_client")

# ── YAML config loader ────────────────────────────────────────────────────
_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_REPO_ROOT = Path(__file__).parent

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass


def reset_llm_config_cache() -> None:
    """Clear cached YAML config so tests or env mutations can reload it."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def _resolve_env_var(value: str) -> str:
    """Resolve ${VAR_NAME} or ${VAR_NAME:default} patterns in config values."""
    if not isinstance(value, str) or "${" not in value:
        return value

    def _replace(match):
        expr = match.group(1)
        if ":" in expr:
            # Split once so defaults may contain ":" values such as URLs.
            var_name, default = expr.split(":", 1)
        else:
            var_name, default = expr, ""
        return os.getenv(var_name.strip(), default)

    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _resolve_env_recursive(obj: Any) -> Any:
    """Recursively resolve env vars in a config dict."""
    if isinstance(obj, str):
        return _resolve_env_var(obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_recursive(item) for item in obj]
    return obj


def load_llm_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and cache the LLM configuration from YAML."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed. Install with: pip install pyyaml")
        return {}

    path = Path(config_path) if config_path else _REPO_ROOT / "llm_config.yaml"
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = _resolve_env_recursive(raw)
    if config_path is None:
        _CONFIG_CACHE = config
    return config


# ── Type coercion helpers ─────────────────────────────────────────────────


def _is_missing(value: Any) -> bool:
    return value is None or value == ""


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y"}:
            return True
        if normalized in {"0", "false", "no", "off", "n"}:
            return False
    raise ValueError(f"Cannot coerce {value!r} to bool")


def _coerce_int(value: Any, default: int = 0) -> int:
    if _is_missing(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot coerce {value!r} to int") from exc


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if _is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot coerce {value!r} to float") from exc


def _coerce_numeric_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(cfg)
    for key in (
        "temperature",
        "top_p",
        "gpu_memory_utilization",
        "retry_delay",
        "timeout",
    ):
        if key in coerced:
            coerced[key] = _coerce_float(coerced[key])
    for key in (
        "max_tokens",
        "max_retries",
        "batch_size",
        "max_model_len",
        "tensor_parallel_size",
        "top_k",
        "seq_len",
        "reviewer_num",
    ):
        if key in coerced:
            coerced[key] = _coerce_int(coerced[key])
    if "disable_ssl_verify" in coerced:
        coerced["disable_ssl_verify"] = _coerce_bool(coerced["disable_ssl_verify"])
    return coerced


# ── Provider name normalization ───────────────────────────────────────────


def _normalize_provider_name(provider_name: str) -> str:
    normalized = (provider_name or "").strip().lower()
    if normalized == "gemini-devmate":
        normalized = "devmate"
    return normalized


def _get_configured_providers() -> set:
    """Return the set of provider names defined in llm_config.yaml."""
    config = load_llm_config()
    return set(config.get("providers", {}).keys())


def _validate_provider_name(provider_name: str) -> str:
    normalized = _normalize_provider_name(provider_name)
    configured = _get_configured_providers()
    if configured and normalized not in configured:
        logger.info(
            "Provider '%s' is not in llm_config.yaml providers section. "
            "Treating as OpenAI-compatible custom provider.",
            normalized,
        )
    return normalized


# ═══════════════════════════════════════════════════════════════════════════
# Config Accessors
# ═══════════════════════════════════════════════════════════════════════════


def get_aspect_config(aspect_name: str, step: Optional[str] = None) -> Dict[str, Any]:
    """Get merged config for a specific aspect."""
    config = load_llm_config()
    defaults = config.get("defaults", {})
    aspects = config.get("aspects", {})
    if aspect_name not in aspects:
        raise KeyError(
            f"Unknown aspect '{aspect_name}'. Available: {', '.join(sorted(aspects))}"
        )

    aspect = aspects[aspect_name] or {}
    if not isinstance(aspect, dict):
        raise ValueError(f"Aspect '{aspect_name}' config must be a mapping")

    aspect_fields = {k: v for k, v in aspect.items() if not isinstance(v, dict)}
    if step is not None:
        if step not in aspect or not isinstance(aspect.get(step), dict):
            raise KeyError(f"Unknown step '{step}' for aspect '{aspect_name}'")
        selected_step = aspect[step]
    else:
        step_names = [
            k for k, v in aspect.items() if isinstance(v, dict) and k != "enabled"
        ]
        if step_names and not aspect_fields:
            raise ValueError(
                f"Aspect '{aspect_name}' requires a step. Available: {', '.join(sorted(step_names))}"
            )
        selected_step = {}

    merged = {**defaults, **aspect_fields, **selected_step}
    merged.pop("enabled", None)
    if "provider" in merged:
        merged["provider"] = _validate_provider_name(merged["provider"])
    return _coerce_numeric_fields(merged)


def get_reviewer_config(reviewer_name: str) -> Dict[str, Any]:
    """Get config for a specific reviewer."""
    config = load_llm_config()
    defaults = config.get("defaults", {})
    reviewers = config.get("reviewers", {})
    if reviewer_name not in reviewers:
        raise KeyError(
            f"Unknown reviewer '{reviewer_name}'. Available: {', '.join(sorted(reviewers))}"
        )
    reviewer = reviewers[reviewer_name] or {}

    if reviewer.get("type") == "api":
        merged = {**defaults, **reviewer}
        merged.pop("enabled", None)
        if "provider" in merged:
            merged["provider"] = _validate_provider_name(merged["provider"])
        return _coerce_numeric_fields(merged)
    result = _coerce_numeric_fields(reviewer)
    result.pop("enabled", None)
    return result


def get_provider_credentials(provider_name: str) -> Dict[str, Any]:
    """Get credentials for a specific provider."""
    config = load_llm_config()
    providers = config.get("providers", {})
    provider = _validate_provider_name(provider_name)
    if provider not in providers:
        raise KeyError(f"Provider '{provider}' is not configured in llm_config.yaml")
    return _coerce_numeric_fields(providers[provider] or {})


def get_provider_defaults(provider_name: str) -> Dict[str, Any]:
    """Get provider-level defaults from YAML."""
    config = load_llm_config()
    provider = _normalize_provider_name(provider_name)
    defaults = config.get("provider_defaults", {})
    if provider not in defaults:
        return {}
    return _coerce_numeric_fields(defaults[provider] or {})


def get_referenced_api_providers() -> Set[str]:
    """Providers referenced by aspects and API reviewers in YAML."""
    config = load_llm_config()
    providers: Set[str] = set()
    for aspect_name, aspect in (config.get("aspects") or {}).items():
        try:
            aspect_cfg = get_aspect_config(aspect_name)
            providers.add(_validate_provider_name(aspect_cfg.get("provider", "openai")))
        except ValueError:
            pass
        if isinstance(aspect, dict):
            for key, value in aspect.items():
                if isinstance(value, dict) and value.get("provider"):
                    providers.add(_validate_provider_name(value["provider"]))
    for reviewer_name, reviewer in (config.get("reviewers") or {}).items():
        if isinstance(reviewer, dict) and reviewer.get("type") == "api":
            reviewer_cfg = get_reviewer_config(reviewer_name)
            providers.add(
                _validate_provider_name(reviewer_cfg.get("provider", "openai"))
            )
    return providers


# ═══════════════════════════════════════════════════════════════════════════
# Enable/Disable, Profiles, Listing
# ═══════════════════════════════════════════════════════════════════════════


def is_enabled(section: str, name: str) -> bool:
    """Check if a specific aspect or reviewer is enabled in YAML.

    Args:
        section: "aspects" or "reviewers"
        name: item name (e.g. "constructiveness", "treereview")

    Returns:
        True if enabled (default True if key missing).
    """
    config = load_llm_config()
    items = config.get(section, {})
    if name not in items:
        return False
    item = items[name]
    if not isinstance(item, dict):
        return False
    return _coerce_bool(item.get("enabled"), True)


def list_all(section: str) -> List[str]:
    """List all defined names in a section (regardless of enabled state)."""
    config = load_llm_config()
    return sorted((config.get(section) or {}).keys())


def list_enabled(section: str) -> List[str]:
    """List only enabled names in a section."""
    return [name for name in list_all(section) if is_enabled(section, name)]


def list_disabled(section: str) -> List[str]:
    """List only disabled names in a section."""
    return [name for name in list_all(section) if not is_enabled(section, name)]


def get_profile_names() -> List[str]:
    """List all available run profile names."""
    config = load_llm_config()
    return sorted((config.get("run_profiles") or {}).keys())


def get_profile_items(profile_name: str) -> Optional[List[str]]:
    """Get the item list for a run profile.

    Returns:
        List of item names, or None for 'all' profile (no filter).
    """
    config = load_llm_config()
    profiles = config.get("run_profiles", {})
    if profile_name not in profiles:
        raise KeyError(
            f"Unknown profile '{profile_name}'. Available: {', '.join(sorted(profiles))}"
        )
    return profiles[profile_name]


def resolve_items(
    section: str,
    *,
    only: Optional[List[str]] = None,
    skip: Optional[List[str]] = None,
    profile: Optional[str] = None,
) -> List[str]:
    """Resolve the final list of items to run, applying all filters.

    Priority: only/skip args > profile > enabled flags.

    Args:
        section: "aspects" or "reviewers"
        only: explicit whitelist (overrides profile and enabled flags)
        skip: explicit blacklist (removed from result)
        profile: profile name to filter by

    Returns:
        Sorted list of item names that should run.
    """
    skip_set = set(skip or [])

    # If explicit --only, use it directly (ignore enabled flags)
    if only:
        available = set(list_all(section))
        result = []
        for name in only:
            if name in available:
                result.append(name)
            # Silently skip items that belong to the other section
        return sorted(set(result) - skip_set)

    # If profile, get the profile's filter list
    profile_filter = None
    if profile:
        profile_items = get_profile_items(profile)
        if profile_items is not None:
            profile_filter = set(profile_items)

    # Start with enabled items
    enabled = set(list_enabled(section))

    # Intersect with profile filter if present
    if profile_filter is not None:
        enabled = enabled.intersection(profile_filter)

    return sorted(enabled - skip_set)


# ═══════════════════════════════════════════════════════════════════════════
# PRISMLLMClient — Unified Client
# ═══════════════════════════════════════════════════════════════════════════


class PRISMLLMClient:
    """
    Unified LLM client for the PRISM project.

    Supports all providers through a single interface. Use factory methods
    for convenient instantiation from llm_config.yaml:

        client = PRISMLLMClient.for_aspect("constructiveness")
        client = PRISMLLMClient.for_aspect("flaw_identification", step="step1")
        client = PRISMLLMClient.for_reviewer("treereview")

    Or direct instantiation:
        client = PRISMLLMClient(provider="openai", model="gpt-4o-mini")
    """

    # Hardcoded providers for direct API usage without llm_config.yaml.
    # Any provider name can also be used — it will be read from the YAML
    # providers section or treated as OpenAI-compatible with a custom base_url.
    BUILTIN_PROVIDERS = {"openai", "gemini", "azure", "mimo", "devmate", "openrouter"}

    # ── Factory Methods ────────────────────────────────────────────────────

    @classmethod
    def for_aspect(
        cls, aspect_name: str, step: Optional[str] = None, **overrides
    ) -> "PRISMLLMClient":
        """Create a client configured for a specific benchmarking aspect."""
        cfg = get_aspect_config(aspect_name, step)
        cfg.update(overrides)
        return cls._from_config(
            cfg, label=f"aspect:{aspect_name}" + (f":{step}" if step else "")
        )

    @classmethod
    def for_reviewer(cls, reviewer_name: str, **overrides) -> "PRISMLLMClient":
        """Create a client configured for a specific LLM reviewer."""
        cfg = get_reviewer_config(reviewer_name)
        if cfg.get("type") == "local":
            raise ValueError(
                f"Reviewer '{reviewer_name}' is type 'local' and cannot be used as an API client"
            )
        cfg.update(overrides)
        return cls._from_config(cfg, label=f"reviewer:{reviewer_name}")

    @classmethod
    def _from_config(cls, cfg: Dict[str, Any], label: str = "") -> "PRISMLLMClient":
        """Create client from a config dict."""
        cfg = _coerce_numeric_fields(cfg)
        provider = _validate_provider_name(cfg.get("provider", "openai"))
        creds = get_provider_credentials(provider)

        api_key = cfg.get("api_key") or creds.get("api_key", "")
        base_url = cfg.get("base_url") or creds.get("base_url")
        model = cfg.get("model", "")
        temperature = _coerce_float(cfg.get("temperature"), 0.0)
        max_tokens = _coerce_int(cfg.get("max_tokens"), 4096)
        max_retries = _coerce_int(cfg.get("max_retries"), 3)
        retry_delay = _coerce_float(cfg.get("retry_delay"), 1.0)
        timeout = _coerce_float(cfg.get("timeout"), 120.0)

        # Azure-specific
        endpoint = cfg.get("endpoint") or creds.get("endpoint")
        api_version = cfg.get("api_version") or creds.get("api_version", "2024-10-21")
        deployment = cfg.get("deployment") or creds.get("deployment")

        # Devmate-specific
        proxy = cfg.get("proxy") or creds.get("proxy", "")
        disable_ssl_verify = _coerce_bool(
            cfg.get("disable_ssl_verify", creds.get("disable_ssl_verify")), False
        )

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url or endpoint,
            temperature=temperature,
            max_tokens=max_tokens,
            api_version=api_version,
            deployment=deployment,
            proxy=proxy,
            disable_ssl_verify=disable_ssl_verify,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            label=label,
        )

    # ── Constructor ────────────────────────────────────────────────────────

    def __init__(
        self,
        provider: str = "openai",
        model: str = "",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_version: str = "2024-10-21",
        deployment: Optional[str] = None,
        proxy: str = "",
        disable_ssl_verify: bool = False,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        label: str = "",
    ):
        self.provider = _validate_provider_name(provider)

        self.model = model
        self.api_key = api_key or ""
        self.base_url = base_url
        self.temperature = _coerce_float(temperature, 0.0)
        self.max_tokens = _coerce_int(max_tokens, 4096)
        self.api_version = api_version
        self.deployment = deployment
        self.proxy = proxy
        self.disable_ssl_verify = _coerce_bool(disable_ssl_verify, False)
        self.timeout = _coerce_float(timeout, 120.0)
        self.max_retries = _coerce_int(max_retries, 3)
        self.retry_delay = _coerce_float(retry_delay, 1.0)
        self.label = label

        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the underlying SDK client."""
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "azure":
            self._init_azure()
        elif self.provider == "mimo":
            self._init_openai_compatible("Mimo")
        elif self.provider == "devmate":
            self._init_devmate()
        elif self.provider == "openrouter":
            self._init_openai_compatible("OpenRouter")
        else:
            raise ValueError(f"Unsupported provider after validation: {self.provider}")

    def _init_openai(self):
        try:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            kwargs["timeout"] = self.timeout
            kwargs["max_retries"] = 0
            self._client = OpenAI(**kwargs)
            logger.info(f"[{self.label}] OpenAI ready (model={self.model})")
        except Exception as e:
            raise RuntimeError(f"Failed to init OpenAI client: {e}") from e

    def _init_gemini(self):
        try:
            from google import genai

            if not self.api_key:
                raise RuntimeError("Missing Gemini API key. Set GOOGLE_API_KEY.")
            self._client = genai.Client(api_key=self.api_key)
            logger.info(f"[{self.label}] Gemini ready (model={self.model})")
        except Exception as e:
            raise RuntimeError(f"Failed to init Gemini client: {e}") from e

    def _init_azure(self):
        try:
            from openai import AzureOpenAI

            kwargs = {
                "api_version": self.api_version,
                "timeout": self.timeout,
                "max_retries": 0,
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["azure_endpoint"] = self.base_url
            if self.deployment:
                kwargs["azure_deployment"] = self.deployment
            self._client = AzureOpenAI(**kwargs)
            logger.info(
                f"[{self.label}] Azure OpenAI ready (deployment={self.deployment or self.model})"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to init Azure OpenAI client: {e}") from e

    def _init_openai_compatible(self, name: str):
        try:
            from openai import OpenAI

            if not self.api_key:
                raise RuntimeError(f"Missing API key for {name}.")
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
            logger.info(f"[{self.label}] {name} ready (model={self.model})")
        except Exception as e:
            raise RuntimeError(f"Failed to init {name} client: {e}") from e

    def _init_devmate(self):
        try:
            import httpx
            from openai import OpenAI

            if not self.api_key:
                raise RuntimeError("Missing Devmate API key.")

            http_kwargs: Dict[str, Any] = {
                "verify": not self.disable_ssl_verify,
                "timeout": self.timeout,
            }
            if self.proxy:
                http_kwargs["proxy"] = self.proxy
            http_client = httpx.Client(**http_kwargs)

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                http_client=http_client,
                timeout=self.timeout,
                max_retries=0,
            )
            logger.info(
                f"[{self.label}] Devmate ready (model={self.model}, proxy={'on' if self.proxy else 'off'})"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to init Devmate client: {e}") from e

    # ── Core Interface ─────────────────────────────────────────────────────

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        deployment: Optional[str] = None,
    ) -> str:
        """
        Generate text from a system+user prompt pair.

        Args:
            system_prompt: System message.
            user_prompt: User message.
            response_format: e.g. {"type": "json_object"} for JSON mode.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            max_output_tokens: Legacy alias for max_tokens.
            model: Override default model.
            deployment: Legacy alias for model/deployment.

        Returns:
            Generated text string.
        """
        temp = self.temperature if temperature is None else temperature
        token_override = max_tokens if max_tokens is not None else max_output_tokens
        model_override = model if model is not None else deployment
        tokens = (
            self.max_tokens
            if token_override is None
            else _coerce_int(token_override, self.max_tokens)
        )
        mdl = self.model if model_override is None else model_override
        json_mode = (
            response_format is not None and response_format.get("type") == "json_object"
        )

        logger.debug(
            "[%s] LLM call provider=%s model=%s temperature=%s max_tokens=%s "
            "json_mode=%s timeout=%s max_retries=%s",
            self.label,
            self.provider,
            mdl,
            temp,
            tokens,
            json_mode,
            self.timeout,
            self.max_retries,
        )

        if self.provider == "gemini":
            return self._call_gemini(
                system_prompt, user_prompt, temp, tokens, mdl, json_mode
            )
        else:
            return self._call_openai_compat(
                system_prompt,
                user_prompt,
                temp,
                tokens,
                mdl,
                json_mode,
                temperature_override=temperature is not None,
            )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate text from a single prompt (convenience method).

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            model: Override default model.

        Returns:
            Generated text string.
        """
        sys_prompt = system_prompt or "You are a helpful assistant."
        return self.generate_text(
            sys_prompt,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate and parse JSON response.

        Returns:
            Parsed JSON dict, or None if parsing fails.
        """
        raw = self.generate_text(
            system_prompt,
            user_prompt,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        return self._parse_json(raw)

    # ── OpenAI-compatible backend ──────────────────────────────────────────

    def _call_openai_compat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        model: str,
        json_mode: bool,
        *,
        temperature_override: bool = False,
    ) -> str:
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        # Reasoning models use max_completion_tokens
        if self._is_reasoning_model(model):
            request_kwargs["max_completion_tokens"] = max_tokens
            if temperature_override:
                logger.warning(
                    "[%s] Ignoring explicit temperature override for reasoning model %s",
                    self.label,
                    model,
                )
        else:
            request_kwargs["temperature"] = temperature
            request_kwargs["max_tokens"] = max_tokens

        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            try:
                resp = self._client.chat.completions.create(**request_kwargs)
                return self._extract_response_text(resp)
            except Exception as exc:
                err = str(exc)
                # Auto-switch to max_completion_tokens
                if "max_tokens" in err and "max_completion_tokens" in err:
                    request_kwargs.pop("max_tokens", None)
                    request_kwargs["max_completion_tokens"] = max_tokens
                    request_kwargs.pop("temperature", None)
                    continue
                # Drop response_format if unsupported
                if json_mode and "response_format" in err.lower():
                    request_kwargs.pop("response_format", None)
                    continue
                logger.warning(
                    f"[{self.label}] API call failed (attempt {attempt + 1}): {exc}"
                )
                if attempt == attempts - 1:
                    raise
                time.sleep(self.retry_delay)
        return ""

    # ── Gemini backend ─────────────────────────────────────────────────────

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        model: str,
        json_mode: bool,
    ) -> str:
        from google import genai

        config_kwargs: Dict[str, Any] = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            try:
                resp = self._client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(**config_kwargs),
                )
                return (resp.text or "").strip()
            except Exception as exc:
                logger.warning(
                    f"[{self.label}] Gemini call failed (attempt {attempt + 1}): {exc}"
                )
                if attempt == attempts - 1:
                    raise
                time.sleep(self.retry_delay)
        return ""

    # ── Helpers ────────────────────────────────────────────────────────────

    _REASONING_PREFIXES = ("o1", "o3", "gpt-5", "gpt-4.5")

    def _is_reasoning_model(self, model: str) -> bool:
        m = (model or "").lower()
        return any(m.startswith(p) for p in self._REASONING_PREFIXES)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text from an OpenAI ChatCompletion response."""
        if isinstance(response, str):
            return response.strip()
        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content.strip()
        return str(response).strip()

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM output with lenient handling.

        Strategies (in order):
          1. Direct parse
          2. Strip code fences, then parse
          3. Extract first balanced JSON object via raw_decode
        """
        if not text:
            return None
        text = text.strip()

        # Strip code fences
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        # Try direct parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Extract first JSON object
        decoder = json.JSONDecoder()
        for start, ch in enumerate(text):
            if ch in "{[":
                try:
                    _, end = decoder.raw_decode(text, start)
                    candidate = text[start : start + end]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
        return None

    def __repr__(self) -> str:
        return (
            f"PRISMLLMClient(provider={self.provider!r}, model={self.model!r}, "
            f"label={self.label!r})"
        )
