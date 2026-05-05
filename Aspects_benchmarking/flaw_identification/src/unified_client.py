"""
UnifiedChatClient – thin adapter so ReviewEvaluatorPipeline can use
OpenAI (native), Google Gemini, or Bosch Devmate Gemini through the
exact same interface as AzureChatClient.generate_text().

Usage:
    client = UnifiedChatClient(provider="openai", model="gpt-4o-mini")
    client = UnifiedChatClient(provider="gemini", model="models/gemini-2.5-flash-lite")
    client = UnifiedChatClient(provider="gemini-devmate", model="gemini-3-flash-preview")
"""

from __future__ import annotations

import json
import os
import re
import time
import sys
from typing import Any, Optional

from dotenv import dotenv_values, find_dotenv, load_dotenv

_DOTENV_PATH = find_dotenv()
load_dotenv(_DOTENV_PATH, override=False)
_DOTENV_VALUES = dotenv_values(_DOTENV_PATH) if _DOTENV_PATH else {}


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    """Extract the first balanced JSON object from arbitrary text."""
    text = _strip_code_fences(text)
    try:
        json.loads(text)
        return text
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch in "{[":
            try:
                _, end = decoder.raw_decode(text, start)
                return text[start:end]
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON object found in response")


def _stringify_content_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        for key in ("text", "content", "value"):
            value = part.get(key)
            if isinstance(value, str):
                return value
        return ""
    for attr in ("text", "content", "value"):
        value = getattr(part, attr, None)
        if isinstance(value, str):
            return value
    return ""


def _extract_openai_response_text(response: Any) -> str:
    """Handle normal ChatCompletion objects, streamed chunks, or raw text."""
    if isinstance(response, str):
        return response.strip()

    choices = getattr(response, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                combined = "".join(_stringify_content_part(part) for part in content).strip()
                if combined:
                    return combined

    if response is not None and not isinstance(response, (str, bytes)):
        try:
            iterator = iter(response)
        except TypeError:
            iterator = None
        if iterator is not None:
            chunks: list[str] = []
            for chunk in iterator:
                chunk_choices = getattr(chunk, "choices", None) or []
                if not chunk_choices:
                    continue
                delta = getattr(chunk_choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    chunks.append("".join(_stringify_content_part(part) for part in content))
            combined = "".join(chunks).strip()
            if combined:
                return combined

    return str(response).strip()


def _looks_like_json_text(text: str) -> bool:
    try:
        _extract_first_json_object(text)
        return True
    except Exception:
        return False


def _env_flag(name: str, default: bool = False) -> bool:
    value = _get_env_value(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_value(*names: str) -> Optional[str]:
    for name in names:
        runtime_value = os.getenv(name)
        if runtime_value:
            return runtime_value
        dotenv_value = _DOTENV_VALUES.get(name)
        if isinstance(dotenv_value, str) and dotenv_value:
            return dotenv_value
    return None


class UnifiedChatClient:
    """
    Drop-in replacement for AzureChatClient that supports native OpenAI or Google Gemini.

    Implements: generate_text(system_prompt, user_prompt, *, response_format,
                               temperature, max_output_tokens, deployment)

    The `deployment` kwarg is silently ignored (Azure-specific); the model is
    set once at construction time.
    """

    SUPPORTED_PROVIDERS = {"openai", "gemini", "gemini-devmate", "mimo"}

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ):
        self.provider = provider.lower()
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(f"provider must be one of {self.SUPPORTED_PROVIDERS}, got '{provider}'")

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        if self.provider == "openai":
            self.model = model or _get_env_value("OPENAI_MODEL") or "gpt-4o-mini"
            self._init_openai(api_key)
        elif self.provider == "gemini":
            self.model = model or _get_env_value("GEMINI_MODEL") or "models/gemini-2.5-flash-lite"
            self._init_gemini(api_key)
        elif self.provider == "gemini-devmate":
            self.model = model or _get_env_value("GEMINI_DEVMATE_MODEL", "DEVMATE_MODEL") or "gemini-3-flash-preview"
            self._init_gemini_devmate(api_key)
        elif self.provider == "mimo":
            self.model = model or _get_env_value("MIMO_MODEL") or "mimo-v2.5-pro"
            self._init_mimo(api_key)

    # ------------------------------------------------------------------
    # Provider setup
    # ------------------------------------------------------------------

    def _init_openai(self, api_key: Optional[str]) -> None:
        try:
            from openai import OpenAI
            key = api_key or _get_env_value("OPENAI_API_KEY")
            base_url = _get_env_value("OPENAI_BASE_URL")
            self._openai_client = OpenAI(api_key=key, base_url=base_url)
            print(f"[INFO] UnifiedChatClient: OpenAI ready (model={self.model})", file=sys.stderr)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise OpenAI client: {exc}") from exc

    def _build_devmate_client(self, proxy_url: Optional[str]):
        try:
            import httpx
            from openai import OpenAI

            http_kwargs: dict[str, Any] = {
                "verify": self._devmate_ssl_verify,
            }
            if proxy_url:
                http_kwargs["proxy"] = proxy_url
            http_client = httpx.Client(**http_kwargs)
            return OpenAI(
                api_key=self._devmate_api_key,
                base_url=self._devmate_base_url,
                http_client=http_client,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise Devmate OpenAI-compatible client: {exc}") from exc

    def _init_gemini_devmate(self, api_key: Optional[str]) -> None:
        self._devmate_api_key = (
            api_key
            or _get_env_value("GEMINI_DEVMATE_API_KEY", "DEVMATE_API_KEY", "GEMINI_API_KEY")
        )
        if not self._devmate_api_key:
            raise RuntimeError(
                "Missing Devmate API key. Set GEMINI_DEVMATE_API_KEY, DEVMATE_API_KEY, or GEMINI_API_KEY."
            )

        self._devmate_base_url = _get_env_value("DEVMATE_BASE_URL") or "https://devmate.bosch.com/api/v3"
        # Prefer an explicit Devmate proxy or the known Bosch corporate proxy.
        # Generic HTTP(S)_PROXY values in local .env files may point to a dead local
        # proxy (e.g. 127.0.0.1:3128), which breaks Devmate calls.
        self._devmate_proxy = (
            _get_env_value("DEVMATE_PROXY")
            or "http://rb-proxy-apac.bosch.com:8080"
        )
        self._devmate_ssl_verify = not _env_flag("DEVMATE_DISABLE_SSL_VERIFY", default=True)
        self._devmate_proxy_fallback_used = False
        self._openai_client = self._build_devmate_client(self._devmate_proxy)
        print(
            f"[INFO] UnifiedChatClient: Devmate Gemini ready (model={self.model}, proxy={'on' if self._devmate_proxy else 'off'})",
            file=sys.stderr,
        )

    def _init_mimo(self, api_key: Optional[str]) -> None:
        try:
            from openai import OpenAI
            key = api_key or _get_env_value("MIMO_API_KEY")
            if not key:
                raise RuntimeError("MIMO_API_KEY environment variable not set")
            base_url = _get_env_value("MIMO_BASE_URL") or "https://api.xiaomimimo.com/v1"
            self._openai_client = OpenAI(api_key=key, base_url=base_url)
            print(f"[INFO] UnifiedChatClient: Mimo ready (model={self.model}, base_url={base_url})", file=sys.stderr)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise Mimo client: {exc}") from exc

    def _init_gemini(self, api_key: Optional[str]) -> None:
        try:
            from google import genai
            key = api_key or _get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable not set")
            self._gemini_client = genai.Client(api_key=key)
            print(f"[INFO] UnifiedChatClient: Gemini ready (model={self.model})", file=sys.stderr)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise Gemini client: {exc}") from exc

    # ------------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------------

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        deployment: Optional[str] = None,  # ignored – Azure-only concept
    ) -> str:
        """
        Generate text and return raw string (matches AzureChatClient.generate_text).
        """
        temp = self.temperature if temperature is None else temperature
        tokens = self.max_output_tokens if max_output_tokens is None else max_output_tokens
        json_mode = response_format is not None and response_format.get("type") == "json_object"

        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt, temp, tokens, json_mode)
        elif self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt, temp, tokens, json_mode)
        elif self.provider == "gemini-devmate":
            return self._call_openai(system_prompt, user_prompt, temp, tokens, json_mode, provider_label="Devmate")
        elif self.provider == "mimo":
            return self._call_openai(system_prompt, user_prompt, temp, tokens, json_mode, provider_label="Mimo")

    # ------------------------------------------------------------------
    # OpenAI backend
    # ------------------------------------------------------------------

    # Models that use max_completion_tokens instead of max_tokens
    _COMPLETION_TOKENS_MODELS = ("o1", "o3", "gpt-5", "gpt-4.5")

    def _uses_completion_tokens(self) -> bool:
        """Return True if this model requires max_completion_tokens instead of max_tokens."""
        m = (self.model or "").lower()
        return any(m.startswith(prefix) for prefix in self._COMPLETION_TOKENS_MODELS)

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        provider_label: str = "OpenAI",
    ) -> str:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if json_mode and self.provider != "gemini-devmate":
            request_kwargs["response_format"] = {"type": "json_object"}

        # Reasoning models (o1/o3/gpt-5) ignore temperature and use
        # max_completion_tokens; classic models use temperature + max_tokens.
        if self._uses_completion_tokens():
            request_kwargs["max_completion_tokens"] = max_tokens
            # temperature is not supported on reasoning models – omit it
        else:
            request_kwargs["temperature"] = temperature
            request_kwargs["max_tokens"] = max_tokens

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(
                    f"  [DEBUG] {provider_label} call attempt {attempt+1}/{max_retries}: model={self.model}",
                    file=sys.stderr,
                )
                resp = self._openai_client.chat.completions.create(**request_kwargs)
                text = _extract_openai_response_text(resp)
                if json_mode and not _looks_like_json_text(text):
                    print(
                        f"  [INFO] {provider_label} returned non-JSON text; attempting JSON repair",
                        file=sys.stderr,
                    )
                    text = self._repair_json_response(
                        original_text=text,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                        provider_label=provider_label,
                    )
                return text
            except Exception as exc:
                err = str(exc)
                # Auto-detect: API explicitly told us to use max_completion_tokens
                if "max_tokens" in err and "max_completion_tokens" in err:
                    print(
                        f"  [INFO] Switching to max_completion_tokens for {provider_label} model={self.model}",
                        file=sys.stderr,
                    )
                    request_kwargs.pop("max_tokens", None)
                    request_kwargs["max_completion_tokens"] = max_tokens
                    request_kwargs.pop("temperature", None)  # also drop temperature
                    continue  # retry immediately, don't count as real attempt
                if request_kwargs.get("response_format") and "response_format" in err.lower():
                    print(
                        f"  [INFO] Retrying {provider_label} call without response_format enforcement",
                        file=sys.stderr,
                    )
                    request_kwargs.pop("response_format", None)
                    continue
                if (
                    self.provider == "gemini-devmate"
                    and not self._devmate_proxy_fallback_used
                    and self._devmate_proxy
                    and any(
                        token in err.lower()
                        for token in ("proxy", "407", "tunnel", "connection refused", "getaddrinfo failed")
                    )
                ):
                    print("  [INFO] Retrying Devmate call without proxy", file=sys.stderr)
                    self._openai_client = self._build_devmate_client(None)
                    self._devmate_proxy_fallback_used = True
                    continue
                if (
                    self.provider == "gemini-devmate"
                    and not self._devmate_proxy_fallback_used
                    and self._devmate_proxy
                    and any(
                        token in err.lower()
                        for token in ("connection timed out", "502", "timed out")
                    )
                ):
                    print("  [INFO] Devmate proxy timed out; retrying without proxy", file=sys.stderr)
                    self._openai_client = self._build_devmate_client(None)
                    self._devmate_proxy_fallback_used = True
                    continue
                print(f"  [WARNING] {provider_label} call failed (attempt {attempt+1}): {exc}", file=sys.stderr)
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""  # unreachable

    def _repair_json_response(
        self,
        *,
        original_text: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        provider_label: str,
    ) -> str:
        repair_request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You repair model outputs into strict JSON. "
                        "Return exactly one valid JSON object, with no markdown fences and no extra commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "The previous response did not follow the required JSON-only format.\n\n"
                        "[ORIGINAL SYSTEM PROMPT]\n"
                        f"{system_prompt}\n\n"
                        "[ORIGINAL USER PROMPT]\n"
                        f"{user_prompt}\n\n"
                        "[INVALID MODEL OUTPUT]\n"
                        f"{original_text}\n\n"
                        "Rewrite the answer as exactly one valid JSON object that follows the original task."
                    ),
                },
            ],
            "stream": False,
        }
        if self._uses_completion_tokens():
            repair_request_kwargs["max_completion_tokens"] = max_tokens
        else:
            repair_request_kwargs["temperature"] = 0.0
            repair_request_kwargs["max_tokens"] = max_tokens

        repair_response = self._openai_client.chat.completions.create(**repair_request_kwargs)
        repaired_text = _extract_openai_response_text(repair_response)
        return _extract_first_json_object(repaired_text)

    # ------------------------------------------------------------------
    # Gemini backend
    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        from google import genai

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(
                    f"  [DEBUG] Gemini call attempt {attempt+1}/{max_retries}: model={self.model}",
                    file=sys.stderr,
                )
                config_kwargs: dict[str, Any] = {
                    "system_instruction": system_prompt,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
                if json_mode:
                    config_kwargs["response_mime_type"] = "application/json"

                resp = self._gemini_client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(**config_kwargs),
                )
                text = (resp.text or "").strip()

                # Detect truncated response (finish_reason == MAX_TOKENS).
                try:
                    finish_reason = resp.candidates[0].finish_reason
                    finish_name = getattr(finish_reason, "name", str(finish_reason))
                    if "MAX_TOKENS" in finish_name.upper() or str(finish_reason) in ("2", "FinishReason.MAX_TOKENS"):
                        print(
                            f"  [WARNING] Gemini response TRUNCATED (finish_reason={finish_name})."
                            f" Response length={len(text)} chars, max_output_tokens={max_tokens}."
                            f" Increase AZURE_OPENAI_STEP1_MAX_OUTPUT_TOKENS or AZURE_OPENAI_STEP2_MAX_OUTPUT_TOKENS.",
                            file=sys.stderr,
                        )
                except Exception:
                    pass

                return text
            except Exception as exc:
                exc_name = type(exc).__name__
                print(f"  [WARNING] Gemini call failed (attempt {attempt+1}): {exc_name}: {exc}", file=sys.stderr)
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""  # unreachable


