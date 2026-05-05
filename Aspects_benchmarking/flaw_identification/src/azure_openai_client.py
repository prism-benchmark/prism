from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv
from openai import AzureOpenAI

try:
    from openai import DefaultHttpxClient
except ImportError:
    DefaultHttpxClient = None

load_dotenv(find_dotenv(), override=False)

# Global cache for token provider (reuse across calls to avoid re-auth overhead)
_cached_token_provider = None
_client_cache: dict[tuple[str, str, str], AzureOpenAI] = {}


class AzureOpenAIConfigError(RuntimeError):
    pass


def _stringify_content_part(part: Any) -> str:
    if isinstance(part, str):
        return part

    if isinstance(part, dict):
        if isinstance(part.get("text"), str):
            return part["text"]
        if isinstance(part.get("content"), str):
            return part["content"]
        if isinstance(part.get("value"), str):
            return part["value"]
        return ""

    text = getattr(part, "text", None)
    if isinstance(text, str):
        return text

    content = getattr(part, "content", None)
    if isinstance(content, str):
        return content

    value = getattr(part, "value", None)
    if isinstance(value, str):
        return value

    return ""


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        combined_text = "".join(_stringify_content_part(part) for part in content).strip()
        if combined_text:
            return combined_text

    refusal = getattr(message, "refusal", None)
    if isinstance(refusal, str) and refusal.strip():
        return refusal.strip()

    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return str(parsed).strip()

    return ""


def _write_debug_response_artifact(response: Any, deployment: str, response_text: str) -> Optional[Path]:
    debug_root = Path(os.getenv("AZURE_OPENAI_DEBUG_DIR", "output_cfi/debug_azure"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    deployment_slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in deployment)
    debug_path = debug_root / f"azure_response_{deployment_slug}_{timestamp}.json"

    try:
        debug_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "deployment": deployment,
            "extracted_text": response_text,
            "response": response.model_dump(mode="json") if hasattr(response, "model_dump") else str(response),
        }
        debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return debug_path
    except Exception as exc:
        print(f"  [WARNING] Failed to write Azure debug artifact: {exc}", file=sys.stderr)
        sys.stderr.flush()
        return None


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise AzureOpenAIConfigError(f"Missing required environment variable: {name}")
    return value


def _get_first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _resolve_ssl_verify_setting() -> str | bool:
    ca_bundle = _get_first_env(
        "AZURE_OPENAI_CA_BUNDLE",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "CURL_CA_BUNDLE",
    )
    if ca_bundle:
        return ca_bundle

    disable_verify = (os.getenv("AZURE_OPENAI_DISABLE_SSL_VERIFY") or "").strip().lower()
    if disable_verify in {"1", "true", "yes", "on"}:
        print("[WARNING] SSL certificate verification is disabled for Azure calls. Use only for local debugging.", file=sys.stderr)
        sys.stderr.flush()
        return False

    return True


def _resolve_api_key(deployment: Optional[str], explicit_api_key: Optional[str] = None) -> Optional[str]:
    if explicit_api_key:
        return explicit_api_key

    return (
        _get_deployment_specific_env("AZURE_OPENAI_API_KEY", deployment)
        or _get_first_env(
            "AZURE_OPENAI_API_KEY",
            "AZURE_API_KEY",
            "AZURE_OPENAI_KEY",
        )
    )


def _normalize_deployment_env_suffix(deployment: str) -> str:
    return "".join(ch for ch in deployment.upper() if ch.isalnum())


def _get_deployment_specific_env(base_name: str, deployment: Optional[str]) -> Optional[str]:
    if not deployment:
        return None
    suffix = _normalize_deployment_env_suffix(deployment)
    return _get_first_env(f"{base_name}_{suffix}")


def resolve_azure_runtime_config(
    deployment: Optional[str] = None,
    explicit_api_key: Optional[str] = None,
) -> dict[str, Optional[str]]:
    endpoint = _get_deployment_specific_env("AZURE_OPENAI_ENDPOINT", deployment) or _get_first_env(
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_ENDPOINT",
    )
    api_key = _resolve_api_key(deployment, explicit_api_key)
    api_version = _get_deployment_specific_env("AZURE_OPENAI_API_VERSION", deployment) or _get_first_env(
        "AZURE_OPENAI_API_VERSION",
        "AZURE_API_VERSION",
    ) or "2024-10-21"
    return {
        "endpoint": endpoint,
        "api_key": api_key,
        "api_version": api_version,
    }


def get_preferred_gpt5mini_deployment() -> Optional[str]:
    gpt5_endpoint = _get_first_env("AZURE_OPENAI_ENDPOINT_GPT5MINI")
    if not gpt5_endpoint:
        return None
    return _get_first_env("AZURE_OPENAI_DEPLOYMENT_GPT5MINI") or "gpt-5-mini"


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _disable_broken_local_proxies() -> None:
    proxy_names = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
    disabled = []

    for name in proxy_names:
        value = os.getenv(name)
        if not value:
            continue

        parsed = urlparse(value)
        host = parsed.hostname
        port = parsed.port

        if host not in {"127.0.0.1", "localhost"} or not port:
            continue

        if _can_connect(host, port):
            continue

        os.environ.pop(name, None)
        disabled.append(f"{name}={value}")

    if disabled:
        print(
            "[WARNING] Disabled unreachable local proxy settings for Azure calls: " + ", ".join(disabled),
            file=sys.stderr,
        )
        sys.stderr.flush()


def _build_azure_ad_token_provider():
    """Build and cache Azure AD token provider to avoid repeated credential creation."""
    global _cached_token_provider
    
    if _cached_token_provider is not None:
        return _cached_token_provider
    
    tenant_id = os.getenv("AZURE_TENANT_ID") or os.getenv("TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID") or os.getenv("CLIENT_ID") or os.getenv("APPLICATION_AI_VOS_USERS_ID")
    client_secret = os.getenv("AZURE_CLIENT_SECRET") or os.getenv("CLIENT_SECRET") or os.getenv("APPLICATION_AI_VOS_USERS_SECRET")

    if not all([tenant_id, client_id, client_secret]):
        return None

    try:
        from azure.identity import ClientSecretCredential, get_bearer_token_provider
        from azure.core.pipeline.transport import RequestsTransport
    except ImportError as exc:
        raise AzureOpenAIConfigError(
            "azure-identity is required for Azure AD authentication. Install it or set AZURE_OPENAI_API_KEY."
        ) from exc

    _disable_broken_local_proxies()

    ssl_verify = _resolve_ssl_verify_setting()

    print(f"[DEBUG] Creating Azure AD token provider (tenant={tenant_id[:20]}...)", file=sys.stderr)
    sys.stderr.flush()
    
    try:
        credential_kwargs = {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "client_secret": client_secret,
            "connection_timeout": 30,
            "read_timeout": 60,
        }
        if ssl_verify is not True:
            credential_kwargs["transport"] = RequestsTransport(connection_verify=ssl_verify)
        credential = ClientSecretCredential(**credential_kwargs)
    except TypeError:
        # Fallback if timeout parameters not supported in older azure-identity
        print(f"[DEBUG] Azure AD timeout params not supported, using defaults", file=sys.stderr)
        credential_kwargs = {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "client_secret": client_secret,
        }
        if ssl_verify is not True:
            credential_kwargs["transport"] = RequestsTransport(connection_verify=ssl_verify)
        credential = ClientSecretCredential(**credential_kwargs)
    
    _cached_token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default",
    )
    print(f"[DEBUG] Token provider created and cached successfully", file=sys.stderr)
    sys.stderr.flush()
    return _cached_token_provider


def get_default_deployment() -> str:
    deployment = _get_first_env("AZURE_OPENAI_DEPLOYMENT", "AZURE_CHAT_DEPLOYMENT")
    if not deployment:
        raise AzureOpenAIConfigError(
            "Missing required environment variable: AZURE_OPENAI_DEPLOYMENT (or AZURE_CHAT_DEPLOYMENT)"
        )
    return deployment


def get_azure_openai_client(deployment: Optional[str] = None, api_key: Optional[str] = None) -> AzureOpenAI:
    _disable_broken_local_proxies()
    runtime_config = resolve_azure_runtime_config(deployment, explicit_api_key=api_key)
    endpoint = runtime_config["endpoint"]
    if not endpoint:
        raise AzureOpenAIConfigError(
            "Missing required environment variable: AZURE_OPENAI_ENDPOINT (or AZURE_ENDPOINT). If using gpt-5-mini on a separate resource, set AZURE_OPENAI_ENDPOINT_GPT5MINI."
        )
    api_version = runtime_config["api_version"] or "2024-10-21"
    api_key = runtime_config["api_key"]

    cache_key = (
        endpoint,
        api_version,
        "api_key" if api_key else "azure_ad",
    )
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    client_kwargs = {
        "azure_endpoint": endpoint,
        "api_version": api_version,
    }

    ssl_verify = _resolve_ssl_verify_setting()
    if ssl_verify is not True and DefaultHttpxClient is not None:
        client_kwargs["http_client"] = DefaultHttpxClient(verify=ssl_verify)

    if api_key:
        print("[DEBUG] Azure OpenAI auth mode: API key", file=sys.stderr)
        sys.stderr.flush()
        client_kwargs["api_key"] = api_key
    else:
        print("[DEBUG] Azure OpenAI auth mode: Azure AD token", file=sys.stderr)
        sys.stderr.flush()
        token_provider = _build_azure_ad_token_provider()
        if token_provider is None:
            raise AzureOpenAIConfigError(
                "Provide AZURE_OPENAI_API_KEY (preferred) or Azure AD creds: AZURE_TENANT_ID + (AZURE_CLIENT_ID or APPLICATION_AI_VOS_USERS_ID) + (AZURE_CLIENT_SECRET or APPLICATION_AI_VOS_USERS_SECRET). If your company network intercepts TLS, also set AZURE_OPENAI_CA_BUNDLE/REQUESTS_CA_BUNDLE to the corporate root CA PEM path."
            )
        client_kwargs["azure_ad_token_provider"] = token_provider

    client = AzureOpenAI(**client_kwargs)
    _client_cache[cache_key] = client
    return client


class AzureChatClient:
    def __init__(
        self,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
    ):
        self.deployment = deployment or get_default_deployment()
        self.api_key = api_key
        self.client = get_azure_openai_client(self.deployment, api_key=api_key)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        deployment: Optional[str] = None,
    ) -> str:
        effective_deployment = deployment or self.deployment
        client = self.client if effective_deployment == self.deployment else get_azure_openai_client(effective_deployment, api_key=self.api_key)

        request_kwargs = {
            "model": effective_deployment,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # "temperature": self.temperature if temperature is None else temperature,
        }
        token_limit = self.max_output_tokens if max_output_tokens is None else max_output_tokens
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        # Add timeout and retry logic
        max_retries = 3
        retry_delay = 2  # seconds, will exponential backoff
        
        for attempt in range(max_retries):
            try:
                print(f"  [DEBUG] Azure call attempt {attempt+1}/{max_retries}: model={request_kwargs['model']}, tokens={token_limit}", file=sys.stderr)
                sys.stderr.flush()
                
                response = client.chat.completions.create(
                    **request_kwargs,
                    max_completion_tokens=token_limit,
                    timeout=60.0,  # 60 second timeout per request
                )
                print(f"  [DEBUG] Azure call succeeded on attempt {attempt+1}", file=sys.stderr)
                sys.stderr.flush()
                response_text = _extract_response_text(response)
                if not response_text:
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                    debug_path = _write_debug_response_artifact(response, effective_deployment, response_text)
                    print(
                        f"  [WARNING] Azure response content was empty (finish_reason={finish_reason})",
                        file=sys.stderr,
                    )
                    if debug_path is not None:
                        print(f"  [WARNING] Raw Azure response saved to: {debug_path}", file=sys.stderr)
                        try:
                            first_choice = response.model_dump(mode="json").get("choices", [None])[0]
                            print(
                                "  [WARNING] Raw first choice: " + json.dumps(first_choice, ensure_ascii=False)[:2000],
                                file=sys.stderr,
                            )
                        except Exception:
                            pass
                    sys.stderr.flush()
                return response_text
                
            except TypeError as e:
                # max_completion_tokens not supported, try max_tokens
                if "max_completion_tokens" in str(e):
                    print(f"  [DEBUG] Retrying with max_tokens instead of max_completion_tokens", file=sys.stderr)
                    sys.stderr.flush()
                    try:
                        response = client.chat.completions.create(
                            **request_kwargs,
                            max_tokens=token_limit,
                            timeout=180.0,
                        )
                        response_text = _extract_response_text(response)
                        if not response_text:
                            finish_reason = getattr(response.choices[0], "finish_reason", None)
                            debug_path = _write_debug_response_artifact(response, effective_deployment, response_text)
                            print(
                                f"  [WARNING] Azure response content was empty (finish_reason={finish_reason})",
                                file=sys.stderr,
                            )
                            if debug_path is not None:
                                print(f"  [WARNING] Raw Azure response saved to: {debug_path}", file=sys.stderr)
                                try:
                                    first_choice = response.model_dump(mode="json").get("choices", [None])[0]
                                    print(
                                        "  [WARNING] Raw first choice: " + json.dumps(first_choice, ensure_ascii=False)[:2000],
                                        file=sys.stderr,
                                    )
                                except Exception:
                                    pass
                            sys.stderr.flush()
                        return response_text
                    except Exception as e2:
                        print(f"  [ERROR] Failed with max_tokens: {type(e2).__name__}: {str(e2)[:200]}", file=sys.stderr)
                        sys.stderr.flush()
                        raise
                else:
                    raise
                    
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                
                # Check if auth error (don't retry, just fail fast)
                is_auth_error = "ClientAuthenticationError" in error_type or "authentication" in error_msg.lower()
                
                # Check if retryable (API timeout, rate limit, server error)
                is_retryable = any(x in error_msg.lower() for x in ["timeout", "connection", "429", "500", "503", "temporary"])
                
                print(f"  [WARNING] Azure call failed (attempt {attempt+1}): {error_type}: {error_msg[:150]}", file=sys.stderr)
                print(f"  [WARNING] Auth Error: {is_auth_error}, Retryable: {is_retryable}", file=sys.stderr)
                sys.stderr.flush()
                
                # Don't retry auth errors - they will fail repeatedly
                if is_auth_error:
                    if "certificate verify failed" in error_msg.lower():
                        print(
                            "  [FATAL] TLS certificate verification failed while authenticating. Prefer AZURE_OPENAI_API_KEY auth, or set AZURE_OPENAI_CA_BUNDLE/REQUESTS_CA_BUNDLE to your corporate root CA PEM path.",
                            file=sys.stderr,
                        )
                    print(f"  [FATAL] Authentication error - not retrying. Check Azure AD credentials and network.", file=sys.stderr)
                    sys.stderr.flush()
                    raise
                
                if not is_retryable or attempt == max_retries - 1:
                    print(f"  [ERROR] Azure API call failed (final attempt): {error_type}", file=sys.stderr)
                    sys.stderr.flush()
                    raise
                
                # Exponential backoff: 2s, 4s, 8s
                wait_time = retry_delay * (2 ** attempt)
                print(f"  [DEBUG] Waiting {wait_time}s before retry...", file=sys.stderr)
                sys.stderr.flush()
                time.sleep(wait_time)