import json
import re
from typing import Any

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    tiktoken = None

try:
    from langchain_core.output_parsers import JsonOutputParser  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    JsonOutputParser = None


def count_token(text: str, model_name: str = "gpt-4o", encoding_name: str = "o200k_base") -> int:
    """Count tokens with a tiktoken fallback.

    For full baseline reproduction, install tiktoken. If it is unavailable,
    this falls back to a whitespace-based approximation so prepare-only flows
    can still run.
    """
    if tiktoken is None:
        return max(1, len(text.split()))
    if model_name:
        try:
            encoding_name = tiktoken.encoding_name_for_model(model_name)
        except Exception:
            pass
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def _extract_json_span(text: str, open_char: str, close_char: str) -> str:
    start = text.index(open_char)
    end = text.rindex(close_char) + 1
    return text[start:end]


def _loads_with_backslash_fix(raw: str) -> Any:
    try:
        return json.loads(raw, strict=False)
    except json.JSONDecodeError:
        pattern = r'(?<!\)\(?![\"])'
        processed = re.sub(pattern, r'\\', raw)
        return json.loads(processed, strict=False)


def load_json_object(text: str):
    try:
        json_text = _extract_json_span(text, '{', '}')
        return _loads_with_backslash_fix(json_text)
    except Exception:
        if JsonOutputParser is not None:
            parser = JsonOutputParser()
            return parser.parse(text)
        raise ValueError('Unable to parse JSON object from model response.')


def load_json_array(text: str):
    try:
        json_text = _extract_json_span(text, '[', ']')
        return _loads_with_backslash_fix(json_text)
    except Exception:
        if JsonOutputParser is not None:
            parser = JsonOutputParser()
            return parser.parse(text)
        raise ValueError('Unable to parse JSON array from model response.')
