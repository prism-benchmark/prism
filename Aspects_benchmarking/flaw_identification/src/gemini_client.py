from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv, find_dotenv

from src.azure_openai_client import AzureChatClient

# Auto-load .env once (project root or parent dirs)
load_dotenv(find_dotenv(), override=False)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _is_hex4(s: str) -> bool:
    return len(s) == 4 and all(c in "0123456789abcdefABCDEF" for c in s)


def _fix_json_backslashes_charwise(s: str) -> str:
    r"""
    Repair invalid JSON backslash escapes by doubling backslashes that do NOT start
    a valid JSON escape sequence.

    Valid JSON escapes:
      \" \\ \/ \b \f \n \r \t \\uXXXX

    Common failure case:
      "\alpha" is invalid JSON (because "\a" is not a valid JSON escape)
      This function transforms it into "\\alpha" in the JSON source so json.loads accepts it.
    """
    out: list[str] = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        # Backslash at end
        if i == n - 1:
            out.append("\\\\")
            i += 1
            continue

        nxt = s[i + 1]

        # Valid simple escapes
        if nxt in ['"', "\\", "/", "b", "f", "n", "r", "t"]:
            out.append("\\")
            out.append(nxt)
            i += 2
            continue

        # Unicode escape must be \uXXXX
        if nxt == "u":
            if i + 6 <= n and _is_hex4(s[i + 2 : i + 6]):
                out.append("\\")
                out.append("u")
                out.append(s[i + 2 : i + 6])
                i += 6
                continue
            # bad unicode escape like \u12GZ or \u1 -> escape the backslash
            out.append("\\\\")
            i += 1
            continue

        # Anything else is invalid JSON escape (e.g., \alpha, \text, \cite, \_)
        out.append("\\\\")
        i += 1

    return "".join(out)


def json_loads_lenient(s: str) -> Any:
    """
    Parse JSON. If it fails due to invalid escapes (common with LaTeX backslashes)
    or raw control characters (literal newlines inside string values from Gemini),
    repair and retry. Includes a final fallback that doubles ALL backslashes.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        msg = str(e)

        # Stage 0: raw control characters (e.g. literal \n or \t inside a JSON string).
        # json.loads(strict=False) accepts these without modification.
        if "Invalid control character" in msg:
            try:
                return json.loads(s, strict=False)
            except json.JSONDecodeError:
                pass  # fall through to backslash repair below

        if ("Invalid \\escape" not in msg) and ("Invalid \\u" not in msg):
            raise

        # Stage A: targeted repair
        fixed = _fix_json_backslashes_charwise(s)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError as e2:
            msg2 = str(e2)
            if ("Invalid \\escape" not in msg2) and ("Invalid \\u" not in msg2):
                raise

            # Stage B (nuclear): double every backslash everywhere
            fixed2 = s.replace("\\", "\\\\")
            return json.loads(fixed2)


def _extract_balanced_json_object(text: str) -> str:
    """
    Extract the first balanced {...} JSON object from text,
    respecting strings and escape sequences.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Unbalanced JSON braces in model output.")


def extract_first_json_object(text: str) -> str:
    """
    Extract the first JSON object from model output and validate it (lenient).
    Returns a JSON string.
    """
    text = strip_code_fences(text)

    # Fast path: entire response is JSON
    try:
        json_loads_lenient(text)
        return text
    except Exception:
        pass

    candidate = _extract_balanced_json_object(text).strip()
    json_loads_lenient(candidate)  # validate
    return candidate


class GeminiClient:
    """
    Backward-compatible wrapper that now routes generation through Azure OpenAI.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
        sleep_on_rate_limit: float = 2.0,
    ):
        self.api_key = api_key
        self.model = model or os.getenv("AZURE_OPENAI_EXTRACTION_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.temperature = float(os.getenv("TEMPERATURE", str(temperature)))
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", str(max_output_tokens)))
        self.sleep_on_rate_limit = sleep_on_rate_limit
        self._client = AzureChatClient(
            deployment=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    def generate_text(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                return self._client.generate_text(
                    "You are a precise research-review analysis assistant.",
                    prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    deployment=self.model,
                )

            except Exception as e:
                last_err = e
                time.sleep(self.sleep_on_rate_limit * (attempt + 1))

        raise RuntimeError(f"Azure generation failed after retries: {last_err}")


def repair_json_with_model(client: GeminiClient, bad_output: str, schema_hint: str) -> Dict[str, Any]:
    repair_prompt = f"""
Your previous output was invalid. Return ONLY valid JSON.

Constraints:
- Output must be a single JSON object.
- No markdown, no code fences, no commentary.

Schema hint:
{schema_hint}

Invalid output:
{bad_output}
""".strip()

    raw = client.generate_text(repair_prompt)
    json_str = extract_first_json_object(raw)
    return json_loads_lenient(json_str)
