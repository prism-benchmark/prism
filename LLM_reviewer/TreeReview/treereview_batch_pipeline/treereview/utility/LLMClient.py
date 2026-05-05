import datetime
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None

try:
    from langchain.llms.base import LLM  # type: ignore
    from langchain_core.callbacks import CallbackManagerForLLMRun  # type: ignore
except Exception:  # pragma: no cover
    class LLM:
        pass

    class CallbackManagerForLLMRun:
        pass

load_dotenv()


class BaseClient:
    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.0-flash",
        default_temperature: Optional[float] = 0.0,
        default_top_p: Optional[float] = 1.0,
        default_max_tokens: Optional[int] = 32768,
        default_frequency_penalty: Optional[float] = 0.0,
        default_presence_penalty: Optional[float] = 0.0,
        system_prompt: Optional[str] = None,
        cache_db_path: Optional[str] = None,
        max_retires: Optional[int] = 4,
        fingerprint: Optional[str] = None,
    ):
        self.base_url = base_url
        self.api_key = api_key
        if not api_key:
            raise ValueError(
                "Missing API key for LLMClient. Set GEMINI_API_KEY in your environment or .env file."
            )
        self.openai_client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.system_prompt = system_prompt
        self.fingerprint = fingerprint or datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.cache_db = sqlite3.connect(cache_db_path) if cache_db_path else None
        if self.cache_db:
            self._init_db()
        self.default_params = {
            "model": default_model or "gpt-4o",
            "temperature": default_temperature,
            "top_p": default_top_p,
            "max_tokens": default_max_tokens,
            "frequency_penalty": default_frequency_penalty,
            "presence_penalty": default_presence_penalty,
        }
        self.max_retires = max(0, max_retires)
        self.tokenizers: Dict[str, Any] = {}
        self.reset()

    @staticmethod
    def from_config(config):
        return LLMClient(
            base_url=config.get("base_url"),
            api_key=config.get("api_key"),
            default_model=config.get("default_model", "gpt-4o"),
            default_temperature=config.get("default_temperature", 0.0),
            default_top_p=config.get("default_top_p", 1.0),
            default_max_tokens=config.get("default_max_tokens", 1024),
            default_frequency_penalty=config.get("default_frequency_penalty", 0.0),
            default_presence_penalty=config.get("default_presence_penalty", 0.0),
            cache_db_path=config.get("cache_db_path"),
            max_retires=config.get("max_retires", 4),
            fingerprint=config.get("fingerprint"),
        )

    def reset(self):
        self.messages = []
        if self.system_prompt is not None:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def _init_db(self):
        cur = self.cache_db.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_cache (
                fingerprint TEXT,
                model TEXT NOT NULL,
                messages_json TEXT NOT NULL,
                temperature REAL NOT NULL,
                top_p REAL NOT NULL,
                max_tokens INTEGER NOT NULL,
                frequency_penalty REAL NOT NULL,
                presence_penalty REAL NOT NULL,
                response_json TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS token_usage (
                fingerprint TEXT,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                is_cached INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self.cache_db.commit()

    def _get_tokenizer(self, model: str):
        if model not in self.tokenizers:
            if tiktoken is None:
                self.tokenizers[model] = None
            else:
                try:
                    self.tokenizers[model] = tiktoken.encoding_for_model(model)
                except KeyError:
                    self.tokenizers[model] = tiktoken.get_encoding("cl100k_base")
        return self.tokenizers[model]

    def estimate_messages_num_tokens(self, messages: List[Dict], model: str) -> int:
        tokenizer = self._get_tokenizer(model)
        if tokenizer is None:
            return sum(max(1, len(str(v).split())) for msg in messages for v in msg.values())
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key in ["role", "content"]:
                    num_tokens += len(tokenizer.encode(value))
                if key in ["name", "role"]:
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def chat(self, prompt: str, **kwargs):
        self.messages.append({"role": "user", "content": prompt})
        response = self._run_prompt_basic(messages=self.messages, **kwargs)
        response_message = response["choices"][0]["message"]
        self.messages.append(response_message.copy())
        return response_message["content"]

    def generate(self, prompt: str, use_system_prompt: bool = False, system_prompt: Optional[str] = None, **kwargs) -> str:
        system_prompt = system_prompt or self.system_prompt
        messages = []
        if use_system_prompt:
            if system_prompt is None:
                raise ValueError("use_system_prompt=True requires a system prompt.")
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._run_prompt_basic(messages=messages, **kwargs)
        return response["choices"][0]["message"]["content"]

    def _run_prompt_basic(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        use_cache: bool = True,
        max_retries: Optional[int] = None,
    ) -> Dict:
        params = {
            "model": model or self.default_params["model"],
            "temperature": self.default_params["temperature"] if temperature is None else temperature,
            "top_p": self.default_params["top_p"] if top_p is None else top_p,
            "max_tokens": self.default_params["max_tokens"] if max_tokens is None else max_tokens,
            "frequency_penalty": self.default_params["frequency_penalty"] if frequency_penalty is None else frequency_penalty,
            "presence_penalty": self.default_params["presence_penalty"] if presence_penalty is None else presence_penalty,
        }
        max_retries = self.max_retires if max_retries is None else max_retries

        db_keyvals = params.copy()
        messages_json = json.dumps(messages, sort_keys=True)
        db_keyvals["messages_json"] = messages_json
        cache_json = None
        token_usage_keyvals = None

        if use_cache and self.cache_db and params["temperature"] == 0:
            cur = self.cache_db.cursor()
            select_keyvals = db_keyvals.copy()
            select_keyvals["messages_token_count"] = self.estimate_messages_num_tokens(messages, params["model"])
            dbrecs = cur.execute(
                """select response_json, model, prompt_tokens, completion_tokens, total_tokens from chat_cache
                   where model = :model and messages_json = :messages_json and temperature = :temperature and
                         ((:messages_token_count+max_tokens) > total_tokens or max_tokens = :max_tokens) and
                         total_tokens <= (:messages_token_count+:max_tokens) and top_p = :top_p and
                         frequency_penalty = :frequency_penalty and presence_penalty = :presence_penalty""",
                select_keyvals,
            ).fetchall()
            if dbrecs:
                cache_json = dbrecs[0][0]
                token_usage_keyvals = {
                    "fingerprint": self.fingerprint,
                    "model": dbrecs[0][1],
                    "prompt_tokens": dbrecs[0][2],
                    "completion_tokens": dbrecs[0][3],
                    "total_tokens": dbrecs[0][4],
                    "is_cached": 1,
                    "created_at": datetime.datetime.timestamp(datetime.datetime.utcnow()),
                }

        if cache_json is None:
            model_keyvals = db_keyvals.copy()
            del model_keyvals["messages_json"]
            call_params = model_keyvals.copy()
            call_params["messages"] = messages
            resp = None
            if max_retries > 0:
                remaining = max_retries
                while resp is None and remaining >= 0:
                    remaining -= 1
                    try:
                        resp = self.openai_client.chat.completions.create(**call_params).model_dump()
                    except openai.RateLimitError:
                        time.sleep(60)
                    except openai.APIError as e:
                        if getattr(e, "code", None) in [502, "RequestTimeOut", 400]:
                            time.sleep(10)
                        else:
                            time.sleep(10)
                    except openai.APITimeoutError:
                        time.sleep(10)
                    except openai.BadRequestError as e:
                        raise ValueError(f"Bad request to LLM API: {e}") from e
            else:
                resp = self.openai_client.chat.completions.create(**call_params).model_dump()
            if resp is None:
                raise RuntimeError("LLM request failed after retries; no response was returned.")
            insert_keyvals = db_keyvals.copy()
            cache_json = json.dumps(resp)
            insert_keyvals.update(
                {
                    "fingerprint": self.fingerprint,
                    "response_json": cache_json,
                    "created_at": datetime.datetime.timestamp(datetime.datetime.utcnow()),
                    "prompt_tokens": resp["usage"]["prompt_tokens"],
                    "completion_tokens": resp["usage"]["completion_tokens"],
                    "total_tokens": resp["usage"]["total_tokens"],
                }
            )
            token_usage_keyvals = {
                "fingerprint": self.fingerprint,
                "model": params["model"],
                "prompt_tokens": resp["usage"]["prompt_tokens"],
                "completion_tokens": resp["usage"]["completion_tokens"],
                "total_tokens": resp["usage"]["total_tokens"],
                "is_cached": 0,
                "created_at": datetime.datetime.timestamp(datetime.datetime.utcnow()),
            }
            if use_cache and self.cache_db:
                cur = self.cache_db.cursor()
                cur.execute(
                    """INSERT INTO chat_cache (fingerprint, model, messages_json, temperature, top_p, max_tokens,
                       frequency_penalty, presence_penalty, response_json, created_at, prompt_tokens,
                       completion_tokens, total_tokens)
                       VALUES (:fingerprint, :model, :messages_json, :temperature, :top_p, :max_tokens,
                               :frequency_penalty, :presence_penalty, :response_json, :created_at,
                               :prompt_tokens, :completion_tokens, :total_tokens)""",
                    insert_keyvals,
                )
                self.cache_db.commit()

        if token_usage_keyvals is not None and use_cache and self.cache_db:
            cur = self.cache_db.cursor()
            cur.execute(
                """INSERT INTO token_usage (fingerprint, model, prompt_tokens, completion_tokens, total_tokens, is_cached, created_at)
                   VALUES (:fingerprint, :model, :prompt_tokens, :completion_tokens, :total_tokens, :is_cached, :created_at)""",
                token_usage_keyvals,
            )
            self.cache_db.commit()
        return json.loads(cache_json)

    def get_token_usage(self, model: Optional[str] = None) -> Dict:
        if not self.cache_db:
            return {"records": []}
        cur = self.cache_db.cursor()
        if model:
            rows = cur.execute(
                "select model, prompt_tokens, completion_tokens, total_tokens, is_cached, created_at from token_usage where model = ?",
                (model,),
            ).fetchall()
        else:
            rows = cur.execute(
                "select model, prompt_tokens, completion_tokens, total_tokens, is_cached, created_at from token_usage"
            ).fetchall()
        return {"records": rows}


class LLMClient(LLM):
    base_client: BaseClient = None

    def __init__(self, **kwargs):
        super().__init__()
        api_key = kwargs.pop("api_key", None) or os.getenv("GEMINI_API_KEY")
        self.base_client = BaseClient(api_key=api_key, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom-openai-compatible"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not supported by this LLMClient implementation.")
        return self.base_client.generate(prompt, **kwargs)

    def invoke(self, input, config=None, **kwargs):
        if isinstance(input, list):
            text_parts = []
            for item in input:
                content = getattr(item, "content", None)
                if content is not None:
                    text_parts.append(str(content))
                else:
                    text_parts.append(str(item))
            prompt = "\n\n".join(text_parts)
        else:
            prompt = str(input)
        return self.base_client.generate(prompt, **kwargs)
