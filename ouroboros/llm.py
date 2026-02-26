"""
Ouroboros — LLM client.

Routes to:
1. Google Vertex AI (google-genai SDK) if GOOGLE_APPLICATION_CREDENTIALS is set and model is Google
2. Google AI Studio (API key) if GOOGLE_API_KEY is set and model is Google
3. OpenRouter for everything else

Vertex uses $300 credits with no daily rate wall.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_LIGHT_MODEL = "google/gemini-3.0-flash"

# Google model name mapping: OpenRouter-style → native
_GOOGLE_MODEL_MAP = {
    "google/gemini-3.1-pro-preview": "publishers/google/models/gemini-3.1-pro-preview",
    "google/gemini-3-pro-preview":   "publishers/google/models/gemini-3-pro-preview",
    "google/gemini-3-flash-preview": "publishers/google/models/gemini-3-flash-preview",
    "google/gemini-2.5-pro":         "gemini-2.5-pro",
    "google/gemini-2.5-pro-preview": "gemini-2.5-pro-preview-05-06",
    "google/gemini-2.5-flash":       "gemini-2.5-flash",
    "google/gemini-2.0-flash":       "gemini-2.0-flash",
}

# Vertex region: 3.x models require "global", 2.x models need a real region
_VERTEX_LOCATION_GLOBAL = "global"
_VERTEX_LOCATION_REGIONAL = "us-central1"
_VERTEX_PROJECT = "gen-lang-client-0999726021"


def _vertex_location_for_model(native_model: str) -> str:
    """Determine the correct Vertex region for a model.
    Gemini 3.x (publishers/google/models/) → global
    Gemini 2.x and others → us-central1
    """
    if "publishers/google/models/" in native_model:
        return _VERTEX_LOCATION_GLOBAL
    return _VERTEX_LOCATION_REGIONAL


def _is_google_model(model: str) -> bool:
    return model.startswith("google/") or model.startswith("gemini-")


def _google_native_model_id(model: str) -> str:
    return _GOOGLE_MODEL_MAP.get(model, model.replace("google/", ""))


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    import logging
    log = logging.getLogger("ouroboros.llm")
    try:
        import requests
    except ImportError:
        log.warning("requests not installed, cannot fetch pricing")
        return {}
    try:
        url = "https://openrouter.ai/api/v1/models"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        prefixes = ("anthropic/", "openai/", "google/", "meta-llama/", "x-ai/", "qwen/")
        pricing_dict = {}
        for model in models:
            model_id = model.get("id", "")
            if not model_id.startswith(prefixes):
                continue
            pricing = model.get("pricing", {})
            if not pricing or not pricing.get("prompt"):
                continue
            raw_prompt = float(pricing.get("prompt", 0))
            raw_completion = float(pricing.get("completion", 0))
            raw_cached_str = pricing.get("input_cache_read")
            raw_cached = float(raw_cached_str) if raw_cached_str else None
            prompt_price = round(raw_prompt * 1_000_000, 4)
            completion_price = round(raw_completion * 1_000_000, 4)
            cached_price = round(raw_cached * 1_000_000, 4) if raw_cached is not None else round(prompt_price * 0.1, 4)
            if prompt_price > 1000 or completion_price > 1000:
                continue
            pricing_dict[model_id] = (prompt_price, cached_price, completion_price)
        log.info(f"Fetched pricing for {len(pricing_dict)} models from OpenRouter")
        return pricing_dict
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning(f"Failed to fetch OpenRouter pricing: {e}")
        return {}


def _build_tool_call_id_to_name(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build mapping from tool_call_id → function name from assistant messages.
    
    Gemini FunctionResponse requires the actual function name (e.g. 'repo_read'),
    but OpenAI tool result messages only carry tool_call_id. This resolves them.
    """
    mapping = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in (msg.get("tool_calls") or []):
                tc_id = tc.get("id", "")
                fn_name = tc.get("function", {}).get("name", "")
                if tc_id and fn_name:
                    mapping[tc_id] = fn_name
    return mapping


def _clean_schema_for_gemini(schema: Any) -> Any:
    """Recursively strip JSON Schema keywords that Gemini doesn't support.
    
    Gemini chokes on: additionalProperties, default, oneOf, anyOf, allOf,
    $ref, $schema, pattern, format, minLength, maxLength, etc.
    """
    if not isinstance(schema, dict):
        return schema

    UNSUPPORTED = {
        "additionalProperties", "$schema", "$ref", "$defs",
        "default", "oneOf", "anyOf", "allOf",
        "if", "then", "else", "not", "patternProperties",
        "minItems", "maxItems", "uniqueItems",
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf", "minLength", "maxLength", "pattern", "format",
        "examples", "title",
    }

    cleaned = {}
    for key, value in schema.items():
        if key in UNSUPPORTED:
            continue
        if key == "type":
            if isinstance(value, list):
                non_null = [t for t in value if t != "null"]
                cleaned["type"] = non_null[0] if non_null else "string"
            elif value == "null":
                cleaned["type"] = "string"
            else:
                cleaned["type"] = value
        elif key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {
                k: _clean_schema_for_gemini(v) for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            cleaned["items"] = _clean_schema_for_gemini(value)
        elif key == "enum" and isinstance(value, list):
            cleaned["enum"] = [v for v in value if v is not None]
        else:
            cleaned[key] = value

    if "type" not in cleaned and "properties" in cleaned:
        cleaned["type"] = "object"

    return cleaned


def _merge_consecutive_roles(contents: list) -> list:
    """Merge consecutive Content objects with the same role.
    
    Gemini requires strictly alternating user/model roles.
    Multiple tool results (each role='user') must be merged into one.
    """
    if not contents:
        return contents
    merged = [contents[0]]
    for c in contents[1:]:
        if merged[-1].role == c.role:
            merged[-1].parts.extend(c.parts)
        else:
            merged.append(c)
    return merged


class LLMClient:
    """LLM API wrapper. Routes Google models to Vertex/AI Studio, others to OpenRouter."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._client = None
        self._google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        self._vertex_creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        self._vertex_clients: Dict[str, Any] = {}

    def _use_vertex(self, model: str) -> bool:
        return bool(self._vertex_creds_path and os.path.exists(self._vertex_creds_path)) and _is_google_model(model)

    def _use_google_studio(self, model: str) -> bool:
        return bool(self._google_api_key) and _is_google_model(model) and not self._use_vertex(model)

    def _get_vertex_client(self, location: str = _VERTEX_LOCATION_GLOBAL):
        """Get or create a Vertex AI client for the given location."""
        if location not in self._vertex_clients:
            import google.oauth2.service_account
            from google import genai
            credentials = google.oauth2.service_account.Credentials.from_service_account_file(
                self._vertex_creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._vertex_clients[location] = genai.Client(
                vertexai=True,
                project=_VERTEX_PROJECT,
                location=location,
                credentials=credentials,
            )
            log.info("Initialized Vertex AI client (project=%s, location=%s)", _VERTEX_PROJECT, location)
        return self._vertex_clients[location]

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/WozzySalmon/ourobruhs",
                    "X-Title": "Ouroboros",
                },
            )
        return self._client

    def _get_google_studio_client(self):
        from openai import OpenAI
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=self._google_api_key,
        )

    def _fetch_generation_cost(self, generation_id: str) -> Optional[float]:
        try:
            import requests
            url = f"{self._base_url.rstrip('/')}/generation?id={generation_id}"
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
            time.sleep(0.5)
            resp = requests.get(url, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("data") or {}
                cost = data.get("total_cost") or data.get("usage", {}).get("cost")
                if cost is not None:
                    return float(cost)
        except Exception:
            log.debug("Failed to fetch generation cost", exc_info=True)
        return None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        effort = normalize_reasoning_effort(reasoning_effort)
        if self._use_vertex(model):
            return self._chat_vertex(messages, model, tools, effort, max_tokens, tool_choice)
        if self._use_google_studio(model):
            return self._chat_google_studio(messages, model, tools, effort, max_tokens, tool_choice)
        return self._chat_openrouter(messages, model, tools, effort, max_tokens, tool_choice)

    # Per-model rate limiter: track last call time to avoid hammering Vertex
    _last_vertex_call: float = 0.0
    _VERTEX_MIN_INTERVAL: float = 1.5  # seconds between calls

    def _chat_vertex(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Chat via Vertex AI using google-genai SDK. Uses $300 credits, no daily wall."""
        from google.genai import types as gtypes

        # Rate limiting: ensure minimum interval between Vertex calls
        now = time.time()
        elapsed = now - self._last_vertex_call
        if elapsed < self._VERTEX_MIN_INTERVAL:
            time.sleep(self._VERTEX_MIN_INTERVAL - elapsed)
        self._last_vertex_call = time.time()

        native_model = _google_native_model_id(model)
        client = self._get_vertex_client(location=_vertex_location_for_model(native_model))
        log.debug(f"Vertex call: {model} → {native_model} (location={_vertex_location_for_model(native_model)})")

        # Build tool_call_id → function name mapping for tool results
        tc_id_to_name = _build_tool_call_id_to_name(messages)

        # Convert OpenAI-style messages to google-genai contents
        # Key insight: Gemini 3.x thinking models attach thought_signatures to
        # function calls. These MUST be replayed verbatim — reconstructing from
        # OpenAI format loses them and causes 400 errors. We stash raw Gemini
        # Content objects on message dicts as '_gemini_content' and use them
        # when available.
        system_parts = []
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""

            # Fast path: use cached raw Gemini Content if available (preserves thought_signatures)
            if "_gemini_content" in msg:
                contents.append(msg["_gemini_content"])
                continue

            if role == "system":
                # Accumulate all system messages (don't overwrite — there may be many)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text.strip():
                                system_parts.append(text)
                        elif isinstance(block, str) and block.strip():
                            system_parts.append(block)
                elif isinstance(content, str) and content.strip():
                    system_parts.append(content)
            elif role == "assistant":
                # Handle tool calls in assistant messages
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    parts = []
                    if content:
                        parts.append(gtypes.Part(text=content))
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        args = json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else (fn.get("arguments") or {})
                        parts.append(gtypes.Part(function_call=gtypes.FunctionCall(name=fn.get("name", ""), args=args)))
                    contents.append(gtypes.Content(role="model", parts=parts))
                else:
                    contents.append(gtypes.Content(role="model", parts=[gtypes.Part(text=content)]))
            elif role == "tool":
                try:
                    result = json.loads(content) if isinstance(content, str) else content
                except Exception:
                    result = {"result": str(content)[:10000]}
                if not isinstance(result, dict):
                    result = {"result": str(result)[:10000]}
                # Resolve function name from tool_call_id (Gemini requires actual name, not just ID)
                tool_call_id = msg.get("tool_call_id", "")
                tool_name = tc_id_to_name.get(tool_call_id) or msg.get("name") or "unknown_tool"
                gemini_content = gtypes.Content(role="user", parts=[
                    gtypes.Part(function_response=gtypes.FunctionResponse(name=tool_name, response=result))
                ])
                # Stash for future replay
                msg["_gemini_content"] = gemini_content
                contents.append(gemini_content)
            else:
                # Handle multipart content (vision)
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part.get("type") == "text":
                            parts.append(gtypes.Part(text=part["text"]))
                        elif part.get("type") == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                import base64 as _b64
                                mime, data = url[5:].split(";base64,", 1)
                                parts.append(gtypes.Part(inline_data=gtypes.Blob(mime_type=mime, data=_b64.b64decode(data))))
                            else:
                                parts.append(gtypes.Part(text=f"[image: {url}]"))
                    contents.append(gtypes.Content(role="user", parts=parts))
                else:
                    contents.append(gtypes.Content(role="user", parts=[gtypes.Part(text=content)]))

        # Build tool declarations (clean schemas for Gemini compatibility)
        tool_list = None
        if tools:
            declarations = []
            for t in tools:
                fn = t.get("function", {})
                raw_params = fn.get("parameters")
                cleaned_params = _clean_schema_for_gemini(raw_params) if raw_params else None
                try:
                    declarations.append(gtypes.FunctionDeclaration(
                        name=fn.get("name", ""),
                        description=(fn.get("description", "") or "")[:1024],
                        parameters=cleaned_params,
                    ))
                except Exception as e:
                    log.warning("Failed to convert tool '%s' for Gemini: %s", fn.get("name"), e)
                    # Fallback: declare without parameters
                    try:
                        declarations.append(gtypes.FunctionDeclaration(
                            name=fn.get("name", ""),
                            description=(fn.get("description", "") or "")[:1024],
                        ))
                    except Exception:
                        pass
            if declarations:
                tool_list = [gtypes.Tool(function_declarations=declarations)]

        config_kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tool_list:
            config_kwargs["tools"] = tool_list

        # Merge consecutive same-role messages (Gemini strict requirement)
        contents = _merge_consecutive_roles(contents)

        try:
            resp = client.models.generate_content(
                model=native_model,
                contents=contents,
                config=gtypes.GenerateContentConfig(**config_kwargs),
            )
        except Exception as e:
            log.error("Vertex AI call failed for %s: %s", native_model, e)
            raise

        # Convert response back to OpenAI-style message dict
        msg_dict: Dict[str, Any] = {"role": "assistant", "content": None}
        tool_calls_out = []

        if not resp.candidates:
            log.warning("Gemini returned no candidates for %s", native_model)
            return msg_dict, {}

        candidate = resp.candidates[0]
        if not candidate.content or not candidate.content.parts:
            log.warning("Gemini candidate has no content/parts for %s", native_model)
            return msg_dict, {}

        # Stash the raw Gemini Content object so we can replay it verbatim
        # on the next round (preserves thought_signatures for thinking models)
        msg_dict["_gemini_content"] = candidate.content

        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_calls_out.append({
                    "id": f"call_{part.function_call.name}_{int(time.time()*1000)}",
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": json.dumps(dict(part.function_call.args) if part.function_call.args else {}),
                    }
                })
            elif hasattr(part, "text") and part.text:
                msg_dict["content"] = (msg_dict.get("content") or "") + part.text

        if tool_calls_out:
            msg_dict["tool_calls"] = tool_calls_out

        # Build usage dict
        usage_meta = resp.usage_metadata or {}
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) or 0
        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        # Estimate cost (Gemini 3.1 Pro: $2/1M input, $12/1M output)
        estimated_cost = (prompt_tokens / 1_000_000 * 2.0) + (completion_tokens / 1_000_000 * 12.0)
        if estimated_cost > 0:
            usage["cost"] = round(estimated_cost, 6)
            usage["cost_estimated"] = True

        return msg_dict, usage

    def _chat_google_studio(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Chat via Google AI Studio (API key, subject to daily limits)."""
        client = self._get_google_studio_client()
        native_model = _google_native_model_id(model)
        kwargs: Dict[str, Any] = {"model": native_model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])
        if not usage.get("cost"):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            estimated_cost = (prompt_tokens / 1_000_000 * 2.0) + (completion_tokens / 1_000_000 * 12.0)
            if estimated_cost > 0:
                usage["cost"] = round(estimated_cost, 6)
                usage["cost_estimated"] = True
        return msg, usage

    def _chat_openrouter(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Chat via OpenRouter."""
        client = self._get_client()
        extra_body: Dict[str, Any] = {"reasoning": {"effort": reasoning_effort, "exclude": True}}
        kwargs: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "extra_body": extra_body}
        if tools:
            tools_with_cache = [t for t in tools]
            if tools_with_cache:
                last_tool = {**tools_with_cache[-1]}
                last_tool["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
                tools_with_cache[-1] = last_tool
            kwargs["tools"] = tools_with_cache
            kwargs["tool_choice"] = tool_choice
        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}
        if not usage.get("cached_tokens"):
            prompt_details = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict) and prompt_details.get("cached_tokens"):
                usage["cached_tokens"] = int(prompt_details["cached_tokens"])
        if not usage.get("cache_write_tokens"):
            prompt_details_for_write = usage.get("prompt_tokens_details") or {}
            if isinstance(prompt_details_for_write, dict):
                cache_write = (prompt_details_for_write.get("cache_write_tokens")
                              or prompt_details_for_write.get("cache_creation_tokens")
                              or prompt_details_for_write.get("cache_creation_input_tokens"))
                if cache_write:
                    usage["cache_write_tokens"] = int(cache_write)
        if not usage.get("cost"):
            gen_id = resp_dict.get("id") or ""
            if gen_id:
                cost = self._fetch_generation_cost(gen_id)
                if cost is not None:
                    usage["cost"] = cost
        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "google/gemini-3.1-pro-preview",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({"type": "image_url", "image_url": {"url": img["url"]}})
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img['base64']}"}})
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))
        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(messages=messages, model=model, tools=None, reasoning_effort=reasoning_effort, max_tokens=max_tokens)
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        return os.environ.get("OUROBOROS_MODEL", "google/gemini-3.1-pro-preview")

    def available_models(self) -> List[str]:
        main = os.environ.get("OUROBOROS_MODEL", "google/gemini-3.1-pro-preview")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
