"""
Ouroboros — LLM client.

Routes to:
1. Google Vertex AI (google-genai SDK) if GOOGLE_APPLICATION_CREDENTIALS is set and model is Google
2. Google AI Studio (API key) if GOOGLE_API_KEY is set and model is Google
3. OpenRouter for everything else

Vertex uses $300 credits with no daily rate wall.
"""

from __future__ import annotations

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
    "google/gemini-2.5-pro-preview": "gemini-2.5-pro-preview",
    "google/gemini-2.5-flash":       "gemini-2.5-flash",
    "google/gemini-2.0-flash":       "gemini-2.0-flash",
}

# Vertex region — global works for preview models
_VERTEX_LOCATION = "global"
_VERTEX_PROJECT  = "gen-lang-client-0999726021"


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


class LLMClient:
    """LLM API wrapper. Routes Google models to Vertex/AI Studio, others to OpenRouter."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._client = None
        self._google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        self._vertex_creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        self._vertex_client = None

    def _use_vertex(self, model: str) -> bool:
        return bool(self._vertex_creds_path and os.path.exists(self._vertex_creds_path)) and _is_google_model(model)

    def _use_google_studio(self, model: str) -> bool:
        return bool(self._google_api_key) and _is_google_model(model) and not self._use_vertex(model)

    def _get_vertex_client(self):
        if self._vertex_client is None:
            import google.oauth2.service_account
            from google import genai
            credentials = google.oauth2.service_account.Credentials.from_service_account_file(
                self._vertex_creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._vertex_client = genai.Client(
                vertexai=True,
                project=_VERTEX_PROJECT,
                location=_VERTEX_LOCATION,
                credentials=credentials,
            )
        return self._vertex_client

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

        client = self._get_vertex_client()
        native_model = _google_native_model_id(model)
        log.debug(f"Vertex call: {model} → {native_model}")

        # Convert OpenAI-style messages to google-genai contents
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                # Handle tool calls in assistant messages
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    parts = []
                    if content:
                        parts.append(gtypes.Part(text=content))
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        import json as _json
                        args = _json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else (fn.get("arguments") or {})
                        parts.append(gtypes.Part(function_call=gtypes.FunctionCall(name=fn.get("name", ""), args=args)))
                    contents.append(gtypes.Content(role="model", parts=parts))
                else:
                    contents.append(gtypes.Content(role="model", parts=[gtypes.Part(text=content)]))
            elif role == "tool":
                import json as _json
                try:
                    result = _json.loads(content) if isinstance(content, str) else content
                except Exception:
                    result = {"result": content}
                tool_name = msg.get("name") or "tool"
                contents.append(gtypes.Content(role="user", parts=[
                    gtypes.Part(function_response=gtypes.FunctionResponse(name=tool_name, response=result))
                ]))
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

        # Build tool declarations
        tool_list = None
        if tools:
            import json as _json
            declarations = []
            for t in tools:
                fn = t.get("function", {})
                declarations.append(gtypes.FunctionDeclaration(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters"),
                ))
            tool_list = [gtypes.Tool(function_declarations=declarations)]

        config_kwargs: Dict[str, Any] = {"max_output_tokens": max_tokens}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tool_list:
            config_kwargs["tools"] = tool_list

        resp = client.models.generate_content(
            model=native_model,
            contents=contents,
            config=gtypes.GenerateContentConfig(**config_kwargs),
        )

        # Convert response back to OpenAI-style message dict
        msg_dict: Dict[str, Any] = {"role": "assistant", "content": None}
        tool_calls_out = []

        for part in (resp.candidates[0].content.parts if resp.candidates else []):
            if hasattr(part, "function_call") and part.function_call:
                import json as _json
                tool_calls_out.append({
                    "id": f"call_{part.function_call.name}_{int(time.time()*1000)}",
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": _json.dumps(dict(part.function_call.args)),
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
