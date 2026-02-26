"""Web search tool."""

from __future__ import annotations

import json
import urllib.parse
from typing import List

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.browser import _browse_page, _ensure_browser

def _web_search(ctx: ToolContext, query: str) -> str:
    try:
        url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        content = _browse_page(ctx, url, output="text", timeout=15000)
        
        # We also want to include the text returned
        cutoff = 10000
        truncated = content[:cutoff] + ("... [truncated]" if len(content) > cutoff else "")
        return json.dumps({"answer": truncated}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": "Search the web autonomously using Playwright via headless browser. Best for real-time lookups without incurring high API costs. Returns raw extracted text answers.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
            }, "required": ["query"]},
        }, _web_search),
    ]
