"""A minimal OpenRouter chat client with image and tool-call support. Standard library only.

    client = OpenRouter("anthropic/claude-sonnet-4.5")      # key from OPENROUTER_API_KEY
    reply = client.chat(messages, tools)  # -> {"message": {...}, "usage": {...}, "latency_s": float}

`messages` use the OpenAI chat format; an image is a content part
{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}.
Requests and replies are appended to `log_path` (images elided) when given.
"""
import json
import os
import time
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    pass


def _elide(messages):
    """Messages with image payloads replaced by a short marker, for the log."""
    out = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = [{"type": "text", "text": "[image]"} if p.get("type") == "image_url" else p for p in c]
            m = dict(m, content=c)
        out.append(m)
    return out


class OpenRouter:
    def __init__(self, model, api_key=None, temperature=0.2, max_tokens=8192, retries=5, timeout=180.0,
                 log_path=None, referer="https://github.com/dexagent", title="dexagent"):
        self.model = model
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set")
        self.temperature, self.max_tokens, self.retries, self.timeout = temperature, max_tokens, retries, timeout
        self.log_path = log_path
        self.headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                        "HTTP-Referer": referer, "X-Title": title}

    def _log(self, record):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def chat(self, messages, tools=None):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
            # Require a tool call every turn (finish is a tool). Empty "thinking" replies otherwise
            # burn the agent-call budget on models that reason in text without calling anything.
            body["tool_choice"] = "required"
        data = json.dumps(body).encode()
        last = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                req = urllib.request.Request(URL, data=data, headers=self.headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read())
                if "error" in payload and "choices" not in payload:
                    raise OpenRouterError(f"api error: {json.dumps(payload['error'])[:500]}")
                choice = payload["choices"][0]
                reply = {"message": choice["message"], "usage": payload.get("usage") or {},
                         "latency_s": round(time.time() - t0, 2), "finish_reason": choice.get("finish_reason")}
                self._log({"t": time.strftime("%H:%M:%S"), "attempt": attempt, "request": _elide(messages)[-3:],
                           "reply": reply})
                return reply
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")[:500]
                last = f"HTTP {e.code}: {text}"
                if e.code not in RETRY_STATUS:
                    raise OpenRouterError(last) from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError, IndexError) as e:
                last = f"{type(e).__name__}: {e}"
            except OpenRouterError as e:
                last = str(e)
            self._log({"t": time.strftime("%H:%M:%S"), "attempt": attempt, "error": last})
            time.sleep(min(30.0, 2.0 ** attempt))
        raise OpenRouterError(f"OpenRouter failed after {self.retries + 1} attempts: {last}")
