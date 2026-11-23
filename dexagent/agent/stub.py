"""A scripted stand-in for the OpenRouter client, for exercising the loop without a model.

A script is a list of turns; each turn is a list of tool calls {"name": ..., "arguments": {...}}
(or a plain string for a reply with no tool call). The client plays them in order and then
calls `finish`.
"""
import json


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.model = "stub"

    def chat(self, messages, tools=None):
        turn = self.script[self.i] if self.i < len(self.script) else [{"name": "finish", "arguments": {"reason": "script ended"}}]
        self.i += 1
        if isinstance(turn, str):
            msg = {"role": "assistant", "content": turn}
        else:
            calls = [{"id": f"stub{self.i}_{j}", "type": "function",
                      "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}}
                     for j, c in enumerate(turn)]
            msg = {"role": "assistant", "content": None, "tool_calls": calls}
        return {"message": msg, "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "latency_s": 0.0, "finish_reason": "stop"}
