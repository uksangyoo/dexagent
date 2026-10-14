from dexagent.gap.tools import ToolResult


def run(ctx, action: dict) -> ToolResult:
    """Invoke the connector's typed, guarded object-motion tool."""
    return ctx.tool("hand.move_object", action=action)
