from dexagent.gap.tools import ToolResult


def run(ctx, request: dict) -> ToolResult:
    """Invoke a measured contact transition through the connector registry."""
    return ctx.tool("hand.transition_contact", request=request)
