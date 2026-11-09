"""Expose signal quality for the bundled recordings over MCP stdio."""
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .analysis import signal_quality
from .data import load_recording

server = FastMCP("wearable-quality")


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
def inspect_signal_quality(recording: Literal["offset", "rising", "steady"]) -> dict:
    """Report coverage and missing measurements for a bundled recording."""
    return {"recording": recording, **signal_quality(load_recording(recording))}


if __name__ == "__main__":
    server.run(transport="stdio")
