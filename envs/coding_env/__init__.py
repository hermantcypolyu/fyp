"""The coding environment: agent tools, sandbox, verifier.

`app.py` (the NeMo Gym adapter) is deliberately the only module here that will import
`nemo_gym`; everything else runs, and is tested, without it. See
docs/decisions/0002-nemo-gym-is-an-optional-extra.md.
"""

from envs.coding_env.limits import ResourceLimits
from envs.coding_env.sandbox import Sandbox, open_sandbox
from envs.coding_env.tools import TOOL_DEFINITIONS, Toolbox, ToolCall, ToolResult
from envs.coding_env.verifier import VerifyResult, verify, verify_sandbox

__all__ = [
    "TOOL_DEFINITIONS",
    "ResourceLimits",
    "Sandbox",
    "ToolCall",
    "ToolResult",
    "Toolbox",
    "VerifyResult",
    "open_sandbox",
    "verify",
    "verify_sandbox",
]
