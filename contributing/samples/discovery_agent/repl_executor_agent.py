"""REPL Executor Agent - A deterministic custom BaseAgent for code execution.

Implements an Extended Finite State Machine (EFSM) that:
  1. Reads submitted code from session state (the 'repl_code_submission' key)
  2. Runs a security check against a block-list of dangerous operations
  3. Executes the code in a sandboxed Python exec environment
  4. Writes execution results (stdout, stderr) back into session state
  5. Emits an Event so downstream agents can consume the output

This agent is deterministic: it does NOT invoke an LLM. It inherits
directly from BaseAgent and overrides _run_async_impl.
"""

from __future__ import annotations

import io
import logging
import sys
import time
import traceback
from typing import Any, AsyncGenerator, Optional

from google.genai import types
from pydantic import Field

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

logger = logging.getLogger("google_adk." + __name__)

# ── Security block-list ──────────────────────────────────────────────────────
BLOCKED_PATTERNS: list[str] = [
    "os.system(",
    "subprocess.",
    "shutil.rmtree(",
    "__import__('os')",
    "__import__('subprocess')",
    "eval(",
    "exec(",
    "compile(",
    "open('/etc",
    "open('/proc",
    "open('/sys",
    "importlib.import_module",
    "ctypes.",
    "socket.",
]


def _security_check(code: str) -> Optional[str]:
    """Return an error message if the code contains blocked patterns."""
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return f"Security violation: code contains blocked pattern '{pattern}'"
    return None


# ── Sandboxed builtins (mirrors rlm-minimal's REPLEnv) ──────────────────────
_SAFE_BUILTINS: dict[str, Any] = {
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "range": range,
    "reversed": reversed,
    "any": any,
    "all": all,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "dir": dir,
    "repr": repr,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "format": format,
    "iter": iter,
    "next": next,
    "slice": slice,
    "hash": hash,
    "id": id,
    "callable": callable,
    "issubclass": issubclass,
    "__import__": __import__,
    # Exceptions
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "FileNotFoundError": FileNotFoundError,
    "ImportError": ImportError,
    "NameError": NameError,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


class REPLExecutorAgent(BaseAgent):
    """Deterministic EFSM agent that executes Python code from session state.

    Protocol (via session state key-value pairs):
        Input:
            state['repl_code_submission'] - code string to execute
        Output:
            state['repl_stdout']          - captured stdout
            state['repl_stderr']          - captured stderr
            state['repl_exec_time']       - execution wall time in seconds
            state['repl_status']          - 'success' | 'error' | 'blocked'

    The agent maintains a persistent namespace across executions within a
    session via state['repl_namespace'] (serialized as JSON-safe dict).
    """

    # Persistent REPL namespace lives across calls within a session.
    # It is NOT stored in pydantic fields because it contains arbitrary objects.
    _namespaces: dict[str, dict[str, Any]] = {}

    model_config = BaseAgent.model_config

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Execute submitted code, write results back to state."""
        session_id = ctx.session.id
        state = ctx.session.state

        code: str = state.get("repl_code_submission", "")
        if not code.strip():
            # Nothing to execute - emit a no-op event
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="[REPL] No code submitted.")],
                ),
            )
            return

        # ── EFSM State 1: Security Check ────────────────────────────────────
        violation = _security_check(code)
        if violation:
            actions = EventActions(
                state_delta={
                    "repl_stdout": "",
                    "repl_stderr": violation,
                    "repl_exec_time": 0.0,
                    "repl_status": "blocked",
                },
            )
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"[REPL] BLOCKED: {violation}")],
                ),
                actions=actions,
            )
            return

        # ── EFSM State 2: Code Execution ────────────────────────────────────
        # Retrieve or create the persistent namespace for this session
        if session_id not in REPLExecutorAgent._namespaces:
            REPLExecutorAgent._namespaces[session_id] = {
                "__builtins__": _SAFE_BUILTINS,
            }
        namespace = REPLExecutorAgent._namespaces[session_id]

        # Inject any downloaded GitHub files from state into the namespace
        for key, value in state.items():
            if isinstance(key, str) and key.startswith("gh_file:") and isinstance(value, str):
                var_name = key.replace("gh_file:", "").replace("/", "_").replace(".", "_").replace("-", "_")
                namespace[var_name] = value

        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        start_time = time.time()
        try:
            sys.stdout = stdout_buf
            sys.stderr = stderr_buf
            exec(code, namespace, namespace)  # noqa: S102
            status = "success"
        except Exception:
            traceback.print_exc(file=stderr_buf)
            status = "error"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        exec_time = time.time() - start_time
        stdout_str = stdout_buf.getvalue()
        stderr_str = stderr_buf.getvalue()

        # Truncate oversized output to avoid blowing up context
        max_output = 50_000
        if len(stdout_str) > max_output:
            stdout_str = stdout_str[:max_output] + "\n... [truncated]"
        if len(stderr_str) > max_output:
            stderr_str = stderr_str[:max_output] + "\n... [truncated]"

        # ── EFSM State 3: Write Results ─────────────────────────────────────
        state_delta = {
            "repl_stdout": stdout_str,
            "repl_stderr": stderr_str,
            "repl_exec_time": exec_time,
            "repl_status": status,
            # Clear the submission so it doesn't re-execute
            "repl_code_submission": "",
        }

        output_text = f"[REPL] status={status} time={exec_time:.2f}s"
        if stdout_str:
            output_text += f"\n--- stdout ---\n{stdout_str}"
        if stderr_str:
            output_text += f"\n--- stderr ---\n{stderr_str}"

        actions = EventActions(state_delta=state_delta)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part(text=output_text)],
            ),
            actions=actions,
        )
