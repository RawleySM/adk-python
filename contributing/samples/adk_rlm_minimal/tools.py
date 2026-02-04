# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for the ADK RLM REPL environment.

This module provides tools for:
- Code execution in the REPL
- Sub-LLM queries
- Security approval for code execution
- Final answer submission
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional

from google.genai import types
from google.adk.tools.tool_context import ToolContext

from .repl_state import (
    REPLStateManager,
    REPLState,
    SecurityLevel,
    REPL_STATE_KEY,
    REPL_LOCALS_KEY,
    REPL_CONTEXT_KEY,
    REPL_LAST_OUTPUT_KEY,
    REPL_LAST_ERROR_KEY,
    REPL_PENDING_CODE_KEY,
    REPL_SECURITY_LEVEL_KEY,
    REPL_ARTIFACT_ENABLED_KEY,
    REPL_CODE_HISTORY_KEY,
    REPL_FINAL_ANSWER_KEY,
    REPL_QUERY_KEY,
)
from .repl_env import ADKREPLEnvironment, REPLResult, SecurityLevel as EnvSecurityLevel

logger = logging.getLogger("google_adk." + __name__)

# Global REPL environment (initialized per session via state)
_repl_environments: dict[str, ADKREPLEnvironment] = {}


def _get_or_create_repl(
    session_id: str,
    state: dict[str, Any],
    sub_lm_callback: Optional[callable] = None,
) -> ADKREPLEnvironment:
    """Get or create a REPL environment for a session.

    Args:
        session_id: Session identifier
        state: State dictionary
        sub_lm_callback: Callback for sub-LLM queries

    Returns:
        ADKREPLEnvironment instance
    """
    if session_id not in _repl_environments:
        security_str = state.get(REPL_SECURITY_LEVEL_KEY, SecurityLevel.BASIC.value)
        security_level = EnvSecurityLevel(security_str)

        repl = ADKREPLEnvironment(security_level=security_level)

        # Initialize with context if available
        context = state.get(REPL_CONTEXT_KEY)
        locals_dict = state.get(REPL_LOCALS_KEY, {})

        repl.initialize(
            context=context,
            locals_dict=locals_dict,
            sub_lm_callback=sub_lm_callback,
        )

        _repl_environments[session_id] = repl

    return _repl_environments[session_id]


def _cleanup_repl(session_id: str) -> None:
    """Clean up REPL environment for a session.

    Args:
        session_id: Session identifier
    """
    if session_id in _repl_environments:
        _repl_environments[session_id].cleanup()
        del _repl_environments[session_id]


async def execute_code(
    code: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Execute Python code in the REPL environment.

    This tool executes Python code in a sandboxed REPL environment.
    The code can use:
    - `context`: The loaded context data
    - `llm_query(prompt)`: Query the sub-LLM
    - `FINAL_VAR(var_name)`: Return a variable as the final answer
    - Standard Python built-ins and imports

    The REPL maintains state between executions, so variables defined
    in one execution are available in subsequent executions.

    Args:
        code: Python code to execute. Wrap code in ```repl``` blocks.
        tool_context: ADK tool context

    Returns:
        Dictionary with execution results including stdout, stderr, and locals
    """
    state = tool_context.state
    session_id = tool_context._invocation_context.session.id

    # Check current state
    current_state = REPLStateManager.get_state_from_dict(state)

    # Transition to CODE_PENDING_REVIEW
    if current_state == REPLState.IDLE:
        REPLStateManager.transition_to(state, REPLState.CODE_PENDING_REVIEW)

    # Get security level
    security_level_str = state.get(REPL_SECURITY_LEVEL_KEY, SecurityLevel.BASIC.value)
    security_level = SecurityLevel(security_level_str)

    # Store pending code for review
    REPLStateManager.set_pending_code(state, code)

    # For NONE or BASIC security, auto-approve
    if security_level in (SecurityLevel.NONE, SecurityLevel.BASIC):
        REPLStateManager.transition_to(state, REPLState.CODE_APPROVED)
    else:
        # For STRICT security, code stays pending until explicitly approved
        return {
            "status": "pending_approval",
            "message": "Code requires security approval before execution",
            "code": code,
        }

    # Get approved code
    approved_code = REPLStateManager.get_pending_code(state)
    if not approved_code:
        return {
            "status": "error",
            "message": "No code pending for execution",
        }

    # Transition to EXECUTING
    REPLStateManager.transition_to(state, REPLState.EXECUTING)

    # Get or create REPL environment
    repl = _get_or_create_repl(session_id, state)

    # Execute code
    result: REPLResult = repl.execute(approved_code)

    # Update state with results
    state[REPL_LAST_OUTPUT_KEY] = result.stdout
    state[REPL_LAST_ERROR_KEY] = result.stderr
    state[REPL_LOCALS_KEY] = result.locals_snapshot

    # Add to code history
    REPLStateManager.add_code_to_history(
        state,
        approved_code,
        result.stdout,
        result.stderr,
    )

    # Save to artifact if enabled
    if state.get(REPL_ARTIFACT_ENABLED_KEY, False):
        await _save_code_artifact(tool_context, approved_code, result)

    # Clear pending code and transition back to IDLE
    REPLStateManager.clear_pending_code(state)
    REPLStateManager.transition_to(state, REPLState.IDLE)

    # Increment iteration
    REPLStateManager.increment_iteration(state)

    # Format response
    response = {
        "status": "success" if result.success else "error",
        "stdout": _truncate_output(result.stdout),
        "stderr": result.stderr if result.stderr else None,
        "execution_time": f"{result.execution_time:.3f}s",
        "variables": list(result.locals_snapshot.keys()),
    }

    if not result.success:
        response["error"] = result.stderr

    return response


async def _save_code_artifact(
    tool_context: ToolContext,
    code: str,
    result: REPLResult,
) -> None:
    """Save executed code to an artifact.

    Args:
        tool_context: ADK tool context
        code: Code that was executed
        result: Execution result
    """
    try:
        artifact_service = tool_context._invocation_context.artifact_service
        if artifact_service is None:
            return

        # Get iteration number for filename
        iteration = tool_context.state.get("repl_iteration", 0)
        filename = f"repl_code_{iteration:04d}.py"

        # Create artifact content
        content = f"""# REPL Execution - Iteration {iteration}
# Execution time: {result.execution_time:.3f}s
# Success: {result.success}

{code}

# --- Output ---
# stdout:
{chr(10).join('# ' + line for line in result.stdout.split(chr(10)) if line)}

# stderr:
{chr(10).join('# ' + line for line in result.stderr.split(chr(10)) if line) if result.stderr else '# (none)'}
"""

        # Save artifact
        artifact = types.Part.from_text(text=content)
        await artifact_service.save_artifact(
            app_name=tool_context._invocation_context.app_name,
            user_id=tool_context._invocation_context.user_id,
            session_id=tool_context._invocation_context.session.id,
            filename=filename,
            artifact=artifact,
        )

        logger.debug(f"Saved code artifact: {filename}")

    except Exception as e:
        logger.warning(f"Failed to save code artifact: {e}")


def _truncate_output(output: str, max_chars: int = 100000) -> str:
    """Truncate output to a maximum length.

    Args:
        output: Output string
        max_chars: Maximum characters

    Returns:
        Truncated output
    """
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + f"\n... [truncated, {len(output) - max_chars} chars omitted]"


async def approve_code_execution(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Approve pending code for execution (for STRICT security mode).

    This tool is used when security_level is STRICT and code requires
    explicit approval before execution.

    Args:
        tool_context: ADK tool context

    Returns:
        Dictionary with approval status
    """
    state = tool_context.state

    # Check if there's pending code
    pending_code = REPLStateManager.get_pending_code(state)
    if not pending_code:
        return {
            "status": "error",
            "message": "No code pending for approval",
        }

    # Check current state
    current_state = REPLStateManager.get_state_from_dict(state)
    if current_state != REPLState.CODE_PENDING_REVIEW:
        return {
            "status": "error",
            "message": f"Invalid state for approval: {current_state.value}",
        }

    # Approve the code
    REPLStateManager.transition_to(state, REPLState.CODE_APPROVED)

    return {
        "status": "approved",
        "message": "Code approved for execution",
        "code": pending_code,
    }


async def reject_code_execution(
    reason: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Reject pending code (for STRICT security mode).

    This tool is used when security_level is STRICT and code should not
    be executed due to security concerns.

    Args:
        reason: Reason for rejection
        tool_context: ADK tool context

    Returns:
        Dictionary with rejection status
    """
    state = tool_context.state

    # Check if there's pending code
    pending_code = REPLStateManager.get_pending_code(state)
    if not pending_code:
        return {
            "status": "error",
            "message": "No code pending for rejection",
        }

    # Reject and transition
    REPLStateManager.clear_pending_code(state)
    REPLStateManager.transition_to(state, REPLState.CODE_REJECTED)

    # Transition back to IDLE for next attempt
    REPLStateManager.transition_to(state, REPLState.IDLE)

    return {
        "status": "rejected",
        "reason": reason,
        "code": pending_code,
    }


async def submit_final_answer(
    answer: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Submit the final answer for the query.

    Use this tool when you have determined the answer to the user's query.
    This will complete the REPL session.

    Args:
        answer: The final answer to the user's query
        tool_context: ADK tool context

    Returns:
        Dictionary confirming the final answer
    """
    state = tool_context.state

    # Set final answer and transition to COMPLETE
    REPLStateManager.set_final_answer(state, answer)

    # Clean up REPL environment
    session_id = tool_context._invocation_context.session.id
    _cleanup_repl(session_id)

    return {
        "status": "complete",
        "final_answer": answer,
    }


async def submit_final_variable(
    variable_name: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Submit a REPL variable as the final answer.

    Use this tool to return a variable from the REPL environment
    as your final answer.

    Args:
        variable_name: Name of the variable to return as the final answer
        tool_context: ADK tool context

    Returns:
        Dictionary with the variable value as the final answer
    """
    state = tool_context.state
    session_id = tool_context._invocation_context.session.id

    # Get REPL environment
    if session_id not in _repl_environments:
        return {
            "status": "error",
            "message": "REPL environment not initialized",
        }

    repl = _repl_environments[session_id]
    locals_dict = repl.get_locals()

    # Get variable value
    variable_name = variable_name.strip().strip('"').strip("'")
    if variable_name not in locals_dict:
        return {
            "status": "error",
            "message": f"Variable '{variable_name}' not found in REPL",
            "available_variables": list(locals_dict.keys()),
        }

    value = locals_dict[variable_name]
    answer = str(value)

    # Set final answer
    REPLStateManager.set_final_answer(state, answer)

    # Clean up
    _cleanup_repl(session_id)

    return {
        "status": "complete",
        "variable_name": variable_name,
        "final_answer": answer,
    }


async def get_repl_state(
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Get the current state of the REPL environment.

    This tool returns information about the current REPL state including
    available variables, iteration count, and execution history.

    Args:
        tool_context: ADK tool context

    Returns:
        Dictionary with REPL state information
    """
    state = tool_context.state

    current_state = REPLStateManager.get_state_from_dict(state)
    locals_dict = state.get(REPL_LOCALS_KEY, {})
    iteration = state.get("repl_iteration", 0)
    max_iterations = state.get("repl_max_iterations", 20)
    code_history = state.get(REPL_CODE_HISTORY_KEY, [])

    return {
        "state": current_state.value,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "variables": list(locals_dict.keys()),
        "history_count": len(code_history),
        "last_output": _truncate_output(state.get(REPL_LAST_OUTPUT_KEY, ""), 500),
        "last_error": state.get(REPL_LAST_ERROR_KEY, ""),
    }


def find_code_blocks(text: str) -> list[str]:
    """Extract code blocks from text.

    Finds code blocks wrapped in ```repl``` or ```python``` markers.

    Args:
        text: Text to search for code blocks

    Returns:
        List of code strings found
    """
    # Match ```repl or ```python blocks
    pattern = r'```(?:repl|python)\s*\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches if matches else []


def check_for_final_answer(text: str) -> Optional[str]:
    """Check if text contains a final answer.

    Looks for FINAL(answer) or FINAL_VAR(variable_name) patterns.

    Args:
        text: Text to check

    Returns:
        Final answer if found, None otherwise
    """
    # Check for FINAL(answer)
    final_pattern = r'FINAL\((.*?)\)'
    match = re.search(final_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Check for FINAL_VAR(variable_name) - handled by submit_final_variable tool
    return None
