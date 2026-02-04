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

"""ADK-native RLM (Recursive Language Model) with REPL environment.

This module provides an ADK-native implementation of the RLM pattern with:
- Extended Finite State Machine (EFSM) for REPL state management
- Security states between code submission and execution
- Python code module-based REPL environment
- Stdout routing to appropriate models (base/sub LLM)
- Artifact storage for code submissions
- DebugLoggingPlugin integration

Usage:
    adk run contributing/samples/adk_rlm_minimal

    Or with custom options:
    adk run contributing/samples/adk_rlm_minimal --config '{"security_level": "strict"}'

Example queries:
    - "What is the magic number in the context?"
    - "Summarize the key points in the context"
    - "Find and extract all numbers from the context"
"""

from __future__ import annotations

import os
from typing import Any, Optional

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins import DebugLoggingPlugin
from google.adk.tools import FunctionTool
from google.adk.agents.callback_context import CallbackContext

from .agents import (
    create_rlm_agent_hierarchy,
    create_sub_llm_agent,
    before_base_agent_callback,
    after_base_agent_callback,
    BASE_RLM_SYSTEM_PROMPT,
)
from .tools import (
    execute_code,
    submit_final_answer,
    submit_final_variable,
    get_repl_state,
    approve_code_execution,
    reject_code_execution,
)
from .repl_state import (
    REPLStateManager,
    REPLState,
    SecurityLevel,
    REPL_STATE_KEY,
    REPL_CONTEXT_KEY,
    REPL_QUERY_KEY,
    REPL_ARTIFACT_ENABLED_KEY,
)


# Configuration from environment variables
DEFAULT_BASE_MODEL = os.environ.get("RLM_BASE_MODEL", "gemini-2.5-flash")
DEFAULT_SUB_MODEL = os.environ.get("RLM_SUB_MODEL", "gemini-2.0-flash")
DEFAULT_SECURITY_LEVEL = os.environ.get("RLM_SECURITY_LEVEL", "basic")
DEFAULT_MAX_ITERATIONS = int(os.environ.get("RLM_MAX_ITERATIONS", "20"))
DEFAULT_ARTIFACT_ENABLED = os.environ.get("RLM_ARTIFACT_ENABLED", "true").lower() == "true"
DEFAULT_DEBUG_LOGGING = os.environ.get("RLM_DEBUG_LOGGING", "true").lower() == "true"


def _get_security_level(level_str: str) -> SecurityLevel:
    """Convert string to SecurityLevel enum.

    Args:
        level_str: Security level string ("none", "basic", "strict")

    Returns:
        SecurityLevel enum value
    """
    level_map = {
        "none": SecurityLevel.NONE,
        "basic": SecurityLevel.BASIC,
        "strict": SecurityLevel.STRICT,
    }
    return level_map.get(level_str.lower(), SecurityLevel.BASIC)


# Create the sub-LLM agent for llm_query() functionality
sub_llm_agent = create_sub_llm_agent(
    name="sub_llm",
    model=DEFAULT_SUB_MODEL,
    description="Sub-LLM for analyzing context chunks. Use this when you need semantic analysis of large text chunks.",
)


# Dynamic instruction that includes context and query from state
def dynamic_instruction(callback_context: CallbackContext) -> str:
    """Generate dynamic instruction based on current state.

    Args:
        callback_context: Callback context with state access

    Returns:
        Instruction string with context-specific guidance
    """
    state = callback_context.state
    query = state.get(REPL_QUERY_KEY, "")
    iteration = state.get("repl_iteration", 0)
    max_iterations = state.get("repl_max_iterations", DEFAULT_MAX_ITERATIONS)

    base_instruction = BASE_RLM_SYSTEM_PROMPT

    # Add iteration-specific guidance
    if iteration == 0:
        guidance = """

## Current Status
This is your first interaction. You have NOT yet seen the context.
Your first action should be to explore the context variable to understand what you're working with.

DO NOT provide a final answer yet - you must first analyze the context!
"""
    elif iteration >= max_iterations - 2:
        guidance = f"""

## Current Status
Iteration {iteration + 1} of {max_iterations}. You are running low on iterations!
Please synthesize your findings and provide a final answer soon.
"""
    else:
        guidance = f"""

## Current Status
Iteration {iteration + 1} of {max_iterations}.
Continue your analysis and work toward answering the query.
"""

    if query:
        guidance += f"\n## Original Query\n{query}\n"

    return base_instruction + guidance


# Create the root agent
root_agent = LlmAgent(
    name="rlm_repl_agent",
    model=DEFAULT_BASE_MODEL,
    description="ADK-native RLM agent with REPL environment for recursive context analysis",
    instruction=dynamic_instruction,
    tools=[
        FunctionTool(execute_code),
        FunctionTool(submit_final_answer),
        FunctionTool(submit_final_variable),
        FunctionTool(get_repl_state),
    ],
    sub_agents=[sub_llm_agent],
    before_agent_callback=before_base_agent_callback,
    after_agent_callback=after_base_agent_callback,
    # Output key to store the final response
    output_key="rlm_response",
)


# Create plugins list
plugins = []

# Add DebugLoggingPlugin if enabled
if DEFAULT_DEBUG_LOGGING:
    plugins.append(
        DebugLoggingPlugin(
            output_path="adk_rlm_debug.yaml",
            include_session_state=True,
            include_system_instruction=True,
        )
    )


# Create the App
app = App(
    name="adk_rlm_minimal",
    root_agent=root_agent,
    plugins=plugins,
)


# Utility functions for programmatic use

def initialize_rlm_session(
    state: dict[str, Any],
    context: str,
    query: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    security_level: SecurityLevel = SecurityLevel.BASIC,
    artifact_enabled: bool = DEFAULT_ARTIFACT_ENABLED,
) -> dict[str, Any]:
    """Initialize a new RLM session with context and query.

    This function should be called to set up the state before running
    the RLM agent.

    Args:
        state: ADK state dictionary to initialize
        context: Context data for analysis
        query: User query to answer
        max_iterations: Maximum iterations allowed
        security_level: Security level for code execution
        artifact_enabled: Whether to save code to artifacts

    Returns:
        Initialized state dictionary
    """
    REPLStateManager.initialize_state(
        state,
        context=context,
        query=query,
        max_iterations=max_iterations,
        security_level=security_level,
        artifact_enabled=artifact_enabled,
    )
    return state


def get_final_answer(state: dict[str, Any]) -> Optional[str]:
    """Get the final answer from session state.

    Args:
        state: ADK state dictionary

    Returns:
        Final answer if available, None otherwise
    """
    return REPLStateManager.get_final_answer(state)


def is_session_complete(state: dict[str, Any]) -> bool:
    """Check if the RLM session is complete.

    Args:
        state: ADK state dictionary

    Returns:
        True if session is complete
    """
    return REPLStateManager.is_complete(state)


# Export for ADK CLI discovery
__all__ = [
    "app",
    "root_agent",
    "initialize_rlm_session",
    "get_final_answer",
    "is_session_complete",
]
