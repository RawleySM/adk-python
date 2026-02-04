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

"""Agent definitions for the ADK RLM system.

This module defines the base LLM agent (orchestrator) and sub-LLM agent
(for recursive context analysis) using ADK's agent framework.
"""

from __future__ import annotations

from typing import Any, Optional
import logging

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent

from .tools import (
    execute_code,
    approve_code_execution,
    reject_code_execution,
    submit_final_answer,
    submit_final_variable,
    get_repl_state,
)
from .repl_state import SecurityLevel

logger = logging.getLogger("google_adk." + __name__)


# System prompt for the base RLM agent
BASE_RLM_SYSTEM_PROMPT = """You are an intelligent assistant with access to a Python REPL environment for analyzing and processing large contexts. Your task is to answer queries by writing and executing Python code.

## REPL Environment

The REPL environment provides:
1. A `context` variable containing the data you need to analyze
2. An `llm_query(prompt)` function to query a sub-LLM for semantic analysis
3. Standard Python built-ins and common libraries (json, re, etc.)

## Key Functions

- `llm_query(prompt: str) -> str`: Query the sub-LLM with any prompt. The sub-LLM can handle ~500K characters, so don't hesitate to send large chunks of context.
- `FINAL_VAR(variable_name)`: Return a REPL variable as your final answer

## Strategy Guidelines

1. **First, explore the context**: Use code to understand the structure and size of your data
2. **Chunk strategically**: If the context is large, break it into manageable chunks for analysis
3. **Use sub-LLM for semantics**: The `llm_query()` function is powerful for understanding content meaning
4. **Build up your answer**: Use variables to accumulate findings before providing your final answer

## Example Workflow

```repl
# Step 1: Understand the context
print(f"Context type: {type(context)}")
print(f"Context length: {len(context)}")
print(f"First 500 chars: {context[:500]}")
```

```repl
# Step 2: Process in chunks
chunk_size = 100000
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
print(f"Number of chunks: {len(chunks)}")

# Step 3: Analyze each chunk
findings = []
for i, chunk in enumerate(chunks):
    result = llm_query(f"Analyze this chunk and find any important information: {chunk}")
    findings.append(result)
    print(f"Chunk {i}: {result[:100]}...")
```

```repl
# Step 4: Synthesize final answer
final_answer = llm_query(f"Based on these findings, answer the original query: {findings}")
print(final_answer)
```

## Important Notes

- Execute code step by step; don't try to do everything at once
- Check execution results before proceeding
- Use print() statements to see intermediate results
- When ready, use `submit_final_answer` or `submit_final_variable` to complete

Think carefully, plan your approach, and execute systematically."""


# System prompt for the sub-LLM agent
SUB_LLM_SYSTEM_PROMPT = """You are a sub-LLM assistant that helps analyze chunks of context data.

Your role is to:
1. Analyze the provided context chunk
2. Extract relevant information based on the query
3. Provide clear, concise summaries or findings

Be thorough but concise. Focus on answering exactly what is asked.
If you find specific information (like numbers, names, or facts), state them clearly.
If you don't find the requested information in the chunk, say so explicitly."""


def create_sub_llm_agent(
    name: str = "sub_llm",
    model: str = "gemini-2.0-flash",
    description: str = "Sub-LLM for analyzing context chunks",
) -> LlmAgent:
    """Create a sub-LLM agent for recursive context analysis.

    This agent is used by the REPL's llm_query() function to analyze
    chunks of context data.

    Args:
        name: Agent name
        model: Model to use
        description: Agent description

    Returns:
        LlmAgent configured for sub-LLM tasks
    """
    return LlmAgent(
        name=name,
        model=model,
        description=description,
        instruction=SUB_LLM_SYSTEM_PROMPT,
        # Sub-LLM doesn't need tools - it just analyzes and responds
        tools=[],
        # Prevent transfer back to parent
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


def create_security_reviewer_agent(
    name: str = "security_reviewer",
    model: str = "gemini-2.0-flash",
) -> LlmAgent:
    """Create a security reviewer agent for STRICT security mode.

    This agent reviews code before execution and can approve or reject it.

    Args:
        name: Agent name
        model: Model to use

    Returns:
        LlmAgent configured for security review
    """
    security_instruction = """You are a security reviewer for Python code execution.

Your job is to review code that will be executed in a sandboxed REPL environment.

## Review Criteria

APPROVE code that:
- Uses standard Python operations (loops, functions, data processing)
- Reads from the provided context variable
- Uses the llm_query() function for analysis
- Performs string/list/dict operations
- Uses safe imports (json, re, math, collections, etc.)

REJECT code that:
- Attempts to access the filesystem beyond the temp directory
- Uses subprocess, os.system, or shell commands
- Tries to access network resources
- Attempts to modify system state
- Uses eval() or exec() on user input
- Contains obfuscated or suspicious patterns

## Response Format

After reviewing, use either:
- `approve_code_execution` to approve the code
- `reject_code_execution` with a reason to reject

Be thorough but efficient. Most data analysis code should be approved."""

    return LlmAgent(
        name=name,
        model=model,
        description="Reviews code for security before execution",
        instruction=security_instruction,
        tools=[
            FunctionTool(approve_code_execution),
            FunctionTool(reject_code_execution),
        ],
        disallow_transfer_to_parent=False,
    )


def create_base_rlm_agent(
    name: str = "base_rlm",
    model: str = "gemini-2.5-flash",
    security_level: SecurityLevel = SecurityLevel.BASIC,
    sub_llm_model: str = "gemini-2.0-flash",
) -> LlmAgent:
    """Create the base RLM agent (main orchestrator).

    This is the primary agent that orchestrates the REPL interaction,
    executing code and building up an answer to the user's query.

    Args:
        name: Agent name
        model: Model to use for the base agent
        security_level: Security level for code execution
        sub_llm_model: Model to use for the sub-LLM

    Returns:
        LlmAgent configured as the base RLM orchestrator
    """
    # Base tools for code execution and final answer
    tools = [
        FunctionTool(execute_code),
        FunctionTool(submit_final_answer),
        FunctionTool(submit_final_variable),
        FunctionTool(get_repl_state),
    ]

    # Create sub-agents
    sub_agents = [
        create_sub_llm_agent(model=sub_llm_model),
    ]

    # Add security reviewer if using STRICT mode
    if security_level == SecurityLevel.STRICT:
        tools.extend([
            FunctionTool(approve_code_execution),
            FunctionTool(reject_code_execution),
        ])
        sub_agents.append(create_security_reviewer_agent())

    return LlmAgent(
        name=name,
        model=model,
        description="Base RLM agent for orchestrating REPL-based context analysis",
        instruction=BASE_RLM_SYSTEM_PROMPT,
        tools=tools,
        sub_agents=sub_agents,
        # Allow transfer to sub-agents but not to parent
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=False,
    )


def create_rlm_agent_hierarchy(
    base_model: str = "gemini-2.5-flash",
    sub_model: str = "gemini-2.0-flash",
    security_level: SecurityLevel = SecurityLevel.BASIC,
) -> LlmAgent:
    """Create the complete RLM agent hierarchy.

    This function creates a properly configured agent hierarchy with:
    - Base RLM agent (orchestrator)
    - Sub-LLM agent (for context analysis)
    - Optional security reviewer (for STRICT mode)

    Args:
        base_model: Model for the base agent
        sub_model: Model for sub-agents
        security_level: Security level for code execution

    Returns:
        Root LlmAgent of the hierarchy
    """
    return create_base_rlm_agent(
        model=base_model,
        security_level=security_level,
        sub_llm_model=sub_model,
    )


# Callback functions for agent lifecycle

async def before_base_agent_callback(callback_context):
    """Callback before base agent execution.

    Initializes REPL state if not already initialized.
    """
    state = callback_context.state
    from .repl_state import REPLStateManager, REPL_STATE_KEY

    if REPL_STATE_KEY not in state:
        logger.debug("Initializing REPL state in before_agent_callback")
        REPLStateManager.initialize_state(state)

    return None


async def after_base_agent_callback(callback_context):
    """Callback after base agent execution.

    Checks if max iterations exceeded and handles cleanup.
    """
    state = callback_context.state
    from .repl_state import REPLStateManager

    if REPLStateManager.has_exceeded_max_iterations(state):
        logger.warning("Max iterations exceeded, forcing final answer")
        # The agent should handle this in its next turn

    return None
