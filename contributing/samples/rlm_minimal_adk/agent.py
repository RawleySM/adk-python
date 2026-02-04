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

"""ADK-native minimal Recursive Language Model (RLM) sample."""

from __future__ import annotations

import code
import contextlib
import io
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins import DebugLoggingPlugin
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
from google.genai import types

BASE_AGENT_NAME = "rlm_base_agent"
SUB_AGENT_NAME = "rlm_sub_agent"

REPL_STATE_KEY = "rlm_repl_state"
JIRA_TICKET_KEY = "Jira_ticket"
FSM_IDLE = "idle"
FSM_SECURITY_REVIEW = "security_review"
FSM_APPROVED = "approved"

RLM_STATIC_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {chunk}")
print(answer)
```

As an example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {header} section: {info}")
    buffers.append(f"{header}: {summary}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {query}\n\nSummaries:\n" + "\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""

_INTERPRETERS: dict[str, code.InteractiveInterpreter] = {}
_LOCALS_BY_SESSION: dict[str, dict[str, Any]] = {}


def _ensure_repl_state(tool_context: ToolContext) -> dict[str, Any]:
  repl_state = tool_context.state.get(REPL_STATE_KEY)
  if not isinstance(repl_state, dict):
    repl_state = {}
  repl_state.setdefault("fsm_state", FSM_IDLE)
  repl_state.setdefault("pending_code", "")
  repl_state.setdefault("stdout_by_agent", {})
  repl_state.setdefault("stdout_target_by_agent", {})
  repl_state.setdefault("submission_count", 0)
  repl_state.setdefault("save_code_as_artifact", False)
  repl_state.setdefault("sub_lm_requests", [])
  tool_context.state[REPL_STATE_KEY] = repl_state
  return repl_state


def _get_interpreter(session_id: str) -> code.InteractiveInterpreter:
  if session_id not in _INTERPRETERS:
    _LOCALS_BY_SESSION[session_id] = {}
    _INTERPRETERS[session_id] = code.InteractiveInterpreter(
        locals=_LOCALS_BY_SESSION[session_id]
    )
  return _INTERPRETERS[session_id]


def _queue_sub_lm_request(
    repl_state: dict[str, Any],
    *,
    prompt: str,
    agent_name: str,
) -> str:
  repl_state["sub_lm_requests"] = repl_state["sub_lm_requests"] + [
      {"from_agent": agent_name, "prompt": prompt}
  ]
  return "Sub-LM request queued."


def _record_stdout(
    repl_state: dict[str, Any],
    *,
    target_agent: str,
    stdout_text: str,
) -> None:
  repl_state["stdout_by_agent"][target_agent] = stdout_text


async def submit_code(
    tool_context: ToolContext, code_text: str
) -> dict[str, Any]:
  """Submits code into the REPL and transitions into security review."""
  repl_state = _ensure_repl_state(tool_context)
  repl_state["pending_code"] = code_text
  repl_state["fsm_state"] = FSM_SECURITY_REVIEW
  repl_state["submission_count"] = repl_state["submission_count"] + 1
  if repl_state["save_code_as_artifact"]:
    artifact_name = (
        f"rlm_repl_submission_{repl_state['submission_count']:04d}.py"
    )
    artifact_part = types.Part(
        inline_data=types.Blob(
            mime_type="text/x-python", data=code_text.encode("utf-8")
        )
    )
    await tool_context.save_artifact(
        artifact_name,
        artifact_part,
        custom_metadata={"submitted_by": tool_context.agent_name},
    )
  return {
      "status": "submitted",
      "fsm_state": repl_state["fsm_state"],
  }


def approve_code_execution(tool_context: ToolContext) -> dict[str, Any]:
  """Approves the pending code and moves to the execution-ready state."""
  repl_state = _ensure_repl_state(tool_context)
  if repl_state["fsm_state"] != FSM_SECURITY_REVIEW:
    return {
        "status": "ignored",
        "message": "No code pending security review.",
        "fsm_state": repl_state["fsm_state"],
    }
  repl_state["fsm_state"] = FSM_APPROVED
  return {
      "status": "approved",
      "fsm_state": repl_state["fsm_state"],
  }


def execute_pending_code(tool_context: ToolContext) -> dict[str, Any]:
  """Executes approved code inside a stateful Python REPL."""
  repl_state = _ensure_repl_state(tool_context)
  if repl_state["fsm_state"] != FSM_APPROVED:
    return {
        "status": "blocked",
        "message": "Code must be approved before execution.",
        "fsm_state": repl_state["fsm_state"],
    }
  pending_code = repl_state.get("pending_code", "")
  if not pending_code:
    repl_state["fsm_state"] = FSM_IDLE
    return {
        "status": "empty",
        "message": "No code available to execute.",
        "fsm_state": repl_state["fsm_state"],
    }
  interpreter = _get_interpreter(tool_context.session.id)
  locals_env = _LOCALS_BY_SESSION[tool_context.session.id]
  locals_env["sub_lm"] = lambda prompt: _queue_sub_lm_request(
      repl_state,
      prompt=prompt,
      agent_name=tool_context.agent_name,
  )
  stdout_buffer = io.StringIO()
  with contextlib.redirect_stdout(stdout_buffer):
    incomplete = interpreter.runsource(pending_code, symbol="exec")
  stdout_text = stdout_buffer.getvalue()
  stdout_targets = repl_state["stdout_target_by_agent"]
  target_agent = stdout_targets.get(
      tool_context.agent_name, tool_context.agent_name
  )
  _record_stdout(repl_state, target_agent=target_agent, stdout_text=stdout_text)
  repl_state["pending_code"] = ""
  repl_state["fsm_state"] = FSM_IDLE
  return {
      "status": "incomplete" if incomplete else "executed",
      "stdout": stdout_text,
      "stdout_target": target_agent,
      "fsm_state": repl_state["fsm_state"],
  }


def set_stdout_target(
    tool_context: ToolContext, target_agent: str
) -> dict[str, Any]:
  """Routes print() output to the specified agent channel."""
  repl_state = _ensure_repl_state(tool_context)
  repl_state["stdout_target_by_agent"][tool_context.agent_name] = target_agent
  return {
      "status": "ok",
      "stdout_target": target_agent,
  }


def request_sub_lm(tool_context: ToolContext, prompt: str) -> dict[str, Any]:
  """Queues a sub-LM request and transfers control to the sub agent."""
  repl_state = _ensure_repl_state(tool_context)
  repl_state["sub_lm_requests"] = repl_state["sub_lm_requests"] + [
      {"from_agent": tool_context.agent_name, "prompt": prompt}
  ]
  tool_context.actions.transfer_to_agent = SUB_AGENT_NAME
  return {"status": "transferring", "to": SUB_AGENT_NAME}


def read_sub_lm_request(tool_context: ToolContext) -> dict[str, Any]:
  """Retrieves the latest queued sub-LM request."""
  repl_state = _ensure_repl_state(tool_context)
  if not repl_state["sub_lm_requests"]:
    return {"status": "empty"}
  request = repl_state["sub_lm_requests"][-1]
  return {"status": "ok", "request": request}


def set_save_code_as_artifact(
    tool_context: ToolContext, enabled: bool
) -> dict[str, Any]:
  """Turns artifact logging on or off for submitted code."""
  repl_state = _ensure_repl_state(tool_context)
  repl_state["save_code_as_artifact"] = enabled
  return {"status": "ok", "save_code_as_artifact": enabled}


def reset_repl(tool_context: ToolContext) -> dict[str, Any]:
  """Clears the REPL interpreter and stateful locals."""
  repl_state = _ensure_repl_state(tool_context)
  session_id = tool_context.session.id
  _INTERPRETERS.pop(session_id, None)
  _LOCALS_BY_SESSION.pop(session_id, None)
  repl_state["pending_code"] = ""
  repl_state["fsm_state"] = FSM_IDLE
  return {"status": "reset"}


def set_jira_ticket(
    tool_context: ToolContext, jira_ticket: str
) -> dict[str, Any]:
  """Stores a Jira ticket identifier in session state."""
  tool_context.state[JIRA_TICKET_KEY] = jira_ticket
  return {"status": "ok", "jira_ticket": jira_ticket}


base_agent = LlmAgent(
    name=BASE_AGENT_NAME,
    model="gemini-2.5-flash",
    description="Base RLM agent with a stateful Python REPL.",
    static_instruction=types.Content(
        role="user", parts=[types.Part(text=RLM_STATIC_PROMPT)]
    ),
    instruction=(
        "You are the base RLM agent working ticket {Jira_ticket}. Use the REPL"
        " tools to submit code, move it through the security review state, and"
        " execute only after approval. Use request_sub_lm or the transfer tool"
        " to delegate tasks to the sub agent when recursion is needed. When"
        " you execute code, capture stdout and summarize the results for the"
        " user."
    ),
    tools=[
        submit_code,
        approve_code_execution,
        execute_pending_code,
        set_stdout_target,
        set_save_code_as_artifact,
        set_jira_ticket,
        request_sub_lm,
        reset_repl,
        TransferToAgentTool(agent_names=[SUB_AGENT_NAME]),
    ],
)


sub_agent = LlmAgent(
    name=SUB_AGENT_NAME,
    model="gemini-2.5-flash",
    description="Sub RLM agent for delegated recursive calls.",
    instruction=(
        "You are the sub RLM agent. Use read_sub_lm_request to fetch the"
        " latest delegated prompt, then handle it using the REPL tools. If"
        " you need to return control, use the transfer tool to go back to"
        " the base agent."
    ),
    tools=[
        read_sub_lm_request,
        submit_code,
        approve_code_execution,
        execute_pending_code,
        set_stdout_target,
        TransferToAgentTool(agent_names=[BASE_AGENT_NAME]),
    ],
)


root_agent = LlmAgent(
    name="rlm_root_agent",
    model="gemini-2.5-flash",
    description="Root agent that coordinates base/sub RLM agents.",
    instruction=(
        "You are the root coordinator. Use the transfer tool to route user"
        " requests to the base agent. Do not execute code directly."
    ),
    sub_agents=[base_agent, sub_agent],
    tools=[TransferToAgentTool(agent_names=[BASE_AGENT_NAME, SUB_AGENT_NAME])],
)


app = App(
    name="rlm_minimal_adk",
    root_agent=root_agent,
    plugins=[
        DebugLoggingPlugin(
            output_path="adk_debug.yaml",
            include_session_state=True,
            include_system_instruction=True,
        ),
    ],
)
