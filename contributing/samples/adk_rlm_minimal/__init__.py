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

This package provides an ADK-native implementation of the Recursive Language
Model (RLM) pattern for handling large contexts through:

1. **EFSM State Management**: Extended Finite State Machine for managing REPL
   states with intermediate security states between code submission and execution.

2. **Python code module REPL**: Sandboxed execution environment using Python's
   built-in code module for interactive interpretation.

3. **Stdout Routing**: Intelligent routing of print() output to appropriate
   models (base LLM or sub-LLM) based on execution context.

4. **Agent Hierarchy**: Base LLM agent orchestrates analysis while sub-LLM
   agent handles recursive context chunk analysis.

5. **Artifact Storage**: Optional artifact storage for all executed code
   for debugging and audit purposes.

6. **DebugLoggingPlugin**: Full integration with ADK's debug logging for
   comprehensive interaction tracking.

## Architecture

```
User Query + Context
        ↓
┌─────────────────────────────────┐
│     Root RLM Agent (Base LLM)   │
│  - Orchestrates REPL interaction│
│  - Executes code via tools      │
│  - Builds up final answer       │
└─────────────┬───────────────────┘
              │ execute_code()
              ↓
┌─────────────────────────────────┐
│     REPL Environment (EFSM)     │
│  States: IDLE → PENDING_REVIEW  │
│        → APPROVED → EXECUTING   │
│        → IDLE or COMPLETE       │
└─────────────┬───────────────────┘
              │ llm_query()
              ↓
┌─────────────────────────────────┐
│      Sub-LLM Agent              │
│  - Analyzes context chunks      │
│  - Returns findings to REPL     │
└─────────────────────────────────┘
```

## State Keys

- `repl_state`: Current EFSM state
- `repl_locals`: REPL local variables
- `repl_context`: Context data
- `repl_query`: User query
- `repl_iteration`: Current iteration count
- `repl_code_history`: History of executed code
- `repl_final_answer`: Final answer when complete

## Usage

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from contributing.samples.adk_rlm_minimal import (
    app,
    initialize_rlm_session,
    get_final_answer,
)

# Create runner
session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()
runner = Runner(
    app_name=app.name,
    agent=app.root_agent,
    session_service=session_service,
    artifact_service=artifact_service,
)

# Create session with context
session = await session_service.create_session(app_name=app.name, user_id="user1")
initialize_rlm_session(
    session.state,
    context="Your large context data here...",
    query="What is the magic number?",
)

# Run the agent
async for event in runner.run_async(session_id=session.id, user_input="Start analysis"):
    print(event)

# Get result
answer = get_final_answer(session.state)
```

## Environment Variables

- `RLM_BASE_MODEL`: Model for base agent (default: gemini-2.5-flash)
- `RLM_SUB_MODEL`: Model for sub-LLM (default: gemini-2.0-flash)
- `RLM_SECURITY_LEVEL`: Security level (none/basic/strict, default: basic)
- `RLM_MAX_ITERATIONS`: Maximum iterations (default: 20)
- `RLM_ARTIFACT_ENABLED`: Enable artifact storage (default: true)
- `RLM_DEBUG_LOGGING`: Enable debug logging (default: true)
"""

from .agent import (
    app,
    root_agent,
    initialize_rlm_session,
    get_final_answer,
    is_session_complete,
)
from .repl_state import (
    REPLState,
    REPLStateManager,
    SecurityLevel,
)
from .repl_env import (
    ADKREPLEnvironment,
    REPLResult,
    OutputRouter,
)
from .tools import (
    execute_code,
    submit_final_answer,
    submit_final_variable,
    get_repl_state,
    find_code_blocks,
    check_for_final_answer,
)
from .agents import (
    create_rlm_agent_hierarchy,
    create_sub_llm_agent,
    create_base_rlm_agent,
)

__all__ = [
    # Main exports
    "app",
    "root_agent",
    "initialize_rlm_session",
    "get_final_answer",
    "is_session_complete",
    # State management
    "REPLState",
    "REPLStateManager",
    "SecurityLevel",
    # REPL environment
    "ADKREPLEnvironment",
    "REPLResult",
    "OutputRouter",
    # Tools
    "execute_code",
    "submit_final_answer",
    "submit_final_variable",
    "get_repl_state",
    "find_code_blocks",
    "check_for_final_answer",
    # Agent factories
    "create_rlm_agent_hierarchy",
    "create_sub_llm_agent",
    "create_base_rlm_agent",
]
