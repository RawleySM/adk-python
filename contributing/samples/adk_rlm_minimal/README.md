# ADK-Native RLM (Recursive Language Model) with REPL

This sample demonstrates an ADK-native implementation of the [RLM-minimal](https://github.com/alexzhang13/rlm-minimal) pattern for handling large contexts through recursive LLM calls within a REPL environment.

## Features

- **Extended Finite State Machine (EFSM)**: Manages REPL states with intermediate security states between code submission and execution
- **Python code module REPL**: Sandboxed execution environment using Python's built-in `code` module
- **Stdout Routing**: Routes print() output to appropriate models (base LLM or sub-LLM)
- **Agent Transfer**: Uses ADK's `transfer_to_agent` pattern for flow management
- **Artifact Storage**: Optional artifact storage for all executed code
- **DebugLoggingPlugin**: Full integration with ADK's debug logging

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

## EFSM State Transitions

```
IDLE ──────────────────┬──────────────────────> COMPLETE
  │                    │
  │ submit code        │ final answer
  ↓                    │
CODE_PENDING_REVIEW ───┤
  │         │          │
  │ approve │ reject   │
  ↓         ↓          │
CODE_APPROVED  CODE_REJECTED
  │              │
  │ execute      │ retry
  ↓              │
EXECUTING ───────┴────────────────────────────> IDLE
```

## Usage

### Using ADK CLI

```bash
# Run interactively
adk run contributing/samples/adk_rlm_minimal

# With environment variable configuration
export RLM_BASE_MODEL=gemini-2.5-flash
export RLM_SUB_MODEL=gemini-2.0-flash
export RLM_SECURITY_LEVEL=basic
export RLM_MAX_ITERATIONS=20
adk run contributing/samples/adk_rlm_minimal
```

### Running the Example

```bash
# Run the needle-in-haystack example
python -m contributing.samples.adk_rlm_minimal.example

# With custom parameters
python -m contributing.samples.adk_rlm_minimal.example \
    --lines 100000 \
    --iterations 15 \
    --security basic
```

### Programmatic Usage

```python
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from contributing.samples.adk_rlm_minimal import (
    app,
    initialize_rlm_session,
    get_final_answer,
    SecurityLevel,
)

async def main():
    # Create services
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()

    # Create runner
    runner = Runner(
        app_name=app.name,
        agent=app.root_agent,
        session_service=session_service,
        artifact_service=artifact_service,
    )

    # Create and initialize session
    session = await session_service.create_session(
        app_name=app.name,
        user_id="user1",
    )

    # Initialize with your context
    large_context = "..." # Your large context here
    initialize_rlm_session(
        session.state,
        context=large_context,
        query="What is the key information in this context?",
        max_iterations=20,
        security_level=SecurityLevel.BASIC,
        artifact_enabled=True,
    )

    # Run the agent
    async for event in runner.run_async(
        session_id=session.id,
        user_input="Analyze the context and answer the query",
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(part.text)

    # Get result
    answer = get_final_answer(session.state)
    print(f"Final answer: {answer}")

asyncio.run(main())
```

## State Keys

| Key | Scope | Description |
|-----|-------|-------------|
| `repl_state` | Session | Current EFSM state |
| `repl_locals` | Session | REPL local variables |
| `repl_context` | Session | Loaded context data |
| `repl_query` | Session | User query |
| `repl_iteration` | Session | Current iteration count |
| `repl_max_iterations` | Session | Maximum allowed iterations |
| `repl_code_history` | Session | History of executed code |
| `repl_final_answer` | Session | Final answer when complete |
| `repl_security_level` | Session | Security level setting |
| `repl_artifact_enabled` | Session | Whether artifacts are saved |
| `temp:repl_pending_code` | Invocation | Code pending security review |

## Security Levels

- **NONE**: No security checks, all code auto-approved
- **BASIC**: Basic checks for dangerous operations (default)
- **STRICT**: Requires explicit approval for each code execution

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RLM_BASE_MODEL` | gemini-2.5-flash | Model for base agent |
| `RLM_SUB_MODEL` | gemini-2.0-flash | Model for sub-LLM |
| `RLM_SECURITY_LEVEL` | basic | Security level |
| `RLM_MAX_ITERATIONS` | 20 | Maximum iterations |
| `RLM_ARTIFACT_ENABLED` | true | Enable artifact storage |
| `RLM_DEBUG_LOGGING` | true | Enable debug logging |

## Tools

### execute_code

Executes Python code in the sandboxed REPL environment.

```python
result = execute_code(
    code="print(len(context))",
    tool_context=ctx,
)
```

### submit_final_answer

Submits the final answer and completes the session.

```python
result = submit_final_answer(
    answer="The magic number is 1234567",
    tool_context=ctx,
)
```

### submit_final_variable

Submits a REPL variable as the final answer.

```python
result = submit_final_variable(
    variable_name="final_answer",
    tool_context=ctx,
)
```

### get_repl_state

Returns current REPL state information.

```python
state_info = get_repl_state(tool_context=ctx)
```

## Debug Output

When debug logging is enabled, all interactions are logged to `adk_rlm_debug.yaml`:

```yaml
---
invocation_id: abc123
session_id: session456
entries:
  - timestamp: "2026-02-04T10:30:00"
    entry_type: tool_call
    tool_name: execute_code
    args:
      code: "print(len(context))"
  - timestamp: "2026-02-04T10:30:01"
    entry_type: tool_response
    result:
      status: success
      stdout: "1000000"
```

## Files

- `agent.py` - Main App and root agent configuration
- `agents.py` - Agent factory functions and prompts
- `tools.py` - Tool implementations for REPL interaction
- `repl_state.py` - EFSM state management
- `repl_env.py` - REPL environment using Python code module
- `example.py` - Needle-in-haystack example script
