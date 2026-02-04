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

"""EFSM (Extended Finite State Machine) state management for the ADK RLM REPL.

This module implements state management using ADK's key-value state system
with an Extended Finite State Machine model for the REPL environment.

State Transitions:
    IDLE -> CODE_PENDING_REVIEW: Agent submits code
    CODE_PENDING_REVIEW -> CODE_APPROVED: Security check passes (or auto-approve)
    CODE_PENDING_REVIEW -> CODE_REJECTED: Security check fails
    CODE_APPROVED -> EXECUTING: Code execution starts
    EXECUTING -> IDLE: Execution completes
    Any State -> COMPLETE: Final answer provided
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field


class REPLState(str, Enum):
    """EFSM states for the REPL environment."""

    # Initial state - waiting for agent action
    IDLE = "idle"

    # Code has been submitted, pending security review
    CODE_PENDING_REVIEW = "code_pending_review"

    # Code has been approved for execution
    CODE_APPROVED = "code_approved"

    # Code rejected by security check
    CODE_REJECTED = "code_rejected"

    # Code is currently executing
    EXECUTING = "executing"

    # REPL session is complete (final answer provided)
    COMPLETE = "complete"


class SecurityLevel(str, Enum):
    """Security levels for code execution."""

    # No security checks - auto-approve all code
    NONE = "none"

    # Basic checks - block dangerous operations
    BASIC = "basic"

    # Strict checks - require explicit approval
    STRICT = "strict"


# State key prefixes following ADK conventions
STATE_PREFIX = ""  # Session-scoped state
TEMP_PREFIX = "temp:"  # Invocation-scoped state

# State keys for REPL management
REPL_STATE_KEY = "repl_state"
REPL_LOCALS_KEY = "repl_locals"
REPL_GLOBALS_KEY = "repl_globals"
REPL_CONTEXT_KEY = "repl_context"
REPL_ITERATION_KEY = "repl_iteration"
REPL_MAX_ITERATIONS_KEY = "repl_max_iterations"
REPL_PENDING_CODE_KEY = "temp:repl_pending_code"
REPL_LAST_OUTPUT_KEY = "repl_last_output"
REPL_LAST_ERROR_KEY = "repl_last_error"
REPL_FINAL_ANSWER_KEY = "repl_final_answer"
REPL_CODE_HISTORY_KEY = "repl_code_history"
REPL_SECURITY_LEVEL_KEY = "repl_security_level"
REPL_QUERY_KEY = "repl_query"
REPL_ARTIFACT_ENABLED_KEY = "repl_artifact_enabled"


@dataclass
class REPLStateManager:
    """Manages REPL state transitions and validation.

    This class provides methods for managing the EFSM state of the REPL,
    including state transitions, validation, and state queries.
    """

    # Valid state transitions
    VALID_TRANSITIONS: dict[REPLState, set[REPLState]] = field(default_factory=lambda: {
        REPLState.IDLE: {REPLState.CODE_PENDING_REVIEW, REPLState.COMPLETE},
        REPLState.CODE_PENDING_REVIEW: {
            REPLState.CODE_APPROVED,
            REPLState.CODE_REJECTED,
            REPLState.COMPLETE
        },
        REPLState.CODE_APPROVED: {REPLState.EXECUTING, REPLState.COMPLETE},
        REPLState.CODE_REJECTED: {REPLState.IDLE, REPLState.COMPLETE},
        REPLState.EXECUTING: {REPLState.IDLE, REPLState.COMPLETE},
        REPLState.COMPLETE: set(),  # Terminal state
    })

    @classmethod
    def can_transition(cls, from_state: REPLState, to_state: REPLState) -> bool:
        """Check if a state transition is valid.

        Args:
            from_state: Current state
            to_state: Target state

        Returns:
            True if the transition is valid, False otherwise
        """
        valid_targets = cls.VALID_TRANSITIONS.get(from_state, set())
        return to_state in valid_targets

    @classmethod
    def get_state_from_dict(cls, state_dict: dict[str, Any]) -> REPLState:
        """Extract REPL state from state dictionary.

        Args:
            state_dict: ADK state dictionary

        Returns:
            Current REPL state, defaults to IDLE if not set
        """
        state_str = state_dict.get(REPL_STATE_KEY, REPLState.IDLE.value)
        try:
            return REPLState(state_str)
        except ValueError:
            return REPLState.IDLE

    @classmethod
    def initialize_state(
        cls,
        state_dict: dict[str, Any],
        context: Optional[str] = None,
        query: Optional[str] = None,
        max_iterations: int = 20,
        security_level: SecurityLevel = SecurityLevel.BASIC,
        artifact_enabled: bool = False,
    ) -> dict[str, Any]:
        """Initialize REPL state in the state dictionary.

        Args:
            state_dict: ADK state dictionary to update
            context: Initial context data
            query: User query
            max_iterations: Maximum iterations allowed
            security_level: Security level for code execution
            artifact_enabled: Whether to save code to artifacts

        Returns:
            Updated state dictionary (also modifies in place)
        """
        state_dict[REPL_STATE_KEY] = REPLState.IDLE.value
        state_dict[REPL_LOCALS_KEY] = {}
        state_dict[REPL_GLOBALS_KEY] = {}
        state_dict[REPL_ITERATION_KEY] = 0
        state_dict[REPL_MAX_ITERATIONS_KEY] = max_iterations
        state_dict[REPL_LAST_OUTPUT_KEY] = ""
        state_dict[REPL_LAST_ERROR_KEY] = ""
        state_dict[REPL_CODE_HISTORY_KEY] = []
        state_dict[REPL_SECURITY_LEVEL_KEY] = security_level.value
        state_dict[REPL_ARTIFACT_ENABLED_KEY] = artifact_enabled

        if context is not None:
            state_dict[REPL_CONTEXT_KEY] = context
        if query is not None:
            state_dict[REPL_QUERY_KEY] = query

        return state_dict

    @classmethod
    def transition_to(
        cls,
        state_dict: dict[str, Any],
        new_state: REPLState,
        validate: bool = True,
    ) -> bool:
        """Transition to a new REPL state.

        Args:
            state_dict: ADK state dictionary to update
            new_state: Target state
            validate: Whether to validate the transition

        Returns:
            True if transition was successful

        Raises:
            ValueError: If transition is invalid and validate=True
        """
        current_state = cls.get_state_from_dict(state_dict)

        if validate and not cls.can_transition(current_state, new_state):
            raise ValueError(
                f"Invalid state transition from {current_state.value} to {new_state.value}"
            )

        state_dict[REPL_STATE_KEY] = new_state.value
        return True

    @classmethod
    def increment_iteration(cls, state_dict: dict[str, Any]) -> int:
        """Increment and return the iteration count.

        Args:
            state_dict: ADK state dictionary

        Returns:
            New iteration count
        """
        current = state_dict.get(REPL_ITERATION_KEY, 0)
        new_count = current + 1
        state_dict[REPL_ITERATION_KEY] = new_count
        return new_count

    @classmethod
    def has_exceeded_max_iterations(cls, state_dict: dict[str, Any]) -> bool:
        """Check if max iterations have been exceeded.

        Args:
            state_dict: ADK state dictionary

        Returns:
            True if max iterations exceeded
        """
        current = state_dict.get(REPL_ITERATION_KEY, 0)
        max_iter = state_dict.get(REPL_MAX_ITERATIONS_KEY, 20)
        return current >= max_iter

    @classmethod
    def add_code_to_history(
        cls,
        state_dict: dict[str, Any],
        code: str,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Add executed code to history.

        Args:
            state_dict: ADK state dictionary
            code: Code that was executed
            stdout: Standard output from execution
            stderr: Standard error from execution
        """
        history = state_dict.get(REPL_CODE_HISTORY_KEY, [])
        history.append({
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "iteration": state_dict.get(REPL_ITERATION_KEY, 0),
        })
        state_dict[REPL_CODE_HISTORY_KEY] = history

    @classmethod
    def set_pending_code(cls, state_dict: dict[str, Any], code: str) -> None:
        """Set code pending for security review.

        Args:
            state_dict: ADK state dictionary
            code: Code pending review
        """
        state_dict[REPL_PENDING_CODE_KEY] = code

    @classmethod
    def get_pending_code(cls, state_dict: dict[str, Any]) -> Optional[str]:
        """Get code pending for security review.

        Args:
            state_dict: ADK state dictionary

        Returns:
            Pending code or None
        """
        return state_dict.get(REPL_PENDING_CODE_KEY)

    @classmethod
    def clear_pending_code(cls, state_dict: dict[str, Any]) -> None:
        """Clear pending code after processing.

        Args:
            state_dict: ADK state dictionary
        """
        if REPL_PENDING_CODE_KEY in state_dict:
            del state_dict[REPL_PENDING_CODE_KEY]

    @classmethod
    def set_final_answer(cls, state_dict: dict[str, Any], answer: str) -> None:
        """Set the final answer and transition to COMPLETE state.

        Args:
            state_dict: ADK state dictionary
            answer: Final answer
        """
        state_dict[REPL_FINAL_ANSWER_KEY] = answer
        cls.transition_to(state_dict, REPLState.COMPLETE, validate=False)

    @classmethod
    def get_final_answer(cls, state_dict: dict[str, Any]) -> Optional[str]:
        """Get the final answer if set.

        Args:
            state_dict: ADK state dictionary

        Returns:
            Final answer or None
        """
        return state_dict.get(REPL_FINAL_ANSWER_KEY)

    @classmethod
    def is_complete(cls, state_dict: dict[str, Any]) -> bool:
        """Check if the REPL session is complete.

        Args:
            state_dict: ADK state dictionary

        Returns:
            True if session is complete
        """
        return cls.get_state_from_dict(state_dict) == REPLState.COMPLETE
