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

"""REPL environment implementation using Python's code module.

This module provides a sandboxed REPL environment for executing Python code
within the ADK RLM system. It uses Python's built-in `code` module for
interactive interpretation with custom stdout/stderr routing.
"""

from __future__ import annotations

import code
import io
import sys
import threading
import json
import tempfile
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional
import logging

from .repl_state import (
    REPLStateManager,
    REPLState,
    SecurityLevel,
    REPL_LOCALS_KEY,
    REPL_CONTEXT_KEY,
    REPL_LAST_OUTPUT_KEY,
    REPL_LAST_ERROR_KEY,
)

logger = logging.getLogger("google_adk." + __name__)


@dataclass
class REPLResult:
    """Result of a REPL code execution."""

    stdout: str
    stderr: str
    locals_snapshot: dict[str, Any]
    execution_time: float
    success: bool = True

    def __str__(self) -> str:
        return (
            f"REPLResult(success={self.success}, "
            f"stdout={self.stdout[:100]}..., "
            f"stderr={self.stderr[:100] if self.stderr else ''}, "
            f"execution_time={self.execution_time:.3f}s)"
        )


class OutputRouter:
    """Routes stdout/stderr to appropriate destinations.

    This class manages the routing of print() output to either the base LLM
    or sub-LLM based on the current execution context.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._base_lm_buffer = io.StringIO()
        self._sub_lm_buffer = io.StringIO()
        self._current_target = "base"  # "base" or "sub"
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def set_target(self, target: str) -> None:
        """Set the current output target.

        Args:
            target: Either "base" or "sub" to indicate which model receives output
        """
        with self._lock:
            self._current_target = target

    def get_buffer(self, target: Optional[str] = None) -> str:
        """Get the contents of a buffer.

        Args:
            target: Buffer to retrieve ("base" or "sub"), defaults to current target

        Returns:
            Contents of the specified buffer
        """
        with self._lock:
            if target is None:
                target = self._current_target
            if target == "base":
                return self._base_lm_buffer.getvalue()
            else:
                return self._sub_lm_buffer.getvalue()

    def clear_buffer(self, target: Optional[str] = None) -> None:
        """Clear a buffer.

        Args:
            target: Buffer to clear ("base" or "sub"), defaults to current target
        """
        with self._lock:
            if target is None:
                target = self._current_target
            if target == "base":
                self._base_lm_buffer = io.StringIO()
            else:
                self._sub_lm_buffer = io.StringIO()

    def write(self, text: str) -> None:
        """Write text to the current target buffer.

        Args:
            text: Text to write
        """
        with self._lock:
            if self._current_target == "base":
                self._base_lm_buffer.write(text)
            else:
                self._sub_lm_buffer.write(text)

    @contextmanager
    def redirect(self, target: str = "base"):
        """Context manager to redirect stdout to a specific target.

        Args:
            target: Target buffer ("base" or "sub")
        """
        old_target = self._current_target
        self.set_target(target)
        try:
            yield
        finally:
            self.set_target(old_target)


class SecureInteractiveConsole(code.InteractiveConsole):
    """An interactive console with security restrictions.

    This extends Python's InteractiveConsole with:
    - Sandboxed execution environment
    - Output capture and routing
    - Configurable security restrictions
    """

    def __init__(
        self,
        locals_dict: Optional[dict[str, Any]] = None,
        security_level: SecurityLevel = SecurityLevel.BASIC,
        output_router: Optional[OutputRouter] = None,
    ):
        """Initialize the secure console.

        Args:
            locals_dict: Initial local namespace
            security_level: Security level for code execution
            output_router: Router for stdout/stderr output
        """
        self._security_level = security_level
        self._output_router = output_router or OutputRouter()
        self._lock = threading.Lock()

        # Build safe globals
        safe_globals = self._build_safe_globals()

        # Initialize with safe environment
        if locals_dict is None:
            locals_dict = {}

        super().__init__(locals={**safe_globals, **locals_dict})

    def _build_safe_globals(self) -> dict[str, Any]:
        """Build a safe globals dictionary with restricted built-ins.

        Returns:
            Dictionary of safe global variables
        """
        # Safe built-ins
        safe_builtins = {
            # Data types
            'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
            'type': type, 'isinstance': isinstance, 'issubclass': issubclass,

            # Iterators and sequences
            'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
            'sorted': sorted, 'reversed': reversed, 'range': range,
            'iter': iter, 'next': next, 'slice': slice,

            # Math
            'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
            'pow': pow, 'divmod': divmod, 'complex': complex,

            # String/char
            'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin, 'oct': oct,
            'repr': repr, 'ascii': ascii, 'format': format,

            # Object introspection
            'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
            'delattr': delattr, 'dir': dir, 'vars': vars,
            'callable': callable, 'hash': hash, 'id': id,

            # Collections
            'any': any, 'all': all,

            # Bytes
            'bytes': bytes, 'bytearray': bytearray, 'memoryview': memoryview,

            # OOP
            'super': super, 'property': property,
            'staticmethod': staticmethod, 'classmethod': classmethod, 'object': object,

            # Allow imports (controlled)
            '__import__': __import__,
            'open': open,  # Allow file access for context loading

            # Exception classes
            'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
            'KeyError': KeyError, 'IndexError': IndexError, 'AttributeError': AttributeError,
            'FileNotFoundError': FileNotFoundError, 'OSError': OSError, 'IOError': IOError,
            'RuntimeError': RuntimeError, 'NameError': NameError, 'ImportError': ImportError,
            'StopIteration': StopIteration, 'AssertionError': AssertionError,
            'NotImplementedError': NotImplementedError,

            # Blocked for security
            'input': None,
            'eval': None,
            'exec': None,
            'compile': None,
            'globals': None,
            'locals': None,
            '__builtins__': None,  # Will be replaced with this dict
        }

        return {'__builtins__': safe_builtins}

    def check_code_security(self, code_str: str) -> tuple[bool, str]:
        """Check code for security violations.

        Args:
            code_str: Code to check

        Returns:
            Tuple of (is_safe, message)
        """
        if self._security_level == SecurityLevel.NONE:
            return True, "Security checks disabled"

        dangerous_patterns = [
            ("import os", "os module import blocked"),
            ("import subprocess", "subprocess module import blocked"),
            ("import sys", "sys module import blocked for security"),
            ("__import__('os')", "dynamic os import blocked"),
            ("__import__('subprocess')", "dynamic subprocess import blocked"),
            ("eval(", "eval() is blocked"),
            ("exec(", "exec() is blocked"),
            ("compile(", "compile() is blocked"),
            ("open('/", "absolute path file access restricted"),
            ("os.system", "os.system() is blocked"),
            ("subprocess.", "subprocess operations blocked"),
        ]

        # Basic security checks
        if self._security_level in (SecurityLevel.BASIC, SecurityLevel.STRICT):
            for pattern, message in dangerous_patterns:
                if pattern in code_str:
                    return False, message

        return True, "Code passed security checks"

    @contextmanager
    def _capture_output(self):
        """Context manager to capture stdout/stderr.

        Yields:
            Tuple of (stdout_buffer, stderr_buffer)
        """
        with self._lock:
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            try:
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                yield stdout_buffer, stderr_buffer
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    def execute_code(self, code_str: str) -> REPLResult:
        """Execute code in the sandboxed environment.

        This implements notebook-style execution where the last expression
        is automatically printed.

        Args:
            code_str: Code to execute

        Returns:
            REPLResult with execution results
        """
        start_time = time.time()

        # Security check
        is_safe, security_msg = self.check_code_security(code_str)
        if not is_safe:
            return REPLResult(
                stdout="",
                stderr=f"Security Error: {security_msg}",
                locals_snapshot=dict(self.locals),
                execution_time=time.time() - start_time,
                success=False,
            )

        with self._capture_output() as (stdout_buffer, stderr_buffer):
            try:
                # Split into import statements and other code
                lines = code_str.split('\n')
                import_lines = []
                other_lines = []

                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)

                # Execute imports first
                if import_lines:
                    import_code = '\n'.join(import_lines)
                    exec(import_code, self.locals, self.locals)

                # Execute remaining code with notebook-style last expression printing
                if other_lines:
                    other_code = '\n'.join(other_lines)
                    non_comment_lines = [
                        line for line in other_lines
                        if line.strip() and not line.strip().startswith('#')
                    ]

                    if non_comment_lines:
                        last_line = non_comment_lines[-1].strip()

                        # Check if last line is an expression (not a statement)
                        is_expression = self._is_expression(last_line)

                        if is_expression:
                            try:
                                # Execute all but last line as statements
                                if len(non_comment_lines) > 1:
                                    # Find where last line starts
                                    last_line_idx = -1
                                    for i, line in enumerate(other_lines):
                                        if line.strip() == last_line:
                                            last_line_idx = i

                                    if last_line_idx > 0:
                                        statements_code = '\n'.join(other_lines[:last_line_idx])
                                        exec(statements_code, self.locals, self.locals)

                                # Evaluate last line as expression
                                result = eval(last_line, self.locals, self.locals)
                                if result is not None:
                                    print(repr(result))
                            except Exception:
                                # Fall back to normal execution
                                exec(other_code, self.locals, self.locals)
                        else:
                            exec(other_code, self.locals, self.locals)

                stdout_content = stdout_buffer.getvalue()
                stderr_content = stderr_buffer.getvalue()
                success = True

            except Exception as e:
                stdout_content = stdout_buffer.getvalue()
                stderr_content = stderr_buffer.getvalue() + str(e)
                success = False

        execution_time = time.time() - start_time

        return REPLResult(
            stdout=stdout_content,
            stderr=stderr_content,
            locals_snapshot=self._safe_locals_snapshot(),
            execution_time=execution_time,
            success=success,
        )

    def _is_expression(self, line: str) -> bool:
        """Check if a line is an expression (not a statement).

        Args:
            line: Line to check

        Returns:
            True if line appears to be an expression
        """
        statement_starters = (
            'import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ',
            'try:', 'with ', 'return ', 'yield ', 'break', 'continue', 'pass',
            'raise ', 'assert ', 'del ', 'global ', 'nonlocal ',
        )

        if line.startswith(statement_starters):
            return False

        # Check for assignment (but not comparison)
        if '=' in line.split('#')[0]:
            # Check if it's a comparison operator
            eq_idx = line.find('=')
            if eq_idx > 0:
                prev_char = line[eq_idx - 1]
                if prev_char in '!<>=':
                    return True  # It's a comparison
                if eq_idx < len(line) - 1 and line[eq_idx + 1] == '=':
                    return True  # It's ==
            return False  # It's an assignment

        if line.endswith(':'):
            return False

        if line.startswith('print('):
            return False  # Explicit print, don't double-print

        return True

    def _safe_locals_snapshot(self) -> dict[str, Any]:
        """Create a serializable snapshot of locals.

        Returns:
            Dictionary of local variables that can be serialized
        """
        snapshot = {}
        for key, value in self.locals.items():
            if key.startswith('_') or key == '__builtins__':
                continue
            try:
                # Try to serialize the value
                json.dumps(value)
                snapshot[key] = value
            except (TypeError, ValueError):
                # Store string representation for non-serializable values
                snapshot[key] = f"<{type(value).__name__}>"
        return snapshot

    def add_to_locals(self, name: str, value: Any) -> None:
        """Add a variable to the local namespace.

        Args:
            name: Variable name
            value: Variable value
        """
        self.locals[name] = value

    def get_from_locals(self, name: str, default: Any = None) -> Any:
        """Get a variable from the local namespace.

        Args:
            name: Variable name
            default: Default value if not found

        Returns:
            Variable value or default
        """
        return self.locals.get(name, default)


class ADKREPLEnvironment:
    """ADK-integrated REPL environment.

    This class provides the main interface for the REPL environment,
    integrating with ADK's state management and tool context.
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.BASIC,
        temp_dir: Optional[str] = None,
    ):
        """Initialize the ADK REPL environment.

        Args:
            security_level: Security level for code execution
            temp_dir: Temporary directory for file operations
        """
        self._security_level = security_level
        self._output_router = OutputRouter()
        self._console: Optional[SecureInteractiveConsole] = None
        self._temp_dir = temp_dir or tempfile.mkdtemp(prefix="adk_repl_")
        self._sub_lm_callback: Optional[Callable[[str], str]] = None

    def initialize(
        self,
        context: Optional[str] = None,
        context_json: Optional[dict | list] = None,
        locals_dict: Optional[dict[str, Any]] = None,
        sub_lm_callback: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Initialize the REPL with context and callbacks.

        Args:
            context: String context to load
            context_json: JSON context to load
            locals_dict: Initial local variables
            sub_lm_callback: Callback function for sub-LLM queries
        """
        self._sub_lm_callback = sub_lm_callback

        # Create console with initial locals
        self._console = SecureInteractiveConsole(
            locals_dict=locals_dict or {},
            security_level=self._security_level,
            output_router=self._output_router,
        )

        # Add special functions
        self._add_special_functions()

        # Load context if provided
        if context_json is not None:
            self._load_json_context(context_json)
        if context is not None:
            self._load_string_context(context)

    def _add_special_functions(self) -> None:
        """Add special functions to the REPL namespace."""
        if self._console is None:
            return

        # Add llm_query function
        def llm_query(prompt: str) -> str:
            """Query the sub-LLM with the given prompt.

            Args:
                prompt: The prompt to send to the sub-LLM

            Returns:
                The sub-LLM's response
            """
            if self._sub_lm_callback is None:
                return "Error: Sub-LLM not available"

            # Route output to sub-LLM buffer during query
            with self._output_router.redirect("sub"):
                result = self._sub_lm_callback(prompt)

            return result

        self._console.add_to_locals('llm_query', llm_query)

        # Add FINAL_VAR function
        def final_var(variable_name: str) -> str:
            """Return a variable from the REPL as the final answer.

            Args:
                variable_name: Name of the variable to return

            Returns:
                String representation of the variable
            """
            variable_name = variable_name.strip().strip('"').strip("'")
            value = self._console.get_from_locals(variable_name)
            if value is None:
                return f"Error: Variable '{variable_name}' not found"
            return str(value)

        self._console.add_to_locals('FINAL_VAR', final_var)

    def _load_json_context(self, context_json: dict | list) -> None:
        """Load JSON context into the REPL.

        Args:
            context_json: JSON data to load
        """
        if self._console is None:
            return

        context_path = os.path.join(self._temp_dir, "context.json")
        with open(context_path, "w") as f:
            json.dump(context_json, f, indent=2)

        # Execute code to load context
        load_code = f"""
import json
with open(r'{context_path}', 'r') as f:
    context = json.load(f)
"""
        self._console.execute_code(load_code)

    def _load_string_context(self, context: str) -> None:
        """Load string context into the REPL.

        Args:
            context: String context to load
        """
        if self._console is None:
            return

        context_path = os.path.join(self._temp_dir, "context.txt")
        with open(context_path, "w") as f:
            f.write(context)

        # Execute code to load context
        load_code = f"""
with open(r'{context_path}', 'r') as f:
    context = f.read()
"""
        self._console.execute_code(load_code)

    def execute(self, code: str) -> REPLResult:
        """Execute code in the REPL.

        Args:
            code: Code to execute

        Returns:
            REPLResult with execution results
        """
        if self._console is None:
            return REPLResult(
                stdout="",
                stderr="Error: REPL not initialized",
                locals_snapshot={},
                execution_time=0.0,
                success=False,
            )

        return self._console.execute_code(code)

    def get_locals(self) -> dict[str, Any]:
        """Get current local variables.

        Returns:
            Dictionary of local variables
        """
        if self._console is None:
            return {}
        return self._console._safe_locals_snapshot()

    def get_output_router(self) -> OutputRouter:
        """Get the output router.

        Returns:
            The OutputRouter instance
        """
        return self._output_router

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        try:
            import shutil
            if os.path.exists(self._temp_dir):
                shutil.rmtree(self._temp_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {e}")

    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()
