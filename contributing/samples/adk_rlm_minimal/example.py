#!/usr/bin/env python3
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

"""Example usage of the ADK-native RLM with needle-in-haystack problem.

This script demonstrates how to use the ADK RLM implementation to solve
a needle-in-haystack problem where a magic number is hidden within a
large context of random text.

Usage:
    python -m contributing.samples.adk_rlm_minimal.example

    Or with custom parameters:
    python -m contributing.samples.adk_rlm_minimal.example --lines 100000 --iterations 15
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from typing import Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService

from . import (
    app,
    initialize_rlm_session,
    get_final_answer,
    is_session_complete,
    SecurityLevel,
)


def generate_massive_context(
    num_lines: int = 100_000,
    magic_number: Optional[str] = None,
) -> tuple[str, str]:
    """Generate a large context with a hidden magic number.

    Args:
        num_lines: Number of lines to generate
        magic_number: The magic number to hide (generated if not provided)

    Returns:
        Tuple of (context string, magic number)
    """
    print(f"Generating context with {num_lines:,} lines...")

    if magic_number is None:
        magic_number = str(random.randint(1_000_000, 9_999_999))

    # Random words to fill the context
    random_words = [
        "data", "text", "random", "content", "information",
        "sample", "example", "filler", "placeholder", "noise",
        "lorem", "ipsum", "dolor", "amet", "consectetur",
    ]

    lines = []
    for _ in range(num_lines):
        num_words = random.randint(3, 10)
        line_words = [random.choice(random_words) for _ in range(num_words)]
        lines.append(" ".join(line_words))

    # Insert the magic number at a random position in the middle
    magic_position = random.randint(num_lines // 3, 2 * num_lines // 3)
    lines[magic_position] = f"The magic number is {magic_number}"

    print(f"Magic number '{magic_number}' inserted at line {magic_position:,}")

    return "\n".join(lines), magic_number


async def run_rlm_example(
    context: str,
    query: str,
    max_iterations: int = 20,
    security_level: SecurityLevel = SecurityLevel.BASIC,
    artifact_enabled: bool = True,
    verbose: bool = True,
) -> Optional[str]:
    """Run the RLM agent on a context with a query.

    Args:
        context: Context data to analyze
        query: Query to answer
        max_iterations: Maximum iterations
        security_level: Security level for code execution
        artifact_enabled: Whether to save code artifacts
        verbose: Whether to print progress

    Returns:
        Final answer or None if not found
    """
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

    # Create session
    session = await session_service.create_session(
        app_name=app.name,
        user_id="test_user",
    )

    if verbose:
        print(f"Created session: {session.id}")
        print(f"Context length: {len(context):,} characters")
        print(f"Query: {query}")
        print("-" * 60)

    # Initialize RLM state with context and query
    initialize_rlm_session(
        session.state,
        context=context,
        query=query,
        max_iterations=max_iterations,
        security_level=security_level,
        artifact_enabled=artifact_enabled,
    )

    # Run the agent
    iteration = 0
    async for event in runner.run_async(
        session_id=session.id,
        user_input=f"Please analyze the context and answer: {query}",
    ):
        if verbose:
            # Print progress
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        # Print first 200 chars of text
                        text_preview = part.text[:200]
                        if len(part.text) > 200:
                            text_preview += "..."
                        print(f"[{event.author}] {text_preview}")
                    elif part.function_call:
                        print(f"[{event.author}] Calling: {part.function_call.name}")
                    elif part.function_response:
                        print(f"[{event.author}] Response from: {part.function_response.name}")

        # Check for completion
        if event.is_final_response():
            iteration += 1
            if verbose:
                print(f"\n--- Iteration {iteration} complete ---\n")

    # Get final answer from state
    # Reload session to get updated state
    session = await session_service.get_session(
        app_name=app.name,
        user_id="test_user",
        session_id=session.id,
    )

    final_answer = get_final_answer(session.state)

    if verbose:
        print("=" * 60)
        print(f"Session complete: {is_session_complete(session.state)}")
        print(f"Final answer: {final_answer}")

    return final_answer


async def main():
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(
        description="ADK RLM needle-in-haystack example"
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=100_000,
        help="Number of lines in the context (default: 100,000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=15,
        help="Maximum iterations (default: 15)",
    )
    parser.add_argument(
        "--security",
        choices=["none", "basic", "strict"],
        default="basic",
        help="Security level (default: basic)",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Disable artifact storage",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--magic-number",
        type=str,
        default=None,
        help="Use a specific magic number instead of random",
    )

    args = parser.parse_args()

    # Generate context
    context, expected_answer = generate_massive_context(
        num_lines=args.lines,
        magic_number=args.magic_number,
    )

    # Map security level
    security_map = {
        "none": SecurityLevel.NONE,
        "basic": SecurityLevel.BASIC,
        "strict": SecurityLevel.STRICT,
    }
    security_level = security_map[args.security]

    # Run the RLM
    print("\n" + "=" * 60)
    print("Starting ADK RLM Analysis")
    print("=" * 60 + "\n")

    result = await run_rlm_example(
        context=context,
        query="What is the magic number hidden in the context?",
        max_iterations=args.iterations,
        security_level=security_level,
        artifact_enabled=not args.no_artifacts,
        verbose=not args.quiet,
    )

    # Check result
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Expected answer: {expected_answer}")
    print(f"RLM answer: {result}")

    if result and expected_answer in result:
        print("\n✓ SUCCESS: Magic number found correctly!")
        return 0
    else:
        print("\n✗ FAILURE: Magic number not found or incorrect")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
