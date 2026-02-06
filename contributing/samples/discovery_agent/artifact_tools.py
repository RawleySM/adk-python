"""Artifact tools for persisting Jira tickets, REPL code, and PR docs.

Each tool wraps the ADK ArtifactService (via ToolContext) to save versioned
binary artifacts into the session-scoped artifact store.
"""

from __future__ import annotations

from typing import Any


async def save_jira_ticket(
    ticket_id: str,
    title: str,
    description: str,
    acceptance_criteria: str = "",
    tool_context: Any = None,
) -> dict:
    """Save a Jira ticket into the artifact store.

    Args:
        ticket_id: The Jira ticket key (e.g. 'PROJ-123').
        title: Short summary of the ticket.
        description: Full ticket description / body.
        acceptance_criteria: Acceptance criteria for the ticket.
        tool_context: ADK ToolContext (injected automatically).

    Returns:
        A dict with the artifact filename and version.
    """
    from google.genai import types as genai_types

    content = (
        f"# {ticket_id}: {title}\n\n"
        f"## Description\n{description}\n\n"
        f"## Acceptance Criteria\n{acceptance_criteria}\n"
    )

    filename = f"jira_{ticket_id.replace('-', '_').lower()}"

    if tool_context is not None:
        artifact = genai_types.Part(text=content)
        version = await tool_context.save_artifact(filename, artifact)
        # Also store in state for quick access by other agents
        tool_context.state[f"jira:{ticket_id}"] = content
        return {"filename": filename, "version": version, "ticket_id": ticket_id}

    return {"error": "No tool_context available", "ticket_id": ticket_id}


async def save_repl_submission(
    label: str,
    code: str,
    stdout: str = "",
    stderr: str = "",
    tool_context: Any = None,
) -> dict:
    """Save a REPL code submission and its output as an artifact.

    Args:
        label: A short descriptive label for this submission.
        code: The Python code that was executed.
        stdout: Captured standard output.
        stderr: Captured standard error.
        tool_context: ADK ToolContext (injected automatically).

    Returns:
        A dict with the artifact filename and version.
    """
    from google.genai import types as genai_types

    content = (
        f"# REPL Submission: {label}\n\n"
        f"```python\n{code}\n```\n\n"
        f"## stdout\n```\n{stdout}\n```\n\n"
        f"## stderr\n```\n{stderr}\n```\n"
    )

    # Sanitize label for filename
    safe_label = label.lower().replace(" ", "_").replace("-", "_")[:40]
    filename = f"repl_{safe_label}"

    if tool_context is not None:
        artifact = genai_types.Part(text=content)
        version = await tool_context.save_artifact(filename, artifact)
        return {"filename": filename, "version": version, "label": label}

    return {"error": "No tool_context available", "label": label}


async def save_pr_document(
    pr_title: str,
    pr_body: str,
    files_changed: str = "",
    tool_context: Any = None,
) -> dict:
    """Save pull request documentation as an artifact.

    Args:
        pr_title: The PR title.
        pr_body: The full PR description / body in markdown.
        files_changed: Summary of files changed.
        tool_context: ADK ToolContext (injected automatically).

    Returns:
        A dict with the artifact filename and version.
    """
    from google.genai import types as genai_types

    content = (
        f"# Pull Request: {pr_title}\n\n"
        f"## Description\n{pr_body}\n\n"
        f"## Files Changed\n{files_changed}\n"
    )

    filename = "pr_document"

    if tool_context is not None:
        artifact = genai_types.Part(text=content)
        version = await tool_context.save_artifact(filename, artifact)
        # Persist final PR output in state for the root agent to surface
        tool_context.state["pr_document"] = content
        return {"filename": filename, "version": version, "pr_title": pr_title}

    return {"error": "No tool_context available", "pr_title": pr_title}


async def save_discovery_artifact(
    filename: str,
    content: str,
    tool_context: Any = None,
) -> dict:
    """Save a generic discovery artifact (file analysis, summaries, etc.).

    Args:
        filename: Artifact filename (no extension needed).
        content: The text content to save.
        tool_context: ADK ToolContext (injected automatically).

    Returns:
        A dict with the artifact filename and version.
    """
    from google.genai import types as genai_types

    if tool_context is not None:
        artifact = genai_types.Part(text=content)
        version = await tool_context.save_artifact(filename, artifact)
        return {"filename": filename, "version": version}

    return {"error": "No tool_context available", "filename": filename}
