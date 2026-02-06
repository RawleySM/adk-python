"""GitHub FuncTools for downloading files and directory listings.

These tools are exposed to LlmAgents so they can pull repository content
into the REPL execution environment and session state.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Optional
from urllib import request, error
import json


def github_list_directory(
    owner: str,
    repo: str,
    path: str = "",
    ref: str = "main",
    tool_context: Any = None,
) -> dict:
    """List the contents of a GitHub repository directory.

    Args:
        owner: The GitHub repository owner (e.g. 'google').
        repo: The repository name (e.g. 'adk-python').
        path: Path within the repo to list. Empty string for root.
        ref: Git ref (branch, tag, or commit SHA). Defaults to 'main'.
        tool_context: ADK ToolContext (injected automatically).

    Returns:
        A dictionary with 'entries' containing name, type, path, and size
        for each item in the directory.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    if ref:
        url += f"?ref={ref}"

    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except error.HTTPError as e:
        return {"error": f"GitHub API error {e.code}: {e.reason}", "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}

    if not isinstance(data, list):
        return {"error": "Path is not a directory", "url": url}

    entries = []
    for item in data:
        entries.append({
            "name": item.get("name"),
            "type": item.get("type"),  # "file" or "dir"
            "path": item.get("path"),
            "size": item.get("size", 0),
        })

    if tool_context is not None:
        listing_text = "\n".join(
            f"{'[dir] ' if e['type'] == 'dir' else ''}{e['path']} ({e['size']}B)"
            for e in entries
        )
        tool_context.state[f"gh_listing:{owner}/{repo}/{path}"] = listing_text

    return {"entries": entries, "count": len(entries)}


def github_download_file(
    owner: str,
    repo: str,
    path: str,
    ref: str = "main",
    tool_context: Any = None,
) -> dict:
    """Download the contents of a single file from a GitHub repository.

    Args:
        owner: The GitHub repository owner.
        repo: The repository name.
        path: Full path to the file within the repo.
        ref: Git ref (branch, tag, or commit SHA). Defaults to 'main'.
        tool_context: ADK ToolContext (injected automatically).

    Returns:
        A dictionary with 'content' (decoded text), 'path', and 'size'.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    if ref:
        url += f"?ref={ref}"

    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except error.HTTPError as e:
        return {"error": f"GitHub API error {e.code}: {e.reason}", "path": path}
    except Exception as e:
        return {"error": str(e), "path": path}

    if data.get("type") != "file":
        return {"error": f"Path '{path}' is not a file", "type": data.get("type")}

    encoding = data.get("encoding", "")
    raw_content = data.get("content", "")

    if encoding == "base64":
        try:
            content = base64.b64decode(raw_content).decode("utf-8", errors="replace")
        except Exception:
            content = raw_content
    else:
        content = raw_content

    # Persist to session state so REPL and other agents can access it
    if tool_context is not None:
        state_key = f"gh_file:{owner}/{repo}/{path}"
        tool_context.state[state_key] = content

    return {
        "content": content,
        "path": path,
        "size": data.get("size", len(content)),
        "sha": data.get("sha", ""),
    }


def github_download_directory(
    owner: str,
    repo: str,
    path: str = "",
    ref: str = "main",
    max_files: int = 50,
    tool_context: Any = None,
) -> dict:
    """Recursively download all files from a GitHub repository directory.

    Walks the directory tree (up to max_files) and stores each file's
    content into session state under 'gh_file:{owner}/{repo}/{filepath}'.

    Args:
        owner: The GitHub repository owner.
        repo: The repository name.
        path: Path within the repo. Empty string for root.
        ref: Git ref. Defaults to 'main'.
        max_files: Maximum number of files to download.
        tool_context: ADK ToolContext (injected automatically).

    Returns:
        A dict with 'files_downloaded' count and 'paths' list.
    """
    downloaded = []
    dirs_to_visit = [path]

    while dirs_to_visit and len(downloaded) < max_files:
        current_dir = dirs_to_visit.pop(0)
        listing = github_list_directory(
            owner=owner, repo=repo, path=current_dir, ref=ref
        )

        if "error" in listing:
            continue

        for entry in listing.get("entries", []):
            if len(downloaded) >= max_files:
                break
            if entry["type"] == "dir":
                dirs_to_visit.append(entry["path"])
            elif entry["type"] == "file":
                result = github_download_file(
                    owner=owner,
                    repo=repo,
                    path=entry["path"],
                    ref=ref,
                    tool_context=tool_context,
                )
                if "error" not in result:
                    downloaded.append(entry["path"])

    return {"files_downloaded": len(downloaded), "paths": downloaded}
