"""Discovery Agent - Main entrypoint with ADK App wrapper.

Implements the Recursive Language Model (RLM) architecture using Google ADK:

Architecture (mapping RLM-minimal → ADK):
  - RLM's recursive loop      → LoopAgent with max_iterations
  - RLM's REPL environment    → REPLExecutorAgent (custom BaseAgent / EFSM)
  - RLM's sub-LM calls        → ParallelAgent for batched reducer queries
  - RLM's state management    → Session state key-value pairs
  - RLM's FINAL output        → output_key on the PR writer agent

The agent is wrapped in an App class to enable `adk run` and `adk web` CLI
functionality. The DebugLoggingPlugin is included for full observability.

Usage:
  adk run contributing/samples/discovery_agent
"""

from google.adk.agents import (
    LlmAgent,
    LoopAgent,
    ParallelAgent,
    SequentialAgent,
)
from google.adk.apps import App
from google.adk.plugins import DebugLoggingPlugin

from .artifact_tools import (
    save_discovery_artifact,
    save_jira_ticket,
    save_pr_document,
    save_repl_submission,
)
from .github_tools import (
    github_download_directory,
    github_download_file,
    github_list_directory,
)
from .prompts import (
    EXPLORER_INSTRUCTION,
    INGEST_INSTRUCTION,
    ORCHESTRATOR_INSTRUCTION,
    PLANNER_INSTRUCTION,
    PR_WRITER_INSTRUCTION,
    REDUCER_INSTRUCTION,
)
from .repl_executor_agent import REPLExecutorAgent

# ── Model configuration ─────────────────────────────────────────────────────
MODEL = "gemini-2.0-flash"

# ── Shared tool sets ─────────────────────────────────────────────────────────
GITHUB_TOOLS = [github_list_directory, github_download_file, github_download_directory]
ARTIFACT_TOOLS = [save_jira_ticket, save_repl_submission, save_pr_document, save_discovery_artifact]

# ── REPL Executor (EFSM custom agent) ───────────────────────────────────────
# This deterministic agent reads code from state, runs a security check,
# executes it, and writes results back to state. No LLM involved.
repl_executor = REPLExecutorAgent(
    name="repl_executor",
    description="Deterministic EFSM agent that executes Python code from session state.",
)

# ── Ingest Agent ─────────────────────────────────────────────────────────────
# Parses user request, extracts Jira ticket and repo info, saves artifacts.
ingest_agent = LlmAgent(
    name="ingest_agent",
    description="Parses user input to extract Jira ticket and GitHub repo coordinates.",
    instruction=INGEST_INSTRUCTION,
    model=MODEL,
    tools=[save_jira_ticket, save_discovery_artifact],
    output_key="ingest_result",
)

# ── Explorer Agent ───────────────────────────────────────────────────────────
# Uses GitHub tools and REPL to investigate the codebase.
explorer_agent = LlmAgent(
    name="explorer_agent",
    description="Explores a GitHub codebase using GitHub API tools and REPL execution.",
    instruction=EXPLORER_INSTRUCTION,
    model=MODEL,
    tools=GITHUB_TOOLS + [save_discovery_artifact, save_repl_submission],
    output_key="exploration_summary",
)

# ── Exploration Loop ─────────────────────────────────────────────────────────
# The RLM pattern: the explorer submits code → the REPL executes it →
# explorer reads results → repeats. This is the core recursive loop.
# SequentialAgent inside a LoopAgent creates the iterative REPL cycle.
exploration_step = SequentialAgent(
    name="exploration_step",
    description="One iteration: explorer submits code, REPL executes, explorer reads results.",
    sub_agents=[explorer_agent, repl_executor],
)

exploration_loop = LoopAgent(
    name="exploration_loop",
    description="Iteratively explore the codebase via REPL (RLM recursive loop).",
    max_iterations=5,
    sub_agents=[exploration_step],
)

# ── Parallel Reducer Agents ──────────────────────────────────────────────────
# Like RLM's sub-LM batched queries: multiple reducers analyze different
# aspects of the codebase in parallel, then their outputs merge via state.
reducer_files_agent = LlmAgent(
    name="reducer_files",
    description="Identifies the minimal set of files relevant to the Jira ticket.",
    instruction=(
        REDUCER_INSTRUCTION
        + "\n\nFocus specifically on identifying which FILES need to be modified. "
        "Write your file list to state['reduced_files']."
    ),
    model=MODEL,
    tools=[save_discovery_artifact],
    output_key="reduced_files",
)

reducer_interfaces_agent = LlmAgent(
    name="reducer_interfaces",
    description="Identifies APIs, interfaces, and dependencies relevant to the ticket.",
    instruction=(
        REDUCER_INSTRUCTION
        + "\n\nFocus specifically on identifying INTERFACES, APIs, and "
        "DEPENDENCIES between modules. Write your analysis to state['reduced_interfaces']."
    ),
    model=MODEL,
    tools=[save_discovery_artifact],
    output_key="reduced_interfaces",
)

parallel_reducer = ParallelAgent(
    name="parallel_reducer",
    description="Parallel batched reduction: files + interfaces analyzed concurrently.",
    sub_agents=[reducer_files_agent, reducer_interfaces_agent],
)

# ── Merge Reducer Agent ──────────────────────────────────────────────────────
# After parallel reduction, merge the results into a single reduced spec.
merge_reducer_agent = LlmAgent(
    name="merge_reducer",
    description="Merges parallel reducer outputs into a single reduced specification.",
    instruction=(
        "You merge the outputs of parallel reducer agents into one cohesive specification.\n\n"
        "Read from session state:\n"
        "- state['reduced_files']: List of files to modify.\n"
        "- state['reduced_interfaces']: APIs and dependency analysis.\n"
        "- state['exploration_summary']: Broader codebase context.\n\n"
        "Combine these into a single, comprehensive reduced specification and "
        "write it to state['reduced_spec']. Save it as an artifact with "
        "filename 'reduced_spec' using save_discovery_artifact."
    ),
    model=MODEL,
    tools=[save_discovery_artifact],
    output_key="reduced_spec",
)

# ── Reduction pipeline: parallel reduce → merge ─────────────────────────────
reduction_pipeline = SequentialAgent(
    name="reduction_pipeline",
    description="Reduce codebase: parallel analysis then merge.",
    sub_agents=[parallel_reducer, merge_reducer_agent],
)

# ── Planner Agent ────────────────────────────────────────────────────────────
planner_agent = LlmAgent(
    name="planner_agent",
    description="Creates a detailed implementation plan from the reduced spec.",
    instruction=PLANNER_INSTRUCTION,
    model=MODEL,
    tools=[save_discovery_artifact],
    output_key="implementation_plan",
)

# ── PR Writer Agent ──────────────────────────────────────────────────────────
pr_writer_agent = LlmAgent(
    name="pr_writer_agent",
    description="Generates complete pull request documentation.",
    instruction=PR_WRITER_INSTRUCTION,
    model=MODEL,
    tools=[save_pr_document, save_discovery_artifact],
    output_key="pr_document",
)

# ── Root Agent: Full Discovery Pipeline ──────────────────────────────────────
# Orchestrates the complete RLM pipeline:
#   Ingest → Explore (loop) → Reduce (parallel) → Plan → Write PR
root_agent = SequentialAgent(
    name="discovery_agent",
    description=(
        "Recursive Language Model agent that explores a brownfield codebase, "
        "reduces it to essential files, plans a Jira ticket implementation, "
        "and outputs a pull request document."
    ),
    sub_agents=[
        ingest_agent,
        exploration_loop,
        reduction_pipeline,
        planner_agent,
        pr_writer_agent,
    ],
)

# ── App wrapper (enables `adk run` / `adk web`) ─────────────────────────────
app = App(
    name="discovery_agent",
    root_agent=root_agent,
    plugins=[
        DebugLoggingPlugin(
            output_path="discovery_agent_debug.yaml",
            include_session_state=True,
            include_system_instruction=True,
        ),
    ],
)
