"""Runtime tests with scripted model output and the real MCP HTTP client."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
from uuid import UUID

import httpx
import pytest
from dotenv import load_dotenv as real_load_dotenv

# The inference package is nested beside the separate training package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agentflow"))

from agentflow.engine import factory
from agentflow.models import executor, planner, verifier
from agentflow.models.formatters import (
    MemoryVerification,
    NextStep,
    QueryAnalysis,
    ToolCommand,
)
from agentflow.solver import construct_solver

NAME = "Parallel_Search_Tool"
URL = "https://search.parallel.ai/mcp"
SOURCE = "https://example.org/france"
REAL_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    # load_dotenv 1.0.1 does not honor PYTHON_DOTENV_DISABLED.
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    for key in ("GOOGLE_API_KEY", "PARALLEL_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


class ScriptedModel:
    """Only model output is scripted; loader, dispatcher and memory are real."""

    def __init__(self, commands):
        self.commands = commands
        self.calls = 0

    def __call__(self, prompt, response_format=None, **kwargs):
        if response_format is QueryAnalysis:
            return QueryAnalysis(
                concise_summary="Find evidence",
                required_skills="search",
                relevant_tools=NAME,
                additional_considerations="Read sources",
            )
        if response_format is NextStep:
            return NextStep(
                justification="Find evidence",
                context="",
                sub_goal="Research",
                tool_name=NAME,
            )
        if response_format is ToolCommand:
            command = self.commands[self.calls % len(self.commands)]
            self.calls += 1
            return ToolCommand(
                analysis="Read evidence", explanation="Research", command=command
            )
        if response_format is MemoryVerification:
            return MemoryVerification(
                analysis="Continue until sources are read",
                stop_signal=self.calls % len(self.commands) == 0,
            )
        return "Answer from evidence: " + str(prompt)


def build_solver(monkeypatch, tmp_path, commands=None, **kwargs):
    model = ScriptedModel(
        commands or ['execution = tool.execute(query="France capital")']
    )
    for module in (factory, executor, planner, verifier):
        monkeypatch.setattr(module, "create_llm_engine", lambda **kw: model)
    return construct_solver(
        output_types="direct",
        verbose=False,
        root_cache_dir=str(tmp_path),
        max_steps=4,
        **kwargs,
    )


class MCPServer:
    def __init__(self):
        self.requests = []
        self.mode = "ok"
        self.cancelled = threading.Event()

    @property
    def calls(self):
        return [
            r[1]["params"] for r in self.requests if r[1].get("method") == "tools/call"
        ]

    async def handle(self, request):
        assert str(request.url) == URL
        assert request.headers["user-agent"] == (
            "AgentFlow/0.3.6 (+https://github.com/lupantech/AgentFlow)"
        )
        assert "authorization" not in request.headers
        if request.method != "POST":
            return httpx.Response(405)
        data = json.loads(request.content)
        self.requests.append((request, data))
        method = data["method"]
        if "id" not in data:
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object"}}
                    for name in ("web_search", "web_fetch")
                ]
            }
        else:
            assert method == "tools/call"
            if self.mode == "hang":
                try:
                    await asyncio.sleep(10)
                finally:
                    self.cancelled.set()
            if self.mode == "http_error":
                return httpx.Response(503, text="Service unavailable")
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Anonymous quota exceeded"
                            if self.mode == "quota"
                            else f"Paris is the capital of France. Source: {SOURCE}"
                        ),
                    }
                ],
                "isError": self.mode == "quota",
            }
        return httpx.Response(
            200, json={"jsonrpc": "2.0", "id": data["id"], "result": result}
        )


@pytest.fixture
def server(monkeypatch):
    server = MCPServer()

    class Client(REAL_CLIENT):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(server.handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    return server


def dispatch(solver, command):
    solver.executor.set_query_cache_dir(solver.root_cache_dir)
    return solver.executor.execute_tool_command(NAME, command)


def test_unconfigured_default_to_memory_and_answer(monkeypatch, tmp_path, server):
    solver = build_solver(monkeypatch, tmp_path)
    assert NAME in solver.planner.available_tools
    assert "Ground_Google_Search_Tool" not in solver.planner.available_tools
    output = solver.solve("What is the capital of France?")
    assert SOURCE in output["direct_output"]
    assert SOURCE in output["memory"]["Action Step 1"]["result"][0]
    assert server.calls[0]["name"] == "web_search"
    assert server.calls[0]["arguments"]["search_queries"] == ["France capital"]
    assert {data["method"] for _, data in server.requests} >= {
        "initialize",
        "tools/list",
        "tools/call",
    }
    for request, _ in server.requests:
        assert str(request.url) == URL
        assert (
            request.headers["user-agent"]
            == "AgentFlow/0.3.6 (+https://github.com/lupantech/AgentFlow)"
        )
        assert "authorization" not in request.headers


def test_configured_google_and_explicit_choices(monkeypatch, tmp_path, server):
    monkeypatch.setenv("GOOGLE_API_KEY", "fixture-google-key")
    solver = build_solver(monkeypatch, tmp_path)
    assert "Ground_Google_Search_Tool" in solver.planner.available_tools
    assert NAME not in solver.planner.available_tools
    assert "not enabled" in dispatch(solver, 'execution = tool.execute(query="France")')
    explicit = build_solver(
        monkeypatch, tmp_path, enabled_tools=[NAME], tool_engine=["self"]
    )
    assert explicit.planner.available_tools == [NAME]
    assert explicit.executor.tool_instances_cache[NAME].model_string == "gpt-4o"
    disabled = build_solver(monkeypatch, tmp_path, enabled_tools=[])
    assert disabled.planner.available_tools == []
    assert "not enabled" in dispatch(
        disabled, 'execution = tool.execute(query="France")'
    )
    assert not server.calls
    all_tools = build_solver(monkeypatch, tmp_path, enabled_tools=["all"])
    assert NAME in all_tools.planner.available_tools
    assert "Ground_Google_Search_Tool" in all_tools.planner.available_tools


def test_explicit_google_failure_does_not_fallback(monkeypatch, tmp_path, server):
    solver = build_solver(monkeypatch, tmp_path, enabled_tools=["Google_Search_Tool"])
    assert NAME not in solver.planner.available_tools
    assert (
        not solver.planner.available_tools
    )  # Missing key, explicit selection stays unavailable.
    assert not server.calls


def test_saved_google_credentials(monkeypatch, tmp_path, server):
    import dotenv

    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=saved-google-key\n")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: real_load_dotenv(env))
    solver = build_solver(monkeypatch, tmp_path)
    assert "Ground_Google_Search_Tool" in solver.planner.available_tools
    assert NAME not in solver.planner.available_tools
    assert not server.calls


def test_session_lifetime_and_fetch(monkeypatch, tmp_path, server):
    commands = [
        'execution = tool.execute(query="France capital")',
        f'execution = tool.execute(query="France capital", urls=["{SOURCE}"], search_queries=["France capital"])',
        'execution = tool.execute(query="Paris population")',
    ]
    solver = build_solver(monkeypatch, tmp_path, commands, enabled_tools=[NAME])
    solver.solve("First question")
    first = [c["arguments"]["session_id"] for c in server.calls]
    assert len(first) == 3 and len(set(first)) == 1
    UUID(first[0])
    assert server.calls[1]["name"] == "web_fetch"
    assert server.calls[1]["arguments"]["urls"] == [SOURCE]
    assert server.calls[1]["arguments"]["search_queries"] == ["France capital"]
    server.mode = "quota"
    solver.solve("New solve uses a new session even on quota errors")
    second = [c["arguments"]["session_id"] for c in server.calls[3:]]
    assert len(set(second)) == 1 and second[0] != first[0]
    assert "quota" in solver.memory.get_actions()["Action Step 1"]["result"][0]


def test_concurrent_solves_share_tool_without_sharing_id(monkeypatch, tmp_path, server):
    def commands(topic):
        return [
            f'execution = tool.execute(query="{topic}")',
            f'execution = tool.execute(query="{topic}", urls=["{SOURCE}"])',
            f'execution = tool.execute(query="{topic}")',
        ]

    a = build_solver(
        monkeypatch, tmp_path / "a", commands("alpha"), enabled_tools=[NAME]
    )
    b = build_solver(
        monkeypatch, tmp_path / "b", commands("beta"), enabled_tools=[NAME]
    )
    b.executor.tool_instances_cache = a.executor.tool_instances_cache
    barrier = threading.Barrier(2)

    async def handle(request):
        if (
            request.method == "POST"
            and json.loads(request.content).get("method") == "tools/call"
        ):
            await asyncio.to_thread(barrier.wait, 5)
        return await server.handle(request)

    class Client(REAL_CLIENT):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda s: s.solve("Find France"), [a, b]))
    assert all(SOURCE in result["direct_output"] for result in results)
    assert len({c["arguments"]["session_id"] for c in server.calls}) == 2
    for topic in ("alpha", "beta"):
        calls = [c for c in server.calls if c["arguments"]["objective"] == topic]
        assert len(calls) == 3
        assert len({c["arguments"]["session_id"] for c in calls}) == 1


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("quota", "MCP error: Anonymous quota"),
        ("http_error", "request failed"),
        ("hang", "timed out"),
    ],
)
def test_dispatch_errors_and_deadline(monkeypatch, tmp_path, server, mode, expected):
    solver = build_solver(monkeypatch, tmp_path, enabled_tools=[NAME])
    server.mode = mode
    solver.executor.tool_instances_cache[NAME].timeout = 0.15
    result = dispatch(solver, 'execution = tool.execute(query="France")')
    assert expected in result[0]
    if mode == "hang":
        assert server.cancelled.wait(1)


def test_executor_timeout_cancels_mcp(monkeypatch, tmp_path, server):
    solver = build_solver(monkeypatch, tmp_path, enabled_tools=[NAME])
    solver.executor.max_time = 0.2
    server.mode = "hang"
    assert (
        "Execution timed out"
        in dispatch(solver, 'execution = tool.execute(query="France")')[0]
    )
    assert server.cancelled.wait(1)


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "q", "urls": []},
        {"query": "q", "urls": ["file:///etc/passwd"]},
        {"query": "q", "urls": ["https://user:pass@example.org"]},
        {"query": "q", "urls": [None]},
        {"query": "q", "urls": ["https://example.org:bad/path"]},
        {"query": "q", "urls": [SOURCE] * 21},
        {"query": "q", "search_queries": []},
        {"query": "q" * 201, "urls": [SOURCE]},
    ],
)
def test_invalid_input_never_requests(monkeypatch, tmp_path, server, arguments):
    solver = build_solver(monkeypatch, tmp_path, enabled_tools=[NAME])
    result = dispatch(solver, f"execution = tool.execute(**{arguments!r})")
    assert "input error" in result[0]
    assert not server.requests


def test_precancelled_request(monkeypatch, tmp_path, server):
    solver = build_solver(monkeypatch, tmp_path, enabled_tools=[NAME])
    from agentflow.tools.parallel_search.context import cancel_event

    event = threading.Event()
    event.set()
    token = cancel_event.set(event)
    try:
        tool = solver.executor.tool_instances_cache[NAME]
        assert "cancelled" in tool.execute("France")
    finally:
        cancel_event.reset(token)
    assert not server.requests
