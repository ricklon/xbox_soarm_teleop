"""The workshop notebook is generated; keep the file and its builder in step."""

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_workshop_notebook", ROOT / "scripts" / "build_workshop_notebook.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _code_cells() -> list[str]:
    notebook = json.loads((ROOT / "notebooks" / "workshop_agent.ipynb").read_text())
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_notebook_file_matches_its_builder():
    built = _builder().build()
    on_disk = json.loads((ROOT / "notebooks" / "workshop_agent.ipynb").read_text())
    assert on_disk == built, "run: uv run python scripts/build_workshop_notebook.py"


def test_every_code_cell_is_valid_python():
    for cell in _code_cells():
        lines = [line for line in cell.splitlines() if not line.startswith("%")]  # notebook magics
        compile("\n".join(lines), "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


def test_both_agent_options_define_the_same_function():
    agents = [cell for cell in _code_cells() if "async def ask_agent(" in cell]
    assert len(agents) == 2
    claude, other = agents
    assert "AsyncAnthropic" in claude and "anthropic" not in other.lower()
    assert "/chat/completions" in other and "MODEL_BASE_URL" in other
    for cell in agents:
        assert "task: str, max_steps: int = 30" in cell


def test_the_arm_key_is_typed_into_a_hidden_prompt():
    setup = next(cell for cell in _code_cells() if "class connect" in cell)
    assert 'getpass("Your arm' in setup
    assert not any("PASTE YOUR KEY" in cell for cell in _code_cells())
