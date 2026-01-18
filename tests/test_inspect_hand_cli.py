from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def test_inspect_hand_importable():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "inspect_hand.py"
    spec = importlib.util.spec_from_file_location("inspect_hand", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module is not None


def test_inspect_hand_help_includes_new_args():
    script_path = Path(__file__).resolve().parents[1] / "tools" / "inspect_hand.py"
    root_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    python_path = os.pathsep.join(
        [str(root_dir / "src"), str(root_dir), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["PYTHONPATH"] = python_path
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0
    stdout = result.stdout
    assert "--hand-y1-ratio" in stdout
    assert "--hand-y2-ratio" in stdout
