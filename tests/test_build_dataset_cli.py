from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_build_dataset_help_includes_hand_args() -> None:
    script_path = Path(__file__).resolve().parents[1] / "tools" / "build_dataset.py"
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
    assert "--with-hand" in stdout
    assert "--hand-template-size" in stdout
    assert "--hand-roi-x1" in stdout
