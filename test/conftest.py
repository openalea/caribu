"""The legacy calls libcaribu's executables by name: the Python's own bin directory put on PATH, so that they are
found when the tests run from the environment's Python without activating it."""
import os
import sys
from pathlib import Path

os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
