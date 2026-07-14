"""Guards the OpenCV variant, because getting it wrong fails *non-deterministically* on a TPU.

The GUI build (`opencv-python`) links libGL.so.1, which does not exist on a headless host. That on
its own would be a clean, loud failure -- but cv2's bootstrap inserts `site-packages/cv2` into
sys.path BEFORE loading its native module and only restores sys.path afterwards:

    for p in reversed(PYTHON_EXTENSIONS_PATHS):
        sys.path.insert(1, p)          # site-packages/cv2 now sits ahead of the stdlib
    ...
    native_module = importlib.import_module("cv2")   # ImportError: libGL.so.1 -- raises HERE
    ...
    sys.path = save_sys_path                          # never reached

So a *failed* `import cv2` leaves that directory on sys.path permanently. cv2 ships a `typing`
subpackage, so any later multiprocessing **spawn** child -- a data-loader worker -- resolves
`import typing` to `cv2/typing`, which imports cv2, which raises again, and the worker dies during
interpreter bootstrap.

lerobot already requires opencv-python-headless. Requiring the GUI build as well installed BOTH, and
they overwrite each other's `cv2/cv2.abi3.so`, so which one won was decided by uv's parallel install
order on each fresh VM. Hence: sometimes the job trains, sometimes its workers die on startup.
"""

import importlib.metadata as md
import multiprocessing as mp
import sys

import pytest


def _opencv_distributions() -> list[str]:
    return sorted(
        d.metadata["Name"]
        for d in md.distributions()
        if (d.metadata["Name"] or "").lower().startswith("opencv")
    )


def test_only_the_headless_opencv_is_installed():
    """Two OpenCV builds fight over the same cv2/ directory; the winner is install-order roulette."""
    installed = _opencv_distributions()
    assert installed == ["opencv-python-headless"], (
        f"expected only opencv-python-headless, found {installed}. The GUI build needs libGL.so.1, "
        "which a TPU VM does not have."
    )


def test_importing_cv2_does_not_leak_sys_path():
    """The actual defect: a cv2 import that leaves site-packages/cv2 ahead of the stdlib."""
    before = list(sys.path)
    import cv2  # noqa: F401

    leaked = [p for p in sys.path if p not in before]
    assert not leaked, f"cv2 left {leaked} on sys.path; a spawned worker would import cv2/typing as `typing`"


def test_stdlib_typing_is_not_shadowed_by_cv2():
    import cv2  # noqa: F401
    import typing

    assert "site-packages" not in typing.__file__, f"`typing` resolved to {typing.__file__}"


def _child(queue):
    # A spawn child re-imports __main__ through runpy -> pkgutil -> functools, which does
    # `from typing import get_origin, Union` before `typing` is in sys.modules. That is exactly the
    # import that used to land in cv2/typing and take the worker down.
    import typing

    queue.put(typing.__file__)


@pytest.mark.skipif(sys.platform != "linux", reason="the failure is specific to headless Linux hosts")
def test_a_spawned_worker_survives_after_cv2_is_imported():
    import cv2  # noqa: F401

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(queue,))
    proc.start()
    proc.join(120)

    assert proc.exitcode == 0, "a spawned data-loader worker died during interpreter bootstrap"
    assert "site-packages" not in queue.get(timeout=10)
