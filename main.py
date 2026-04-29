#!/usr/bin/env python3
"""Dock Analytics — entry point."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _resolve_pyqt5_plugins_dir() -> Path | None:
    """PyQt5 wheel/conda layout: .../site-packages/PyQt5/Qt5/plugins."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    lib = Path(sys.prefix) / "lib"
    if lib.is_dir():
        for child in sorted(lib.iterdir()):
            if not child.is_dir() or not child.name.startswith("python"):
                continue
            cand = child / "site-packages" / "PyQt5" / "Qt5" / "plugins"
            if cand.is_dir():
                rp = cand.resolve()
                if rp not in seen:
                    seen.add(rp)
                    ordered.append(rp)
    try:
        for sp in site.getsitepackages():
            cand = Path(sp) / "PyQt5" / "Qt5" / "plugins"
            if cand.is_dir():
                rp = cand.resolve()
                if rp not in seen:
                    seen.add(rp)
                    ordered.append(rp)
    except Exception:
        pass
    return ordered[0] if ordered else None


def _align_qt_platform_plugins_with_pyqt5() -> None:
    """Avoid system Qt platform plugins with Conda PyQt5 (fixes xcb 'found but not loaded')."""
    target = _resolve_pyqt5_plugins_dir()
    if target is None:
        return
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(target)


def main():
    _align_qt_platform_plugins_with_pyqt5()

    from app.ui_main import run_app

    run_app()


if __name__ == "__main__":
    main()
