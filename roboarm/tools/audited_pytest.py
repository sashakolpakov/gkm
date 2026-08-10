"""Run pytest with a process-level project write boundary.

The audit hook rejects Python filesystem mutation attempts outside the roboarm
root and records all observed in-bound writes.  It complements the static path
and symlink tests without depending on the parent repository's dirty status.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

try:
    import fcntl
except ImportError:  # pragma: no cover - available on the target macOS runtime
    fcntl = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_REPORT = PROJECT_ROOT / "artifacts" / "write-audit.json"
NULL_DEVICE = Path(os.devnull).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_OPEN_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_RDWR
    | os.O_APPEND
    | os.O_CREAT
    | os.O_TRUNC
)
if hasattr(os, "O_EXCL"):
    _OPEN_WRITE_FLAGS |= os.O_EXCL


class ProjectWriteAudit:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve(strict=True)
        self.events: list[tuple[str, str]] = []

    @staticmethod
    def _directory_for_fd(dir_fd: Any) -> Path | None:
        if not isinstance(dir_fd, int) or dir_fd < 0:
            return None
        if sys.platform == "darwin" and fcntl is not None:
            try:
                buffer = fcntl.fcntl(dir_fd, 50, b"\0" * 1024)  # F_GETPATH
                value = buffer.split(b"\0", 1)[0]
                if value:
                    return Path(os.fsdecode(value)).resolve(strict=True)
            except (OSError, ValueError):
                pass
        for descriptor_root in (Path("/proc/self/fd"), Path("/dev/fd")):
            descriptor = descriptor_root / str(dir_fd)
            try:
                resolved = descriptor.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if resolved != descriptor:
                return resolved
        raise PermissionError(f"Phase-0 write audit could not resolve dir_fd={dir_fd}")

    def _path(self, value: Any, dir_fd: Any = None) -> Path | None:
        if isinstance(value, int):
            return None
        try:
            path = Path(os.fsdecode(value))
        except (TypeError, ValueError):
            return None
        if not path.is_absolute():
            path = (self._directory_for_fd(dir_fd) or Path.cwd()) / path
        return path.resolve(strict=False)

    def _record(self, event: str, value: Any, dir_fd: Any = None) -> None:
        path = self._path(value, dir_fd)
        if path is None or path == NULL_DEVICE:
            return
        if not path.is_relative_to(self.project_root):
            raise PermissionError(
                "Phase-0 write audit blocked "
                f"{event} outside roboarm: path={path!r}, "
                f"root={self.project_root!r}"
            )
        self.events.append((event, str(path.relative_to(self.project_root))))

    @staticmethod
    def _open_is_write(mode: Any, flags: Any) -> bool:
        if isinstance(mode, str) and any(character in mode for character in "wax+"):
            return True
        return isinstance(flags, int) and bool(flags & _OPEN_WRITE_FLAGS)

    def __call__(self, event: str, args: tuple[Any, ...]) -> None:
        if event == "open":
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            if self._open_is_write(mode, flags):
                self._record(event, args[0])
            return

        if event == "os.chdir" and args:
            path = self._path(args[0])
            if path is not None and not path.is_relative_to(self.project_root):
                raise PermissionError(
                    f"Phase-0 write audit blocked chdir outside roboarm: {path}"
                )
            return

        dir_fd_by_event = {
            "os.chmod": 2,
            "os.mkdir": 2,
            "os.remove": 1,
            "os.rmdir": 1,
            "os.unlink": 1,
            "os.utime": 3,
        }
        if event in dir_fd_by_event and args:
            index = dir_fd_by_event[event]
            dir_fd = args[index] if len(args) > index else None
            self._record(event, args[0], dir_fd)
            return

        if event == "os.truncate" and args:
            self._record(event, args[0])
            return

        if event in {"os.rename", "os.replace"} and len(args) >= 2:
            source_dir_fd = args[2] if len(args) > 2 else None
            destination_dir_fd = args[3] if len(args) > 3 else None
            self._record(f"{event}:source", args[0], source_dir_fd)
            self._record(f"{event}:destination", args[1], destination_dir_fd)
            return

        if event == "os.symlink" and len(args) >= 2:
            dir_fd = args[2] if len(args) > 2 else None
            self._record(event, args[1], dir_fd)
            return

        if event == "os.link" and len(args) >= 2:
            source_dir_fd = args[2] if len(args) > 2 else None
            destination_dir_fd = args[3] if len(args) > 3 else None
            self._record(f"{event}:source", args[0], source_dir_fd)
            self._record(f"{event}:destination", args[1], destination_dir_fd)

    def report(self, exit_code: int) -> dict[str, object]:
        by_event = Counter(event for event, _ in self.events)
        by_top_level = Counter(
            path.partition("/")[0] for _, path in self.events if path
        )
        return {
            "schema_version": 1,
            "project_root": str(self.project_root),
            "outside_writes_blocked": 0,
            "pytest_exit_code": exit_code,
            "observed_write_events": len(self.events),
            "unique_written_paths": sorted({path for _, path in self.events}),
            "events_by_type": dict(sorted(by_event.items())),
            "events_by_top_level": dict(sorted(by_top_level.items())),
        }


def main(arguments: list[str]) -> int:
    audit = ProjectWriteAudit(PROJECT_ROOT)
    sys.addaudithook(audit)
    exit_code = int(pytest.main(arguments))
    AUDIT_REPORT.parent.mkdir(exist_ok=True)
    AUDIT_REPORT.write_text(
        json.dumps(audit.report(exit_code), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
