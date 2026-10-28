"""Managed child processes with clean termination. Arguments are always lists (never a shell)."""
from __future__ import annotations

import logging
import subprocess
import threading
from collections import deque

from app.core.security import redact_text

log = logging.getLogger(__name__)
_REGISTRY: "set[ManagedProcess]" = set()
_REG_LOCK = threading.Lock()


class ManagedProcess:
    def __init__(self, args: list[str], name: str, stdin: bool = False, stdout: bool = True):
        if not all(isinstance(a, str) for a in args):
            raise TypeError("subprocess args must be strings")
        self.args, self.name = args, name
        self._want_stdin, self._want_stdout = stdin, stdout
        self.proc: subprocess.Popen | None = None
        self.stderr_tail: deque[str] = deque(maxlen=20)
        self._err_thread: threading.Thread | None = None

    def start(self) -> "ManagedProcess":
        self.proc = subprocess.Popen(
            self.args,
            stdin=subprocess.PIPE if self._want_stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE if self._want_stdout else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._err_thread = threading.Thread(target=self._drain_stderr, name=f"{self.name}-stderr", daemon=True)
        self._err_thread.start()
        with _REG_LOCK:
            _REGISTRY.add(self)
        return self

    def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for raw in iter(self.proc.stderr.readline, b""):
            self.stderr_tail.append(redact_text(raw.decode(errors="replace").rstrip()))

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def last_error(self) -> str:
        return " | ".join(list(self.stderr_tail)[-3:])

    def stop(self, timeout: float = 3.0) -> None:
        p = self.proc
        if p is None:
            return
        try:
            if p.stdin:
                try:
                    p.stdin.close()
                except OSError:
                    pass
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout)
                except subprocess.TimeoutExpired:
                    log.warning("process %s ignored SIGTERM; killing", self.name)
                    p.kill()
                    p.wait(timeout)
        finally:
            for s in (p.stdout, p.stderr):
                try:
                    if s:
                        s.close()
                except OSError:
                    pass
            with _REG_LOCK:
                _REGISTRY.discard(self)


def terminate_all(timeout: float = 3.0) -> None:
    with _REG_LOCK:
        procs = list(_REGISTRY)
    for p in procs:
        p.stop(timeout)
