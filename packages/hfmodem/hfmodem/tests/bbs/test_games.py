# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json
import os
from pathlib import Path
import struct
import sys
import time

import pytest

from hfmodem.bbs.games.door import InputProfile, ProcessAdapter
from hfmodem.bbs.games.zmachine import Checkpoints
from hfmodem.bbs.store import OwnershipError, WorldLock


def collect(adapter, kind, timeout=5):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        batch = adapter.poll()
        events.extend(batch)
        if any(event.kind == kind for event in batch):
            return events
        time.sleep(.005)
    pytest.fail(f"no {kind}: {events}")


def fixture_process(tmp_path, source, profile=None):
    script = tmp_path / "fixture.py"
    script.write_text(source)
    adapter = ProcessAdapter([str(Path(sys.executable).resolve()), "-u", str(script)], profile=profile)
    adapter.start("opaque", tmp_path / "world")
    return adapter


def test_opaque_input_contract_and_native_eof(tmp_path):
    adapter = fixture_process(tmp_path, "import sys\nsys.stdout.write('Prompt without newline>')\nsys.stdout.flush()\nfor line in sys.stdin:\n print('got:'+repr(line),flush=True)\n if line.strip()=='quit': break\n")
    try:
        first = collect(adapter, "prompt")
        assert first[-1].mode == "unknown"
        assert not adapter.submit("one\ntwo")
        assert not adapter.submit("\x1b")
        assert not adapter.submit("x" * 257)
        assert adapter.submit("north")
        collect(adapter, "prompt")
        assert adapter.submit("quit")
        result = collect(adapter, "exited")
        assert result[-1].outcome == "eof"
        assert not adapter.alive
        assert not adapter.submit("again")
    finally:
        adapter.close("test")


def test_fixed_key_profile_does_not_append_enter(tmp_path):
    adapter = fixture_process(tmp_path, "import os\nprint(repr(os.read(0,1)),flush=True)\n", InputProfile(mode="key"))
    collect(adapter, "prompt")
    assert not adapter.submit("AB", "key")
    assert adapter.submit("F", "key")
    result = collect(adapter, "exited")
    assert "b'F'" in "".join(event.text for event in result)


def test_cp437_output_and_pause_profile(tmp_path):
    adapter = fixture_process(tmp_path, "import os\nos.write(1,b'\\xb3')\nassert os.read(0,1)==b'\\r'\n", InputProfile(mode="pause", encoding="cp437", terminator=b"\r"))
    first = collect(adapter, "prompt")
    assert not adapter.submit("x", "pause")
    assert adapter.submit("", "pause")
    result = collect(adapter, "exited")
    assert "│" in "".join(event.text for event in first + result)


def test_output_flood_is_stopped_with_lifecycle_event(tmp_path):
    adapter = fixture_process(tmp_path, "import os\nwhile True: os.write(1,b'x'*4096)\n", InputProfile(max_output_bytes=1024))
    result = collect(adapter, "exited")
    assert result[-1].outcome == "output_limit"
    assert sum(len(event.text) for event in result) <= 1024


def test_world_lock_contention_is_not_a_second_writer(tmp_path):
    world = tmp_path / "world"
    with WorldLock(world / ".world.lock"):
        adapter = fixture_process(tmp_path, "raise RuntimeError('must never spawn')")
        result = collect(adapter, "exited")
        assert result[-1].outcome == "error"


def test_close_is_nonblocking_and_reaps_group(tmp_path):
    adapter = fixture_process(tmp_path, "import time\ntime.sleep(60)\n")
    collect(adapter, "prompt")
    adapter.close("forced disconnect")
    result = collect(adapter, "exited")
    assert result[-1].outcome == "closed"
    with WorldLock(tmp_path / "world/.world.lock"):
        pass


def quetzal(value=b"state"):
    def chunk(name, data):
        return name + struct.pack(">I", len(data)) + data + (b"\0" if len(data) % 2 else b"")
    body = b"IFZS" + chunk(b"IFhd", b"1234567890123") + chunk(b"Stks", b"") + chunk(b"CMem", value)
    return b"FORM" + struct.pack(">I", len(body)) + body


def test_complete_generation_publication_keeps_previous(tmp_path):
    manager = Checkpoints(tmp_path, {"story": "one", "runtime": "v1"})
    source = tmp_path / "pending"
    source.write_bytes(quetzal())
    first = manager.publish(source)
    source.write_bytes(quetzal(b"next"))
    second = manager.publish(source)
    selected, path = manager.selected()
    assert selected == second
    assert path.read_bytes() == quetzal(b"next")
    assert json.loads((manager.root / "manifest.json").read_text()) == {"current": second, "previous": first}
    assert (manager.root / first / "payload.qzl").exists()


@pytest.mark.parametrize("step", ["payload.qzl", "metadata.json", "manifest"])
def test_interrupted_checkpoint_does_not_select_partial_generation(tmp_path, monkeypatch, step):
    import hfmodem.bbs.games.zmachine as module
    manager = Checkpoints(tmp_path, {"story": "one"})
    source = tmp_path / "pending"
    source.write_bytes(quetzal())
    first = manager.publish(source)
    original = module._write
    def fail(path, data):
        if step in path.name:
            raise OSError("full disk")
        original(path, data)
    monkeypatch.setattr(module, "_write", fail)
    with pytest.raises(OSError):
        manager.publish(source)
    assert manager.selected()[0] == first


def test_changed_story_or_corrupt_payload_never_restores(tmp_path):
    manager = Checkpoints(tmp_path, {"story": "one"})
    source = tmp_path / "pending"
    source.write_bytes(quetzal())
    manager.publish(source)
    with pytest.raises(ValueError, match="another runtime"):
        Checkpoints(tmp_path, {"story": "two"}).selected()
    _, path = manager.selected()
    path.write_bytes(quetzal(b"corrupt"))
    with pytest.raises(ValueError, match="hash"):
        manager.selected()


def test_symlink_checkpoint_and_invalid_quetzal_rejected(tmp_path):
    manager = Checkpoints(tmp_path, {})
    source = tmp_path / "pending"
    source.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError):
        manager.publish(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    manager.root.symlink_to(outside, target_is_directory=True)
    source.write_bytes(quetzal())
    with pytest.raises(ValueError, match="root"):
        manager.publish(source)


def test_lock_survives_owner_process_exit(tmp_path):
    """A direct subprocess retains an inherited writer lock after owner dies."""
    import subprocess
    marker = tmp_path / "pid"
    lock = tmp_path / ".world.lock"
    owner = tmp_path / "owner.py"
    owner.write_text("from hfmodem.bbs.store import WorldLock\nimport subprocess,sys\nfrom pathlib import Path\nlock=WorldLock(Path(sys.argv[1]))\nchild=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],pass_fds=(lock.fileno(),))\nPath(sys.argv[2]).write_text(str(child.pid))\n")
    parent = subprocess.run([sys.executable, str(owner), str(lock), str(marker)], check=True)
    assert parent.returncode == 0
    child_pid = int(marker.read_text())
    try:
        with pytest.raises(OwnershipError):
            WorldLock(lock)
    finally:
        import signal
        os.kill(child_pid, signal.SIGKILL)


def test_cancelled_prepare_cannot_spawn_backend(tmp_path):
    import threading
    entered, release = threading.Event(), threading.Event()
    sentinel = tmp_path / "must-not-exist"
    class SlowPrepare(ProcessAdapter):
        def _prepare(self):
            entered.set()
            assert release.wait(3)
    adapter = SlowPrepare([str(Path(sys.executable).resolve()), "-c", f"from pathlib import Path;Path({str(sentinel)!r}).touch()"])
    adapter.start("opaque", tmp_path / "world")
    assert entered.wait(3)
    adapter.close("cancel startup")
    release.set()
    collect(adapter, "exited")
    assert not sentinel.exists()


def test_structured_prompt_barrier_drains_stdout_first(tmp_path, monkeypatch):
    import selectors
    original = selectors.DefaultSelector
    class ControlFirst:
        def __init__(self): self.inner = original()
        def register(self, *args): return self.inner.register(*args)
        def unregister(self, *args): return self.inner.unregister(*args)
        def get_map(self): return self.inner.get_map()
        def close(self): self.inner.close()
        def select(self, timeout=None):
            return sorted(self.inner.select(timeout), key=lambda pair: pair[0].data != "err")
    monkeypatch.setattr(selectors, "DefaultSelector", ControlFirst)
    script = tmp_path / "fixture.py"
    script.write_text("import os,time\nos.write(1,b'previous output')\nos.write(2,b'INPUT\\n')\ntime.sleep(.1)\n")
    class Structured(ProcessAdapter):
        def _control(self, line):
            if line == "INPUT": self._prompt("line")
    adapter = Structured([str(Path(sys.executable).resolve()), "-u", str(script)], profile=InputProfile(structured=True))
    adapter.start("opaque", tmp_path / "world")
    result = collect(adapter, "exited")
    text_index = next(i for i,e in enumerate(result) if e.kind == "text")
    prompt_index = next(i for i,e in enumerate(result) if e.kind == "prompt")
    assert text_index < prompt_index


def test_explicit_line_does_not_override_profile_unless_requested(tmp_path):
    adapter = fixture_process(tmp_path, "import os\nprint(repr(os.read(0,2)),flush=True)\n", InputProfile(mode="key"))
    collect(adapter, "prompt")
    assert adapter.submit("F", "line")
    result = collect(adapter, "exited")
    assert "b'F\\n'" in "".join(event.text for event in result)


def test_cleanup_kills_term_ignoring_descendant_after_leader_exits(tmp_path):
    child_ready = tmp_path / "descendant-ready"
    child_source = (
        "import os,signal,sys,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(.01)\n"
    )
    parent_source = (
        "import subprocess,sys\n"
        f"subprocess.Popen([sys.executable,'-u','-c',{child_source!r},{str(child_ready)!r}],"
        "close_fds=False,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "sys.stdin.readline()\n"
    )
    adapter = fixture_process(tmp_path, parent_source)
    collect(adapter, "prompt")
    deadline = time.monotonic() + 3
    while not child_ready.exists() and time.monotonic() < deadline:
        time.sleep(.005)
    assert child_ready.exists()
    adapter.close("disconnect")
    collect(adapter, "exited")
    # Descendant inherited the same lock and never voluntarily releases it.
    # Acquiring it proves the writer exited, even if its PID is briefly a zombie.
    deadline = time.monotonic() + 3
    while True:
        try:
            with WorldLock(tmp_path / "world/.world.lock"):
                break
        except OwnershipError:
            if time.monotonic() >= deadline:
                pytest.fail("descendant retained writer ownership after exited")
            time.sleep(.005)
