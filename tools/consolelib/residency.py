# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""WHAT THE INTERPRETER LOADED, which is not what the filesystem holds.

A manifest checks files on disk, and a file on disk is not proof of an import.
The venv carries PEP 660 editable installs whose generated finder ends with
`sys.meta_path.append(_EditableFinder)`. Because it appends, the default path
finder -- which reads `sys.path`, and therefore `PYTHONPATH` -- is consulted
first, and a snapshot named on `PYTHONPATH` wins. That is the whole of what
binds a frozen-snapshot arm to its snapshot, and it is a packaging backend's
choice in a generated file rather than anything this tree states. Written
`insert(0, ...)` instead, every frozen arm ever flown would have been the working
tree, and every manifest would still have verified.

So the question is put to the interpreter rather than to the directory. The
launcher is a subprocess and the console cannot read its `sys.modules`, so the
same interpreter is started under the same environment and the same working
directory, the arm's entry points are imported -- imported, never run; each one
guards its `main` behind `__name__ == "__main__"` -- and its `sys.modules` is
walked and sent back.

Two readings are taken, one before the lock and one at teardown, because the
count of runtime files differing from the staged candidate has moved eight times
inside a single day's flying. A reading that changed under a flying arm is a
reading about no single artifact.

An ordinary arm records what came back and flies. `--qualify` refuses instead.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import station

#: The distributions a release is about. `hfmodem.core` and the bare `hfmodem`
#: both count; `hfmodemish` does not.
CANDIDATE = ('hfmodem', 'hfhost', 'creance')

PROBE_TIMEOUT_S = 180

PROBE = """
import importlib.util, json, os, sys
for entry in sys.argv[1:]:
    if entry.endswith('.py'):
        sys.path.insert(0, os.path.dirname(entry))
        spec = importlib.util.spec_from_file_location('__arm_entry__', entry)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    else:
        importlib.import_module(entry)
json.dump({name: getattr(module, '__file__', None)
           for name, module in sys.modules.items()}, sys.stdout)
"""


def ours(name):
    return any(name == root or name.startswith(root + '.') for root in CANDIDATE)


def digest(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def resident(paths):
    """Every candidate module in a name -> `__file__` mapping, with its digest.

    A module with no `__file__` is kept rather than dropped: a candidate package
    that resolved as a namespace is split across trees, which is the thing this
    is here to catch.
    """
    return {name: dict(file=path, sha256=digest(path) if path else None)
            for name, path in sorted(paths.items()) if ours(name)}


def imported(modules=None):
    """The same reading taken by a process of its own imports."""
    return resident({name: getattr(module, '__file__', None)
                     for name, module in (modules or sys.modules).items()})


def set_sha256(modules):
    """One digest over what was loaded, spelled the way a manifest spells one."""
    return hashlib.sha256(json.dumps({name: found['sha256']
                                      for name, found in modules.items()},
                                     sort_keys=True).encode()).hexdigest()


def label(modules):
    """The line an instrument's results carry to say which artifact produced them."""
    if not modules:
        return 'no candidate module loaded'
    found = [found['file'] for found in modules.values() if found['file']]
    return (f'{len(modules)} candidate module(s) under '
            f'{os.path.commonpath(found) if found else "no file on disk"}, '
            f'set sha256 {set_sha256(modules)[:12]}')


def probe(entries, env, python=None):
    """Ask the arm's interpreter, in the arm's environment, what it resolves.

    `-c` puts the working directory first on `sys.path` exactly as `-m` does, so
    the search this measures is the one `onair.sh` hands the launcher rather than
    a near neighbour of it.
    """
    entries = [str(entry) for entry in entries]
    done = subprocess.run([str(python or station.VENV), '-c', PROBE, *entries],
                          cwd=station.ROOT, env=env, timeout=PROBE_TIMEOUT_S,
                          capture_output=True, text=True)
    if done.returncode:
        said = done.stderr.strip().splitlines()
        raise RuntimeError(f'the interpreter could not import {", ".join(entries)}: '
                           + (said[-1] if said else f'exit {done.returncode}'))
    return resident(json.loads(done.stdout))


def snapshot_root(manifest, named):
    """Where `freeze_v32.py` puts the files a manifest describes."""
    return (Path(named).parent / 'snapshots' / manifest['variant']).resolve()


def against(modules, named):
    """Why this residency is not the named snapshot's, or None if it is.

    Both halves matter and they fail differently: a module that resolved outside
    the snapshot root is the wrong tree, and one inside it that hashes wrong is a
    snapshot somebody wrote to.
    """
    manifest = json.loads(Path(named).read_text())
    root = snapshot_root(manifest, named)
    strayed, altered = [], []
    for name, found in sorted(modules.items()):
        here = Path(found['file']).resolve() if found['file'] else None
        if here is None or not here.is_relative_to(root):
            strayed.append(f'{name} <- {here or "no file; a namespace package"}')
        elif manifest['files'].get(here.relative_to(root).as_posix()) != found['sha256']:
            altered.append(name)
    said = []
    if strayed:
        said.append(f'{len(strayed)} module(s) resolved outside {root}: '
                    + ', '.join(strayed[:4]) + (', ...' if len(strayed) > 4 else ''))
    if altered:
        said.append(f'{len(altered)} module(s) under the snapshot do not match '
                    f'{Path(named).name}: ' + ', '.join(altered[:4])
                    + (', ...' if len(altered) > 4 else ''))
    if not modules:
        said.append(f'nothing candidate was imported at all, which is not a '
                    f'residency in {root}')
    return '; '.join(said) or None


def evidence(mode, plan, named=None):
    """One residency reading for an arm, and what it is worth against a snapshot.

    A probe that fails is recorded and never refuses an ordinary arm -- the
    operator's experiments are not this instrument's business. Under `--qualify`
    the same failure is a refusal, because an arm whose residency could not be
    read is not evidence about a build.
    """
    env = dict(os.environ, **plan['env'])
    record = dict(entries=[str(entry) for entry in mode.entry],
                  python=str(station.VENV), pythonpath=env.get('PYTHONPATH'),
                  snapshot=str(named) if named else None)
    try:
        record['modules'] = probe(mode.entry, env)
        record['refusal'] = against(record['modules'], named) if named else None
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as trouble:
        # Including a manifest that went away under a flying arm: this is called
        # from a teardown that still owes the operator a `provenance.json`.
        return dict(record, modules=record.get('modules'),
                    refusal=f'residency could not be read: {trouble}')
    return record
