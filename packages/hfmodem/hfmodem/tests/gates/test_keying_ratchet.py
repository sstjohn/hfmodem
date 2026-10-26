# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[5]
P = "packages/hfmodem/hfmodem/"
OWNERS = {P + name for name in (
    "station/arbiter.py", "core/rig.py", "core/ptt.py",
    "station/preflight.py", "station/process.py",
)}
MIGRATING = {
    P + "cli.py": "6.7",
    P + "shrike/ota.py": "6.7",
    P + "shrike/onair.py": "6.7",
    P + "shrike/beacon.py": "6.7",
    P + "besra/radio.py": "6.4",
    P + "sabir/radio.py": "6.6",
    P + "lanner/qso.py": "6.2",
    P + "contact/radio.py": "6.2",
    P + "contact/runtime.py": "6.2",
    P + "sabir/onair.py": "6.6",
    P + "besra/host/run_server.py": "6.4",
    "tools/ptt_tail_check.py": "6.7",
    "tools/consolelib/voice.py": "6.2",
    "tools/vara_rig_bridge.py": "6.5",
    "tools/kestrel_connect.py": "6.5",
    "tools/onair_session.py": "6.5",
    "tools/alc_knee.py": "6.7",
    "tools/drive_sweep.py": "6.7",
}
#: The claim's line proof opens the line to read it low before admission and never
#: raises it. claim.line_factory builds that opener for claim's CLI and hfcapture.
PROOF_FACTORY = P + "core/claim.py"


def violations(source):
    tree = ast.parse(source)
    names = {"Rig": "Rig", "RtsPtt": "RtsPtt", "LineKeyer": "LineKeyer",
             "Keyer": "Keyer", "drop_rts": "drop_rts", "TIOCMBIS": "TIOCMBIS"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names[alias.asname or alias.name] = alias.name

    def name(node):
        return names.get(node.id, node.id) if isinstance(node, ast.Name) else (
            node.attr if isinstance(node, ast.Attribute) else "")

    rigs = {"rig"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
            if name(node.value.func) == "Rig":
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    rigs.add(ast.unparse(target))

    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"key", "unkey", "ptt"}:
            parent = parents.get(node)
            if node.attr == "ptt" and not isinstance(parent, (ast.Call, ast.keyword, ast.Assign)):
                continue
            if name(node.value) == "rig" or ast.unparse(node.value) in rigs:
                found.append((node.lineno, "rig edge"))
        if isinstance(node, (ast.Name, ast.Attribute)) and name(node) == "TIOCMBIS":
            found.append((node.lineno, "TIOCMBIS"))
        if isinstance(node, ast.Call):
            called = name(node.func)
            if called in {"RtsPtt", "LineKeyer", "Keyer", "drop_rts"}:
                found.append((node.lineno, called))
            if (called == "assert_" and node.args and
                    isinstance(node.args[0], ast.Constant) and node.args[0].value is True):
                found.append((node.lineno, "assert high"))
    return found


def test_keying_paths_have_named_cutovers():
    unexpected = []
    for folder in (ROOT / "packages", ROOT / "tools"):
        for path in folder.rglob("*.py"):
            rel = path.relative_to(ROOT).parts
            relative = "/".join(rel)
            if ("tests" in rel or rel[:3:2] == ("packages", "build")
                    or path.name.startswith("test_") or path.name.endswith("_test.py")
                    or relative in OWNERS):
                continue
            hits = violations(path.read_text())
            if relative == P + "cli.py":
                tree = ast.parse(path.read_text())
                cmd = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name == "cmd_rig")
                hits = [(line, kind) for line, kind in hits if not (
                    kind == "drop_rts" and cmd.lineno <= line <= cmd.end_lineno)]
            if relative == PROOF_FACTORY:
                tree = ast.parse(path.read_text())
                factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                               and n.name == "line_factory")
                hits = [(line, kind) for line, kind in hits if not (
                    kind == "RtsPtt" and factory.lineno <= line <= factory.end_lineno)]
            if hits and relative not in MIGRATING:
                unexpected.append((relative, hits))
    assert not unexpected


def shell_proofs(source):
    pattern = r"<<['\"]?(\w+)['\"]?[^\n]*\n(.*?)^\1$"
    for match in re.finditer(pattern, source, re.MULTILINE | re.DOTALL):
        body = match[2]
        opener = re.search(r"\bline_factory\s*\(|\bline_proof\s*\(\s*(?!None\s*\))[^\s)]", body)
        if not opener and re.search(r"(?:from\s+hfmodem\.core\s+import\s+ptt|hfmodem\.core\.ptt)", body):
            imports = [n for n in ast.walk(ast.parse(body)) if isinstance(n, ast.ImportFrom)]
            opener = not (imports and all(n.module == "hfmodem.core.ptt" and
                                          all(a.name == "derive_ptt_port" for a in n.names)
                                          for n in imports))
        if opener:
            functions = re.findall(r"(?m)^(\w+)\(\)\s*\{", source[:match.start()])
            yield functions[-1] if functions else None


def test_shell_line_proofs_are_named():
    found = [(path.relative_to(ROOT).as_posix(), function)
             for path in (ROOT / "tools").rglob("*.sh") for function in shell_proofs(path.read_text())]
    assert found == [("tools/lib/rig.sh", "prove_down")]


@pytest.mark.parametrize("statement", ["from hfmodem.core import ptt",
                                        "from hfmodem.core.ptt import drop_rts",
                                        "claim.line_proof(claim.line_factory(sys.argv[1]))",
                                        "claim.line_factory(port)()",
                                        "claim.line_proof(opener)"])
def test_shell_import_spellings(statement):
    source = "bad() {\npython - <<'PY'\n" + statement + "\nPY\n}\n"
    assert list(shell_proofs(source)) == ["bad"]


def test_shell_line_proof_without_an_opener_is_not_a_proof():
    source = "fine() {\npython - <<'PY'\nclaim.line_proof(None)\nclaim.line_proof()\nPY\n}\n"
    assert list(shell_proofs(source)) == []


@pytest.mark.parametrize("source", [
    "rig.key(em)", "k = rig.key\nk(em)", "install(key=self.rig.ptt)",
    "LineKeyer('fake')", "x = Rig()\nx.key(em)",
    "from hfmodem.core.ptt import LineKeyer as LK\nLK('fake')",
    "line.assert_(True)", "ioctl(fd, termios.TIOCMBIS, bits)", "drop_rts('fake')",
])
def test_positive_fixtures(source):
    assert violations(source)
