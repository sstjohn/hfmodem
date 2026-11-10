# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A target that stops resolving must cost a red run, not silent coverage.

`selftest` skips a target whose path does not resolve, which is right when the
target is genuinely optional and wrong when it is the point of the run — a rename
or a moved venv takes an entire graded target with it and the gate still exits 0.
These tests pin the two ways that goes wrong: a known target that vanished, and a
required name nothing answers to.

They drive `Selftest` directly with a stub stage list rather than spawning modems,
so they stay fast and hermetic; `test_selftest.py::test_m1_gate` is the live one.
"""

import creance.selftest as selftest


def _run(require=(), stages=()):
    """A Selftest with a canned stage list, summarised."""
    run = selftest.Selftest(size="1k", require=require)
    for name, status in stages:
        run.stages.append(selftest.Stage(name, status, ""))
    rc = run.summarize()
    return rc, run.stages


def test_absent_required_target_fails_instead_of_skipping():
    run = selftest.Selftest(size="1k", require=("sabir",))
    run.skip("sabir virtual pair", "no venv interpreter at /nonexistent")
    assert run.stages[0].status == selftest.FAIL
    assert run.summarize() == 1


def test_absent_optional_target_still_skips():
    run = selftest.Selftest(size="1k", require=("kestrel",))
    run.skip("sabir virtual pair", "no venv interpreter at /nonexistent")
    assert run.stages[0].status == selftest.SKIP


def test_required_name_no_stage_answers_to_is_a_failure():
    """The case a rename actually produces: nothing skipped, nothing ran, and
    without this check the run is green on the targets that still resolve."""
    rc, stages = _run(require=("besra",),
                      stages=[("conform kestrel-loopback", selftest.PASS)])
    assert rc == 1
    assert any("besra" in s.name and s.status == selftest.FAIL for s in stages)


def test_required_target_matched_anywhere_in_the_stage_name():
    """Stage names read 'conform kestrel-loopback' / 'responder on sabir-B', so
    the target is inside the name rather than its first word."""
    rc, _ = _run(require=("kestrel", "sabir"),
                 stages=[("conform kestrel-loopback", selftest.PASS),
                         ("responder on sabir-B", selftest.PASS)])
    assert rc == 0


def test_no_require_preserves_the_old_skipping_behaviour():
    run = selftest.Selftest(size="1k")
    run.skip("sabir virtual pair", "absent")
    assert run.stages[0].status == selftest.SKIP
    assert run.summarize() == 0


def test_no_capability_probe_reads_a_sibling_repos_source():
    """A capability probe that greps another repo's source is absence in
    disguise: when that path moves the probe returns False, the stage it gates
    is never recorded, and `require` has nothing to match. `_supports_hostapi`
    did exactly this and cost the structured pair silently. Guard the shape, not
    just the one function."""
    import inspect
    src = inspect.getsource(selftest)
    assert "_supports_hostapi" not in src
    # no reading of a sibling repo's source text to infer a capability
    assert "read_text" not in src.split("def write_configs")[0]
