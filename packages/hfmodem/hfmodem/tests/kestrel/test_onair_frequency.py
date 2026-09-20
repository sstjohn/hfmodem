# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the session tunes, and where the emission lands.

Winlink publishes the signal centre; the dial is 1500 Hz below it. The session
used to send the published number straight to ``F``, tuning every attempt 1500 Hz
high — our bursts above where the gateway listens, its answer under the SSB
filter. The arithmetic is one subtraction, which is exactly the kind of thing
that gets re-derived wrongly, so it is pinned here along with the segment guard
that depends on it.

The last stage of that subtraction happens in `bash`, and until now nothing read
it. `onair.sh` builds the `--freq` argument with `bc`, and `bc` truncates: at
`scale=0` the dial goes out as whole kHz, up to 999 Hz below where the gateway
listens, which is silent in both directions and costs the whole rig window. Both
`scale=0` and `scale=1` -- the exact defect that was fixed -- passed every test
in this tree, because the only one that named the launcher computed the
truncation itself and never opened the file. So `launcher_dial_khz` below runs
the launcher's own expression through `bc` and the answer is compared against the
arithmetic, in both verbs that key a transmitter through it.
"""
import re
import shutil
import subprocess

import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.corpora import harness

S = harness("onair_session")

_LAUNCHER = corpora.TOOLS / "onair.sh"


#: A verb's block ends at the `;;` that closes it, which the launcher writes on its
#: own line at two spaces. Cutting at the first `;;` anywhere reads only as far as
#: the first NESTED case arm -- `vara` grew one on 2026-08-14 to tell `--force` from
#: a bare `force`, and the dial line sitting after it went unread while five tests
#: reported the launcher no longer built `--freq` at all.
_ARM_END = re.compile(r"^  ;;", re.M)


def launcher_dial_khz(centre_hz: int, verb: str) -> float:
    """The dial `onair.sh <verb>` hands the tool, in kHz, from its own `bc` line."""
    rest = _LAUNCHER.read_text().split(f"\n{verb})")[1]
    end = _ARM_END.search(rest)
    assert end, f"onair.sh {verb} has no closing ';;' -- the block cannot be read"
    block = rest[:end.start()]
    expr = re.search(r'echo "(scale=[^"]*)" \| bc', block)
    assert expr, f"onair.sh {verb} no longer builds --freq with bc: {block[:200]}"
    r = subprocess.run(["bash", "-c", f'centre={centre_hz}\necho "{expr.group(1)}" | bc'],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


@pytest.mark.skipif(not _LAUNCHER.exists() or not shutil.which("bc"),
                    reason="onair.sh or bc not present")
@pytest.mark.parametrize("verb", ["vara", "vara-mail"])
@pytest.mark.parametrize("centre_hz", [7102850, 7101550, 7103500, 14108000])
def test_the_launcher_hands_over_the_whole_dial(verb, centre_hz):
    """Every digit of it. 7102.850 kHz is a published centre and its dial is
    7101.350 -- a number `bc` will happily give back as 7101 or 7101.3, and the
    rig will happily tune."""
    assert launcher_dial_khz(centre_hz, verb) == pytest.approx(
        (centre_hz - 1500) / 1000.0, abs=1e-9)


@pytest.mark.skipif(not _LAUNCHER.exists() or not shutil.which("bc"),
                    reason="onair.sh or bc not present")
def test_both_launcher_verbs_tune_the_same_dial():
    """`vara` and `vara-mail` build the dial in two separate lines, so one can be
    corrected and the other left, and a mail run would then call a frequency the
    connect run had already proved."""
    assert (launcher_dial_khz(7102850, "vara")
            == launcher_dial_khz(7102850, "vara-mail"))


def test_dial_is_the_published_centre_less_1500_hz():
    assert S.DIAL_OFFSET_HZ == 1500
    assert S.dial_hz(7103500) == 7102000        # KC9GHZ, the connect that worked
    assert S.dial_hz(7102000) == 7100500        # NS0A, the other one


def test_emission_sits_above_the_dial():
    lo, hi = S.emission_khz(7103500)
    assert (lo, hi) == pytest.approx((7102.75, 7104.3))


def _csv(tmp_path, *rows):
    p = tmp_path / "gw.csv"
    p.write_text("Callsign,Frequency,GridSquare,Hours,Mode\n"
                 + "".join(f"{c},{f},EN62,,VARA\n" for c, f in rows))
    return p


def test_a_channel_whose_emission_leaves_the_data_segment_is_dropped(tmp_path, capsys):
    """7124.5 is inside 40 m and inside the CSV's band, and still illegal to use:
    the dial is 7123.0 and the emission runs to 7125.3, past the 7125 edge."""
    picked = S.pick_targets(_csv(tmp_path, ("W1AAA", 7101.0), ("W3CCC", 7124.5)),
                            "EN63", "40", 10, 5000.0)
    assert [p[0] for p in picked] == ["W1AAA"]
    assert "outside the 40 m data segment" in capsys.readouterr().out


def test_targets_are_listed_by_published_centre(tmp_path):
    """The operator thinks in published channels; only the VFO gets the dial."""
    (call, hz, _d, _m), = S.pick_targets(_csv(tmp_path, ("W1AAA", 7101.0)),
                                         "EN63", "40", 10, 5000.0)
    assert (call, hz) == ("W1AAA", 7101000)
    assert S.dial_hz(hz) == 7099500
