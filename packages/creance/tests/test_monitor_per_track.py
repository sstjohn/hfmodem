# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Per-track dispatch over real audio: the runners moved home by ``--centre``,
and the whole path from tracker to stamped detection.

Every recording here is off-air or a fixture whose content is known, moved by
an amount chosen here, so the right answer is the recording's own reading plus
that amount. They need the rf-corpus beside the tree and skip without it.
"""

import io
import json
import os
import resource
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from hfmodem.tests import evidence
from hfhost.audio import WavReplaySource
from creance.monitor import default_specs, run
from creance.monitor.activity import CONFIRMED

CORPUS = Path(os.environ.get("CREANCE_CORPUS_DIR", evidence.CORPUS))
FIXTURES = CORPUS / "regress" / "fixtures"
#: 2026-07-26: a third station calling somebody at BW500, -46.9 Hz on kestrel's
#: tone grid, at 100.81 s.
KB9MMT_0726 = CORPUS / "offair" / "kb9mmt_reply_20260726" / "rig_rx.wav"
MARYLAND = FIXTURES / "watch_pactor3_maryland.wav"
SPECS = {s.name: s for s in default_specs()}
DIAL = 7100000.0
#: kestrel's tone step, which its handshake scanner reads offsets on
TONE_HZ = 23.4375
#: A replay here takes eight to eleven minutes on a loaded machine, and a runner
#: killed at the default two-minute drain loses the end of its stream.
DRAIN_S = 3600.0

needs = pytest.mark.skipif(
    not (FIXTURES.is_dir() and KB9MMT_0726.exists()
         and all(SPECS[n].available() for n in ("kestrel", "shrike", "besra"))),
    reason="the rf-corpus or a runner is not here")
#: kestrel names a caller only from the station's cached Winlink export, which is
#: gitignored: without it every request on the tape reads unattributed.
GATEWAY_LIST = Path(SPECS["kestrel"].cwd) / "winlink-vara-gateways.csv"
attributes = pytest.mark.skipif(not GATEWAY_LIST.exists(),
                                reason="no cached Winlink gateway list to attribute against")


def _audio(path: Path):
    import numpy as np

    pcm = b"".join(p for _, p in WavReplaySource(path, realtime=False).frames())
    return np.frombuffer(pcm, "<i2").astype(float) / 32768.0


def _moved(x, hz: float, fs: int = 48000):
    import numpy as np
    from scipy.signal import hilbert

    return (hilbert(x) * np.exp(2j * np.pi * hz * np.arange(len(x)) / fs)).real


def _write(path: Path, x, fs: int = 48000) -> Path:
    import numpy as np

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _dispatch(path: Path, dispatch: str, dial: float | None = None,
              cap: int = 12, candidates: Path | None = None) -> list:
    got = []
    specs = default_specs()
    if candidates is not None:
        specs = [replace(s, cwd=str(candidates)) if s.name == 'kestrel' else s for s in specs]
    run(WavReplaySource(path, realtime=False, dial_hz=dial), specs,
        sinks=[got.append], out=io.StringIO(), live=False, dispatch=dispatch, cap=cap,
        drain_timeout=DRAIN_S)
    return got


def _per_track(path: Path, dial: float | None = None, cap: int = 12) -> list:
    return _dispatch(path, "per-track", dial, cap)


def _key(d) -> tuple:
    return (d.modem, d.protocol, d.kind, d.station)


def _decoded(got) -> list:
    return [d for d in got if d.grade == CONFIRMED and d.modem not in ("tracker", "cw")]


def _found(d, among) -> bool:
    return any(_key(e) == _key(d) and abs(e.t - d.t) < 0.6 for e in among)


def _runner(name: str, pcm: bytes, *args: str) -> list[dict]:
    spec = SPECS[name]
    done = subprocess.run([spec.interpreter, spec.runner, *spec.args, *args], input=pcm,
                          cwd=spec.cwd, stdout=subprocess.PIPE, timeout=900, check=True)
    return [d for ln in done.stdout.splitlines() if ln.strip()
            if "t" in (d := json.loads(ln))]


def _pcm(x) -> bytes:
    import numpy as np

    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def _crs(got) -> list:
    return sorted((d for d in got if d.modem == "kestrel" and d.kind == "CR"),
                  key=lambda d: d.t)


def _third_station() -> dict:
    """The connect request the 2026-07-26 tape carries from a station calling
    somebody else, as the whole-band kestrel runner reads it."""
    # The external gateway cache need not exist in an isolated worktree. Without
    # it all requests are unnamed, including ours; identify this recorded burst
    # by its independently measured time and offset instead of cache membership.
    [cr] = [d for d in _runner("kestrel", _pcm(_audio(KB9MMT_0726)))
            if d["kind"] == "CR" and abs(d["t"] - 100.81) < .05]
    assert abs(cr["offset_hz"] - (-46.9)) < .1
    return cr


# -- the runners, moved home -------------------------------------------------

@needs
def test_besra_reads_a_conreq_700_hz_up_from_where_it_sits():
    import numpy as np

    x = np.r_[np.zeros(24000), _audio(FIXTURES / "oracle_ardop_conreq.wav"), np.zeros(24000)]
    got = _runner("besra", _pcm(_moved(x, 700.0)), "--centre", "2200")
    assert [(d["kind"], d["station"]) for d in got] == [("ConReq500M", "M7TFF")]
    assert abs(got[0]["offset_hz"] - 700.0) <= 3.0


@needs
def test_shrike_reads_ke5yta_700_hz_up_from_where_it_sits():
    x = _audio(FIXTURES / "pos_pactor1_local.wav")

    def connects(got):
        return {(d["station"], d["offset_hz"]) for d in got if d["kind"] == "CONNECT"}
    home = connects(_runner("shrike", _pcm(x)))
    moved = connects(_runner("shrike", _pcm(_moved(x, 700.0)), "--centre", "2200",
                             "--shape", "pair"))
    assert home and moved == {(call, off + 700.0) for call, off in home}


# -- the acceptance ----------------------------------------------------------

@needs
@attributes
@pytest.mark.parametrize("up_hz", [700.0, 900.0], ids=["32-hz-guard", "232-hz-guard"])
def test_two_bw500_links_side_by_side_are_two_tracks_at_their_own_offsets(tmp_path, up_hz):
    """The recording and a copy of it moved up and 7.3 s later, summed: each
    connect request is read on its own track at its own offset. A BW500 request
    spans 1113-1781 Hz here, so 700 Hz up leaves a 32 Hz guard between the two
    bands, under one 47 Hz tone step, and 900 Hz leaves 232 Hz. At both the
    tracker can put the copy in a track with a neighbour's overs, centred up to
    150 Hz off it, which kestrel's tiled scan reaches. Laid over the original
    at the same instant the two requests are one emission to the tracker, as the
    design's rule for coincident bursts says, and read as one wide UNKNOWN track.

    The copy is moved by a whole number of hertz and not of kestrel's 23.4375 Hz
    tone bins, so its request is off the grid the scanner reads on and its
    offset is good to half a bin."""
    import numpy as np

    x = _audio(KB9MMT_0726)
    lag = int(7.3 * 48000)
    copy = np.r_[np.zeros(lag), _moved(x, up_hz)][:len(x)]
    got = _per_track(_write(tmp_path / "two.wav", 0.5 * (x + copy)))
    cr = _third_station()
    [first] = [d for d in _crs(got) if abs(d.t - cr["t"]) < 0.05]
    [second] = [d for d in _crs(got) if abs(d.t - cr["t"] - 7.3) < 0.05]
    assert first.offset_hz == cr["offset_hz"]
    assert abs(second.offset_hz - (cr["offset_hz"] + up_hz)) <= TONE_HZ / 2
    assert first.track != second.track


@needs
@attributes
def test_the_same_request_at_8200_hz_in_a_12_khz_window_is_at_the_same_dial(tmp_path):
    from scipy.signal import resample_poly

    x = _audio(KB9MMT_0726)
    rig = _crs(_per_track(KB9MMT_0726, dial=DIAL))
    kiwi = _crs(_per_track(_write(tmp_path / "kiwi.wav",
                                  resample_poly(_moved(x, 6700.0), 1, 2), 24000),
                           dial=DIAL - 6700.0))
    cr = _third_station()
    rf = DIAL + 1500.0 + cr["offset_hz"]
    [at_rig] = [d for d in rig if abs(d.t - cr["t"]) < 0.05]
    [at_kiwi] = [d for d in kiwi if abs(d.t - cr["t"]) < 0.05]
    assert (at_rig.offset_hz, round(at_rig.rf_hz, 1)) == (cr["offset_hz"], round(rf, 1))
    # 6700 Hz is 285.9 tone bins: the request sits off kestrel's grid there
    assert abs(at_kiwi.offset_hz - (cr["offset_hz"] + 6700.0)) <= TONE_HZ / 2
    assert abs(at_kiwi.rf_hz - rf) <= TONE_HZ / 2


def _confirmed(got) -> set:
    return {(d.modem, d.protocol, d.kind, d.station) for d in got
            if d.grade == CONFIRMED and d.modem not in ("tracker", "cw")}


@needs
def test_per_track_finds_every_pactor1_connect_and_control_signal_the_whole_band_does():
    path = FIXTURES / "pos_pactor1_local.wav"
    whole = [d for d in _decoded(_dispatch(path, "whole-band"))
             if d.protocol == "PACTOR-1" and d.kind in ("CONNECT", "CS")]
    per = _decoded(_per_track(path))
    assert whole and all(_found(d, per) for d in whole), whole


@needs
def test_per_track_reads_the_ardop_oracle(tmp_path):
    """Set in a stream: the oracle is 1.7 s long, and the tracker learns a
    floor over its first two seconds before it raises anything."""
    import numpy as np

    rng = np.random.default_rng(7)
    x = np.r_[np.zeros(4 * 48000), _audio(FIXTURES / "oracle_ardop_conreq.wav"),
              np.zeros(3 * 48000)]
    x += rng.normal(0, 1e-3, len(x))
    got = _per_track(_write(tmp_path / "ardop.wav", x))
    assert ("besra", "ARDOP", "ConReq500M", "M7TFF") in _confirmed(got)


@needs
def test_a_shrike_instance_keeps_pace_on_the_maryland_tape():
    """The busiest labelled PACTOR tape, where the whole-band runner spends 2.1x
    real time: one instance on its loudest narrow track, from start to end."""
    x = _audio(MARYLAND)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    _runner("shrike", _pcm(x), "--centre", "1200", "--shape", "pair")
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    assert cpu < 0.5 * len(x) / 48000


@needs
def test_an_upgraded_link_gets_its_pactor3_search_and_every_header_is_read():
    """DL6MAA opens in PACTOR-1 on a track 490 Hz wide and widens as it upgrades;
    the wide instance started then is fed from before the link began, on the
    centre the link opened on, and reads every header while the link is that
    track. Past a minute in the tracker can split the link into fragments off
    its centre, where PACTOR-3 has no search, which is why the console runs the
    whole-band decoders beside these. Uncapped, since the fragments' waiting on
    one another is the cap's question and not this one."""
    path = FIXTURES / "oracle_pactor3_dl6maa.wav"
    whole = _dispatch(path, "whole-band")
    got = _per_track(path, cap=60)

    def headers(got):
        return [d for d in got if d.grade == CONFIRMED
                and (d.protocol, d.kind) == ("PACTOR-3", "HEADER")]
    found = headers(got)
    link = found[0].track
    [end] = [d.t for d in got if d.modem == "tracker" and d.kind == "track-end"
             and d.track == link]
    # A track ends TRACK_GAP_S, 2 s, after its last burst, so this is where the
    # link's own track stopped holding it; later headers sit in other tracks.
    held = [d.t for d in headers(whole) if d.t < end - 2.0]
    assert held
    assert all(any(abs(t - f.t) < 0.3 for f in found) for t in held)


@needs
def test_a_dragon_chirp_is_read_on_its_track(tmp_path):
    import numpy as np
    from hfmodem import tests
    from scipy.signal import resample_poly

    with np.load(Path(tests.__file__).parent / "shrike/fixtures/p4-chirps/audio.npz") as z:
        fs, x = int(z["fs"]), z["audio"][0].astype(float)
    x = np.r_[np.zeros(6 * 48000), resample_poly(_moved(x, 242.4, fs), 48000, fs),
              np.zeros(4800)]
    path = _write(tmp_path / "chirp.wav", x / np.abs(x).max() * 0.5)
    [whole] = [d for d in _decoded(_dispatch(path, "whole-band")) if d.modem == "p4"]
    [chirp] = [d for d in _decoded(_per_track(path)) if d.modem == "p4"]
    assert (chirp.kind, round(chirp.t, 2)) == (whole.kind, round(whole.t, 2))
    assert abs(chirp.offset_hz - whole.offset_hz) <= 4.0 and chirp.track is not None


@needs
@pytest.mark.parametrize("path", [
    pytest.param(FIXTURES / "pos_pactor1_local.wav", id="pactor1-local"),
    pytest.param(KB9MMT_0726, id="kb9mmt", marks=attributes)])
def test_both_dispatches_find_every_frame_the_whole_band_finds_and_say_each_once(path, tmp_path):
    """The two recordings the per-track path alone falls short on: our own
    connect requests inside the receiver's mute raise no track, and a PACTOR-3
    header the tracker merges into a neighbour is moved off its tones."""
    # Naming a keyed VARA handshake requires a candidate dictionary. Supply this
    # recording's known called station instead of depending on a private cache.
    (tmp_path / 'winlink-vara-gateways.csv').write_text(
        'Callsign,GridSquare,Mode\nKB9MMT,,VARA\n')
    whole = _decoded(_dispatch(path, "whole-band", candidates=tmp_path))
    both = _decoded(_dispatch(path, "both", candidates=tmp_path))
    assert whole and all(_found(d, both) for d in whole)
    for i, d in enumerate(both):
        assert not any(_key(e) == _key(d) and abs(e.t - d.t) < 0.6
                       and (e.offset_hz is None or d.offset_hz is None
                            or abs(e.offset_hz - d.offset_hz) <= 10)
                       for e in both[i + 1:]), d
    assert {d.via for d in both} <= {"whole-band", "per-track"}
