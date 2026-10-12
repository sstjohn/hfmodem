# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Per-track dispatch using stub runners, with slow integration checks using
synthetic modem audio and an optional station-Q recording.
"""

import json
import queue
import sys
import textwrap
import time

import pytest
from hfhost.audio import FS, Window

from creance.monitor.activity import CONFIRMED, TENTATIVE, TRACKER, Detection
from creance.monitor.channelizer import ADMITS, Channelizer
from creance.monitor.driver import MonitorSpec, default_specs

FRAME = 4800
WINDOW = Window(300.0, 2900.0, "test", 7100000.0)

TRACKER_STUB = textwrap.dedent(r'''
    import json, sys
    script = json.loads(sys.argv[1])
    seen = 0
    stdin = sys.stdin.buffer
    def say(upto):
        global script
        while script and script[0]["at"] <= upto:
            rec = script.pop(0)
            rec.pop("at")
            sys.stdout.write(json.dumps(rec) + "\n")
            sys.stdout.flush()
    while True:
        raw = stdin.read(9600)
        if not raw:
            break
        seen += len(raw) // 2
        say(seen / 48000.0)
    say(float("inf"))
''')

MODEM_STUB = textwrap.dedent(r'''
    import json, sys
    late = 15.0 if "late" in sys.argv else 0.0
    seen = 0
    stdin = sys.stdin.buffer
    moved = float(sys.argv[sys.argv.index("--centre") + 1]) - 1500.0
    def say(t, kind, protocol="VARA", offset=moved + 3.0):
        sys.stdout.write(json.dumps({"t": t, "modem": sys.argv[1], "protocol": protocol,
            "kind": kind, "grade": "confirmed", "station": "W1AW",
            "detail": " ".join(sys.argv[2:]), "offset_hz": offset}) + "\n")
        sys.stdout.flush()
    while True:
        raw = stdin.read(9600)
        if not raw:
            break
        if seen <= late * 48000 < seen + len(raw) // 2:
            say(0.0, "FIRST")
            say(0.0, "SHAPE", protocol="UNKNOWN")
        seen += len(raw) // 2
        sys.stdout.write(json.dumps({"heard": seen / 48000.0 - late - 1.0}) + "\n")
        sys.stdout.flush()
    say(seen / 48000.0, "LAST", offset=None)
''')


def _track(at, track, kind="track", *, t_start, centre, bw, level=-30.0, over=30.0,
           label="500-wide"):
    return dict(at=at, t=at, modem=TRACKER, protocol=label, kind=kind, grade=TENTATIVE,
                track=track, t_start=t_start, centre_hz=centre, bw_hz=bw,
                level_dbfs=level, over_db=over)


def _specs(tmp_path, script, *extra):
    (tmp_path / "tracker.py").write_text(TRACKER_STUB)
    (tmp_path / "modem.py").write_text(MODEM_STUB)
    tracker = MonitorSpec(TRACKER, sys.executable, str(tmp_path / "tracker.py"),
                          str(tmp_path), (json.dumps(script),))
    modems = [MonitorSpec(name, sys.executable, str(tmp_path / "modem.py"),
                          str(tmp_path), (name, *extra)) for name in ADMITS]
    return tracker, modems


def _run(tmp_path, script, seconds, cap=12):
    out: "queue.Queue[Detection]" = queue.Queue()
    tracker, modems = _specs(tmp_path, script)
    fan = Channelizer(WINDOW, tracker, modems, out, cap=cap, lossy=False)
    fan.start()
    frame = bytes(2 * FRAME)
    for t0 in range(0, int(seconds * FS), FRAME):
        fan.feed(t0, frame)
    assert fan.stop(30.0)
    got = []
    while not out.empty():
        got.append(out.get())
    return fan, got


def _said(got, kind):
    return {d.detail: d for d in got if d.kind == kind}


def test_a_track_gets_every_modem_its_width_admits_on_the_grid(tmp_path):
    script = [_track(4.0, 1, t_start=3.0, centre=1188.4, bw=600.0),
              _track(4.0, 2, t_start=3.0, centre=2012.0, bw=2300.0, label="2k-multicarrier"),
              _track(4.0, 3, t_start=3.0, centre=1605.5, bw=35.0, label="CW-shaped"),
              _track(9.0, 1, "track-end", t_start=3.0, centre=1188.4, bw=600.0),
              _track(9.0, 2, "track-end", t_start=3.0, centre=2012.0, bw=2300.0),
              _track(9.0, 3, "track-end", t_start=3.0, centre=1605.5, bw=35.0)]
    fan, got = _run(tmp_path, script, 12.0)
    assert set(_said(got, "FIRST")) == {
        "--centre 1200 --bw 840",
        "--centre 1200 --bw 2800 --shape pair",
        "--centre 1200 --bw 2600",
        "--centre 1200 --bw 700",
        "--centre 2000 --bw 2540",
        "--centre 2000 --bw 2800 --shape wide",
        "--centre 2000 --bw 2600",
        "--centre 2000 --bw 2800"}
    assert fan.seen == 3 and fan.denied() == []


def test_a_track_that_widens_gets_the_wide_search_beside_the_narrow(tmp_path):
    script = [_track(4.0, 1, t_start=3.0, centre=1500.0, bw=400.0, label="FSK-shaped"),
              _track(6.0, 1, t_start=3.0, centre=1500.0, bw=2200.0, label="2k-multicarrier"),
              _track(9.0, 1, "track-end", t_start=3.0, centre=1500.0, bw=2200.0)]
    fan, _ = _run(tmp_path, script, 12.0)
    assert [s.name for s in fan.started if s.name.startswith("shrike")] == [
        "shrike#1/pair", "shrike#1/wide"]


def test_lanner_base_args_reach_the_per_track_runner(tmp_path):
    from dataclasses import replace

    script = [_track(4.0, 1, t_start=3.0, centre=1500.0, bw=190.0,
                     label='FSK-shaped'),
              _track(9.0, 1, 'track-end', t_start=3.0, centre=1500.0, bw=190.0)]
    tracker, modems = _specs(tmp_path, script)
    base = next(s for s in default_specs(no_rtty_usos=True) if s.name == 'lanner')
    stub = next(s for s in modems if s.name == 'lanner')
    out = queue.Queue()
    fan = Channelizer(WINDOW, tracker,
                      [replace(stub, args=stub.args + base.args)], out, lossy=False)
    fan.start()
    for t0 in range(0, 12 * FS, FRAME):
        fan.feed(t0, bytes(2 * FRAME))
    assert fan.stop(30.0)
    got = []
    while not out.empty():
        got.append(out.get())
    assert [d.detail for d in got if d.kind == 'FIRST'] == [
        '--no-rtty-usos --centre 1500 --bw 700']


def test_a_track_whose_centroid_walks_off_homes_its_next_instance_there(tmp_path):
    """A small walk keeps the home, as an upgrading link needs; a walk past the
    grid is another emission, and the wide search is centred on it."""
    script = [_track(4.0, 1, t_start=3.0, centre=1634.0, bw=190.0, label="FSK-shaped"),
              _track(4.0, 2, t_start=3.0, centre=1500.0, bw=400.0, label="FSK-shaped"),
              _track(6.0, 1, t_start=3.0, centre=2079.0, bw=2040.0, label="2k-multicarrier"),
              _track(6.0, 2, t_start=3.0, centre=1555.0, bw=1000.0, label="2k-multicarrier"),
              _track(9.0, 1, "track-end", t_start=3.0, centre=2079.0, bw=2040.0),
              _track(9.0, 2, "track-end", t_start=3.0, centre=1555.0, bw=1000.0)]
    fan, got = _run(tmp_path, script, 12.0)
    wide = {s.name: s.args[s.args.index("--centre") + 1]
            for s in fan.started if s.name.endswith("/wide")}
    assert wide == {"shrike#1/wide": "2100", "shrike#2/wide": "1500"}


def test_a_detection_carries_its_track_and_where_its_decoder_put_it(tmp_path):
    script = [_track(4.0, 7, t_start=3.0, centre=2188.4, bw=600.0),
              _track(9.0, 7, "track-end", t_start=3.0, centre=2188.4, bw=600.0)]
    _, got = _run(tmp_path, script, 12.0)
    d = next(d for d in got if d.modem == "kestrel" and d.kind == "FIRST")
    assert (d.track, d.bw_hz, d.offset_hz, d.grade) == (7, 600.0, 703.0, CONFIRMED)
    assert d.centre_hz == 2203.0 and d.rf_hz == 7102203.0
    last = next(d for d in got if d.modem == "kestrel" and d.kind == "LAST")
    assert last.centre_hz == 2188.4 and last.rf_hz == 7102188.4
    assert not [d for d in got if d.kind == "SHAPE"]


def test_an_instance_hears_from_before_its_track_and_stops_with_it(tmp_path):
    script = [_track(6.0, 1, t_start=4.5, centre=1500.0, bw=600.0),
              _track(10.0, 1, "track-end", t_start=4.5, centre=1500.0, bw=600.0)]
    _, got = _run(tmp_path, script, 30.0)
    first = next(d for d in got if d.modem == "kestrel" and d.kind == "FIRST")
    last = next(d for d in got if d.modem == "kestrel" and d.kind == "LAST")
    assert first.t == pytest.approx(2.5, abs=FRAME / FS)
    assert 10.0 <= last.t < 13.0


def test_a_freed_slot_goes_to_the_loudest_waiting_and_the_rest_are_named(tmp_path):
    script = [_track(3.0, 3, t_start=2.0, centre=2500.0, bw=600.0, level=-40.0),
              _track(4.0, 1, t_start=3.0, centre=1000.0, bw=600.0, level=-50.0),
              _track(4.5, 2, t_start=3.0, centre=2000.0, bw=600.0, level=-20.0),
              _track(6.0, 3, "track-end", t_start=2.0, centre=2500.0, bw=600.0, level=-40.0),
              _track(9.0, 1, "track-end", t_start=3.0, centre=1000.0, bw=600.0, level=-50.0),
              _track(9.0, 2, "track-end", t_start=3.0, centre=2000.0, bw=600.0, level=-20.0)]
    fan, got = _run(tmp_path, script, 12.0, cap=4)
    assert {d.track for d in got if d.kind == "FIRST"} == {2, 3}
    denied = fan.denied()
    [line] = [line for line in denied if line.startswith("# track#1 ")]
    # An ended lanner still catching up occupies its slot until its watermark
    # catches track-end; the other three decoders remain available immediately.
    assert all("no lanner --" in line for line in denied if not line.startswith("# track#1 "))
    assert line.startswith("# track#1 700-1300 Hz 500-wide -50 dBFS")
    assert "no besra, kestrel, lanner, shrike" in line and fan.peak == 4


def test_a_track_far_louder_takes_the_slots_of_the_quietest(tmp_path):
    script = [_track(3.0, 1, t_start=2.0, centre=5000.0, bw=600.0, level=-75.0),
              _track(4.0, 2, t_start=3.0, centre=1500.0, bw=600.0, level=-23.0),
              _track(9.0, 1, "track-end", t_start=2.0, centre=5000.0, bw=600.0, level=-75.0),
              _track(9.0, 2, "track-end", t_start=3.0, centre=1500.0, bw=600.0, level=-23.0)]
    fan, got = _run(tmp_path, script, 12.0, cap=4)
    assert [s.name for s in fan.started] == ["besra#1", "kestrel#1", "lanner#1", "shrike#1/pair",
                                             "besra#2", "kestrel#2", "lanner#2", "shrike#2/pair"]
    assert {d.track for d in got if d.kind == "FIRST" and d.modem == "kestrel"} == {1, 2}
    [line] = fan.denied()
    assert line.startswith("# track#1 ") and "stopped for a louder track" in line


def test_one_frame_heard_by_two_instances_is_said_once(tmp_path):
    script = [_track(4.0, 1, t_start=3.0, centre=1480.0, bw=600.0),
              _track(4.0, 2, t_start=3.0, centre=1530.0, bw=600.0),
              _track(9.0, 1, "track-end", t_start=3.0, centre=1480.0, bw=600.0),
              _track(9.0, 2, "track-end", t_start=3.0, centre=1530.0, bw=600.0)]
    _, got = _run(tmp_path, script, 12.0)
    assert len([d for d in got if d.modem == "kestrel" and d.kind == "FIRST"]) == 1


def test_every_tracker_record_goes_out_placed_on_the_dial(tmp_path):
    script = [_track(4.0, 1, t_start=3.0, centre=1605.5, bw=35.0, label="CW-shaped")]
    _, got = _run(tmp_path, script, 6.0)
    [rec] = [d for d in got if d.modem == TRACKER]
    assert rec.rf_hz == 7101605.5 and rec.t_start == 3.0


WHOLE_STUB = textwrap.dedent(r'''
    import json, sys
    said = json.loads(sys.argv[1])
    seen = 0
    stdin = sys.stdin.buffer
    while True:
        raw = stdin.read(9600)
        if not raw:
            break
        seen += len(raw) // 2
        while said and said[0]["t"] <= seen / 48000.0:
            sys.stdout.write(json.dumps(said.pop(0)) + "\n")
            sys.stdout.flush()
''')


def _both(tmp_path, monkeypatch, script, seconds, whole_says, *extra, pace=0.0):
    """``aggregate.run`` with both dispatches over silence, the whole-band stub
    saying ``whole_says`` when the stream reaches each."""
    import io

    from hfhost.audio import AudioSource

    from creance.monitor import aggregate

    class Quiet(AudioSource):
        window = WINDOW

        def frames(self):
            for t0 in range(0, int(seconds * FS), FRAME):
                time.sleep(pace)
                yield t0, bytes(2 * FRAME)

    (tmp_path / "whole.py").write_text(WHOLE_STUB)
    tracker, modems = _specs(tmp_path, script, *extra)
    monkeypatch.setattr(aggregate, "tracker_spec", lambda base, window: tracker)
    monkeypatch.setattr(aggregate, "Channelizer",
                        lambda window, tracker, specs, out, **kw:
                        Channelizer(window, tracker, modems, out, **kw))
    whole = MonitorSpec("kestrel", sys.executable, str(tmp_path / "whole.py"),
                        str(tmp_path), (json.dumps(whole_says),))
    got = []
    aggregate.run(Quiet(), [whole], sinks=[got.append], out=io.StringIO(), live=False,
                  dispatch="both")
    return [d for d in got if d.modem == "kestrel"]


def test_both_dispatches_say_a_frame_once_and_the_track_says_it(tmp_path, monkeypatch):
    script = [_track(4.0, 1, t_start=3.0, centre=2188.4, bw=600.0),
              _track(9.0, 1, "track-end", t_start=3.0, centre=2188.4, bw=600.0)]
    twin = dict(t=1.0, modem="kestrel", protocol="VARA", kind="FIRST", grade="confirmed",
                station="W1AW", offset_hz=703.0)
    alone = dict(twin, t=5.0, kind="ONLY-WHOLE", offset_hz=None)
    got = _both(tmp_path, monkeypatch, script, 12.0, [twin, alone])
    first = [d for d in got if d.kind == "FIRST"]
    assert [(d.via, d.track) for d in first] == [("per-track", 1)]
    [only] = [d for d in got if d.kind == "ONLY-WHOLE"]
    assert only.via == "whole-band" and only.track is None


def test_a_whole_band_detection_is_placed_on_the_track_it_sat_in(tmp_path, monkeypatch):
    script = [_track(4.0, 1, t_start=3.0, centre=2188.4, bw=600.0),
              _track(9.0, 1, "track-end", t_start=3.0, centre=2188.4, bw=600.0)]
    placed = dict(t=5.0, modem="kestrel", protocol="VARA", kind="PLACED",
                  grade="confirmed", station="W1AW", offset_hz=690.0)
    [d] = [d for d in _both(tmp_path, monkeypatch, script, 12.0, [placed])
           if d.kind == "PLACED"]
    assert (d.via, d.track, d.centre_hz, d.bw_hz, d.rf_hz) == (
        "whole-band", 1, 2190.0, 600.0, 7102190.0)


def test_the_track_says_its_frame_however_late_its_decoder_says_it(tmp_path, monkeypatch):
    """The instance says its frame only after 15 s more audio, as a decoder
    behind on a loaded machine would; its twin is still the one said, because
    the hold waits on what the instance says it has decoded and not on what it
    has been handed."""
    script = [_track(at, 1, t_start=3.0, centre=2188.4, bw=600.0) for at in range(4, 30)]
    script.append(_track(30.0, 1, "track-end", t_start=3.0, centre=2188.4, bw=600.0))
    twin = dict(t=1.0, modem="kestrel", protocol="VARA", kind="FIRST", grade="confirmed",
                station="W1AW", offset_hz=703.0)
    got = _both(tmp_path, monkeypatch, script, 40.0, [twin], "late", pace=0.01)
    assert [(d.via, d.track) for d in got if d.kind == "FIRST"] == [("per-track", 1)]


@pytest.mark.parametrize("level, over", [(-95.0, 30.0), (-40.0, 6.0)],
                         ids=["under-the-absolute-floor", "too-near-its-own-floor"])
def test_a_track_that_does_not_stand_out_is_offered_no_decoder(tmp_path, level, over):
    script = [_track(4.0, 1, t_start=3.0, centre=1500.0, bw=600.0, level=level, over=over),
              _track(9.0, 1, "track-end", t_start=3.0, centre=1500.0, bw=600.0,
                     level=level, over=over)]
    fan, _ = _run(tmp_path, script, 12.0)
    assert fan.started == [] and fan.denied() == []


@pytest.mark.parametrize('width, admitted', [(149, False), (150, True), (250, True),
                                            (400, True), (600, True), (601, False)])
def test_lanner_admits_narrow_fsk_with_a_700_hz_cut(width, admitted):
    rule = ADMITS['lanner']
    assert (rule.admits(width) is not None) is admitted
    assert rule.channel(width) == 700


def test_lanner_is_started_for_each_narrow_track_and_placed_on_its_dial(tmp_path):
    script = [_track(4, n, t_start=3, centre=centre, bw=300)
              for n, centre in [(1, 1100), (2, 1600), (3, 2100)]]
    fan, got = _run(tmp_path, script, 8)
    assert {s.name for s in fan.started if s.name.startswith('lanner')} == {
        'lanner#1', 'lanner#2', 'lanner#3'}
    assert {(d.track, d.centre_hz, d.rf_hz) for d in got
            if d.modem == 'lanner' and d.kind == 'FIRST'} == {
        (1, 1103, 7101103), (2, 1603, 7101603), (3, 2103, 7102103)}


def test_a_running_lanner_instance_covers_a_successor_reported_before_its_track_ends(tmp_path):
    script = [_track(4, 1, t_start=3, centre=2300, bw=330),
              _track(5, 2, t_start=4.5, centre=2540, bw=250),
              _track(6, 1, 'track-end', t_start=3, centre=2300, bw=330),
              _track(9, 2, 'track-end', t_start=4.5, centre=2540, bw=250)]
    fan, _ = _run(tmp_path, script, 12)
    assert [s.name for s in fan.started if s.name.startswith('lanner')] == ['lanner#1']


@pytest.mark.parametrize('protocol, kind, station', [('RTTY', 'TEXT', ''),
                                                    ('SITOR-B', 'TEXT', ''),
                                                    ('NAVTEX', 'MESSAGE', 'Q')])
def test_text_merge_keeps_distinct_records_and_prefers_the_track_twin(protocol, kind, station):
    from dataclasses import replace
    from types import SimpleNamespace
    from creance.monitor.aggregate import _Merge
    from creance.monitor.activity import ActivityView

    merge = _Merge(SimpleNamespace(decoded_s=10, place=lambda d: d))
    whole = Detection(t=1, modem='lanner', protocol=protocol, kind=kind,
                      grade=CONFIRMED, station=station, role='sender' if station else '',
                      detail='A 17' if station else '45.45 Bd 170 Hz', text='ONE',
                      offset_hz=100, via='whole-band')
    track = replace(whole, via='per-track', track=2)
    assert merge.take(whole) == []
    assert merge.take(track) == [track]
    repeated = replace(track, t=1.1)
    assert merge.take(repeated) == [repeated]
    different_text = replace(track, t=1.2, text='TWO')
    different_detail = replace(track, t=1.3, detail='A 20')
    assert merge.take(different_text) == [different_text]
    assert merge.take(different_detail) == [different_detail]
    assert merge.release(True) == []
    view = ActivityView()
    for d in [track, different_text, different_detail]:
        assert Detection.from_json(d.to_json()) == d
        view.add(d)
    summary = view.summary(2)
    assert summary['protocols'][protocol]['state'] == 'active_confirmed'
    assert summary['unidentified_energy'] is False
    if station:
        assert summary['protocols'][protocol]['stations'][0]['call'] == 'Q'
        assert summary['protocols'][protocol]['stations'][0]['role'] == 'sender'


@pytest.mark.parametrize('track, via', [(None, 'whole-band'), (2, 'per-track')])
def test_lanner_repeated_lines_from_one_instance_are_not_twins(track, via):
    from dataclasses import replace
    from creance.monitor.channelizer import same_frame

    first = Detection(t=0.528, modem='lanner', protocol='RTTY', kind='TEXT',
                      grade=CONFIRMED, text='A\n', track=track, via=via)
    second = replace(first, t=1.023)
    third = replace(first, t=1.518)
    assert not same_frame(first, second)
    assert not same_frame(second, third)
    twin = replace(first, track=3, via='per-track', t=0.6)
    assert same_frame(first, twin)


@pytest.mark.parametrize('arrival', ['held-whole', 'track-first', 'released-whole'])
def test_navtex_merge_uses_identity_and_keeps_later_broadcast(arrival):
    from dataclasses import replace
    from types import SimpleNamespace
    from creance.monitor.aggregate import _Merge
    from creance.monitor.channelizer import same_frame

    merge = _Merge(SimpleNamespace(decoded_s=0, place=lambda d: d))
    whole = Detection(t=100, modem='lanner', protocol='NAVTEX', kind='MESSAGE',
                      grade=CONFIRMED, station='Q', detail='A 17',
                      text='COAST GUARD NOTICE TO MARINERS',
                      offset_hz=-30, via='whole-band')
    track = replace(whole, t=102, text='COAST GU_RD NOTICE TO M_RINERS',
                    offset_hz=-9, via='per-track', track=2)
    # Independent dating and garbles still identify copies of one transmission.
    assert same_frame(whole, track) and same_frame(track, whole)
    assert not same_frame(whole, replace(track, station='R'))
    assert not same_frame(whole, replace(track, detail='B 17'))
    assert not same_frame(whole, replace(track, detail='A 20'))
    assert not same_frame(whole, replace(track, station='', detail=''))
    assert same_frame(whole, replace(track, t=103))
    assert not same_frame(whole, replace(track, t=103.01))
    assert not same_frame(whole, replace(track, text='COAST GUARD NOTICE TO FISHERS'))
    urgent = replace(whole, detail='D 00', text='SEARCH AREA ONE')
    assert not same_frame(urgent, replace(track, detail='D 00', text='SEARCH AREA TWO'))
    assert not same_frame(urgent, replace(urgent, via='per-track', track=2, t=150))
    if arrival == 'track-first':
        got = merge.take(track) + merge.take(whole) + merge.release(True)
    else:
        got = merge.take(whole)
        if arrival == 'released-whole':
            got += merge.release(True)
        got += merge.take(track) + merge.release(True)
    assert got == [whole if arrival == 'released-whole' else track]
    later = replace(track, t=400, track=3)
    later_whole = replace(whole, t=402)
    got += merge.take(later_whole) + merge.take(later) + merge.release(True)
    assert got[-1] == later and len(got) == 2


@pytest.mark.parametrize('protocol', ['RTTY', 'SITOR-B'])
def test_lanner_similar_lines_merge_only_across_instances(protocol):
    from dataclasses import replace
    from types import SimpleNamespace
    from creance.monitor.aggregate import _Merge
    from creance.monitor.channelizer import same_frame

    merge = _Merge(SimpleNamespace(decoded_s=0, place=lambda d: d))
    whole = Detection(t=10, modem='lanner', protocol=protocol, kind='TEXT',
                      grade=CONFIRMED, text='THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG',
                      detail='170 Hz', offset_hz=0, via='whole-band')
    track = replace(whole, t=12, text='THE QUICK BR_WN FOX\nJUMPS  OVER THE LA_Y DOG',
                    offset_hz=10, track=1, via='per-track')
    # Two corrupt characters out of 43 match above 90%, with layout ignored.
    assert same_frame(whole, track) and same_frame(track, whole)
    assert not same_frame(whole, replace(track, t=13.01))
    assert not same_frame(whole, replace(track, offset_hz=10.01))
    assert not same_frame(whole, replace(track, text='THE QUICK RED HEN WALKS HOME'))
    assert merge.take(whole) == []
    assert merge.take(track) == [track]
    repeated = replace(track, t=12.1)
    different = replace(track, t=12.2, text='ANOTHER LINE')
    assert merge.take(repeated) == [repeated]
    assert merge.take(different) == [different]
    assert merge.release(True) == []
    # One substitution in 20 characters is allowed (95%); three exceed the tolerance (85%).
    short = replace(whole, text='ABCDEFGHIJKLMNOPQRST')
    assert same_frame(short, replace(track, text='ABCDEFGHI_KLMNOPQRST'))
    assert not same_frame(short, replace(track, text='ABCDEFGHI_KL_NOPQ_ST'))
    assert same_frame(replace(whole, text='HELLO'), replace(track, text='HELLOX'))
    # Popular-character filtering would miss a long repetitive radio line.
    long = replace(whole, text='RY' * 200)
    assert same_frame(long, replace(track, text='RY' * 100 + '_' + 'RY' * 99))


def test_cw_twins_ignore_detail_and_text():
    from dataclasses import replace
    from creance.monitor.channelizer import same_frame

    first = Detection(t=1, modem='cw', protocol='CW', kind='text', station='W1AW',
                      grade=CONFIRMED, text='W1AW', detail='20 WPM', offset_hz=5)
    assert same_frame(first, replace(first, t=1.1, detail='21 WPM'))
    assert same_frame(first, replace(first, t=1.1, text='W1A_'))
    assert not same_frame(first, replace(first, t=2))
    assert not same_frame(first, replace(first, offset_hz=16))


def test_lanner_retirement_waits_for_completeness_and_suppresses_replacement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    tracker, modems = _specs(tmp_path, [])
    fan = Channelizer(WINDOW, tracker, modems, queue.Queue())
    stopped = []
    monkeypatch.setattr(fan, '_stop_all', lambda monitors, grace: stopped.extend(monitors))
    fan._follow(Detection.from_json(_track(4, 1, t_start=3, centre=1500, bw=300)))
    tr = fan.live[1]
    lanner = SimpleNamespace(decoded_s=3)
    kestrel = SimpleNamespace(decoded_s=3)
    tr.started = {('lanner', ''): lanner, ('kestrel', ''): kestrel}
    tr.wanted = set(tr.started)
    fan.modems = {}
    fan._follow(Detection.from_json(_track(10, 1, 'track-end', t_start=3,
                                          centre=1500, bw=300)))
    assert stopped == [kestrel]
    assert fan.running == [lanner]
    fan._follow(Detection.from_json(_track(12, 2, t_start=11, centre=1510, bw=300)))
    fan.live[2].wanted = {('lanner', '')}
    monkeypatch.setattr(fan, '_spawn', lambda *args: pytest.fail('overlapping replacement'))
    fan._grant()
    lanner.decoded_s = 10
    fan._lanner_state[1] = (9, False)
    fan._dispatch()
    assert fan._held
    fan.report(Detection(t=10, modem='lanner', protocol='UNKNOWN',
                         kind='emission-status', detail='open', track=1, grade=TENTATIVE))
    fan._dispatch()
    assert fan._held
    fan.report(Detection(t=10, modem='lanner', protocol='UNKNOWN',
                         kind='emission-status', detail='closed', track=1, grade=TENTATIVE))
    fan._dispatch()
    assert fan.out.empty()
    assert stopped == [kestrel, lanner]
    assert fan._held == {}
    # A lost terminator cannot keep a process beyond the emission bound.
    from dataclasses import replace
    lanner.decoded_s = 3
    fan._held[1] = replace(tr, started={('lanner', ''): lanner}, ended_at=10)
    fan.tracker = SimpleNamespace(decoded_s=610)
    fan._dispatch()
    assert fan._held == {}
    assert stopped == [kestrel, lanner, lanner]


def _run_lanner_retirement_emission(high, monkeypatch):
    import io
    import numpy as np
    from pathlib import Path
    from hfhost.audio import AudioSource
    from creance.monitor import aggregate
    from creance.monitor.driver import default_specs

    samples = np.random.default_rng(17).normal(0, .002, 10 * FS + len(high) + 3 * FS)
    samples[10 * FS:10 * FS + len(high)] += high
    pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes()

    class Bulletin(AudioSource):
        window = WINDOW

        def frames(self):
            for start in range(0, len(samples), FRAME):
                yield start, pcm[2 * start:2 * (start + FRAME)]

    root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv('PYTHONPATH', ':'.join(str(root / 'packages' / name)
                                           for name in ('hfmodem', 'creance', 'hfhost')))
    monkeypatch.setenv('CREANCE_MODEM_PYTHON', sys.executable)
    records = []
    aggregate.run(Bulletin(), [s for s in default_specs() if s.name == 'lanner'],
                  sinks=[records.append], out=io.StringIO(), live=False,
                  dispatch='per-track', drain_timeout=120)
    return records


@pytest.mark.slow
def test_real_channelizer_keeps_a_600_character_navtex_emission(tmp_path, monkeypatch):
    import numpy as np
    from scipy.signal import hilbert, resample_poly
    from hfmodem.lanner import navtex

    body = 'LONG MESSAGE 0123456789\n' * 25
    assert len(body) == 600
    low = navtex.modulate(navtex.Message('Q', 'A', 17, body), fs=6000, phasing_pairs=72)
    high = resample_poly(np.real(hilbert(low) * np.exp(
        2j * np.pi * 600 * np.arange(len(low)) / 6000)), 8, 1)
    records = _run_lanner_retirement_emission(high, monkeypatch)
    messages = [d for d in records if d.protocol == 'NAVTEX']
    assert [(d.station, d.detail, d.text) for d in messages] == [('Q', 'A 17', body)]
    assert any(d.kind == 'track-end' and d.t < 60 for d in records)


@pytest.mark.slow
def test_real_channelizer_keeps_a_630_character_rtty_emission(tmp_path, monkeypatch):
    from hfmodem.lanner import rtty

    sent = 'LONG RTTY 0123456789\n' * 30
    assert len(sent) == 630
    high = rtty.modulate(sent, mark_hz=2385, space_hz=2215)
    records = _run_lanner_retirement_emission(high, monkeypatch)
    assert ''.join(d.text for d in records if d.protocol == 'RTTY') == sent
    assert any(d.kind == 'track-end' and 35 < d.t < 45 for d in records)


@pytest.mark.slow
@pytest.mark.parametrize('dispatch', [[], ['--per-track'], ['--whole-band']])
def test_real_station_q_console(tmp_path, dispatch):
    """Opt-in recording fixture: build it with scratch/real/convert.py first."""
    import os
    import re
    import subprocess
    from pathlib import Path

    wav = Path('/Users/saul/src/radio/worktrees/lanner-scratch/real/station-q-48k.wav')
    if not wav.is_file():
        pytest.skip('converted station-Q recording is not installed')
    root = Path(__file__).resolve().parents[3]
    config = (root / 'examples/station.toml').read_text()
    config = config.replace('capture_dir = "~/radio/captures"',
                            f'capture_dir = "{tmp_path / "captures"}"')
    station = tmp_path / 'station.toml'
    station.write_text(config)
    env = dict(os.environ, HFMODEM_STATION=str(station),
               CREANCE_MODEM_PYTHON=sys.executable,
               PYTHONPATH=':'.join(str(root / 'packages' / name)
                                  for name in ('hfmodem', 'creance', 'hfhost')))
    completed = subprocess.run(
        [sys.executable, 'tools/console.py', 'monitor', '--wav', str(wav),
         '--no-record', *dispatch], cwd=root, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    (tmp_path / 'console.log').write_text(completed.stdout)
    assert completed.returncode == 0, completed.stdout[-4000:]
    serials = re.findall(r'navtex\s+[^\n]*?from Q\s+A (\d\d)\s', completed.stdout)
    assert sorted(serials) == ['05', '15', '17', '20'], serials
    confirmed_rtty = re.findall(r'^\s*\d+\.\d+s\s+rtty\b.*$', completed.stdout, re.MULTILINE)
    assert confirmed_rtty == [], confirmed_rtty


def test_navtex_copies_allow_local_garble_damage_but_reject_intact_changes():
    from dataclasses import replace
    from creance.monitor.channelizer import same_frame

    whole = Detection(t=100, modem='lanner', protocol='NAVTEX', kind='MESSAGE',
                      grade=CONFIRMED, station='Q', detail='D 00', via='whole-band',
                      text='COAST GUARD _NOTICE TO MARINERS. ' + 'SEARCH AREA ONE. ' * 10)
    track = replace(whole, via='per-track', track=2, t=101,
                    text=whole.text.replace('_NOTICE', '__N_TICE'))
    assert same_frame(whole, track)
    assert not same_frame(whole, replace(track, text=track.text[:-5] + 'TWO. '))
    assert not same_frame(whole, replace(track, text=track.text.replace('ONE', 'TWO')))


def test_navtex_copies_allow_shift_damage_near_a_garble():
    from dataclasses import replace
    from creance.monitor.channelizer import same_frame

    text = 'PREVIOUS BULLETIN. ' * 10 + 'D. SEC SSD BN_ 0104-26 : SAFETF, ISSUED 312247Z AUG 26.'
    first = Detection(t=689.2, modem='lanner', protocol='NAVTEX', kind='MESSAGE',
                      grade=CONFIRMED, station='Q', detail='A 05', via='whole-band', text=text)
    copy = replace(first, via='per-track', track=2,
                   text=text.replace('BN_', 'BNH').replace('SAFETF', 'SAFETY'))
    assert same_frame(first, copy)


@pytest.mark.parametrize('frequency_field', ['rf_hz', 'centre_hz', 'offset_hz'])
def test_navtex_same_identity_on_distinct_frequencies_survives_every_filter(frequency_field):
    from dataclasses import replace
    from types import SimpleNamespace
    from creance.monitor.aggregate import _Merge
    from creance.monitor.channelizer import same_frame, tracker_spec
    from creance.monitor.driver import default_specs

    spec = next(s for s in default_specs() if s.name == 'lanner')
    out = queue.Queue()
    window = Window(300, 2900, 'probe', 7100000)
    fan = Channelizer(window, tracker_spec(spec, window), [spec], out)
    first = Detection(t=100, modem='lanner', protocol='NAVTEX', kind='MESSAGE',
                      grade=CONFIRMED, station='Q', detail='D 00',
                      text='URGENT SEARCH AREA ONE. ' * 5, via='per-track', track=1,
                      **{frequency_field: 1300})
    second = replace(first, track=2, **{frequency_field: 2300})
    assert not same_frame(first, second)
    assert not same_frame(second, first)
    merge = _Merge(SimpleNamespace(decoded_s=1000, place=lambda d: d))
    assert merge.take(first) + merge.take(second) + merge.release(True) == [first, second]
    fan.report(first)
    fan.report(second)
    assert [out.get_nowait(), out.get_nowait()] == [first, second]


@pytest.mark.parametrize('delta, matches', [(24, True), (40, True), (40.1, False)])
def test_navtex_frequency_tolerance_covers_real_dispatch_fits(delta, matches):
    from dataclasses import replace
    from creance.monitor.channelizer import same_frame

    first = Detection(t=100, modem='lanner', protocol='NAVTEX', kind='MESSAGE',
                      grade=CONFIRMED, station='Q', detail='A 05', text='NOTICE TEXT',
                      centre_hz=1479, via='whole-band')
    second = replace(first, centre_hz=1479 + delta, via='per-track', track=2)
    assert same_frame(first, second) is matches
    assert same_frame(second, first) is matches


@pytest.mark.slow
def test_live_track_material_move_recovers_all_five_rtty_lines(monkeypatch):
    import numpy as np
    from hfmodem.lanner import rtty

    first = 'FIRST STATION 12345\n' * 2
    second = 'SECOND STATION 67890\n' * 3
    high = np.r_[rtty.modulate(first, mark_hz=2385, space_hz=2215),
                 rtty.modulate(second, mark_hz=2430, space_hz=2260)]
    records = _run_lanner_retirement_emission(high, monkeypatch)
    text = [r for r in records if r.protocol == 'RTTY']
    assert [r.text for r in text] == first.splitlines(True) + second.splitlines(True)
    assert all(r.grade == CONFIRMED for r in text)
    assert len({r.track for r in text}) == 1


@pytest.mark.slow
def test_live_track_mark_idle_preserves_figures(monkeypatch):
    import numpy as np
    from hfmodem.lanner import ita2, rtty

    sent = '12345678901234567890\n'
    schedule = [(10, True)]
    for i, code in enumerate(ita2.encode(sent)):
        schedule.extend([(1, False)] + [(1, bool(code >> k & 1)) for k in range(5)]
                        + [(1.5, True)])
        if i == 7:
            schedule.append((10 * rtty.BAUD, True))
    schedule.append((10, True))
    bounds = np.rint(np.r_[0, np.cumsum([n for n, _ in schedule])] * FS / rtty.BAUD).astype(int)
    frequencies = np.repeat([2385 if mark else 2215 for _, mark in schedule], np.diff(bounds))
    high = .5 * np.sin(2 * np.pi * np.cumsum(frequencies) / FS)
    records = _run_lanner_retirement_emission(high, monkeypatch)
    text = [r for r in records if r.protocol == 'RTTY']
    assert ''.join(r.text for r in text) == sent
    assert all(r.grade == CONFIRMED for r in text)
    assert len({r.track for r in text}) == 1


@pytest.mark.slow
def test_live_track_short_gap_recovers_a_weak_successor(monkeypatch):
    import numpy as np
    from hfmodem.lanner import rtty

    first, second = 'FIRST STATION 12345\n', 'SECOND STATION 67890\n'
    high = np.r_[rtty.modulate(first, mark_hz=2385, space_hz=2215),
                 np.zeros(FS // 10),
                 .05 * rtty.modulate(second, mark_hz=2385, space_hz=2215)]
    records = _run_lanner_retirement_emission(high, monkeypatch)
    text = [r for r in records if r.protocol == 'RTTY']
    assert [r.text for r in text] == [first, second]
    assert all(r.grade == CONFIRMED for r in text)


@pytest.mark.slow
@pytest.mark.parametrize('keyed', ['hard', 'shaped'], indirect=True)
def test_live_track_shared_mark_successor_is_a_new_transmission(monkeypatch, keyed):
    import numpy as np

    first, second = 'FIRST 12345\n', 'SECOND STATION 67890\n' * 3
    high = np.r_[keyed(first, mark_hz=2385, space_hz=2215), np.zeros(FS // 2),
                 .7 * keyed(second, mark_hz=2555, space_hz=2385)]
    records = _run_lanner_retirement_emission(high, monkeypatch)
    text = [r for r in records if r.protocol == 'RTTY']
    assert [r.text for r in text] == first.splitlines(True) + second.splitlines(True)
    assert all(r.grade == CONFIRMED for r in text)


@pytest.mark.slow
def test_live_track_starting_mid_sitor_never_confirms_rtty(monkeypatch):
    import numpy as np
    from scipy.signal import hilbert
    from hfmodem.lanner import sitor

    sent = sitor.modulate('LONG BULLETIN 0123456789\n' * 12)[5 * FS:]
    high = np.real(hilbert(sent) * np.exp(2j * np.pi * 600 * np.arange(len(sent)) / FS))
    records = _run_lanner_retirement_emission(high, monkeypatch)
    assert any(r.modem == 'lanner' and r.protocol == 'RTTY' for r in records)
    assert all(r.grade == TENTATIVE for r in records if r.modem == 'lanner' and r.protocol == 'RTTY')
