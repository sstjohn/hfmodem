# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace
import os
import signal
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core.audio import Block
from hfmodem.core import wav
from hfmodem.station import process
from hfmodem.station.capture import StationCapture
from hfmodem.tests.station.test_process import configuration, fake_claim, station  # noqa: F401 -- fixtures
from hfmodem.tests.core.test_rig import fake as fake_cat  # noqa: F401 -- a fixture


@pytest.fixture
def policy_station(station, monkeypatch, fake_cat):  # noqa: F811
    from hfmodem.core.rig import Cat
    from hfmodem.tests.core.fakerig import FakePtt
    station.cfg = replace(station.cfg,
        station=replace(station.cfg.station, regulatory='part97',
                        regulatory_settings={'licence': 'general'}),
        rig=replace(station.cfg.rig, model='ft891', identity='ft891',
                    power_w=25, power_clamp_w=25,
                    ptt=replace(station.cfg.rig.ptt, port='/fake/ptt')))
    line = FakePtt()
    line.open = lambda: None
    fake_cat.ptt_type = 'None'
    monkeypatch.setattr(process, 'cat_driver', lambda cfg: Cat('127.0.0.1', fake_cat.port))
    monkeypatch.setattr(process, 'RtsPtt', lambda port: line)
    monkeypatch.setattr(station, 'open_rig', lambda **kw: process.Station.open_rig(station, **kw))
    yield station, fake_cat, line
    if station.rig is not None:
        station.rig.close()


@pytest.mark.parametrize('launcher', [False, True])
@pytest.mark.parametrize('model', ['', '?', 'error', 'mismatch', 'Yaesu FTDX10'])
def test_launcher_unreadable_model_is_advisory_through_real_rig_arm(
        policy_station, monkeypatch, capsys, launcher, model):
    from hfmodem.core.rig import Cat, IdentityMismatch, RigError
    st, cat, line = policy_station
    st.ownership_preflight = not launcher
    cat.model = model
    # A serial CAT driver reads the radio's ID and raises when it is the wrong one:
    # read, not unreadable, so it refuses on a launcher arm too.
    failure = {'error': RigError('missing Model name field'),
               'mismatch': IdentityMismatch('unexpected Yaesu CAT identity')}.get(model)
    if failure:
        def identify(self):
            raise failure
        monkeypatch.setattr(Cat, 'identify', identify)
    if launcher and model not in ('mismatch', 'Yaesu FTDX10'):
        rig = st.open_rig()
        assert rig._armed and not rig.keyed
        assert 'warning: rig model via rigctld is unreadable' in capsys.readouterr().out
    else:
        with pytest.raises(RigError):
            st.open_rig()
    assert not line.line and 'assert up' not in line.calls


@pytest.mark.parametrize('launcher', [False, True])
def test_a_profile_refusal_never_keys_a_launcher_arm_either(policy_station, capsys, launcher):
    """rpr-mail's launcher arm once printed part97's objection as a warning and
    keyed anyway; every other station refuses the emission."""
    from hfmodem.core.regulatory import centred
    from hfmodem.core.rig import RigError
    st, cat, line = policy_station
    cfg = st.cfg
    st.ownership_preflight = not launcher
    cat.freq = 10138500
    rig = st.open_rig()
    emission = centred(rig.dial(), 500, power_w=None)
    with pytest.raises(RigError, match='refused by part97: part97 power:'):
        rig.key(emission, why='fake RPR', duration_s=.1)
    assert not line.line and 'assert up' not in line.calls and not rig.keyed
    st._poll_stop.set()
    process.Station._tick(st)
    assert 'PTT ON' not in capsys.readouterr().out
    assert st.cfg is cfg and cfg.station.transmit and cfg.rig.power_clamp_w == 25


def test_launcher_arm_and_keying_print_the_line_like_every_other_keyer(policy_station, capsys):
    """rpr-mail on the Windows rig printed PTT UNPROVEN, "ownership NOT verified"
    and "proof skipped", then keyed three bursts and an ident with not one line
    saying the keying line went up or came back down."""
    from hfmodem.core.regulatory import centred
    st, cat, line = policy_station
    st.ownership_preflight = False
    cat.freq = 10138500
    rig = st.open_rig()
    arm = capsys.readouterr().out
    assert 'PTT line ownership verified' in arm
    assert 'keying line reads LOW at arm' in arm
    assert 'PTT is the keying line on /dev/fake-ptt; CAT tunes only' in arm
    assert 'proof skipped' not in arm and 'assert up' not in line.calls
    emission = centred(rig.dial(), 500, power_w=st.cfg.declared_power_w)
    rig.unkey(rig.key(emission, why='rpr transmit', duration_s=.1))
    st._poll_stop.set()
    process.Station._tick(st)
    assert capsys.readouterr().out.splitlines()[-2:] == [
        'PTT ON -> line (rpr transmit)', 'PTT OFF -> line (reads low)']


def test_a_stalled_transcript_holds_neither_the_key_nor_the_unkey(policy_station, monkeypatch, capsys):
    """print(flush=True) waits on whatever reads the transcript. Under the CAT
    hold with the line up, that wait was the burst's: the audio started late, or
    the arbiter's key deadline went by, and an unkey could not report done."""
    import builtins
    import threading
    from hfmodem.core.regulatory import centred
    st, cat, line = policy_station
    st.ownership_preflight = False
    cat.freq = 10138500
    rig = st.open_rig()
    capsys.readouterr()
    stalled, release = threading.Event(), threading.Event()
    def stall(*args, **kwargs):
        if release.is_set():
            return builtins.print(*args, **kwargs)
        stalled.set()
        release.wait(5)
    monkeypatch.setattr(process, 'print', stall, raising=False)
    emission = centred(rig.dial(), 500, power_w=st.cfg.declared_power_w)
    done = threading.Event()
    def burst():
        rig.unkey(rig.key(emission, why='rpr transmit', duration_s=.1))
        done.set()
    threading.Thread(target=burst, daemon=True).start()
    try:
        assert done.wait(2), 'key or unkey waited on the transcript'
        assert not stalled.is_set() and not line.line
    finally:
        release.set()
    st._poll_stop.set()
    process.Station._tick(st)
    out = capsys.readouterr().out.splitlines()
    assert out.index('PTT ON -> line (rpr transmit)') < out.index('PTT OFF -> line (reads low)')


@pytest.mark.parametrize('transcript', ['open', 'broken'])
def test_owed_line_reports_print_after_the_rig_closes(policy_station, monkeypatch, capsys, transcript):
    from hfmodem.core.regulatory import centred
    st, cat, line = policy_station
    st.ownership_preflight = False
    cat.freq = 10138500
    rig = st.open_rig()
    emission = centred(rig.dial(), 500, power_w=st.cfg.declared_power_w)
    for _ in range(40):
        rig.unkey(rig.key(emission, why='fake RPR', duration_s=.1))
    assert len(rig.advisories) == rig.advisories.maxlen
    owed = list(rig.advisories)
    closed = []
    close = rig.close
    monkeypatch.setattr(rig, 'close', lambda: closed.append(len(rig.advisories)) or close())
    if transcript == 'broken':
        def broken(*args, **kwargs):
            raise BrokenPipeError
        monkeypatch.setattr(process, 'print', broken, raising=False)
    st._teardown(reason='done')
    assert closed == [rig.advisories.maxlen] and st.rig is None and not line.line
    out = capsys.readouterr().out
    if transcript == 'open':
        assert '\n'.join(owed) in out
    else:
        assert not any(line in out for line in owed)
    assert transcript == 'open' or len(rig.advisories) == rig.advisories.maxlen - 1


def test_a_failed_close_escalates_without_waiting_on_advisory_output(policy_station, monkeypatch):
    from hfmodem.core.operations import LeaseTimeout
    from hfmodem.core.regulatory import centred
    st, cat, line = policy_station
    st.ownership_preflight = False
    cat.freq = 10138500
    rig = st.open_rig()
    rig.unkey(rig.key(centred(rig.dial(), 500, power_w=st.cfg.declared_power_w),
                      why='fake RPR', duration_s=.1))
    assert rig.advisories
    close = rig.close
    def failed():
        close()
        monkeypatch.setattr(rig, 'close', close)
        raise OSError('fake close failure')
    monkeypatch.setattr(rig, 'close', failed)
    def stalled(*args, file=None, **kwargs):
        if file is None:
            pytest.fail('advisory output before escalation')
    monkeypatch.setattr(process, 'print', stalled, raising=False)
    with pytest.raises(LeaseTimeout, match='fake close failure'):
        st._teardown(reason='done')
    assert any(call[0] == 'fatal' for call in st.calls if isinstance(call, tuple))
    assert rig.advisories


def test_capture_keeps_keyed_intervals_and_reports_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr(wav, 'store', lambda *a, **kw: {})
    capture = StationCapture(tmp_path, 'memory')
    capture.push(Block(0., 0, np.ones(10, np.float32)*.1))
    capture.flush_to(10)
    capture.resync(10)
    capture.push(Block(0., 10, np.ones(10, np.float32)*.2))
    capture.push(Block(0., 25, np.ones(5, np.float32)*.3))
    capture.drain()
    capture.close()
    from hfhost.recordings import offset_segments
    row = json.loads((capture.directory/'recording.json').read_text())['lanes']['rx']
    parts = offset_segments(capture.directory / row['source'])
    assert list(parts) == [0, 25]
    blocks = [wavfile.read(wav.source(segment)) for segment in parts.values()]
    rate = blocks[0][0]
    samples = np.concatenate([block for _, block in blocks])
    assert rate == 48000 and len(samples) == 25
    np.testing.assert_allclose(samples/32768., [.1]*10+[.2]*10+[.3]*5, atol=1/32768)
    assert row['missing_samples'] == 5 and row['gaps'] == [[20, 25]]


def test_station_subscribes_capture_before_audio_opens_and_drains_it(tmp_path, monkeypatch):
    calls = []
    cfg = configuration(transmit=False)
    cfg = replace(cfg, audio=replace(cfg.audio, capture_dir=str(tmp_path)))
    st = process.Station(cfg)
    class Audio:
        def __init__(self, **kwargs):
            pass
        def subscribe(self, lane):
            self.lane = lane
            calls.append('subscribe')
        def open(self):
            calls.append('open')
            self.lane.push(Block(0., 0, np.zeros(10, np.float32)))
    monkeypatch.setattr(process, 'StationAudio', Audio)
    monkeypatch.setattr(process.devices, 'find_device', lambda *a, **kw: 0)
    monkeypatch.setattr(wav, 'store', lambda *a, **kw: {})
    st.open_audio()
    assert calls == ['subscribe', 'open']
    st.capture.drain()
    st.capture.close()
    assert wav.read(st.capture.directory / st.capture.rx.row['source']).shape == (10,)


def test_capture_preserves_recordings_across_release_and_reown(station, tmp_path, monkeypatch):  # noqa: F811
    for name in ('stream.wav', 'stream.json', 'stream.flac'):
        (tmp_path/name).write_bytes(b'earlier recording')
    station.cfg = replace(station.cfg, audio=replace(station.cfg.audio, capture_dir=str(tmp_path)))
    class Audio:
        def __init__(self, **kwargs):
            pass
        def subscribe(self, lane):
            self.lane = lane
        def open(self):
            self.lane.push(Block(0., 0, np.ones(10, np.float32)*.1))
        def breathe(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(process, 'StationAudio', Audio)
    monkeypatch.setattr(process.devices, 'find_device', lambda *a, **kw: 0)
    monkeypatch.setattr(wav, 'store', lambda *a, **kw: {})
    monkeypatch.setattr(station, 'open_audio', lambda: process.Station.open_audio(station))
    station.own()
    first = station.capture.directory
    station.release()
    before = {p.relative_to(first): p.read_bytes() for p in first.rglob('*') if p.is_file()}
    station.own()
    second = station.capture.directory
    station.release()
    assert first != second
    assert {p.relative_to(first): p.read_bytes() for p in first.rglob('*') if p.is_file()} == before
    row = json.loads((second / 'recording.json').read_text())['lanes']['rx']
    assert wav.read(second / row['source']).shape == (10,)
    for name in ('stream.wav', 'stream.json', 'stream.flac'):
        assert (tmp_path/name).read_bytes() == b'earlier recording'


def test_capture_records_callback_loss_and_overflow_with_contiguous_indices(tmp_path, monkeypatch):
    from hfmodem.core.audio import StationAudio
    monkeypatch.setattr(wav, 'store', lambda *a, **kw: {})
    audio = StationAudio(input_device=None, output_device=None)
    capture = audio.subscribe(StationCapture(tmp_path, 'callback', audio=audio))
    samples = np.zeros((128, 1), np.float32)
    for stamp, overflow in [(1., False), (1.+256/48000, True)]:
        audio._callback(samples, np.zeros_like(samples), 128,
                        SimpleNamespace(inputBufferAdcTime=stamp, outputBufferDacTime=stamp),
                        SimpleNamespace(input_overflow=overflow))
    assert audio.failure is None
    capture.close()
    facts = json.loads((capture.directory/'recording.json').read_text())['lanes']['rx']
    assert facts['audio_lost_samples'] == audio.lost == 128
    assert facts['audio_overflows'] == audio.overflows == 1
    assert facts['audio_underruns'] == 0
    assert facts['missing_samples'] == 0


@pytest.mark.parametrize('transmit', [False, True])
def test_launcher_skips_measurements_without_changing_transmit(station, transmit):  # noqa: F811
    station.cfg = replace(station.cfg, station=replace(station.cfg.station, transmit=transmit))
    station.ownership_preflight = False
    assert station.own()['state'] == ('owning' if transmit else 'owning-receive')
    assert station.cfg.station.transmit is transmit
    assert 'reference' not in station.calls and 'measure' not in station.calls
    assert ('rig', False) in station.calls
    station.release()


@pytest.mark.parametrize('launcher', [False, True])
@pytest.mark.parametrize('mail_exit', ['success', 'failure', 'start-failure', 'interruption'])
def test_closing_identity_finishes_before_station_devices_close(
        station, monkeypatch, tools, tmp_path, capsys, launcher, mail_exit):  # noqa: F811
    rpr = pytest.importorskip('consolelib.rpr', reason='the RPR arm verdict ships from the repo, not the package')
    from hfmodem.core import cwid, identification
    from hfmodem.core.tx_events import read_line
    from hfmodem.station.arbiter import TxArbiter
    station.cfg = replace(station.cfg, station=replace(station.cfg.station,
                          mycall='TACTIC', idcall='N1CALL', id_mode='cw'))
    station.cw_identity = identification.from_station(station.cfg.station)
    station.closing_cwid = True
    station.ownership_preflight = not launcher
    monkeypatch.setattr(station, 'build_arbiter', lambda: process.Station.build_arbiter(station))
    audio = cwid.audio
    tone = 1500 if launcher else cwid.TONE_HZ
    def cw(call, **kw):
        station.calls.append(('cw', call))
        return audio(call, **kw)
    monkeypatch.setattr(cwid, 'audio', cw)
    def submit(arbiter, request):
        np.testing.assert_array_equal(request.audio, audio('N1CALL', tone_hz=tone))
        spectrum = np.abs(np.fft.rfft(request.audio))
        frequency = np.fft.rfftfreq(len(request.audio), 1/48000)[np.argmax(spectrum)]
        assert frequency == pytest.approx(tone, abs=1)
        assert request.emission.audio_lo_hz == tone - cwid.BANDWIDTH_HZ/2
        assert request.emission.audio_hi_hz == tone + cwid.BANDWIDTH_HZ/2
        station.calls.append(('identified', request.protocol))
        return 123
    monkeypatch.setattr(TxArbiter, 'submit', submit)
    station.own()
    station.rig.dial = lambda: 3597000
    assert not station.arbiter.identity.due
    def exchange(st):
        station.calls.append('mail-start')
        if mail_exit == 'interruption':
            import signal
            station.graceful_signals = True
            station._signal(signal.SIGINT, None)
            station._handle_signals()
            if hasattr(signal, 'SIGUSR1'):
                station._signal(signal.SIGUSR1, None)
                station._handle_signals()
            assert station._stop.is_set() and station._panic_thread is None
        if mail_exit == 'start-failure':
            raise OSError('fake mail start failure')
        station._stop.set()
        return 0 if mail_exit == 'success' else 1
    monkeypatch.setattr(station, '_install_signals', lambda: None)
    if mail_exit == 'start-failure':
        with pytest.raises(OSError, match='fake mail start failure'):
            station.run(on_ready=exchange)
    else:
        assert station.run(on_ready=exchange) == (130 if mail_exit == 'interruption' else 0)
    assert station.calls.index('mail-start') < station.calls.index('lane-stop')
    assert station.calls.index('lane-stop') < station.calls.index(('cw', 'N1CALL'))
    assert station.calls.index(('identified', 'ident')) < station.calls.index('rig-close')
    assert station.calls.count(('cw', 'N1CALL')) == 1
    station.release()
    assert station.calls.count(('cw', 'N1CALL')) == 1
    transcript = capsys.readouterr().out
    events = [event for line in transcript.splitlines() if (event := read_line(line))]
    assert len(events) == 1
    assert events[0]['protocol'] == 'ident' and events[0]['outcome'] == 'sent'
    assert events[0]['event'] == 'completed' and events[0]['call'] == 'N1CALL'
    (tmp_path/'arm.log').write_text(transcript)
    verdict = rpr.verdict(rpr.PROFILE, tmp_path)
    assert verdict['keyed'] and verdict['reached'] == ['keyed']
    assert not verdict['connected']


@pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='launcher stop uses POSIX SIGUSR1')
@pytest.mark.parametrize('first', [*process.FATAL_SIGNALS, getattr(signal, 'SIGUSR1', None)])
@pytest.mark.parametrize('second', process.FATAL_SIGNALS)
@pytest.mark.parametrize('handled', [False, True])
def test_mail_arm_stop_then_independent_signal_panics(station, monkeypatch, first, second, handled):  # noqa: F811
    import threading
    station.graceful_signals = True
    panicked = threading.Event()
    station.rig = SimpleNamespace(panic=panicked.set)
    monkeypatch.setattr(process, 'arm_deadman', lambda *a: None)
    station._signal(first, None)
    if handled:
        station._handle_signals()
        assert station._stop.is_set() and station._own_cancel.is_set()
        assert not panicked.is_set()
    station._signal(second, None)
    station._handle_signals()
    assert station._signals == 2 and panicked.wait(1)
    station._panic_thread.join(1)


@pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='launcher stop uses POSIX SIGUSR1')
@pytest.mark.parametrize('first', [*process.FATAL_SIGNALS, getattr(signal, 'SIGUSR1', None)])
@pytest.mark.parametrize('handled', [False, True])
def test_mail_arm_launcher_stop_is_idempotent(station, first, handled):  # noqa: F811
    station.graceful_signals = True
    station._signal(first, None)
    if handled:
        station._handle_signals()
    for _ in range(3):
        station._signal(signal.SIGUSR1, None)
        station._handle_signals()
    assert station._signal_number == first and station._signals == 1
    assert station._stop.is_set() and station._own_cancel.is_set()
    assert station._panic_thread is None


@pytest.mark.skipif(not hasattr(signal, 'SIGUSR1'), reason='launcher stop uses POSIX SIGUSR1')
@pytest.mark.parametrize('graceful', [False, True])
def test_launcher_stop_handler_is_only_installed_for_mail_arm(station, monkeypatch, graceful):  # noqa: F811
    installed = {}
    station.graceful_signals = graceful
    monkeypatch.setattr(signal, 'signal', lambda sig, handler: installed.setdefault(sig, handler))
    station._install_signals()
    assert set(installed) == set(process.FATAL_SIGNALS) | ({signal.SIGUSR1} if graceful else set())


def test_ordinary_station_int_then_term_still_panics(station, monkeypatch):  # noqa: F811
    import signal
    import threading
    panicked = threading.Event()
    station.rig = SimpleNamespace(panic=panicked.set)
    monkeypatch.setattr(process, 'arm_deadman', lambda *a: None)
    station._signal(signal.SIGINT, None)
    station._signal(signal.SIGTERM, None)
    station._handle_signals()
    assert station._signals == 2 and panicked.wait(1)
    station._panic_thread.join(1)


def test_fatal_shutdown_term_is_counted_after_graceful_stop(station, monkeypatch):  # noqa: F811
    import signal
    station.graceful_signals = True
    station._signal(signal.SIGINT, None)
    station._handle_signals()
    station._fatal_reason = 'close failed'
    station._signal(signal.SIGTERM, None)
    assert station._signals == 2 and station._signal_number == signal.SIGTERM


def test_failed_identification_does_not_publish_completed_transmission(station, monkeypatch, capsys):  # noqa: F811
    from hfmodem.station.arbiter import Identity, Refused, TxArbiter
    station.arbiter = TxArbiter(None, None, identity=Identity('N1CALL'))
    station.rig = SimpleNamespace(dial=lambda: 3597000)
    def submit(*args):
        raise Refused('fake ID refusal')
    monkeypatch.setattr(TxArbiter, 'submit', submit)
    with pytest.raises(Refused, match='fake ID refusal'):
        station.identify()
    assert 'tx_event' not in capsys.readouterr().out


@pytest.mark.parametrize('fail', [False, True])
def test_station_inherits_launcher_admission_and_closes_without_releasing_it(station, monkeypatch, fail):  # noqa: F811
    calls = []
    admission = SimpleNamespace(claim=object(), active=True, opened=lambda: calls.append('opened'),
                                failed=lambda: calls.append('failed'))
    monkeypatch.setenv('RIG_CLAIMED', '1')
    if os.name == 'nt':
        adopted = object()
        def adopt():
            calls.append('inherited')
            return adopted
        def received_admission(descriptor):
            assert descriptor is adopted
            return admission
        monkeypatch.setattr(process.claim, 'adopt', adopt)
        monkeypatch.setattr(process.claim.winclaim, 'received_admission', received_admission)
    else:
        def inherited(fd, launch=False):
            assert fd == 9 and launch is True
            calls.append('inherited')
            return admission
        monkeypatch.setattr(process.claim, 'inherited', inherited)
    monkeypatch.setattr(process.claim, 'acquire', lambda *a, **kw: pytest.fail('acquired another claim'))
    monkeypatch.setattr(process.claim, 'release', lambda *a: pytest.fail('released parent claim'))
    monkeypatch.setattr(process.claim, 'close', lambda descriptor: calls.append('closed'))
    if fail:
        monkeypatch.setattr(station, 'open_audio', lambda: (_ for _ in ()).throw(ValueError('fixture')))
        with pytest.raises(ValueError, match='fixture'):
            station.own()
        assert calls == ['inherited', 'closed', 'failed']
    else:
        assert station.own()['ready']
        assert calls == ['inherited', 'opened', 'failed']
        station.release()
        assert calls[-1] == 'closed'


def test_rpr_receive_line_identifies_the_actual_sender_without_payload(capsys):
    from hfmodem.ax25.frame import Frame, address
    from hfmodem.station.link import RprLink
    delivered = []
    link = SimpleNamespace(station=SimpleNamespace(arbiter=None), host=None, _on_frame=delivered.append)
    raw = Frame((address('W9SSJ'), address('W9OTR', last=True)), 0x73, None, b'').encode()
    RprLink.on_frame(link, raw)
    assert capsys.readouterr().err == 'rpr RX from W9OTR control=0x73\n'
    assert delivered == [raw]


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
def test_station_capture_persists_timestamp_uncertainty(tmp_path, monkeypatch, host):
    from hfmodem.core.audio import timestamp_limitation
    limitation = timestamp_limitation(host)
    monkeypatch.setattr(wav, 'store', lambda *a, **kw: {})
    audio = SimpleNamespace(lost=0 if limitation else 1920, overflows=0, underruns=0,
        timestamp_limitation=limitation, estimated_lost=1920 if limitation else 0)
    capture = StationCapture(tmp_path, 'mock stamps', audio=audio)
    capture.push(Block(0., 0, np.full(8192, .1, np.float32)))
    capture.close()
    stats = json.loads((capture.directory/'recording.json').read_text())['lanes']['rx']
    assert stats['missing_samples'] == 0
    if limitation:
        assert stats['timestamp_limitation'] == limitation
        assert stats['estimated_lost_samples'] == 1920
        assert stats['continuous'] is None
        assert stats['audio_lost_samples'] == 0
    else:
        assert stats['audio_lost_samples'] == 1920
        assert 'timestamp_limitation' not in stats and 'continuous' not in stats
