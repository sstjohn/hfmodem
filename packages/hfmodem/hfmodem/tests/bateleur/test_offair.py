# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The WAV tool through its argument parser."""
from __future__ import annotations

import json

from hfmodem.core import levels
from hfmodem.bateleur import offair

def test_render_then_decode(tmp_path, capsys):
    out = tmp_path / "tx.wav"
    offair.main(["render", "110", str(out), "--mode", "1200S", "--text", "round trip",
                 "--callsign", "N0CALL"])
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["samplerate"] == 48000
    assert abs(side["peak"] - levels.TX_DRIVE) < 1e-3
    got = tmp_path / "rx.bin"
    offair.main(["decode", "110", str(out), "--output", str(got)])
    assert got.read_bytes() == b"round trip"
    assert "1200S" in capsys.readouterr().out


def test_render_the_untrimmed_pulse(tmp_path, capsys):
    out = tmp_path / "wide.wav"
    offair.main(["render", "110", str(out), "--mode", "2400S", "--text", "wide",
                 "--bandwidth", "3200"])
    assert "350 - 3140 Hz" in capsys.readouterr().out
    got = tmp_path / "rx.bin"
    offair.main(["decode", "110", str(out), "--output", str(got)])
    assert got.read_bytes() == b"wide"


def test_39_tone_render_then_decode(tmp_path, capsys):
    """The mode is found from the data; the identification after it is not
    taken for a second transmission."""
    out = tmp_path / "tx.wav"
    offair.main(["render", "39", str(out), "--mode", "600/33", "--text", "thirty-nine",
                 "--callsign", "N0CALL"])
    got = tmp_path / "rx.bin"
    offair.main(["decode", "39", str(out), "--all", "--output", str(got)])
    assert got.read_bytes() == b"thirty-nine"
    printed = capsys.readouterr().out
    assert "39-tone 600/33/A10.8" in printed and "EOM" in printed


def test_16_tone_render_then_decode(tmp_path, capsys):
    """The rate is not on the air; the fill after the message is trimmed;
    the identification after it is not taken for a second transmission."""
    out = tmp_path / "tx.wav"
    offair.main(["render", "16", str(out), "--mode", "1200", "--text", "sixteen",
                 "--callsign", "N0CALL"])
    try:
        offair.main(["decode", "16", str(out)])
    except SystemExit as e:
        assert "--mode" in str(e)
    else:
        raise AssertionError("decoded 16-tone without a rate")
    got = tmp_path / "rx.bin"
    offair.main(["decode", "16", str(out), "--mode", "1200", "--all", "--trim",
                 "--output", str(got)])
    assert got.read_bytes() == b"sixteen"
    assert "16-tone 1200 bit/s" in capsys.readouterr().out


def test_4285_needs_its_mode(tmp_path):
    out = tmp_path / "tx.wav"
    offair.main(["render", "4285", str(out), "--mode", "600S", "--text", "x"])
    try:
        offair.main(["decode", "4285", str(out)])
    except SystemExit as e:
        assert "--mode" in str(e)
    else:
        raise AssertionError("decoded 4285 without a mode")
    got = tmp_path / "rx.bin"
    offair.main(["decode", "4285", str(out), "--mode", "600S", "--trim",
                 "--output", str(got)])
    assert got.read_bytes() == b"x"


def test_39_tone_late_entry(tmp_path, capsys):
    """A recording that starts after the preamble: --late joins it at the
    next framing insertion."""
    from hfmodem.bateleur import tone39
    from hfmodem.bateleur.spec import FS
    from hfmodem.core import wav
    text = b"".join(b"line %04d, late entry\r\n" % i for i in range(120))
    x = tone39.transmit(text, "1200/63")[5 * FS:]
    out = tmp_path / "late.wav"
    wav.write(str(out), x * 0.5, FS, source="test")
    got = tmp_path / "rx.bin"
    offair.main(["decode", "39", str(out), "--mode", "1200/63", "--late",
                 "--output", str(got)])
    data = got.read_bytes()
    assert data and text.endswith(data)


def test_fsk_render_then_decode(tmp_path, capsys):
    out = tmp_path / "fsk.wav"
    offair.main(["render", "FSK", str(out), "--mode", "75w", "--text", "naval broadcast",
                 "--callsign", "N0CALL"])
    got = tmp_path / "rx.bin"
    offair.main(["decode", "FSK", str(out), "--mode", "75W", "--all", "--output", str(got)])
    assert got.read_bytes() == b"naval broadcast"
    assert "FSK 75W" in capsys.readouterr().out


def test_4529_render_then_decode(tmp_path, capsys):
    out = tmp_path / "s4529.wav"
    offair.main(["render", "4529", str(out), "--mode", "300s/2000", "--text", "maritime"])
    got = tmp_path / "rx.bin"
    offair.main(["decode", "4529", str(out), "--mode", "300S/2000", "--output", str(got)])
    assert got.read_bytes() == b"maritime"
    assert "4529 300S/2000" in capsys.readouterr().out
