# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The ALE controller: stations linking, talking and hanging up through
simulated channels (``tests.bateleur.ale.ether``)."""
from __future__ import annotations

from hfmodem.tests.bateleur.ale import ether as E
from hfmodem.bateleur.ale.controller import Config

CHANNELS = [f"C{i}" for i in range(10)]


def _pair(snr=10.0, preset="awgn", seed=0, scanning=True, **kw):
    eth = E.Ether(snr_db=snr, preset=preset, seed=seed)
    chans = CHANNELS if scanning else CHANNELS[:1]
    sam = eth.add(Config(["SAM"], chans, scanning=scanning, listen_s=0.0, **kw))
    joe = eth.add(Config(["JOE"], chans, scanning=scanning, listen_s=0.0, **kw))
    eth.run(1.0)
    return eth, sam, joe


def _both_linked(sam, joe):
    return sam.ctl.state == "linked" and joe.ctl.state == "linked"


def _kinds(st):
    return [e.kind for e in st.ctl.events if e.kind != "STATE"]


def test_a_scanning_station_is_called_and_linked_within_14_seconds():
    """A.4.2.3(e): 10 channels scanned at 2 a second; the call's scanning part
    catches the scan, and the handshake completes inside 14 s."""
    eth, sam, joe = _pair()
    t0 = eth.now
    sam.ctl.call("JOE", eth.now, channel=0)
    eth.run(20, until=lambda: _both_linked(sam, joe))
    assert _both_linked(sam, joe)
    assert eth.now - t0 <= 14.2
    assert "CALLED" in _kinds(joe) and "LINKED" in _kinds(sam)


def test_an_amd_rides_the_call_and_reaches_the_operator():
    eth, sam, joe = _pair(scanning=False)
    sam.ctl.call("JOE", eth.now, amd="qsy 7.102")
    eth.run(10, until=lambda: _both_linked(sam, joe))
    assert joe.ctl.amd_log == [("SAM", "QSY 7.102")]


def test_hanging_up_sends_twas_and_both_return_to_available():
    eth, sam, joe = _pair(scanning=False)
    sam.ctl.call("JOE", eth.now)
    eth.run(10, until=lambda: _both_linked(sam, joe))
    sam.ctl.hang_up(eth.now)
    eth.run(6, until=lambda: joe.ctl.state == "available")
    assert sam.ctl.state == "available" and joe.ctl.state == "available"
    assert "LINK_ENDED" in _kinds(joe)


def test_a_quiet_link_is_dropped_after_twa():
    eth, sam, joe = _pair(scanning=False, activity_s=5.0)
    sam.ctl.call("JOE", eth.now)
    eth.run(10, until=lambda: _both_linked(sam, joe))
    eth.run(8)
    assert sam.ctl.state == "available" and joe.ctl.state == "available"


def test_nobody_home_fails_after_every_channel():
    eth = E.Ether(snr_db=10)
    sam = eth.add(Config(["SAM"], CHANNELS[:3], scanning=True, listen_s=0.0))
    eth.run(1.0)
    sam.ctl.call("JOE", eth.now)
    eth.run(60, until=lambda: "CALL_FAILED" in _kinds(sam))
    assert "CALL_FAILED" in _kinds(sam) and sam.ctl.state == "available"


def test_a_sound_is_heard_and_logged():
    eth, sam, joe = _pair(scanning=False)
    sam.ctl.sound(eth.now)
    eth.run(6)
    heard = [e for e in joe.ctl.events if e.kind == "HEARD"]
    assert heard and heard[0].peer == "SAM" and heard[0].lqa


def test_the_poor_channel_links():
    """Table A-II's poor channel at +11 dB asks for 95% of links; four
    fades here."""
    ok = 0
    for seed in range(4):
        eth, sam, joe = _pair(snr=11.0, preset="ale-poor", seed=seed)
        sam.ctl.call("JOE", eth.now, channel=0)
        eth.run(20, until=lambda: _both_linked(sam, joe))
        ok += _both_linked(sam, joe)
    assert ok >= 3


def test_a_call_asking_for_reports_gets_them_in_the_response():
    """A.5.4.2 (LQA, KA1 set), A.5.6.6.1 (version, answered as AMD) and
    A.5.6.6.2 (capabilities): each answered in the called station's
    response."""
    eth, sam, joe = _pair(scanning=False)
    sam.ctl.call("JOE", eth.now, lqa=True, version=True, capabilities=True)
    eth.run(15, until=lambda: _both_linked(sam, joe))
    ev = {e.kind: e for e in sam.ctl.events}
    assert ev["LQA"].lqa["sinad"] > 0 and ev["LQA"].lqa["ber"] < 10
    assert ev["AMD"].text.startswith("HFMODEM BATELEUR")
    assert "channels=1" in ev["CAPABILITIES"].text


def test_do_not_respond_is_honoured():
    from hfmodem.bateleur.ale import words as W
    eth, sam, joe = _pair(scanning=False)
    dnr = W.word(W.CMD, bits=(ord("t") << 14) | (ord("x") << 7))
    sam.ctl._call_message = [dnr]
    sam.ctl.peer = "JOE"
    sam.ctl._send_call(eth.now)
    assert sam.tx_until > eth.now
    eth.run(6)
    assert ("SAM", 0) in joe.ctl.lqa
    assert "CALLED" not in _kinds(joe)
    assert joe.tx_until == 0


def test_each_channel_is_sounded_periodically_and_heard():
    """A.5.3.1: periodic sounding on the prearranged channels."""
    eth = E.Ether(snr_db=10)
    sam = eth.add(Config(["SAM"], CHANNELS[:2], scanning=True, listen_s=0.5, sound_s=60.0))
    joe = eth.add(Config(["JOE"], CHANNELS[:2], scanning=True, listen_s=0.5))
    eth.run(120)
    sounded = [e.channel for e in sam.ctl.events if e.kind == "SOUNDED"]
    assert sorted(set(sounded)) == [0, 1]
    assert any(e.kind == "HEARD" and e.peer == "SAM" for e in joe.ctl.events)


# -- one-to-many (A.5.5.4) -----------------------------------------------------

def _three(nets=None, snr=10.0):
    eth = E.Ether(snr_db=snr)
    nets = nets or {}
    out = []
    for name in ("SAM", "JOE", "BOB"):
        out.append(eth.add(Config([name], CHANNELS[:1], scanning=False, listen_s=0.0,
                                  nets=nets.get(name, {}))))
    for i, st in enumerate(out):
        st.ctl._rng.seed(i)          # JOE and BOB pick different random slots
    eth.run(1.0)
    return (eth, *out)


def _linked_to(st):
    return {e.peer for e in st.ctl.events if e.kind == "LINKED"}


def test_a_star_group_call_links_every_member_that_answers_in_its_slot():
    """A.5.5.4.3: TO JOE, REP BOB twice; each answers in its slot; the
    acknowledgment names both."""
    eth, sam, joe, bob = _three()
    sam.ctl.call(["JOE", "BOB"], eth.now)
    eth.run(20, until=lambda: joe.ctl.state == bob.ctl.state == "linked"
            and sam.ctl.state == "linked")
    assert sam.ctl.linked == {"JOE", "BOB"}
    assert _linked_to(joe) == {"SAM"} and _linked_to(bob) == {"SAM"}


def test_a_group_member_that_is_not_heard_is_not_linked():
    eth, sam, joe, bob = _three()
    eth.reachable = lambda a, b, c: not (a is bob and b is sam)   # BOB's reply is lost
    sam.ctl.call(["JOE", "BOB"], eth.now)
    eth.run(25, until=lambda: bob.ctl.state == "available" and sam.ctl.state == "linked")
    assert sam.ctl.linked == {"JOE"} and joe.ctl.state == "linked"
    assert bob.ctl.state == "available"
    assert any(e.kind == "CALL_FAILED" for e in bob.ctl.events)


def test_a_star_net_call_is_answered_by_members_in_their_slots():
    """A.5.5.4.2: one net address; each member answers with its own, in the
    slot it was assigned; the caller acknowledges the net."""
    nets = {"JOE": {"NET": ("JOE", 1)}, "BOB": {"NET": ("BOB", 2)}}
    eth, sam, joe, bob = _three(nets)
    sam.ctl.call("NET", eth.now, members=["JOE", "BOB"])
    eth.run(20, until=lambda: joe.ctl.state == bob.ctl.state == sam.ctl.state == "linked")
    assert sam.ctl.linked == {"JOE", "BOB"}
    assert _linked_to(joe) == {"SAM"} and _linked_to(bob) == {"SAM"}


def test_an_anycall_is_answered_in_random_slots():
    """A.5.5.4.5: whoever hears it answers in a slot of its choosing."""
    eth, sam, joe, bob = _three()
    sam.ctl.anycall(eth.now)
    # Sixteen slots of 20 Tw: the caller waits some 45 s for the last.
    eth.run(60, until=lambda: joe.ctl.state == bob.ctl.state == sam.ctl.state == "linked")
    assert sam.ctl.linked == {"JOE", "BOB"}


def test_a_selective_anycall_is_answered_only_by_addresses_ending_so():
    eth, sam, joe, bob = _three()
    sam.ctl.anycall(eth.now, selective="E")
    eth.run(60, until=lambda: sam.ctl.state == "linked" and joe.ctl.state == "linked")
    assert sam.ctl.linked == {"JOE"} and bob.ctl.state == "available"


def test_an_allcall_is_heard_by_all_and_answered_by_none():
    """A.5.5.4.4: TIS concludes it; every station hearing it is linked."""
    eth, sam, joe, bob = _three()
    assert sam.ctl.allcall(eth.now, amd="net check")
    eth.run(10)
    assert joe.ctl.state == bob.ctl.state == "linked"
    assert ("SAM", "NET CHECK") in joe.ctl.amd_log
    assert not any(e.kind == "CALLED" for st in (joe, bob) for e in st.ctl.events)


def test_a_wildcard_call_is_answered_by_the_addresses_it_matches():
    """A.5.5.4.6: "?" stands for any one character; answers in random slots."""
    eth, sam, joe, bob = _three()
    sam.ctl.call("?O?", eth.now)
    eth.run(60, until=lambda: sam.ctl.state == "linked" and joe.ctl.state == "linked"
            and bob.ctl.state == "linked")
    assert sam.ctl.linked == {"JOE", "BOB"}


# -- DTM (A.5.7.3) -------------------------------------------------------------

def _dtm_events(st, kind):
    return [e for e in st.ctl.events if e.kind == kind]


def test_a_dtm_message_crosses_the_link_block_by_block_with_arq():
    eth, sam, joe = _pair(scanning=False, dtm_words=20)
    text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789. " * 3
    sam.ctl.send_dtm("JOE", text, eth.now)                  # calls first
    eth.run(200, until=lambda: _dtm_events(sam, "DTM_SENT"))
    assert [e.data for e in _dtm_events(joe, "DTM")] == [text.upper()]
    assert _dtm_events(sam, "DTM_SENT")[0].data == text
    assert sam.ctl.state == "linked" and joe.ctl.state == "linked"


def test_binary_dtm_without_arq():
    eth, sam, joe = _pair(scanning=False)
    sam.ctl.call("JOE", eth.now)
    eth.run(10, until=lambda: _both_linked(sam, joe))
    data = bytes(range(200))
    sam.ctl.send_dtm("JOE", data, eth.now, arq=False)
    eth.run(120, until=lambda: _dtm_events(joe, "DTM"))
    assert [e.data for e in _dtm_events(joe, "DTM")] == [data]


def test_a_lost_block_is_sent_again_and_read_once():
    """The first copy of the block is lost on the way; the second is ACKed and
    delivered, once."""
    eth, sam, joe = _pair(scanning=False)
    sam.ctl.call("JOE", eth.now)
    eth.run(10, until=lambda: _both_linked(sam, joe))
    lost = {"n": 0}
    real = eth.transmit

    def transmit(sender, channel, start, audio):
        if sender is sam and audio.size > 20 * 3136 and lost["n"] == 0:
            lost["n"] += 1
            return                                            # faded out entirely
        real(sender, channel, start, audio)
    eth.transmit = transmit
    sam.ctl.send_dtm("JOE", "HELLO THERE, THIS IS A DTM MESSAGE OF SOME LENGTH", eth.now)
    eth.run(120, until=lambda: _dtm_events(sam, "DTM_SENT"))
    assert lost["n"] == 1
    assert len(_dtm_events(joe, "DTM")) == 1 and _dtm_events(sam, "DTM_SENT")


def test_a_dtm_null_interrogation_is_answered():
    """A.5.7.3: a CMD DTM NULL with ARQ asks whether a station takes DTM; a DTM
    ARQ with KD4 set says it does."""
    from hfmodem.bateleur.ale import words as W
    eth, sam, joe = _pair(scanning=False)
    sam.ctl.call("JOE", eth.now)
    eth.run(10, until=lambda: _both_linked(sam, joe))
    heard = []
    real = sam.ctl._frame
    sam.ctl._frame = lambda r, now, end=None: (heard.append(r.frame), real(r, now, end))
    sam.ctl._transmit(W.call("JOE", "SAM", message=[W.dtm_null(kd1=1)]), eth.now)
    eth.run(6, until=lambda: any(f.dtm for f in heard))
    arq = [d for f in heard for d in f.dtm]
    assert arq and arq[0].kind == "arq" and arq[0].kd4 == 1 and arq[0].kd1 == 1


# -- DBM (A.5.7.4) -------------------------------------------------------------

def test_a_dbm_message_crosses_the_link_with_arq():
    eth, sam, joe = _pair(scanning=False, dbm_depth=49 * 2)
    data = bytes(range(256)) * 2
    sam.ctl.send_dbm("JOE", data, eth.now)
    eth.run(200, until=lambda: _dtm_events(sam, "DBM_SENT"))
    assert [e.data for e in _dtm_events(joe, "DBM")] == [data]
    assert sam.ctl.state == "linked"


def test_a_dbm_block_failed_twice_is_read_from_both_copies_together():
    """A.5.7.4: a failed block is kept, and the next copy combined with it
    (here, soft values added). Each copy faded for half its length: the
    interleaver spreads either fade over every row, too much for any to
    decode, but no bit was lost in both."""
    import numpy as np
    from hfmodem.bateleur.ale import words as W
    text = "HELLO FROM A DEEPLY INTERLEAVED BLOCK"
    (cmd, bits), = W.dbm_blocks(text, arq=True)
    d = W.dbm_fields(cmd)
    rng = np.random.default_rng(0)

    def copy(faded):
        llr = 1.0 - 2.0 * bits + rng.normal(0, 0.3, bits.size)
        llr[faded] = rng.normal(0, 0.3, faded.stop - faded.start)
        return W.Dbm(d["kind"], d["kd4"], d["kd3"], d["kd2"], d["kd1"],
                     d["depth"], d["bits"]).decode(llr)
    half = bits.size // 2
    a, b = copy(slice(0, half)), copy(slice(half, bits.size))
    assert not a.ok and not b.ok
    assert a.combined(b).ok and a.combined(b).data == text


# -- linking protection (appendix B) -------------------------------------------

KEY = "c2284a1ce7be2f"
HZ = [7_102_000 + 1000 * i for i in range(10)]


def _protected_pair(level=2, key_joe=KEY, scanning=True):
    eth = E.Ether(snr_db=10, seed=0)
    chans = CHANNELS if scanning else CHANNELS[:1]
    hz = HZ[:len(chans)]
    sam = eth.add(Config(["SAM"], chans, scanning=scanning, listen_s=0.0,
                         lp_level=level, lp_key=KEY, channel_hz=hz))
    joe = eth.add(Config(["JOE"], chans, scanning=scanning, listen_s=0.0,
                         lp_level=level, lp_key=key_joe, channel_hz=hz))
    eth.run(1.0)
    return eth, sam, joe


def test_protected_stations_link_and_exchange_traffic():
    """AL-2: every word encrypted under a 2 s PI; the scanning station
    finds the seed without any sync words (B.4.2)."""
    eth, sam, joe = _protected_pair()
    sam.ctl.call("JOE", eth.now, channel=0, amd="protected")
    eth.run(20, until=lambda: _both_linked(sam, joe))
    assert _both_linked(sam, joe) and joe.ctl.amd_log == [("SAM", "PROTECTED")]
    sam.ctl.send_dtm("JOE", "SECRET TEXT", eth.now)
    eth.run(60, until=lambda: _dtm_events(sam, "DTM_SENT"))
    assert [e.data for e in _dtm_events(joe, "DTM")] == ["SECRET TEXT"]


def test_al1_links_too():
    eth, sam, joe = _protected_pair(level=1, scanning=False)
    sam.ctl.call("JOE", eth.now)
    eth.run(10, until=lambda: _both_linked(sam, joe))
    assert _both_linked(sam, joe)


def test_a_station_with_another_key_hears_nothing():
    eth, sam, joe = _protected_pair(key_joe="00112233445566", scanning=False)
    sam.ctl.call("JOE", eth.now)
    eth.run(10)
    assert "CALLED" not in _kinds(joe) and joe.ctl.state == "available"
    assert not any(e.kind == "HEARD" for e in joe.ctl.events)


# -- time exchange (A.5.6.4, B.5.5) --------------------------------------------

def test_the_time_words_match_figures_a38_and_a39():
    """8 May, 15:57:34.12, quality 4: the Time Is, coarse time and
    authenticator words the figures give. Figure A-39 prints the minute
    field 011101111101, twelve digits for an eleven-bit field; 957 minutes is
    01110111101, and that is what reproduces its authenticator."""
    from hfmodem.bateleur.ale import words as W
    cmd = W.time_is(4, 34, 3)
    crs = W.coarse_time(5, 8, 15 * 60 + 57)
    assert f"{cmd:024b}" == "110" "1111110" "100" "100010" "00011"
    assert f"{crs:024b}" == "000" "0" "0101" "01000" "01110111101"
    auth = W.authenticator([cmd, crs])
    assert W.preamble_of(auth) == W.REP and f"{auth & 0x1FFFFF:021b}" == "110101110011111111110"


def _time_pair(joe_off, joe_quality, level=2):
    eth = E.Ether(snr_db=15, seed=1)
    sam = eth.add(Config(["SAM"], CHANNELS[:1], scanning=False, listen_s=0.0,
                         lp_level=level, lp_key=KEY, channel_hz=HZ[:1], time_quality=2))
    joe = eth.add(Config(["JOE"], CHANNELS[:1], scanning=False, listen_s=0.0,
                         lp_level=level, lp_key=KEY, channel_hz=HZ[:1],
                         time_quality=joe_quality))
    joe.ctl._offset = joe_off
    eth.run(1.0)
    return eth, sam, joe


def test_a_station_with_coarse_time_gets_fine_time_protected():
    """B.5.5.2: 20 s out, quality 6; the request goes under the coarse seed,
    the server answers, and the requester's clock comes right."""
    eth, sam, joe = _time_pair(20.0, 6)
    joe.ctl.time_request("SAM", eth.now)
    eth.run(15, until=lambda: any(e.kind in ("TIME", "TIME_FAILED") for e in joe.ctl.events))
    ev = {e.kind: e for e in joe.ctl.events}
    assert ev["TIME_REQUESTED"].text == "coarse" and "TIME" in ev, joe.ctl.events
    assert abs(joe.ctl.utc(eth.now) - sam.ctl.utc(eth.now)) < 0.1
    assert joe.ctl.time_quality(eth.now) <= 4
    assert any(e.kind == "TIME_SERVED" for e in sam.ctl.events)


def test_a_station_without_time_gets_it_in_the_clear():
    """B.5.5.3: no idea of the time; a clear Time Request, answered with an
    authenticator only the key holder could make."""
    eth, sam, joe = _time_pair(-3000.0, 7)
    joe.ctl.time_request("SAM", eth.now)
    eth.run(15, until=lambda: any(e.kind in ("TIME", "TIME_FAILED") for e in joe.ctl.events))
    ev = {e.kind: e for e in joe.ctl.events}
    assert ev["TIME_REQUESTED"].text == "clear" and "TIME" in ev, joe.ctl.events
    assert abs(joe.ctl.utc(eth.now) - sam.ctl.utc(eth.now)) < 0.1
    # And now it can link, protected.
    joe.ctl.call("SAM", eth.now)
    eth.run(10, until=lambda: _both_linked(sam, joe))
    assert _both_linked(sam, joe)


def test_a_clear_time_answer_from_another_key_is_not_believed():
    eth, sam, joe = _time_pair(-3000.0, 7)
    from hfmodem.bateleur.ale.lp import Lattice
    sam.ctl.lp.cipher = Lattice(bytes.fromhex("00112233445566"))
    joe.ctl.time_request("SAM", eth.now)
    eth.run(15, until=lambda: any(e.kind in ("TIME", "TIME_FAILED") for e in joe.ctl.events))
    assert "TIME" not in [e.kind for e in joe.ctl.events]
    assert any(e.kind == "TIME_FAILED" for e in joe.ctl.events)


def test_ack_to_our_long_address_arrives_before_our_deadline():
    eth = E.Ether(snr_db=30)
    caller = eth.add(Config(["SAM"], ["A"], scanning=False, listen_s=0))
    called = eth.add(Config(["LONGADDRESS1234"], ["A"], scanning=False, listen_s=0))
    eth.run(1)
    caller.ctl.call("LONGADDRESS1234", eth.now)
    eth.run(25, until=lambda: _both_linked(caller, called))
    assert _both_linked(caller, called)
    assert "CALL_FAILED" not in _kinds(called)
