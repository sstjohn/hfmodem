"""Entry timing evidence and later decoded traffic are separate diagnostics."""
from copy import deepcopy
from hfmodem.shrike import arq, onair
from hfmodem.tests.shrike.test_repeat_gear_stall import arrives, linked

def _grid():
    grid = onair._MasterGrid(0, 60000, 12000, packet_n=48000, cs_n=5760, d_max_n=4800)
    grid.entry_at = 1
    grid.answer_unread_ms = 110.0
    grid.entry_end_n = grid.p1_data_n + 960
    return grid

def _evidence(grid):
    return deepcopy({name: getattr(grid, name) for name in
                     ("entry_at", "answer_unread_ms", "entry_answers", "entry_tracked", "reading")})

def test_empty_entry_timing_record_scopes_its_no_answer_message():
    grid = _grid()
    before = _evidence(grid)
    message = grid.entry_verdict()
    assert message.startswith("NO TIMED ENTRY ANSWER RECORDED:")
    assert "no timed answer burst was registered after it" in message
    assert "Later decoded packet traffic is reported separately" in message
    assert "has not answered since" not in message
    assert _evidence(grid) == before

def test_later_modeled_decoded_packet_does_not_fill_the_entry_timing_record():
    grid = _grid()
    host, seam = linked()
    token = host.arq.begin_receive_opportunity()
    arrives(host)
    host.arq.finish_receive_opportunity(token)
    assert host.arq.rx_progress == 1 and host.rcvd_total > 0
    assert not grid.entry_answers
    before = (host.arq.state, host.arq.rx_seq, host.arq.tx_seq,
              host.arq.rx_progress, host.rcvd_total, host.sent_total,
              host.arq.tx_acked_bytes, len(seam.words), len(seam.packets))
    message = grid.entry_verdict()
    assert message.startswith("NO TIMED ENTRY ANSWER RECORDED:")
    assert "Later decoded packet traffic is reported separately" in message
    assert before == (host.arq.state, host.arq.rx_seq, host.arq.tx_seq,
                      host.arq.rx_progress, host.rcvd_total, host.sent_total,
                      host.arq.tx_acked_bytes, len(seam.words), len(seam.packets))
    assert host.arq.state is arq.State.CONNECTED

def test_nonempty_entry_position_verdict_and_evidence_are_unchanged():
    grid = _grid()
    grid.entry_end_n = grid.p1_data_n + round(3 * onair.MAX_PULL_S * onair.FS)
    entry_ms = grid.entry_read_ms
    grid.entry_answers = [(cycle, grid.boundary(cycle) + round(entry_ms * .001 * onair.FS), entry_ms)
                          for cycle in (2, 3)]
    before = _evidence(grid)
    message = grid.entry_verdict()
    assert grid._population(entry_ms) == ("AT THE ENTRY POSITION", 0.0)
    assert "THE ANSWER MOVED TO THE ENTRY POSITION" in message
    assert "NO TIMED ENTRY ANSWER RECORDED" not in message
    assert _evidence(grid) == before
