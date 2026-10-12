import pytest
from hfmodem.core.tx_events import TxEvents, read_line


def test_modems_share_envelope_and_session_scoped_ids():
    for protocol in ('VARA','PACTOR','ARDOP'):
        journal=TxEvents(protocol,session_id='session-a')
        journal.record('completed',1,outcome='sent',recording='/capture.wav',sample_rate=48000)
        line,=journal.lines()
        r=read_line(line)
        assert r['protocol']==protocol and r['direction']=='tx'
        assert r['session_id']=='session-a' and r['tx_id']==1
        assert journal.lines()==[]
        assert read_line(line.replace('"schema":1','"schema":999')) is None
    assert read_line('tx_event []') is None


def test_invalid_outcome_is_not_serialized():
    with pytest.raises(ValueError):
        TxEvents('VARA').record('completed',1,outcome='maybe')


def test_vara_metadata_hook_keeps_waveform_identical():
    from types import SimpleNamespace
    import numpy as np
    from hfmodem.kestrel.vara.vara_arq import VaraStationHandshake,VaraState
    station=object.__new__(VaraStationHandshake)
    station.bw='2750';station.state=VaraState.CONNECTED
    station.role='initiator';station.caller='LOCAL';station.called='PEER'
    notes=[];audio=[]
    station.io=SimpleNamespace(tx_metadata=lambda **kw:notes.append(kw),tx=audio.append)
    x=np.arange(12,dtype=float)
    station._transmit_audio(x,packet_kind='data',level=6,modulation='DQPSK',retry=True)
    assert audio[0] is x
    assert notes[0]['packet_kind']=='data' and notes[0]['speed_level']==6
    assert notes[0]['link_state']==VaraState.CONNECTED and notes[0]['retry']
    station.io=SimpleNamespace(tx=audio.append)
    station._transmit_audio(x,packet_kind='control')
    assert audio[-1] is x
