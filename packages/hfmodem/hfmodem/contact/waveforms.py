# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Waveform adapters for a shared contact session. Imports are lazy."""
from .frame import reconcile


class VoiceWaveform:
    name = 'voice'
    designator = '2K40J3E'

    def __init__(self):
        from hfmodem.macagua import script
        self.codec = script

    def open_modem(self, settings, directory):
        from hfmodem.macagua.speech import Speech
        return Speech(settings, directory)

    def receive_chain(self):
        from hfmodem.macagua.chain import ReceiveChain
        return ReceiveChain()

    def condition(self, audio, rate, drive, *, compression='soft'):
        from hfmodem.macagua.chain import transmit
        return transmit(audio, rate, drive, compression=compression)

    def callsigns(self, text):
        from hfmodem.macagua.phonetics import callsigns
        return callsigns(text)

    def parse(self, language, result, candidates, own_call):
        raw = language.extract(result['text'], own_call, candidates)
        return raw, reconcile(raw, candidates, own_call, result['probability'], result['text'])

    def silence_s(self, default):
        return default

    def emission_edges(self, dial):
        return (-2700, -300) if dial < 10_000_000 else (300, 2700)


def select_waveform(name='voice', wpm=20., tone=700.):
    if name == 'voice':
        return VoiceWaveform()
    if name == 'cw':
        from hfmodem.merlin.modem import CwWaveform
        return CwWaveform(wpm, tone)
    raise ValueError('waveform must be voice or cw')
