# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Packet integrity, erased symbols and offline-only watermark boundaries."""
from dataclasses import replace
from types import SimpleNamespace
import random
import json

import pytest

from hfmodem.contact.watermark import Codebook, Encoder, Options, WORDS, frame_from_symbols, pack, symbols, unpack
from hfmodem.contact.runtime import Session
from hfmodem.tests.macagua.test_voice import cfg


@pytest.mark.parametrize('width', [1, 2, 4])
def test_arbitrary_binary_packets_and_all_byte_values(width):
    payloads = [b'', bytes(range(255)), b'\x00\xff\xc3\xa9', random.Random(44).randbytes(37)]
    for payload in payloads:
        encoded = pack(payload, sequence=255)
        assert unpack(frame_from_symbols(symbols(encoded, width), width)) == (255, payload)


def test_checksum_rejects_corrupted_payload_and_symbols_reject_erasures():
    frame = bytearray(pack(b'hello'))
    frame[3] ^= 1
    with pytest.raises(ValueError, match='checksum'):
        unpack(frame)
    values = symbols(pack(b'hello'), 4)
    values[2] = None
    with pytest.raises(ValueError, match='erased'):
        frame_from_symbols(values, 4)
    with pytest.raises(ValueError):
        pack(bytes(256))


@pytest.mark.parametrize('width', [1, 2, 4])
def test_all_symbols_survive_case_punctuation_and_unrelated_words(width):
    codebook = Codebook(Options(symbol_bits=width))
    for symbol in range(codebook.size):
        chosen = [w for w in codebook.words if w in codebook.patterns[symbol]][:28]
        text = ' '.join(chosen).upper() + ', 59! ZXQ.'
        assert codebook.score(text)['symbol'] == symbol
        # Repeating one word is not independent evidence.
        assert codebook.score((chosen[0] + ' ') * 100)['symbol'] is None


def test_missing_and_conflicting_evidence_are_erasures():
    codebook = Codebook()
    assert codebook.score('KD0DLK 579 EN63BB')['symbol'] is None
    one = [w for w in codebook.words if w in codebook.patterns[1]
           and all(w not in p for i, p in enumerate(codebook.patterns) if i != 1)][:8]
    assert codebook.score(' '.join(codebook.words))['symbol'] is None
    observations = [codebook.score(' '.join(one[:4])), codebook.score(' '.join(one[4:]))]
    assert all(o['symbol'] is None for o in observations)
    assert codebook.combine(observations)['symbol'] == 1


def test_codebook_is_stable_and_changed_seed_changes_word_mapping():
    first = Codebook()
    same = Codebook(vocabulary=reversed(WORDS))
    other = Codebook(Options(codebook='different'))
    assert first.patterns == same.patterns and first.digest == same.digest
    assert first.patterns != other.patterns


def test_encoder_biases_whole_tokens_without_putting_payload_in_prompt():
    class Model:
        def request(self, path, data):
            word = data['content'].strip().lower()
            if word == 'wonderful':
                return dict(tokens=[dict(id=9001, piece=' wonder'), dict(id=9002, piece='ful')])
            return dict(tokens=[dict(id=WORDS.index(word), piece=data['content'])])

        def complete(self, instruction, content, schema, sampling=None):
            self.sampling, self.prompt = sampling, instruction + content
            return dict(reply=' '.join(w for w in WORDS if str(WORDS.index(w)) in sampling['logit_bias'])[:100])

    model = Model()
    encoder = Encoder(model, Options(method='logit', cover_mode='free'))
    # A split word must not bias its fragments, even if one fragment is a word.
    assert 'wonderful' not in encoder.codebook.words and 9001 not in encoder.tokens
    symbol = 2
    # Mock completion with enough selected words; actual generation is rehearsed separately.
    selected = [w for w in encoder.codebook.words if w in encoder.codebook.patterns[symbol]][:20]
    def complete(instruction, content, schema, sampling=None):
        model.sampling, model.prompt = sampling, instruction + content
        return dict(reply=' '.join(selected))
    model.complete = complete
    assert encoder.generate(symbol)['score']['symbol'] == symbol
    assert model.sampling['temperature'] > 0
    assert all(encoder.tokens[int(t)] in encoder.codebook.patterns[symbol] for t in model.sampling['logit_bias'])
    assert 'symbol' not in model.prompt and 'payload' not in model.prompt
    encoder.generate(symbol, marked=False)
    assert model.sampling['logit_bias'] == {}


def test_plausible_cover_and_keyed_selection_are_independent_of_payload():
    from hfmodem.contact import cover
    from hfmodem.contact.watermark import SentenceCodebook
    options = Options(symbol_bits=4)
    public = SentenceCodebook(options)
    keyed = SentenceCodebook(options, key=b'x'*32)
    wrong = SentenceCodebook(options, key=b'y'*32)
    assert public.mapping != keyed.mapping and keyed.mapping != wrong.mapping
    for topic in cover.TOPICS:
        choices = cover.candidates(topic)
        assert set(keyed.score(s)['symbol'] for s in choices) == set(range(16))
        assert all(8 <= len(s.split()) <= 24 and not any(c.isdigit() for c in s) for s in choices)
    assert cover.topic_from('There was a solar flare') == 'solar'
    assert cover.topic_from('My XYL is waiting') == 'home'
    assert keyed.score('I am using a made up amplifier')['symbol'] is None
    assert keyed.score(cover.candidates('antenna')[0])['presence_detection'] is False


def test_receiver_uses_recognized_text_not_transmitter_truth_or_payload(tmp_path):
    from dataclasses import asdict
    from hfmodem.contact.channel import replay
    from hfmodem.contact.watermark import SentenceCodebook
    options = Options(symbol_bits=4)
    key = b'x'*32
    codebook = SentenceCodebook(options, key=key)
    payload = b'\x00\xffhi'
    values = symbols(pack(payload), 4)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(options=asdict(options),
        active_vocabulary=codebook.words, payload_hex='deliberately incorrect')))
    for index, value in enumerate(values):
        phrase = next(s for s in codebook.phrases if codebook.score(s)['symbol'] == value)
        (tmp_path / f'block-{index:03d}-round-0-repeat-0.json').write_text(json.dumps(dict(
            requested_symbol=(value + 1) % 16, generated=dict(text='Wrong transmitter truth'),
            recognized='Wrong cached scoring text',
            recognition=dict(text=phrase + ' This is Whiskey nine Sierra Sierra Juliet. Over.'))))
    result = replay(tmp_path, key=key)
    assert result['status'] == 'crc_pass' and result['recovered_hex'] == payload.hex()
    assert replay(tmp_path, key=b'y'*32)['status'] == 'crc_fail'
    assert replay(tmp_path)['status'] == 'crc_fail'


def test_cover_distribution_measurements_have_defined_limits():
    from hfmodem.contact.channel import cover_statistics
    same = [dict(recognized='weather antenna weather')]
    assert cover_statistics(same, same)['jensen_shannon_bits'] == 0
    separate = cover_statistics([dict(recognized='antenna')], [dict(recognized='weather')])
    assert separate['total_variation'] == 1 and separate['jensen_shannon_bits'] == 1
    assert cover_statistics(same, [])['total_variation'] is None


@pytest.mark.parametrize('kwargs', [dict(test=False), dict(test=True, waveform='cw'),
                                  dict(test=True, monitor=True), dict(test=True, replay='recording.wav')])
def test_watermark_cannot_run_on_radio_cw_or_monitor(tmp_path, kwargs):
    with pytest.raises(ValueError, match='computer-audio'):
        Session(cfg(), tmp_path, watermark='hfmodem-v1', **kwargs)


def test_signature_keeps_trusted_facts_and_final_station_id(tmp_path):
    from hfmodem.macagua import script
    session = Session(cfg(), tmp_path, test=True, watermark='hfmodem-v1')
    session.watermark_encoder = SimpleNamespace(generate=lambda *a, **k: dict(text='I enjoy sharing ideas and talking together.'))
    trusted = script.exchange('W9SSJ', 'KD0DLK', '57', session.facts)
    try:
        marked, record = session.mark_text(trusted)
        assert marked.endswith(trusted) and record['original_text'] == trusted
        assert 'Anyway, ' in marked and script.phonetic('KD0DLK') in marked
        session.rig = object()
        with pytest.raises(ValueError, match='rig'):
            session.mark_text(trusted)
        session.rig = None
    finally:
        session.close()


@pytest.mark.parametrize('values', [dict(bias=float('nan')), dict(symbol_bits=3), dict(min_fraction=0),
                                  dict(temperature=0), dict(max_words=100), dict(attempts=0)])
def test_experiment_parameters_are_bounded(values):
    with pytest.raises(ValueError):
        replace(Options(), **values)
