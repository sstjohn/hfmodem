# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Checks for the homepage PACTOR-3 figure, its VE3KPG example and the older VE1YZ excerpt."""
import hashlib
import json
import re
import unittest
import exchange_figure
import repair_pactor_excerpt as excerpt


EXAMPLE = excerpt.ROOT/'site/examples/pactor3-long'
# Our own message IDs may appear here in clear text; the subject is stored only as a digest.
MIDS = ('WDPURMGSL24K', 'N7EMB1JPLLC4')
SUBJECTS = {45: {'c590f5a4651ff49f33180cd7a879dd3332d21ab7af0839631b0780b6893e3a86'}}
# Where every published payload copy shows "#". A change here means curation unmasked or moved a private span.
MASKS = '892b3ae04864b4ad6746a4d7cc393b3a44b2fda4b67a0c817f156d0072525baf'
# The session log's answers to our long-cycle packets, all reacquired from the receiver audio: (id, t0, CS).
ANSWERS = [('excerpt-cs-rx-0', 94.734, 1), ('excerpt-cs-rx-1', 95.984, 6), ('excerpt-cs-rx-2', 99.735, 1),
           ('excerpt-cs-rx-3', 103.484, 2), ('excerpt-cs-rx-4', 107.234, 1), ('excerpt-cs-rx-5', 110.984, 2),
           ('excerpt-cs-rx-6', 114.734, 1), ('excerpt-cs-rx-7', 118.484, 4), ('excerpt-cs-rx-8', 122.234, 1),
           ('excerpt-cs-rx-9', 125.985, 4), ('excerpt-cs-rx-10', 129.736, 1), ('excerpt-cs-rx-11', 133.484, 3)]


def sha(text):
    return hashlib.sha256(text.encode('latin-1')).hexdigest()


def payloads(e):
    return [('event', p) for p in e.get('payloads', [])] + [
        (source, p) for source, view in sorted(e['views'].items()) for d in view['decoded'] for p in d['payloads']]


def reacquire(test, example, events):
    """Each published acquisition reproduces from the example's retained audio."""
    manifest = json.loads((example/'manifest.json').read_text())
    for source in manifest['sources'][:2]:
        audio, _ = excerpt.load_audio(example/source['url'])
        for e in events:
            if not e['id'].startswith('excerpt-cs-') or e['source_id'] != source['id']:
                continue
            candidate, t = excerpt.acquire(audio, e['acquisition']['search_hint_s'], source['offset_s'], e['control_number'])
            test.assertAlmostEqual(t, e['t0'], places=5)
            test.assertEqual(candidate.event.cs+1, e['control_number'])
            test.assertGreater(candidate.quality, .75)
            for view in e['views'].values():
                test.assertTrue((example/'data'/view['signal']).is_file())


def mask_rows(events):
    return [f"{e['id']}/{source}:" + ''.join('#' if c == 0x23 else '.' for c in bytes.fromhex(p['hex']))
            for e in events for source, p in payloads(e)]


def streams(events):
    """Each recording's payload bytes in time order, so a field split across packets stays whole."""
    copies = {}
    for e in events:
        for source, p in payloads(e):
            copies.setdefault(source, []).append(p['hex'])
    return {k: bytes.fromhex(''.join(v)).decode('latin-1') for k, v in copies.items()}


def leaks(text, subjects=SUBJECTS):
    """Private B2 fields that a published payload stream shows unmasked."""
    found = re.findall(r';PR: (?!#{8})\S*|;PQ: (?!#)\S*|;PM: \S+ [^\r]*[^#\d \r][^\r]*|FC EM (?!#)\S*|SMTP:|@', text)
    found += [mid for mid in MIDS if mid in text]
    found += [text[i:i + n] for n, digests in subjects.items()
              for i in range(len(text) - n + 1) if sha(text[i:i + n]) in digests]
    # Compressed message bodies are binary, so an unmasked non-text byte is a leaked body.
    return found + re.findall(r'[^\t\r\n\x20-\x7e]', text)


class HomepageFigure(unittest.TestCase):
    def test_chart_draws_every_recovered_frame(self):
        page = (excerpt.ROOT/'site/index.html').read_text()
        figure = re.search(r'<div id="evidence-pactor"><figure class="cycle">.*?</figure>', page, re.S)[0]
        events = exchange_figure.events('pactor')
        lo, hi = exchange_figure.FIGURES['pactor']['window']
        self.assertIn(exchange_figure.diagram('pactor', events), figure)
        self.assertEqual(set(re.findall(r'data-event="([^"]+)"', figure)),
                         {e['id'] for e in events if exchange_figure.drawn(e, lo, hi)})
        self.assertNotIn('signal candidate', figure)
        # Our outbound long-cycle packets climb to SL3; the empty last SL3 packet carries no level.
        self.assertEqual(re.findall(r'font-size="11">([^<]+)<', figure),
                         ['SL1'] * 8 + ['SL2', 'SL2', 'SL3', 'SL1'])


class LongPacketExemplar(unittest.TestCase):
    """The VE3KPG example exists to show mail moving in PACTOR-3 long-cycle packets."""

    @classmethod
    def setUpClass(cls):
        cls.events = json.loads((EXAMPLE/'events.json').read_text())

    def test_mail_moves_both_ways_in_long_cycle_packets(self):
        for direction in ('tx', 'rx'):
            long = [e for e in self.events if e['direction'] == direction and e['kind'].endswith(' data')
                    and e['status'] == 'CRC pass' and e['t1'] - e['t0'] > 3]
            # A 3.29 s packet fits only the 3.75 s cycle; the status byte's cycle bit is not set on every one.
            self.assertGreaterEqual(len(long), 9, direction)
            self.assertTrue(any(p['length'] for e in long for _, p in payloads(e)), direction)

    def test_published_bytes_keep_private_fields_masked(self):
        self.assertNotIn('@', (EXAMPLE/'events.json').read_text())
        found = streams(self.events)
        self.assertGreater(len(found), 1)
        for source, stream in found.items():
            self.assertEqual(leaks(stream), [], source)

    def test_masked_bytes_stay_masked(self):
        self.assertEqual(sha('\n'.join(mask_rows(self.events))), MASKS, '\n'.join(mask_rows(self.events)))

    def test_mask_pin_covers_top_level_and_nested_copies(self):
        for event_id, nested in (('source0-p3-00113', True), ('source0-p3-00119', False)):
            events = json.loads(json.dumps(self.events))
            e = next(e for e in events if e['id'] == event_id)
            p = (e['views']['rx']['decoded'][0]['payloads'] if nested else e['payloads'])[0]
            data = bytearray.fromhex(p['hex'])
            data[data.index(0x23)] = 0x41
            p['hex'] = data.hex()
            self.assertNotEqual(sha('\n'.join(mask_rows(events))), MASKS, event_id)

    def test_leak_check_catches_each_private_field(self):
        stream = streams(self.events)['rx']
        subject = 'A subject in the clear'
        planted = {len(subject): {sha(subject)}}
        for leaked in (';PR: #1234567\r', ';PQ: 12345678\r', ';PM: W9SSJ ABCDEFGHIJKL 475 #### test\r',
                       'FC EM ABCDEFGHIJKL 608 450 0\r', 'To: SMTP:x\r', '\x1f\x8b\x00', 'WDPURMGSL24K', subject):
            self.assertNotEqual(leaks(stream + leaked, planted), [], leaked)

    def test_peer_answers_every_outbound_packet(self):
        answers = [e for e in self.events if e['id'].startswith('excerpt-cs-')]
        self.assertEqual([(e['id'], round(e['t0'], 3), e['control_number']) for e in answers], ANSWERS)
        self.assertTrue(all(e['direction'] == 'rx' for e in answers))
        self.assertEqual(answers[-1]['payloads'][0]['hex'], '464320')
        sent = [e for e in self.events if e['direction'] == 'tx' and e['kind'].endswith(' data') and e['t1'] > e['t0']
                and 93 < e['t0'] < 134]
        self.assertEqual(len(sent), 11)
        for e in sent:
            self.assertTrue(any(e['t1'] < a['t0'] < e['t1'] + .3 for a in answers), e['id'])

    def test_reacquisition_from_retained_audio_matches_published_evidence(self):
        reacquire(self, EXAMPLE, self.events)


class PactorExcerpt(unittest.TestCase):
    EXAMPLE = excerpt.ROOT/'site/examples/pactor3-send'

    @classmethod
    def setUpClass(cls):
        cls.events = json.loads((cls.EXAMPLE/'events.json').read_text())
        cls.window = [e for e in cls.events if 110 <= e['t0'] < 151]

    def test_answers_and_breakin_precede_role_reversal(self):
        peer = [e for e in self.window if e['direction']=='rx' and e.get('control_number')]
        self.assertEqual([e['control_number'] for e in peer], [1,6,1,4,1,1,1,3,3,3])
        for e in peer[-3:]:
            self.assertEqual(e['status'], 'CRC pass')
            self.assertEqual(e['payloads'][0]['hex'], '3b504d')
        ack = next(e for e in self.window if e['id']=='excerpt-cs-tx-66')
        self.assertGreater(ack['t0'], peer[-1]['t1'])
        self.assertLess(ack['t1'], next(e['t0'] for e in self.window if e['id']=='p3-00089'))
        # The old export showed TX data with no answers and misattributed TX59.
        data = [e for e in self.window if e['direction']=='tx' and e.get('frame_type')=='data']
        self.assertEqual(len(data), 7)
        for e in data:
            self.assertTrue(any(e['t1'] < a['t0'] < e['t1']+.3 for a in peer), e['id'])

    def test_half_duplex_frames_do_not_overlap(self):
        for a,b in zip(self.window,self.window[1:]):
            self.assertLessEqual(a['t1'], b['t0'], (a['id'],b['id']))

    def test_reacquisition_from_retained_audio_matches_published_evidence(self):
        reacquire(self, self.EXAMPLE, self.window)


if __name__=='__main__':
    unittest.main()
