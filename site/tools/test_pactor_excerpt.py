# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Recorded-frame regression checks for the homepage PACTOR exchange."""
import json
import re
import unittest
import repair_pactor_excerpt as excerpt


class PactorExcerpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = json.loads((excerpt.EXAMPLE/'events.json').read_text())
        cls.window = [e for e in cls.events if 110 <= e['t0'] < 151]

    def test_chart_draws_every_recovered_frame(self):
        page = (excerpt.ROOT/'site/index.html').read_text()
        figure = re.search(r'<figure class="cycle" id="evidence-pactor".*?</figure>', page, re.S)[0]
        self.assertIn(excerpt.diagram(self.events), figure)
        self.assertEqual(set(re.findall(r'data-event="([^"]+)"', figure)), {e['id'] for e in self.window})

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
        manifest=json.loads((excerpt.EXAMPLE/'manifest.json').read_text())
        for source in manifest['sources'][:2]:
            audio,_=excerpt.load_audio(excerpt.EXAMPLE/source['url'])
            for e in self.window:
                if not e['id'].startswith('excerpt-cs-') or e['source_id']!=source['id']:
                    continue
                candidate,t=excerpt.acquire(audio,e['acquisition']['search_hint_s'],source['offset_s'],e['control_number'])
                self.assertAlmostEqual(t,e['t0'],places=5)
                self.assertEqual(candidate.event.cs+1,e['control_number'])
                self.assertGreater(candidate.quality,.75)
                for view in e['views'].values():
                    self.assertTrue((excerpt.EXAMPLE/'data'/view['signal']).is_file())


if __name__=='__main__':
    unittest.main()
