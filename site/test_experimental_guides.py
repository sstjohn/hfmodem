"""Illustrated experimental guides keep real constants and explicit evidence limits."""
from pathlib import Path
import importlib.util
import ast
import json
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class ExperimentalGuides(unittest.TestCase):
    def test_lanner_diagram_tables_match_both_alphabets(self):
        source=(ROOT/'docs/viz/rtty-explained.html').read_text()
        table=json.loads(re.search(r'const CODES=(\[.*?\]);',source,re.S)[1])
        ita=load('ita_reference','packages/hfmodem/hfmodem/lanner/ita2.py')
        # The seven-unit alphabet maps to ITA2 codes; load it without importing the modem.
        tree=ast.parse((ROOT/'packages/hfmodem/hfmodem/lanner/ccir476.py').read_text())
        code_table=next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CCIR476_TABLE' for t in node.targets))
        seven={combo:word for combo,_,_,_,word,_ in code_table}
        self.assertEqual(len(table),32)
        for row, (combo,letter,figure,bits) in zip(table,ita.TABLE):
            self.assertEqual(row['combo'],combo)
            self.assertEqual(row['letter'],letter)
            self.assertEqual(row['figure'],figure)
            self.assertEqual(row['us'],ita.US_FIGURE_OVERRIDES.get(combo,(None,figure))[1])
            self.assertEqual(row['ita'],bits.translate(str.maketrans('AZ','01')))
            self.assertEqual(row['sitor'],seven[combo])
            self.assertEqual(row['sitor'].count('B'),4)
            self.assertEqual(row['sitor'].count('Y'),3)

    def test_stanag_headline_constants_and_interleaver_match_source(self):
        source=(ROOT/'docs/viz/stanag-explained.html').read_text()
        facts=json.loads(re.search(r'const GUIDE_FACTS = (.*?);',source)[1])
        def values(name):
            tree=ast.parse((ROOT/f'packages/hfmodem/hfmodem/bateleur/{name}.py').read_text())
            result={}
            for node in tree.body:
                if isinstance(node,ast.Assign):
                    try:value=ast.literal_eval(node.value)
                    except (ValueError,TypeError):continue
                    for target in node.targets:
                        if isinstance(target,ast.Name):result[target.id]=value
            return result
        spec=values('spec');s4529=values('s4529');s4539=values('s4539')
        self.assertEqual(facts['serial'],{'baud':spec['BAUD'],'carrier_hz':spec['CARRIER_HZ'],
                                        'fec_k':spec['CONV_K'],'fec_masks':list(spec['CONV_MASKS'])})
        for key,name in [('frame_symbols','FRAME4285'),('sync_symbols','SYNC4285_LEN'),
                         ('data_symbols','DATA4285_PER_FRAME'),('som','SOM4285'),('eom','EOM110')]:
            self.assertEqual(facts['s4285'][key],spec[name])
        self.assertEqual(facts['s4529']['baud'],s4529['BAUD'])
        for key,name in [('data_symbols','DATA_PER_FRAME'),('probe_symbols','PROBE_LEN'),
                         ('superframe_frames','SUPERFRAME'),('reinsert_symbols','REINSERT')]:
            self.assertEqual(facts['s4539'][key],s4539[name])
        self.assertEqual(facts['s4539']['rates'],list(s4539['RATES']))
        self.assertEqual(facts['s4539']['interleaver_frames'],list(s4539['INTERLEAVERS']))
        example=facts['s4539']
        self.assertEqual(example['example_stride'],s4539['INCREMENT'][example['example_rate']][example['example_frames']])
        self.assertEqual(example['example_slots'],s4539['DATA_PER_FRAME']*example['example_frames']*s4539['RATES'][example['example_rate']][0])
        geometry=json.loads(re.search(r'<script id="stanag-geometry" type="application/json">(.*?)</script>',source,re.S)[1])
        self.assertEqual({name:len(points) for name,points in geometry['points'].items()},facts['constellation_sizes'])
        # Every embedded constellation has distinct, finite I/Q points and a full table.
        for points in geometry['points'].values():
            self.assertEqual(len(points),len(set(map(tuple,points))))
            self.assertTrue(all(len(p)==2 and all(isinstance(v,(int,float)) and abs(v)<=1.001 for v in p) for p in points))

    def test_robust_packet_geometry_matches_waveform_constants(self):
        source=(ROOT/'docs/viz/robust-packet-explained.html').read_text()
        facts=json.loads(re.search(r'const GUIDE_FACTS = (.*?);',source)[1])
        tree=ast.parse((ROOT/'packages/hfmodem/hfmodem/brubru/waveform.py').read_text())
        nodes={target.id:node.value for node in tree.body if isinstance(node,ast.Assign)
               for target in node.targets if isinstance(target,ast.Name)}
        self.assertEqual(facts['nativeSampleRateHz'],ast.literal_eval(nodes['RATE']))
        self.assertEqual(facts['samplesPerSymbol'],ast.literal_eval(nodes['SPS']))
        self.assertEqual(facts['symbolRateBaud'],facts['nativeSampleRateHz']/facts['samplesPerSymbol'])
        tones=nodes['TONES'];count=ast.literal_eval(tones.left.left.args[0]);spacing=ast.literal_eval(tones.left.right)
        offset=ast.literal_eval(tones.right);centre=ast.literal_eval(nodes['CENTRE'])
        self.assertEqual(facts['carrierCount'],count)
        self.assertEqual(facts['carrierSpacingHz'],spacing)
        self.assertEqual(facts['audioCentreHz'],centre)
        self.assertEqual(facts['carrierFrequenciesHz'],[centre+i*spacing-offset for i in range(count)])
        for coded,useful,bits,rate in zip(facts['codedBitRates'],facts['usefulBitRates'],facts['bitsPerCarrierSymbol'],facts['codeRates']):
            self.assertEqual(coded,count*facts['symbolRateBaud']*bits)
            self.assertEqual(useful,coded*rate)

    def test_new_guides_include_diagrams_and_public_reference_paths(self):
        for name in ('robust-packet','stanag','rtty'):
            with self.subTest(mode=name):
                text=(ROOT/f'docs/viz/{name}-explained.html').read_text()
                self.assertGreaterEqual(len(re.findall(r'<svg\b',text)),3)
                self.assertIn('experimental',text.lower())
                self.assertIn('schematic',text.lower())
                self.assertGreaterEqual(len(re.findall(r'<h2\b',text)),5)
                self.assertGreaterEqual(len(re.findall(r'<figcaption\b',text)),3)
                self.assertGreaterEqual(len(re.findall(r'<summary\b',text)),2)
                self.assertNotIn('autoplay',text)
                self.assertIn('<!--colophon-->',text)

if __name__=='__main__':unittest.main()
