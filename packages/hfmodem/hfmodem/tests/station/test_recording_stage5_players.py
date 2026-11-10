# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Both logbook players honor physical offsets and unavailable contact intervals."""
import json
import os
import shutil
import subprocess

import pytest

from hfmodem.tests import hermetic
from hfmodem.tests.station.recording_stage5_support import ROOT

HERE = ROOT / 'tools/logbook_view'


def function(source, name):
    start = source.index('function ' + name + '(')
    line = source[start:source.find('\n', start)]
    if line.endswith('}'):
        return line
    return source[start:source.index('\n}', start) + 2]


def node(script):
    executable = shutil.which('node')
    if executable is None and os.name == 'nt':
        pytest.skip('node is not installed on this Windows host')
    assert executable, 'the existing player test runner requires node'
    hermetic.guard.admit([executable, '-e'], os.environ)
    result = subprocess.run([executable, '-e', script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


AUDIO = '''
const assert=require('node:assert/strict');
class Audio extends EventTarget {
  currentTime=0; readyState=1; playbackRate=1; url='first.wav'; plays=0; paused=false;
  getAttribute(){return this.url;}
  set src(url){this.url=url;this.currentTime=0;}
  play(){this.plays++;return Promise.resolve();}
  pause(){this.paused=true;}
}
'''


@pytest.mark.parametrize('player', ['logbook', 'viewer'])
def test_players_seek_physical_source_offset_and_stop_at_contact_end(player):
    source = (HERE / 'recording_audio.js').read_text(encoding='utf-8')
    setup = '''
const a=new Audio();
const parts=[{url:'first.wav',start_s:0,duration_s:1,source_offset_s:59},
             {url:'second.wav',start_s:1,duration_s:2,source_offset_s:0}];
'''
    if player == 'viewer':
        setup += function((HERE / 'viewer.js').read_text(encoding='utf-8'), 'audioOffset') + '\n'
        setup += function((HERE / 'viewer.js').read_text(encoding='utf-8'), 'syncAudio') + '\n'
        setup += '''
const elements={audio:a,listenSource:{value:'rx'},speed:{value:'1'}};
const $=id=>elements[id],state={time:.5};
const manifest={recording:{url:'first.wav',segments:parts},sources:[{id:'rx',url:'first.wav',segments:parts,offset_s:0}]};
syncAudio();
'''
    else:
        setup += 'RecordingAudio.bind(a,parts); RecordingAudio.seek(a,.5);\n'
    node(AUDIO + source + setup + '''
a.dispatchEvent(new Event('loadedmetadata'));
assert.equal(a.currentTime,59.5);
assert.equal(RecordingAudio.position(a),.5);
a.currentTime=60;
a.dispatchEvent(new Event('timeupdate'));
a.dispatchEvent(new Event('loadedmetadata'));
assert.equal(a.url,'second.wav');
assert.equal(RecordingAudio.position(a),1);
a.currentTime=2;
a.dispatchEvent(new Event('timeupdate'));
assert.equal(a.paused,true);
assert.ok(RecordingAudio.position(a)<=3);
''')


def test_logbook_player_renders_disabled_and_unavailable_states_without_audio_action():
    source = (HERE / 'index.html').read_text(encoding='utf-8')
    selected = {'audio_state': 'disabled', 'recording': {'policy': 'off'}, 'audio': []}
    node(AUDIO + '''
const esc=x=>String(x),fmtDur=x=>String(x);
''' + function(source, 'recordingDuration') + '\n' + function(source, 'recordings') + '\n' + f'''
let html=recordings({json.dumps(selected)});
assert.match(html,/recording off|disabled/i);
assert.doesNotMatch(html,/<audio/);
html=recordings({{audio_state:'unavailable',audio:[{{kind:'rig rx',label:'Our receiver',state:'unavailable',url:null}}]}});
assert.match(html,/unavailable.*possibly moved/i);
assert.doesNotMatch(html,/<audio/);
''')


@pytest.mark.parametrize('player', ['logbook', 'viewer'])
def test_players_keep_gaps_as_gaps_and_never_seek_missing_audio(player):
    source = (HERE / 'recording_audio.js').read_text(encoding='utf-8')
    setup = '''
const a=new Audio();
const parts=[{url:'first.wav',start_s:0,duration_s:1,source_offset_s:0},
             {url:'last.wav',start_s:3,duration_s:1,source_offset_s:0}];
RecordingAudio.bind(a,parts);
'''
    if player == 'viewer':
        setup += function((HERE / 'viewer.js').read_text(encoding='utf-8'), 'audioOffset') + '\n'
        setup += function((HERE / 'viewer.js').read_text(encoding='utf-8'), 'syncAudio') + '\n'
        setup += '''
const elements={audio:a,listenSource:{value:'rx'},speed:{value:'1'}};
const $=id=>elements[id],state={time:2};
const manifest={recording:{url:'first.wav',segments:parts},sources:[{id:'rx',url:'first.wav',segments:parts,offset_s:0}]};
'''
    setup += 'let missing=false; a.addEventListener("error",()=>missing=true);\n'
    setup += 'syncAudio();\n' if player == 'viewer' else 'RecordingAudio.seek(a,2);\n'
    node(AUDIO + source + setup + '''

assert.equal(missing,true);
assert.equal(a.plays,0);
assert.ok(RecordingAudio.position(a)<=1);
''')
