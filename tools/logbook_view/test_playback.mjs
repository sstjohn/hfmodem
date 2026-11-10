// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function recordingFixture(readyState=1) {
 const listeners=new Map();
 const audio={readyState,currentTime:0,paused:true,src:'part.wav',playbackRate:1,
  getAttribute(n){return this[n];},
  addEventListener(n,f){if(!listeners.has(n))listeners.set(n,new Set());listeners.get(n).add(f);},
  removeEventListener(n,f){listeners.get(n)?.delete(f);},
  pause(){this.paused=true;},play(){this.paused=false;return Promise.resolve();},dispatchEvent(){}};
 const context=vm.createContext({Event});
 vm.runInContext(readFileSync(new URL('recording_audio.js',import.meta.url),'utf8')+';globalThis.player=RecordingAudio;',context);
 return {audio,player:context.player,emit(n){for(const f of [...listeners.get(n)||[]])f();}};
}
for(const ready of [0,1]) {
 test(`recording bind starts native playback at selected offset (ready=${ready})`,()=>{
  const f=recordingFixture(ready);
  f.player.bind(f.audio,[{url:'part.wav',start_s:0,source_offset_s:59,duration_s:4}]);
  if(!ready){f.audio.readyState=1;f.emit('loadedmetadata');}
  assert.equal(f.audio.currentTime,59);
 });
}
test('recording bind clamps native seeking to both selection bounds',()=>{
 const f=recordingFixture();
 f.player.bind(f.audio,[{url:'part.wav',start_s:0,source_offset_s:59,duration_s:4}]);
 f.audio.currentTime=0;f.emit('seeking');
 assert.equal(f.audio.currentTime,59);
 f.audio.currentTime=70;f.audio.paused=false;f.emit('seeking');
 assert.equal(f.audio.currentTime,63);assert.equal(f.audio.paused,true);
});

// Exercise the shipped player with delayed browser metadata, seek and play events.
for (const name of ['viewer.js', '../../site/js/examples/viewer.js']) {
 const source=readFileSync(new URL(name, import.meta.url),'utf8');
 const player=source.slice(source.indexOf('function stop()'),source.indexOf('function select('));
 function fixture(){
  const listeners=new Map();let position=0,resolvePlay;
  const audio={readyState:0,seeking:false,paused:true,
   addEventListener:(n,f)=>listeners.set(n,f),removeEventListener:n=>listeners.delete(n),
   load(){},pause(){this.paused=true;},
   get currentTime(){return position;},set currentTime(t){position=t;this.seeking=true;},
   play(){this.paused=false;return new Promise(r=>{resolvePlay=r;});}};
  const elements={audio,play:{attributes:{},setAttribute(n,v){this.attributes[n]=v;}},listen:{checked:true},notice:{textContent:''},selection:{},seek:{},clock:{}};
  const context=vm.createContext({state:{time:8,playing:false},manifest:{duration_s:100},lastTick:0,
   $:n=>elements[n],syncAudio(){return 30.22;},audioOffset:()=>22.22,setTime(){},setTimeout,clearTimeout,
   document:{querySelectorAll:()=>[]}});
  vm.runInContext(player,context);
  return {context,audio,elements,emit(n){listeners.get(n)?.();},finishPlay(){resolvePlay();}};
 }
 test(`${name}: seeks to the recording offset before starting playback`,async()=>{
  const f=fixture(),pending=f.context.play();
  assert.equal(f.audio.paused,true);assert.equal(f.context.state.playing,false);
  f.audio.readyState=1;f.emit('loadedmetadata');await Promise.resolve();
  assert.equal(f.audio.currentTime,30.22);assert.equal(f.audio.paused,true);
  f.audio.seeking=false;f.emit('seeked');await Promise.resolve();
  assert.equal(f.audio.paused,false);assert.equal(f.context.state.playing,false);
  f.finishPlay();await pending;
  assert.equal(f.context.state.playing,true);assert.ok(f.elements.play.disabled===false||f.elements.play.attributes['aria-disabled']==='false');
 });
 test(`${name}: stop cancels a pending play without leaving audio running`,async()=>{
  const f=fixture();f.audio.readyState=1;
  const pending=f.context.play();f.audio.seeking=false;f.emit('seeked');await Promise.resolve();
  f.context.stop();f.finishPlay();await pending;
  assert.equal(f.audio.paused,true);assert.equal(f.context.state.playing,false);
 });
}
