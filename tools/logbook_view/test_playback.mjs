// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

// Exercise the shipped player with delayed browser metadata, seek and play events.
for (const name of ['viewer.js', '../../site/examples/viewer.js']) {
 const source=readFileSync(new URL(name, import.meta.url),'utf8');
 const player=source.slice(source.indexOf('function stop()'),source.indexOf('function select('));
 function fixture(){
  const listeners=new Map();let position=0,resolvePlay;
  const audio={readyState:0,seeking:false,paused:true,
   addEventListener:(n,f)=>listeners.set(n,f),removeEventListener:n=>listeners.delete(n),
   load(){},pause(){this.paused=true;},
   get currentTime(){return position;},set currentTime(t){position=t;this.seeking=true;},
   play(){this.paused=false;return new Promise(r=>{resolvePlay=r;});}};
  const elements={audio,play:{},listen:{checked:true},notice:{textContent:''},selection:{},seek:{},clock:{}};
  const context=vm.createContext({state:{time:8,playing:false},manifest:{duration_s:100},lastTick:0,
   $:n=>elements[n],syncAudio(){},audioOffset:()=>22.22,setTime(){},setTimeout,clearTimeout,
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
  assert.equal(f.context.state.playing,true);assert.equal(f.elements.play.disabled,false);
 });
 test(`${name}: stop cancels a pending play without leaving audio running`,async()=>{
  const f=fixture();f.audio.readyState=1;
  const pending=f.context.play();f.audio.seeking=false;f.emit('seeked');await Promise.resolve();
  f.context.stop();f.finishPlay();await pending;
  assert.equal(f.audio.paused,true);assert.equal(f.context.state.playing,false);
 });
}
