// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
const RecordingAudio = (() => {
  const players = new WeakMap();
  function segmentAt(parts, time) {
    return parts.findLastIndex(p => p.start_s <= time);
  }
  function seek(audio, time) {
    const player = players.get(audio);
    if (!player) { audio.currentTime = time; return time; }
    const generation = ++player.generation;
    if (player.pending) audio.removeEventListener('loadedmetadata', player.pending);
    player.pending = null;
    const index = Math.max(0, segmentAt(player.parts, time));
    const part = player.parts[index];
    if (time < part.start_s || time >= part.start_s + (part.duration_s ?? Infinity)) {
      audio.pause?.(); audio.dispatchEvent(new Event('error')); return null;
    }
    const offset = (part.source_offset_s || 0) + Math.max(0, time - part.start_s);
    player.index = index;
    const changed = audio.getAttribute('src') !== part.url;
    if (changed) audio.src = part.url;
    if (changed || audio.readyState === 0) {
      player.pending = () => {
        if (players.get(audio) !== player || player.generation !== generation) return;
        audio.currentTime = offset;
        player.pending = null;
      };
      audio.addEventListener('loadedmetadata', player.pending, {once:true});
    } else audio.currentTime = offset;
    return offset;
  }
  function position(audio) {
    const player = players.get(audio);
    return player ? player.parts[player.index].start_s + Math.min(player.parts[player.index].duration_s ?? Infinity,
      Math.max(0, audio.currentTime - (player.parts[player.index].source_offset_s || 0))) : audio.currentTime;
  }
  function bind(audio, parts) {
    const existing = players.get(audio);
    if (existing) {
      ++existing.generation;
      audio.removeEventListener('ended', existing.ended);
      audio.removeEventListener('timeupdate', existing.update);
      audio.removeEventListener('seeking', existing.clamp);
      audio.removeEventListener('play', existing.clamp);
      if (existing.pending) audio.removeEventListener('loadedmetadata', existing.pending);
      players.delete(audio);
    }
    if (!parts?.length) return;
    const player = {parts, index:0, generation:0, pending:null};
    players.set(audio, player);
    player.ended = () => {
      const player = players.get(audio), next = player?.parts[player.index + 1];
      if (!next) { audio.pause?.(); return; }
      if (next.start_s > player.parts[player.index].start_s + player.parts[player.index].duration_s) {
        audio.pause?.(); audio.dispatchEvent(new Event("error")); return;
      }
      const rate = audio.playbackRate;
      seek(audio, next.start_s);
      audio.playbackRate = rate;
      audio.play().catch(() => audio.dispatchEvent(new Event('error')));
    };
    player.update = () => {
      const part = player.parts[player.index];
      if (audio.currentTime < (part.source_offset_s || 0)) audio.currentTime = part.source_offset_s || 0;
      if (audio.currentTime >= (part.source_offset_s || 0) + part.duration_s) player.ended();
    };
    player.clamp = () => {
      const part = player.parts[player.index], first = part.source_offset_s || 0;
      if (audio.currentTime < first) audio.currentTime = first;
      if (audio.currentTime >= first + part.duration_s) {
        if (audio.currentTime > first + part.duration_s) audio.currentTime = first + part.duration_s;
        audio.pause?.();
      }
    };
    audio.addEventListener('ended', player.ended);
    audio.addEventListener('timeupdate', player.update);
    audio.addEventListener('seeking', player.clamp);
    audio.addEventListener('play', player.clamp);
    seek(audio, parts[0].start_s);
  }
  return {bind, seek, position, segmentAt};
})();
