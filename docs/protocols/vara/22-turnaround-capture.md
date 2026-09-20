# Turnaround capture boundary

The capture cursor is anchored to the end of our monitored transmission, then
advanced by the configured TX hold and echo guard. Detecting that end using only
an absolute mute threshold confuses the codec's decaying DC offset with audio.
If the decay has not reached the threshold when playback returns, the detector
falls back to the capture callback boundary. Both cases can discard recoverable
receive samples and make the hold calculation sleep longer than intended.

The detector now first looks for a sustained relative drop in the first
difference of the monitored samples. This rejects DC settling without depending
on the monitor's absolute gain. It requires 12 ms of live audio followed by at
least 8 ms at least 20 dB below the recent monitor level. The earliest possible
transmitted end bounds the search, and the last qualifying edge avoids the
startup mute. An incomplete or unsupported edge uses the existing fallback.

The TX hold and echo-guard constants are unchanged. Playback still drains before
the key can be released. This does not remove the radio's own receive mute or
establish an RF emission length from a local monitor recording.

An offline replay of 1449 capture snapshots from 119 September 17/18 recordings
found 1325 supported AC edges. The cursor moves earlier on 1273 snapshots, by
22 ms at the median across all snapshots; eight move later by one 2 ms analysis
frame. These are differences in the software cursor, not measurements of extra
RF symbols recovered or a claim of improved mail completion.

Three native control-tail fixtures retain the cases where the absolute detector
fell back to the callback boundary. Tests also cover gain variation, delayed
monitor audio, incomplete callbacks, isolated low frames and startup mute.
The transport and key-refusal regressions continue to pass. The full turnaround
audit remains in the private station archive.
