# WS8EOC, September 20, 2026, 20:08 CDT

Unmodified 48 kHz mono PCM16 excerpts from `captures/onair-0920-2008/stream.wav`.
`metadata.json` records source and excerpt SHA-256 hashes, absolute half-open
sample ranges, measured peer phases, and the associated transmit sidecars.
The log is `working/onair-0920-2008/pactor-current-ws8eoc-80-pounce-20260921T010819Z.log`.

Eight control excerpts precede recorded long-frame deadline overruns. Three
receive windows contain the current repeated changeover, although the live
pre-key scan often deferred its body and retained an older frame.

The regression feeds real audio through the control/body decoders, ARQ and
RadioTx. Outgoing payloads are synthetic. An elapsed sample clock charges the
recorded aggregate overrun at rendering, and models the observed three-callback
batch. This is a deterministic deadline/recovery test, not a CPU benchmark or
proof of remote reception. The initial missed opportunity can remain; a
completed waveform must not force the additional parity wait. The changeover
cases charge six milliseconds to the speculative ordinary-header read which
previously ran before the known repeated CS3 body. CRC and confirmation gates
remain required, including erased-body and truncated-head negatives.
