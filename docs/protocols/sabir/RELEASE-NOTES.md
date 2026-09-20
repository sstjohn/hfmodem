# Implementation status

Sabir is unpublished. Its normative behavior is specified in [SPEC.md](SPEC.md),
[NEGOTIATION.md](NEGOTIATION.md) and [HOST-API.md](HOST-API.md).

The implementation includes waveform generation and reception, capability
negotiation, bidirectional selective ARQ, retransmission and recovery, mandatory
record integrity, and unacknowledged fragmented objects. Host applications use
a single framed CBOR connection. No VARA host adapter is provided.

Deterministic tests and audio-channel simulations exercise the protocol.
[ONAIR.md](ONAIR.md) distinguishes beacon reception, station integration and
uncompleted two-station session validation. Simulated throughput includes only
the overhead stated for each measurement and is not a measured RF link rate.

Run the Sabir tests with:

```
python -m pytest packages/hfmodem/hfmodem/tests/sabir
```

Session throughput experiments are available through:

```
python -m hfmodem.sabir.sim.release --bytes 250000 --snr 40
python -m hfmodem.sabir.sim.release --channel good --snr 30 --bytes 30000
```

These report virtual-air session goodput separately from error-free cycle rate
and CPU time. The channel option uses a continuous fading trace across packet
and acknowledgement gaps. A two-station radio session, hardware spectral
measurements and behavior under real adjacent-channel traffic remain untested.
