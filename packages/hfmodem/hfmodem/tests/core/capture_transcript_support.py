# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Synthetic PACTOR transcript builders for capture identity reader tests."""

HEAD = """  argv: --transmit --mycall W9SSJ --dxcall KB5LZK --outdir {outdir}
  resolved args: --band=40 --center=7101600 --dxcall=KB5LZK --p1-status-bits45=3 \
--pactor1-only=False --replay=None --transmit=True --tx-offset=0.185
cycle budget: settle 0.040 + packet 0.960 = 1.000 keyed of 1.250 s, leaving 250 ms clear; \
holds an answer whole for d 55-130 ms, of which 55-107 can be read, against a measured peer \
at 87-134 (median 96); TX offset 0.185 assumes d 105 ms
master grid: anchor @ sample 0, cycle 1.2500 s, free-running; a turnaround up to 130 ms \
is searchable
** CONNECTED to KB5LZK **
"""


def cycle(slot, d_ms, word='CS1', shift='normal', counter=1):
    """One keyed cycle: our packet, the peer's onset, and the word we read off it."""
    at = slot * 60000 + round((960 + d_ms) * 48)
    return (f'  TX[{slot}] P1 pkt#{counter} 100Bd 7B  (1.0s)  -- keying\n'
            f'    [grid] d {d_ms:.1f} ms @ sample {at}: error +0.0 ms, window +0.0 ms\n'
            f'    HOLD RX  {slot * 1.25:.2f}  cs       {word}/at anchor (0 bit errors, shif\n'
            f'    [host] rx unassigned: {word}/unassigned at anchor '
            f'(0 bit errors, PACTOR-1, shift {shift})\n')


def arm(tmp_path, name, body, outdir=None):
    path = tmp_path / 'working' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEAD.format(outdir=outdir or f'/captures/{name}') + body)
    return path
