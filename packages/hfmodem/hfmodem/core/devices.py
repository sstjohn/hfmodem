# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Which sound card, and why an unset one is a hard error.

`sounddevice` reads `None` as "the system default". On this station's laptop that
is the built-in microphone and speakers, so a transmit run started with the flag
missing modulates the speakers and records the room — and **every symptom of that
looks like a radio fault**: nobody answers, nothing decodes, and the levels look
plausible because a room is not silent. A transmit path must not be able to fall
back to the default, so `required=True` refuses instead and says which device is
missing.

A name substring resolves as well as an index, because a name is how the operator
and the station launcher refer to the interface ("USB Audio Device") and an index
is not.
"""
from __future__ import annotations


def list_devices() -> None:
    """Print the host's audio devices — what every `--list-devices` shows."""
    import sounddevice as sd
    print(sd.query_devices())


def find_device(name_or_index, kind: str, *, required: bool = False):
    """Resolve an audio device by name substring or index. `kind` is "in" or "out".

    Returns None for an unspecified device — the system default — unless
    `required`, which is what anything that keys a transmitter passes.
    """
    if kind not in ("in", "out"):
        # "output" reads as correct and silently matched on input channels, so a
        # pure-output interface never resolved and the refusal named the wrong
        # flag. Nothing downstream can tell a device that does not exist from one
        # asked about the wrong way.
        raise ValueError(f"kind is 'in' or 'out', not {kind!r}")
    if name_or_index is None:
        if required:
            raise SystemExit(
                f"--audio-{'out' if kind == 'out' else 'in'} is required when "
                f"transmitting: unset means the system default, which is the "
                f"laptop's own {'speakers' if kind == 'out' else 'microphone'}, "
                f"not the radio. Pass the interface by name (--list-devices).")
        return None
    try:
        return int(name_or_index)
    except ValueError:
        pass
    import sounddevice as sd
    want = name_or_index.lower()
    usable = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
              if (d["max_output_channels"] if kind == "out"
                  else d["max_input_channels"]) > 0]
    # An exact name beats a substring, so a device whose name is contained in
    # another's can still be asked for by its own.
    exact = [(i, n) for i, n in usable if n.lower() == want]
    hits = exact or [(i, n) for i, n in usable if want in n.lower()]
    if not hits:
        # WHICH ABSENCE IT IS. A device that is present and has no channels of
        # this direction reads as missing otherwise, and the operator has the
        # listing open showing it: `KT USB Audio` is `0 in, 2 out`, so asking it
        # for input answered "no device matching" about a name three lines above.
        other = "out" if kind == "in" else "in"
        wrong = [n for d in sd.query_devices()
                 if (n := d["name"]).lower() == want or want in n.lower()]
        if wrong:
            raise SystemExit(
                f"{name_or_index!r} is not an {kind} device -- it is "
                f"{other}-only. Name a device with {kind} channels "
                "(use --list-devices)")
        raise SystemExit(
            f"no {kind} audio device matching {name_or_index!r} (use --list-devices)")
    if len(hits) > 1:
        # AMBIGUITY IS A REFUSAL, not a coin toss. This took the first match, and
        # on 2026-08-14 a second interface arrived whose name -- `KT USB Audio` --
        # contains the `USB Audio` the station had always used for the rig's
        # codec. It kept working only because USB enumeration happened to put the
        # codec first, which is luck that a replug or a reboot spends: the same
        # first-match on a transmit path is a station modulating an interface
        # nobody is listening to, past a preflight that passes because *a*
        # matching device is present.
        named = ", ".join(f"{i}: {n}" for i, n in hits)
        raise SystemExit(
            f"{name_or_index!r} matches {len(hits)} {kind} devices ({named}) -- "
            "name one precisely; an exact name always wins over a substring")
    return hits[0][0]
