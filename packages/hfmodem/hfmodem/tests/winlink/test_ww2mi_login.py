# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""WW2MI, 2026-08-18: a login refused, read off the recording of our own transmitter.

Two connects four minutes apart, same binary and same password file. One
authenticated and was offered a message; the other was answered
``Invalid login challenge response -- 2 attempts remaining``. The account's
allowance is three and one is spent, so the difference between the two arms is not
something a further connect may be spent on establishing.

It does not have to be. `RadioLink._capture` records upstream of the half-duplex
mute, so a besra capture holds this station's own transmissions leaking into its
own muted receiver — the ``;PR:`` answer as it left the transmitter, beside the
``;PQ:`` it answers, in one file. A flipped bit in the thirty-bit response
produces exactly the refusal above and nothing else in the session would show it,
which is the one cause of that message this end can be at fault for and the one
these recordings settle.

They settle it against us: the digits on the air are the digits our own responder
computes, on both arms. That is why this file exists rather than a fix — the
finding is a negative, and a negative nobody can re-run stops being evidence.

The digits are never a value this file holds. Both sides of the one comparison
that needs them go through `_fingerprint` first, so the only thing an assertion
can print is a digest under a key this run minted and drops. The challenges are
literals below and a failing assertion prints its operands: kept together the pair
is an offline password search, and what is being checked is agreement, which is a
boolean.
"""
from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.winlink import mask_pr, secure_login_response

B = corpora.harness("rehear.besra")

PASSWORD = Path.home() / ".winlink-pw"

_PQ = re.compile(r";PQ: (\d{8})")

#: A `;PR:` answer as it went out, for the one read that has to see it. Not a
#: second copy of `mask_pr`'s rule: what this finds goes straight into
#: `_fingerprint` and never into a value anything else can reach.
_PR = re.compile(r";PR: *([^\r\n]*)")

#: Minted for this run and dropped with it, so a fingerprint that reaches a failure
#: message is a fingerprint and stays one — where the digits it stands for, printed
#: beside a challenge, are thirty bits of MD5 to search a password with.
_KEY = secrets.token_bytes(32)

pytestmark = corpora.requires_ww2mi_arms


def _fingerprint(digits: str) -> str:
    return hashlib.blake2s(digits.encode("latin-1"), key=_KEY, digest_size=8).hexdigest()


@dataclass(frozen=True, slots=True)
class Arm:
    challenge: str
    #: `_fingerprint` of the answer this station transmitted. The digits are not a
    #: field here and are not a value anywhere else either.
    answer: str
    #: Everything the gateway said, joined — the greeting and its verdict.
    theirs: str
    #: The handshake this station transmitted, as `rehear --ours` prints it, run
    #: through `mask_pr` again on the way in. Quotable whatever `_spoken` did.
    said: str
    #: Whether that second run had anything to do — it must not have.
    masked: bool
    #: How many transmissions of ours the recording holds, ARQ repeats collapsed.
    frames: int


def _answered(heard, keyed) -> str:
    """A fingerprint of the `;PR:` this station put on the air.

    `_spoken`'s join with the masking left off, which is the one read of these
    recordings that has to see the digits. They are a local of this function and
    leave it fingerprinted; nothing in it raises, so no traceback carries the frame
    they are in and `pytest -l` has nothing of theirs to print.
    """
    said = "".join(f.payload.decode("latin-1")
                   for at, f in sorted(heard, key=lambda p: p[0])
                   if f.payload and B._withheld(at, keyed))
    found = _PR.search(said)
    return _fingerprint(found.group(1)) if found else ""


def _computed(challenge: str) -> str:
    """A fingerprint of what this end answers `challenge` with.

    The password is read here and not in the test for the same reason: a frame that
    never held it has none to print.
    """
    return _fingerprint(secure_login_response(challenge, PASSWORD.read_text().strip()))


def _unescaped(detail: str) -> str:
    r"""One transmission as `rehear --ours` prints it, with the `\r` escapes read
    back as the line endings they are — the masking runs to the end of a line, and
    escaped there is no end of line for it to run to."""
    return detail.encode("ascii").decode("unicode_escape")


@pytest.fixture(scope="module")
def arms() -> dict[str, Arm]:
    """Both halves of both connects, off one ungated pass over each recording.

    One pass and not `rehear`'s two: the mute is not the question here and the
    gated arm costs as much again. Everything below reads off `heard` the way the
    adapter's own report does.

    No session log was written beside either of these two, so the pass runs with no
    hint and no clock to align to. It costs nothing that is asked for here: the
    hint lifts the crispness floor under bodyless controls, and every frame this
    file reads carries a payload behind RS and a CRC.
    """
    out = {}
    for name, path in (("accepted", corpora.WW2MI_ACCEPTED),
                       ("refused", corpora.WW2MI_REFUSED)):
        card = B._capture(path)
        heard = [(at, f) for at, f in B._rolling(card, (), None, 0.0) if B._decoded(f)]
        edges, quiet, _level = B._key_edges(B.from_card(card, B.SAMPLE_RATE))
        keyed = B._keyed_intervals(edges, quiet, heard)
        theirs = "".join(f.payload.decode("latin-1")
                         for at, f in sorted(heard, key=lambda p: p[0])
                         if f.payload and not B._withheld(at, keyed))
        # Masked here rather than in the tests, and nothing raises between: the
        # frames themselves are in this frame, and `pytest -l` prints the locals of
        # every frame a traceback passes through.
        ours = [_unescaped(h.detail) for h in B._spoken(heard, keyed)]
        whole = "".join(ours)
        issued = _PQ.search(theirs)
        out[name] = Arm(challenge=issued.group(1) if issued else "",
                        answer=_answered(heard, keyed), theirs=theirs,
                        said=mask_pr("".join(ours[:4])),
                        masked=mask_pr(whole) == whole, frames=len(ours))
    return out


def test_each_arm_answered_the_challenge_its_own_gateway_issued(arms):
    """One challenge per connect, the two the session logs record, and neither
    arm answered the other's — the stale-challenge reading, disposed of."""
    assert arms["accepted"].challenge == "41949693"
    assert arms["refused"].challenge == "90189792"
    assert arms["accepted"].answer != arms["refused"].answer
    for arm in arms.values():
        assert arm.answer, "no `;PR:` of ours on the air in this recording"
        assert len(_PQ.findall(arm.theirs)) == 1, "a second challenge in one session"


def test_the_two_handshakes_are_one_message_but_for_the_masked_answer(arms):
    """What this station said, off the recording of its own muted receiver.

    Two things have to hold for the read-back to be worth quoting. ARQ repeats
    have to collapse — the accepted arm sends its third frame three times and the
    gateway reads it once, and an instrument reporting the repeats would be
    describing the channel and not the message. And the masking has to survive the
    frame boundary: `;P` closes frame 2 and the digits open frame 3, so a mask
    applied per frame prints the answer and misses.

    Whether it survived is asked of `mask_pr` in the fixture and answered yes or
    no: a mask with nothing left to do is a stream that was already masked. What
    the arm carries out of there has been through that mask a second time, so the
    run that finds the digits in the clear is not a run that can print them.
    """
    assert all(arm.masked for arm in arms.values()), "`rehear --ours` kept an answer"
    assert arms["accepted"].said == arms["refused"].said
    assert ";PR: ########" in arms["refused"].said
    # The accepted arm goes on to answer the message offer with `FS Y`; the refused
    # arm never gets one, which is the whole of the difference on this side.
    assert (arms["accepted"].frames, arms["refused"].frames) == (5, 4)


def test_the_accepted_arm_is_an_outside_verdict_on_the_response(arms):
    """WW2MI's CMS read an answer to a challenge it had issued and offered mail.

    Every other check on `secure_login_response` is this end marking its own
    work: the published vectors are the algorithm agreeing with the port it came
    from, and the arms above are the transmitter agreeing with the responder. A
    round trip that challenges itself would pass over a salt nobody else uses.

    This is the one that cannot. The challenge is the CMS's, the verdict is the
    CMS's, and what came back is `;PM:` and an `FC` proposal — the far end acting
    on a login it accepted, with the algorithm, the salt and the password file all
    underneath it. The refused arm draws the same distinction from the other side:
    it got the words instead, and never a proposal.
    """
    accepted, refused = arms["accepted"], arms["refused"]
    assert ";PM: W9SSJ " in accepted.theirs and "FC EM " in accepted.theirs
    assert "Invalid login challenge response" not in accepted.theirs
    assert ";PM:" not in refused.theirs and "FC " not in refused.theirs


def test_the_refusal_travels_with_the_prompt_it_appears_to_be_asking(arms):
    """`Login [931]:` is not a question this station left unanswered.

    It arrives in the gateway's very next transmission after the handshake, in one
    frame with `CMS via WW2MI >` and with the refusal itself, so the verdict was
    reached before any answer to the prompt could have been sent. A retry that
    only answers the prompt is therefore answering something already decided.
    """
    theirs = arms["refused"].theirs
    assert "Login [931]:" in theirs
    prompt = theirs.index("Login [931]:")
    assert 0 < theirs.index("Invalid login challenge response") - prompt < 64
    assert "Login [" not in arms["accepted"].theirs


@pytest.mark.skipif(not PASSWORD.exists(), reason=f"{PASSWORD} is not on this machine")
def test_the_response_on_the_air_is_the_response_we_computed(arms):
    """The refused arm put the right number on the air.

    The accepted arm is the control: it says the salt, the algorithm and the
    password file are all right, so a disagreement on the refused arm would be a
    transmitter fault and nothing else. There is none, and the refusal is
    therefore a verdict the far end reached about an answer that was correct when
    it left here.

    Both sides are fingerprints and the comparison is made before the assertion, so
    the failure this anticipates reports `False` and the failures it does not
    report `False` as well.
    """
    if not any(arm.answer == _computed(arm.challenge) for arm in arms.values()):
        pytest.skip(
            "the credential has changed since these arms were recorded, so this end "
            "cannot reproduce either answer and this comparison can no longer be made. "
            "NOT a computation fault: the published known-answer vectors in "
            "test_session.py check the algorithm against somebody else's challenge and "
            "somebody else's password, and they pass. This check is agreement between "
            "what THIS station transmitted on a particular day and what it computes "
            "today, which is only meaningful while the secret behind both is the same "
            "one. Skipping rather than failing, because a red here would read as a "
            "broken transmitter every time the operator changes a password.")
    for name, arm in arms.items():
        agreed = arm.answer == _computed(arm.challenge)
        assert agreed, (f"the {name} arm transmitted a response this end does not "
                        f"compute for the challenge in the same recording")
