# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""What an arm's transcript says the link did: how far it got, how wide it ran, how fast.

Speed is counted in each direction only where the other end is shown to have
read it: `decoded` is the fastest the peer sent that we decoded with a clean
CRC, `acknowledged` the fastest we sent that the peer acknowledged. Either one
is the link carrying that speed; neither is a level merely keyed or requested.

Each protocol keeps its own vocabulary: VARA's host levels (L1..L10; at BW2300
a record R is level R+1), PACTOR-1's bit rate and PACTOR-2/3's speed levels,
ARDOP's frame types.
"""
from __future__ import annotations

import re
from collections import Counter

ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

# The B2F exchange as the mail client logs it, both sides' commands in the order they went.
B2F = re.compile(r'\bmail: (we|peer) (?:sent|queued|said): (?:FC \w+ (\w+)|FS (\S+)|(FF|FQ)\b)')
FS_ANSWER = re.compile(r'([+Yy])|[-Nn]|[=Ll]|([!Aa])\d+')
# A message we took in whole; a .partial.lzh is one that stopped short.
RECEIVED = re.compile(r'\bmail: received (\w{12})\b|\bmail: wrote \S*?(\w{12})\.b2f\b')
# The client's closing line for a message of ours once the gateway came back with FF.
FAR_END_FF = re.compile(r'\bmail: (?:prepared|transmitted) (\w{12}) — accepted FS [^\n]*far end FF received')


# The live verdicts' milestones (tools/consolelib/{pactor,vara,ardop}.py), unanchored where
# these transcripts put a log stamp or an indent before the line, and with the older spellings
# of the same event. A windowed sense line reads one window, so only a full count of them clears.
CLEARED = re.compile(r'occupancy decision complete|channel is free(?! -- one more window)|'
                     r'channel sense: [^\n]*(?:-> clear|\(clear\)(?:$|;[^\n]*clear windows (\d+)/\1\b))', re.M)
# The line the peer's grant leaves, and the submode the link entered on it.
PACTOR_ENTRY = {'0x59A grant -> PACTOR-3 entry packet': 'PAC3',
                '0x6A9 experimental entry response -> PACTOR-2 SL1': 'PAC2'}
MILESTONES = {
    'pactor': (
        ('cleared', CLEARED),
        ('keyed', re.compile(r'TX\[\d+\][^\n]*-- keying|^keyed [1-9]\d* of', re.M)),
        ('answer', re.compile(r'^control signals decoded [1-9]', re.M)),
        ('connect', re.compile(r'\*\* CONNECTED to |state CONNECTING -> CONNECTED')),
        ('grant', re.compile('|'.join(map(re.escape, PACTOR_ENTRY)))),
        ('entry read', re.compile(r'the peer answered the entry packet')),
        ('packet acknowledged', re.compile(r'^verdict: ACKNOWLEDGED', re.M)),
        ('greeting', re.compile(r'\bmail: remote SID ')),
        ('login', re.compile(r'\bmail: stage (?:awaiting proposal answer|their turn|receiving|closed)')),
        # "mail: accepted <MID> and never got it" is a message that did not arrive.
        ('mail', RECEIVED),
    ),
    'vara': (
        ('cleared', CLEARED),
        ('keyed', re.compile(r'PTT ON|^originating: ', re.M)),
        ('answer', re.compile(r'lattice frame \d+|rx connect-response confirmed for ')),
        ('connect', re.compile(r'— CONNECTED|— the link is up|\bCONNECTED\s+[A-Z0-9]')),
        ('greeting', re.compile(r'\bmail: remote SID ')),
        ('mail', re.compile(r'\bmail: exchange complete')),
    ),
    'ardop': (
        ('cleared', CLEARED),
        ('keyed', re.compile(r'PTT ON|INFO TX |\bTX ConReq')),
        ('answer', re.compile(r'INFO RX Con\w+|CONNECTION REJECTED BY ')),
        ('connect', re.compile(r'ARQ CONNECTION ESTABLISHED WITH |^\s*CONNECTED [A-Z0-9/-]+ @ \d+ Hz', re.M)),
        ('greeting', re.compile(r'\bmail: remote SID ')),
        ('mail', re.compile(r'\bmail: exchange complete')),
    ),
}


def milestones(text: str, protocol: str) -> list[str]:
    """The milestones a transcript proves, in the protocol's order. Mail is the last of them,
    reached by the protocol's own line or by a message that mail() shows moving either way."""
    found = [name for name, pattern in MILESTONES[protocol] if pattern.search(text)]
    if 'mail' not in found and any(mail(text).values()):
        found.append('mail')
    return found


def mail(text: str) -> dict[str, list[str]]:
    """The MIDs of the messages that moved, each way. One of ours went once the gateway
    accepted its proposal and then took the turn itself (its own FC, FF or FQ), which B2F
    only lets it do after the message is in. One came to us when the client wrote it out."""
    sent = [m[1] for m in FAR_END_FF.finditer(text)]
    for session in text.split('mail: remote SID '):
        offered: list[str] = []
        accepted: list[str] = []
        for m in B2F.finditer(session):
            if m[1] == 'we':
                if m[2]:
                    offered.append(m[2])
            elif m[3] is not None:
                answers = [bool(a[1] or a[2]) for a in FS_ANSWER.finditer(m[3])]
                accepted += [mid for mid, yes in zip(offered, answers) if yes]
                offered = []
            else:
                sent += accepted
                accepted = []
    received = [m[1] or m[2] for m in RECEIVED.finditer(text)]
    return {'sent': list(dict.fromkeys(sent)), 'received': list(dict.fromkeys(received))}


def merged(recorded: list[str], found: list[str], protocol: str) -> list[str]:
    """The recorded milestones, each one the transcript adds set in before the first that follows it."""
    order = [name for name, _ in MILESTONES[protocol]]
    out = list(recorded)
    for m in found:
        if m not in out:
            out.insert(next((i for i, r in enumerate(out) if r in order and order.index(r) > order.index(m)), len(out)), m)
    return out


# TX power, as a transcript states it: RFPOWER read back over CAT, never a measured
# output. onair.sh's wrapper reads it once before the gate and once after the modem
# exits ("TX power: RFPOWER 0.6 -> 60.0 W", older transcripts without the trailing
# "setting (not measured RF output)", and "CAT did not answer" when the rig refused
# it). A PACTOR arm's own phase power (shrike/onair.py's _PhasePower) additionally
# reads the rig back before touching it ("RFPOWER 0.50 -> 50 W at arm start;
# restored at teardown"), may put it on a different level for the phase it is
# about to key ("RFPOWER 0.25 -> 25 W (PACTOR-1, before the first call), handed to
# rigctl in ... ms"), for a later grant ("(PACTOR-3, the peer's 0x59A grant)") or
# for the link ending, and restores the level it found the rig on afterwards.
TX_POWER_OUTER = re.compile(r'TX power: RFPOWER [\d.]+ -> ([\d.]+) W')
TX_POWER_BASELINE = re.compile(r'RFPOWER [\d.]+ -> ([\d.]+) W at arm start; restored at teardown')
TX_POWER_PHASE = re.compile(r"RFPOWER [\d.]+ -> ([\d.]+) W \(([^)]*)\), handed to rigctl")


def tx_power(text: str) -> dict:
    """The TX power setting a transcript states, in watts: `tx_w` is what was in
    effect when the arm first keyed, and `tx_w_max` is the top of it when a later
    PACTOR phase, or a hand change between the two outer readings, moved it higher.
    None when the transcript never says, rather than guessed."""
    text = ANSI.sub('', text)
    phases = [(float(m[1]), m[2]) for m in TX_POWER_PHASE.finditer(text)]
    before_first = next((w for w, why in phases if 'before the first call' in why), None)
    baseline = (m := TX_POWER_BASELINE.search(text)) and float(m[1])
    outer = [float(w) for w in TX_POWER_OUTER.findall(text)]
    tx_w = before_first if before_first is not None else baseline if baseline is not None else (outer[0] if outer else None)
    if tx_w is None:
        return {'tx_w': None, 'tx_w_max': None}
    # A phase change is read during the arm and counts toward the top; the pre-arm
    # baseline and the outer readings duplicate it and are only trusted when nothing
    # else says more -- a rig moved by hand with no phase data of its own.
    candidates = [tx_w] + [w for w, _ in phases] + ([] if phases or baseline is not None else outer)
    top = max(candidates)
    return {'tx_w': tx_w, 'tx_w_max': top if top > tx_w else None}


# consolelib/vara.py `account` writes these clauses from analyze.py's by_bw groups.
VARA_DECODED = re.compile(r'\bdecoded overs at ((?:L\d+(?:, )?)+)')
VARA_ACKED = re.compile(r'\backnowledgement read for ((?:L\d+(?:, )?)+)')
# kestrel's IRS names the level each of the peer's overs came at, and what we had asked for;
# the over is read once its DATA line follows. A NAKed one says "0 of 1 frame(s) clean" instead.
VARA_PEER_OVER = re.compile(r"\brx \S+'s over at level (\d+) \(asked \d+\)|\brx DATA over #\d+|\brx \S+'s over at level \d+:")
VARA_BW = re.compile(r'^== VARA\b.*\bBW(500|2300|2750)\b|^originating: .*\(BW(500|2300|2750)\)', re.M)

# PACTOR, as shrike logs it. A link starts as PACTOR-1 and an upgrade names the mode it enters.
P_CONNECTED = re.compile(r'\*\* CONNECTED to |peer answered \(CS\d\) -> link at ')
P_MODE = re.compile(r'link upgraded to PACTOR-(\d)|link falls back to PACTOR-(1)')
# A data packet read off the peer. rxfront and onair print a packet event only once its CRC
# checks, and older transcripts cut the line short of saying so; traffic.py's RX line
# also reports failures, so it has to say CRC-VALID.
P_RX = re.compile(r'RX \([^)]*\) P1 (?:DATA|BREAK-IN) (\d+)Bd '
                  r'|RX \([^)]*\) (?:P2 )?SL(\d)[- ](?!.*CHANGEOVER)'
                  r'|\[traffic\] RX P3 SL(\d) DATA .*\[CRC-VALID\]')
P_ENTRY = re.compile(r'the peer answered the entry packet -> SL(\d)')
# traffic.py: a TX data packet, and the peer's control signals that acknowledge it
# (arq.on_rx_cs: CS3, CS4 and CS6 acknowledge; CS1/CS2 as ACK only when they match its counter).
P_TX = re.compile(r'\[traffic\] TX P3 SL(\d) DATA \S+ seq=(\d)')
P_ACK = re.compile(r'\[traffic\] RX P3 CS\d (?:ACK seq=(\d)|SPEED-UP|CYCLE-TOG|BREAK-IN) ')
P_ANSWERED = re.compile(r'\[traffic\] RX P3 CS\d ')
# onair.py keys a PACTOR-1 packet as `P1 pkt#N <baud>Bd`; the counter moves on only
# once the peer acknowledges, so pkt#N+1 means pkt#N's last copy was read.
P1_PKT = re.compile(r'P1 pkt#(\d+) (\d+)Bd')
# PACTOR-4 after a PACTOR-1 connect: what we keyed in it (a probe, an entry chirp, the link's
# own words and shorts), and what onair's P4 receiver registered of the peer's. A word is
# the peer's once confirmed; a body is a packet read whole, and a banner that follows the
# first P4 line came in one.
P4_OURS = re.compile(r'TX\[\d+\] P4 (?:PROBE (\w+)|LINK (\w+ \S+)|SL\d (CHIRP) ENTRY)[^\n]*-- keying')
P4_WORD = re.compile(r'\[p4 link\] word (\S+) confirmed|\[p4 probe\] heard (\S+) in slot')
P4_BODY = re.compile(r'\[p4 link\] (?P<kind>body|normal)\s+(?:(?P<modern>-)\s+)?(?:new|enriched|confirmed|extended) ')
P4_LINE = re.compile(r'\[p4 \w+\]|TX\[\d+\] P4 ')
BANNER = re.compile(r'\bmail: banner: (.+)')

# ARDOP, as our ARDOP modem traces it.
A_CONNECTED = re.compile(r'ARQ CONNECTION ESTABLISHED WITH \S+, SESSION BW = (\d+) HZ|^\s*CONNECTED \S+ @ (\d+) Hz', re.M)
A_REQUESTED = re.compile(r'^== ARDOP\b.*\bBW(\d+)\b|\bTX ConReq(\d+)', re.M)
A_FRAME = r'((?:4FSK|4PSK|8PSK|16QAM)\.\d+\.\d+S?)\.[EO]'
A_RX = re.compile(rf'\bRX {A_FRAME} (?:sess=\S+ )?ok=True')
A_TX = re.compile(rf'\bTX {A_FRAME} ')
A_ACK = re.compile(r'\bRX DATAACK (?:sess=\S+ )?ok=True')
A_BITS = {'4FSK': 2, '4PSK': 2, '8PSK': 3, '16QAM': 4}


def rank(level: str | None) -> float | None:
    """Faster sorts higher, within a protocol."""
    if not level:
        return None
    if m := re.fullmatch(r'L(\d+)', level):
        return int(m[1])
    if m := re.fullmatch(r'P1 (\d+)Bd', level):
        return 10 + int(m[1]) / 100
    if m := re.fullmatch(r'P([234]) SL(\d)', level):
        return 10 * int(m[1]) + int(m[2])
    if level == 'P4':
        return 40
    if m := re.fullmatch(r'(4FSK|4PSK|8PSK|16QAM)\.(\d+)\.(\d+)S?', level):
        # Gross bit rate: one carrier per 200 Hz for PSK and QAM, a single one for FSK.
        carriers = 1 if m[1] == '4FSK' else max(1, int(m[2]) // 200)
        return carriers * int(m[3]) * A_BITS[m[1]]
    return None


def fastest(*levels: str | None) -> str | None:
    return max((lv for lv in levels if rank(lv) is not None), key=rank, default=None)


def vara_levels(line: str | None) -> dict:
    """The verdict line consolelib/vara.py wrote at teardown."""
    def top(pattern):
        m = pattern.search(line or '')
        return m and fastest(*m[1].split(', '))
    return {'decoded': top(VARA_DECODED), 'acknowledged': top(VARA_ACKED)}


def vara_overs(text: str) -> str | None:
    """The fastest over the peer sent that we read, at the level the transcript states it came at."""
    read, due = [], None
    for m in VARA_PEER_OVER.finditer(ANSI.sub('', text)):
        if m[1]:
            due = f'L{m[1]}'
        elif m[0].startswith('rx DATA') and due:
            read.append(due)
            due = None
        else:
            due = None
    return fastest(*read)


def vara_bw(text: str) -> int | None:
    m = VARA_BW.search(text)
    return m and int(m[1] or m[2])


def pactor(text: str) -> dict:
    """The fastest each way, and once the link went on to PACTOR-4, `p4`: how far that got,
    the level the link had opened at, and what each side sent in it."""
    decoded, acked, sent, mode, linked = [], [], None, 1, False
    pkts: dict[int, int] = {}
    ours, peer, banner, opening = [], [], None, None
    for line in ANSI.sub('', text).splitlines():
        if m := P1_PKT.search(line):
            # Our own counter, and the only one: this reads the same before a connect line as after it.
            n = int(m[1])
            if n - 1 in pkts and n not in pkts:
                acked.append(f'P1 {pkts[n - 1]}Bd')
            pkts[n] = int(m[2])
        elif not linked:
            # A connect is the peer reading our 100 Bd call; anything decoded before it was someone else's.
            if P_CONNECTED.search(line):
                linked = True
                acked.append('P1 100Bd')
        elif P4_LINE.search(line) or (opening and BANNER.search(line)):
            opening = opening or fastest(*decoded, *acked)
            if m := P4_OURS.search(line):
                ours.append(f'{m[1].lower()} probe' if m[1] else m[2] or 'entry chirp')
            elif m := P4_WORD.search(line):
                peer.append(f'word {m[1] or m[2]}')
            elif (body := P4_BODY.search(line)) and (
                    re.search(r'\bCRC=valid\b', line) or
                    (body['kind'] == 'body' and not body['modern'] and 'CRC=' not in line)):
                peer.append('body')
                decoded.append('P4')
            elif m := BANNER.search(line):
                banner = banner or m[1].strip()
        elif m := P_MODE.search(line):
            mode = int(m[1] or m[2])
        elif m := P_RX.search(line):
            decoded.append(f'P1 {m[1]}Bd' if m[1] else f'P{mode} SL{m[2] or m[3]}')
        elif m := P_ENTRY.search(line):
            acked.append(f'P{mode} SL{m[1]}')
        elif m := P_TX.search(line):
            sent = m
        elif P_ANSWERED.search(line):
            if sent and (m := P_ACK.search(line)) and m[1] in (None, sent[2]):
                acked.append(f'P{mode} SL{sent[1]}')
            sent = None
    got = {'decoded': fastest(*decoded), 'acknowledged': fastest(*acked)}
    if not (ours or peer):
        return got
    # Two-way needs a packet of the peer's read whole; its words alone answer, they carry nothing.
    reached = ('probed' if not peer else 'heard' if not ours
               else 'exchange' if 'body' in peer else 'answered')
    def tally(sent):
        return [f'{what} ×{n}' if n > 1 else what for what, n in Counter(sent).items()]
    return got | {'p4': {'reached': reached, 'opening': opening,
                         'peer': tally(peer) + [f'banner {banner}'] * bool(banner and 'body' in peer),
                         'ours': tally(ours)}}


def ardop(text: str) -> dict:
    text = ANSI.sub('', text)
    connected = A_CONNECTED.search(text)
    if not connected:
        requested = A_REQUESTED.search(text)
        return {'bw_hz': requested and int(requested[1] or requested[2]), 'decoded': None, 'acknowledged': None}
    decoded, acked, sent = [], [], None
    for line in text[connected.start():].splitlines():
        if m := A_RX.search(line):
            decoded.append(m[1])
        elif m := A_TX.search(line):
            sent = m[1]
        elif sent and A_ACK.search(line):
            acked.append(sent)
            sent = None
    return {'bw_hz': int(connected[1] or connected[2]), 'decoded': fastest(*decoded), 'acknowledged': fastest(*acked)}
