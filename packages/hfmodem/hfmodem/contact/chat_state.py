# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import json
import types
from typing import Literal, Union, get_args, get_origin, get_type_hints


class State(str, Enum):
    CALLING = 'calling-frequency'
    MONITORING = 'monitoring'
    SNIFFING = 'sniffing-slot'
    CQ = 'calling-cq'
    WAITING = 'waiting-on-slot'
    CONNECTING = 'connecting'
    CONNECTED = 'connected'
    RETURNED = 'returned-to-calling-frequency'
    FAILED = 'failed'


class Effect(str, Enum):
    R = 'R'
    L = 'L'
    T = 'T'
    X = 'X'


class ProjectionSection(str, Enum):
    LIVE = 'live'
    APPLICATION = 'application'
    RECEIVE = 'receive'
    CAPABILITIES = 'capabilities'
    REQUESTS = 'requests'


class Phase(str, Enum):
    CLOSING = 'closing'
    HOST_CLOSED = 'host-closed'
    SETTLING = 'settling'
    BROADCAST = 'broadcast'
    BROADCAST_SLOT = 'broadcast-slot'
    BAND_FAULT = 'band-fault'
    EMAIL_COMPOSE = 'email-compose'
    EMAIL = 'email'
    COMPOSING = 'composing'
    QSY = 'qsy'
    BBS_SEND = 'bbs-send'
    FILE_DOWNLOAD = 'file-download'
    FILE_CAPTURE = 'file-capture'
    BBS = 'bbs'
    OPENING = 'opening'
    VMAIL_ACK = 'vmail-ack'
    CQ = 'cq'
    BEACON = 'beacon'
    CONNECTING = 'connecting'
    EXCHANGE = 'exchange'
    READY_CF = 'ready-cf'
    READY_SLOT = 'ready-slot'
    MONITORING = 'monitoring'


@dataclass(frozen=True, slots=True)
class Identity:
    mycall: str
    idcall: str
    band_m: int
    channel_hz: int|None
    calling_hz: int
    slot: int|str|None
    bandwidth_hz: int|None
    band_fault: bool
    setup: tuple[tuple[str,str],...]


@dataclass(frozen=True, slots=True)
class Link:
    generation: int
    state: str
    flow_state: State
    peer: str|None
    peer_source: str|None
    contact_number: int|None
    how: str|None
    started_utc: str|None
    connected: bool
    pending: bool
    host_closed: bool
    last_transition_at: float|None


@dataclass(frozen=True, slots=True)
class Operation:
    phase: Phase
    subphase: str|None
    description: str
    next_action: str
    intent: str|None
    active: bool
    qsy_slot: int|None
    qsy_receiving: bool
    deadline: float|None
    closing: bool
    quitting: bool
    close_error: str|None


@dataclass(frozen=True, slots=True)
class Credit:
    generation: int|None
    acknowledged_bytes: int
    discarded_bytes: int
    attribution: str


@dataclass(frozen=True, slots=True)
class Tx:
    host_ptt: bool
    physical_line: bool|None
    line_observed_at: float|None
    buffer_bytes: int
    buffer_sequence: int
    host_sequence: int
    sent_after: int|None
    writer_pending: bool
    unreported: bool
    pending: bool
    description: str
    stream_poisoned: bool
    credit: tuple[Credit,...]


@dataclass(frozen=True, slots=True)
class Policy:
    auto_requested: bool
    auto_effective: bool
    auto_state: str
    suspension_reason: str|None
    away_text: str|None
    away_owed_generation: int|None
    new_vmails: int
    following: int|str|None
    follow_tuned: bool
    verbose: bool
    reporter_status: tuple[tuple[str,str],...]


@dataclass(frozen=True, slots=True)
class Hop:
    active: bool
    auto: bool
    bands: tuple[int,...]
    minutes: int|None
    due: float|None
    pending_band: int|None
    paused: str|None
    readiness: tuple[tuple[int,str],...]
    last_tx: tuple[tuple[int,float],...]


@dataclass(frozen=True, slots=True)
class Beacon:
    minutes: int|None
    due: float|None
    expires: float|None
    last_tx: float|None
    once: str|None
    once_request_id: str|None
    advanced_available: bool
    generation: int
    pause_reason: str|None
    hop: Hop


@dataclass(frozen=True, slots=True)
class Broadcast:
    selected_hz: int|None
    active: bool
    phase: str|None
    cancelled: bool
    possibly_outstanding: int
    result: str|None


@dataclass(frozen=True, slots=True)
class Draft:
    id: str
    revision: int
    owner: str
    kind: str
    generation: int
    peer: str
    subject: str
    recipient: str|None
    reply_to: str|None
    lines: tuple[str,...]


@dataclass(frozen=True, slots=True)
class EmailState:
    waiting: bool
    phase: str|None
    sent_generation: int|None


@dataclass(frozen=True, slots=True)
class BBSState:
    active: bool
    remote_view: bool
    owned: bool
    sid: str|None
    generation: int|None
    remote_ansi: bool
    dimensions: tuple[int,int]
    requested_term: tuple[str,int,int]|None
    reload_pending: bool
    sending_name: str|None


@dataclass(frozen=True, slots=True)
class FileEntry:
    name: str
    created: str
    size: int


@dataclass(frozen=True, slots=True)
class FileState:
    active: bool
    downloading: bool
    receiving: bool
    opaque: bool
    generation: int|None
    list_generation: int|None
    requested: str|None
    entries: tuple[FileEntry,...]
    offer: str|None
    received_bytes: int
    offer_deadline: float|None
    failure: str|None


@dataclass(frozen=True, slots=True)
class MailHeader:
    number: int
    id: str
    direction: str
    sender: str
    recipient: str
    subject: str
    utc: str


@dataclass(frozen=True, slots=True)
class MailNotice:
    source: str
    destination: str
    utc: str
    dial_hz: int
    received: int
    urgent: bool = False


@dataclass(frozen=True, slots=True)
class MailAckState:
    generation: int|None
    state: str
    kind: str|None
    result: str|None
    reason: str|None


@dataclass(frozen=True, slots=True)
class ParkedMail:
    id: int
    sender: str
    recipient: str
    subject: str
    state: str
    attempts: int
    expires_unix: float
    last_error: str|None
    notification: tuple[tuple[str,float|None],...]


@dataclass(frozen=True, slots=True)
class ParkingState:
    available: bool
    error: str|None
    enabled: bool
    relay: bool
    expiry_days: int
    per_destination: int
    total: int
    active: bool
    started_generation: int|None
    stopped_generation: int|None
    notifying: tuple[str,str,int,int]|None
    candidates: tuple[str,...]
    draft: tuple[str,str,tuple[str,...]]|None
    mail: tuple[ParkedMail,...]


@dataclass(frozen=True, slots=True)
class Application:
    draft: Draft|None
    vmail_ack_pending: bool
    bbs_list_generation: int|None
    bbs_waiting: bool
    email: EmailState
    bbs: BBSState
    files: FileState
    mailbox: tuple[MailHeader,...]
    notices: tuple[MailNotice,...]
    mail_ack: MailAckState|None = None
    parking: ParkingState|None = None


@dataclass(frozen=True, slots=True)
class HeardRow:
    n: int
    call: str
    slot: int|None
    bw: int
    sn: float|None
    at: float
    beacon: bool
    grid: str
    status: str
    via: str
    band: int|None
    frequency_hz: int


@dataclass(frozen=True, slots=True)
class SlotRow:
    slot: int|str|None
    audio_hz: int
    state: str
    heard: float|None
    calls: tuple[str,...]
    level: int|None
    tags: tuple[str,...]
    home: bool
    caller: str
    called: str
    stage: str
    since: float|None
    at: float
    sn: float|None
    one_side: bool


@dataclass(frozen=True, slots=True)
class Coverage:
    slot: int|str|None
    last_note: str
    observed_at: float
    audio_time: float
    lost_seconds_60s: float|None
    share_60s: float|None
    accounting: str


@dataclass(frozen=True, slots=True)
class Receive:
    host_busy: bool|None
    snr: float|None
    bitrate: str|None
    heard: tuple[HeardRow,...]
    slots: tuple[SlotRow,...]
    watched: tuple[int|str|None,...]
    passband_edges: tuple[tuple[str,tuple[float,float]],...]
    wide_offset: float|None
    table_audio_time: float|None
    observed_at: float|None
    coverage: tuple[Coverage,...]
    monitor_error: str|None


@dataclass(frozen=True, slots=True)
class Ownership:
    mode: str
    actor: str|None
    control_id: str|None
    control_generation: int|None
    lease_expires_utc: str|None
    health: str
    accepting: bool


@dataclass(frozen=True, slots=True)
class ServiceReservation:
    family: str
    state: str
    request_id: str|None


@dataclass(frozen=True, slots=True)
class Capability:
    kind: str
    effect: Effect
    availability: str
    reservation: ServiceReservation|None
    reason_code: str|None
    reason: str|None


@dataclass(frozen=True, slots=True)
class RequestState:
    id: str
    kind: str
    submitted_generation: int
    bound_generation: int|None
    attempt_id: str|None
    state: str
    stage: str|None
    reason_code: str|None
    detail: str
    updated_at: float


@dataclass(frozen=True, slots=True)
class StyledSpan:
    text: str
    foreground: int|None
    background: int|None
    bold: bool


@dataclass(frozen=True, slots=True)
class TextRecord:
    id: str
    direction: str
    text: str
    peer: str|None
    source: str


@dataclass(frozen=True, slots=True)
class Notice:
    text: str
    severity: str


@dataclass(frozen=True, slots=True)
class SlotRecord:
    slot: int|str|None
    audio_time: float
    kind: str
    text: str
    station: str|None


@dataclass(frozen=True, slots=True)
class BBSPage:
    page_id: str
    mode: str
    width: int
    height: int
    spans: tuple[StyledSpan,...]


@dataclass(frozen=True, slots=True)
class TxTextOutcome:
    text_id: str
    write_id: int|None
    status: str
    framed_bytes: int
    acknowledged_bytes: int
    discarded_bytes: int
    unknown_bytes: int
    attribution: str


@dataclass(frozen=True, slots=True)
class TxCredit:
    bytes: int
    disposition: Literal['acknowledged','discarded']
    evidence_sequence: int|None


@dataclass(frozen=True, slots=True)
class DisplayRecord:
    id: str
    created_seq: int
    updated_seq: int
    link_generation: int|None
    bbs_context: str|None
    bbs_revision: int|None
    payload: TextRecord|Notice|SlotRecord|BBSPage
    tx: TxTextOutcome|None


@dataclass(frozen=True, slots=True)
class PresentationState:
    records: tuple[DisplayRecord,...]
    evicted_through: int|None
    bbs_context: str|None
    bbs_revision: int
    bbs_complete: bool
    bbs_reconnect_required: bool


@dataclass(frozen=True, slots=True)
class LiveState:
    sampled_at: float
    identity: Identity
    link: Link
    operation: Operation
    tx: Tx
    policy: Policy
    beacon: Beacon
    broadcast: Broadcast
    ownership: Ownership
    bbs_context: str|None
    bbs_revision: int
    bbs_complete: bool
    bbs_reconnect_required: bool


@dataclass(frozen=True, slots=True)
class Snapshot:
    schema: int
    controller_id: str
    revision: int
    through_seq: int
    sampled_at: float
    utc: str
    identity: Identity
    link: Link
    operation: Operation
    tx: Tx
    policy: Policy
    beacon: Beacon
    broadcast: Broadcast
    application: Application
    receive: Receive
    ownership: Ownership
    capabilities: tuple[Capability,...]
    requests: tuple[RequestState,...]
    presentation: PresentationState
    omitted_sections: tuple[str,...]


@dataclass(frozen=True, slots=True)
class Projection:
    controller_id: str
    revision: int
    through_seq: int
    sampled_at: float
    utc: str
    live: LiveState
    application: Application
    receive: Receive
    capabilities: tuple[Capability,...]
    requests: tuple[RequestState,...]
    omitted_sections: tuple[str,...]


@dataclass(frozen=True, slots=True)
class Receipt:
    id: str
    queued: bool
    reason: str|None


@dataclass(frozen=True, slots=True)
class Applied:
    revision: int


@dataclass(frozen=True, slots=True)
class DraftAccepted:
    draft: Draft|None


@dataclass(frozen=True, slots=True)
class WriteAccepted:
    write_id: int|None
    text_id: str


@dataclass(frozen=True, slots=True)
class ConnectionResult:
    attempt_id: str|None
    generation: int|None
    result: Literal['connected','no-answer','busy']


@dataclass(frozen=True, slots=True)
class WaitStopped:
    target_id: str|None
    effect: Literal['no-op','wait-only','link-closed']


@dataclass(frozen=True, slots=True)
class LinkClosed:
    unkey: Literal['low','high','unknown']


@dataclass(frozen=True, slots=True)
class BeaconResult:
    result: Literal['scheduled','sent','not-sent']


@dataclass(frozen=True, slots=True)
class BroadcastResult:
    possibly_outstanding: int
    result: Literal['queued','sent','failed']


@dataclass(frozen=True, slots=True)
class DirectoryResult:
    entries: tuple[FileEntry,...]
    result: Literal['complete','rejected','incomplete']


@dataclass(frozen=True, slots=True)
class FileResult:
    result: Literal['offer-recorded','validated-saved','capture-started','failed']
    name: str|None
    evidence_id: str|None
    received_bytes: int


@dataclass(frozen=True, slots=True)
class MailResult:
    kind: Literal['vmail','email']
    stage: Literal['queued','smr','modem-ack','er','rejected','unconfirmed']


@dataclass(frozen=True, slots=True)
class TermResult:
    mode: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ReloadResult:
    revision: int


@dataclass(frozen=True, slots=True)
class NotFound:
    id: str


@dataclass(frozen=True, slots=True)
class HelpText:
    topic: str|None
    text: str


@dataclass(frozen=True, slots=True)
class HeardRows:
    rows: tuple[HeardRow,...]


@dataclass(frozen=True, slots=True)
class SlotRows:
    rows: tuple[SlotRow,...]


@dataclass(frozen=True, slots=True)
class FileEntries:
    generation: int|None
    entries: tuple[FileEntry,...]


@dataclass(frozen=True, slots=True)
class MailHeaders:
    headers: tuple[MailHeader,...]


@dataclass(frozen=True, slots=True)
class MailBody:
    header: MailHeader
    body: str


@dataclass(frozen=True, slots=True)
class MailNotices:
    notices: tuple[MailNotice,...]
    mail_ack: MailAckState|None = None
    parking: ParkingState|None = None


@dataclass(frozen=True, slots=True)
class ReporterStatus:
    status: tuple[tuple[str,str],...]


@dataclass(frozen=True, slots=True)
class RequestResult:
    state: RequestState
    result: ApplicationResult|ReadValue|None
    text_outcomes: tuple[TxTextOutcome,...]


@dataclass(frozen=True, slots=True)
class CapabilityCheck:
    controller_id: str
    revision: int
    through_seq: int
    action: OperationalAction
    capability: Capability


@dataclass(frozen=True, slots=True)
class StateChanged:
    revision: int
    through_seq: int
    state: LiveState|None
    changed_sections: tuple[ProjectionSection,...]


@dataclass(frozen=True, slots=True)
class RequestChanged:
    request: RequestState


@dataclass(frozen=True, slots=True)
class ReadResult:
    id: str
    value: ReadValue


@dataclass(frozen=True, slots=True)
class ApplicationOutcome:
    result: ApplicationResult


@dataclass(frozen=True, slots=True)
class DisplayChanged:
    record: DisplayRecord


@dataclass(frozen=True, slots=True)
class Gap:
    after_seq: int
    through_seq: int
    reason: str


@dataclass(frozen=True, slots=True)
class Lifecycle:
    state: str
    error: str|None
    unkey: Literal['low','high','unknown']


@dataclass(frozen=True, slots=True)
class ResultAvailable:
    request_id: str


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    seq: int
    controller_id: str
    utc: str
    observed_at: float
    link_generation: int|None
    request_id: str|None
    kind: str
    payload: EventPayload


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    controller_id: str
    link_generation: int
    actor: str
    control_id: str|None
    control_generation: int|None
    action: Action


@dataclass(frozen=True, slots=True)
class ChatSnapshot:
    kind: Literal['chat.snapshot'] = 'chat.snapshot'


@dataclass(frozen=True, slots=True)
class ChatProjection:
    kind: Literal['chat.projection'] = 'chat.projection'


@dataclass(frozen=True, slots=True)
class ChatCheck:
    action: OperationalAction
    kind: Literal['chat.check'] = 'chat.check'


@dataclass(frozen=True, slots=True)
class ChatOutcome:
    request_id: str
    kind: Literal['chat.outcome'] = 'chat.outcome'


@dataclass(frozen=True, slots=True)
class ChatHelp:
    topic: str|None
    kind: Literal['chat.help'] = 'chat.help'


@dataclass(frozen=True, slots=True)
class HeardList:
    kind: Literal['heard.list'] = 'heard.list'


@dataclass(frozen=True, slots=True)
class SlotsList:
    kind: Literal['slots.list'] = 'slots.list'


@dataclass(frozen=True, slots=True)
class FilesList:
    kind: Literal['files.list'] = 'files.list'


@dataclass(frozen=True, slots=True)
class VMailList:
    kind: Literal['vmail.list'] = 'vmail.list'


@dataclass(frozen=True, slots=True)
class VMailRead:
    number: int
    kind: Literal['vmail.read'] = 'vmail.read'


@dataclass(frozen=True, slots=True)
class VMailPending:
    kind: Literal['vmail.pending'] = 'vmail.pending'


@dataclass(frozen=True, slots=True)
class BroadcastStatus:
    kind: Literal['broadcast.status'] = 'broadcast.status'


@dataclass(frozen=True, slots=True)
class BeaconHopStatus:
    kind: Literal['beacon.hop.status'] = 'beacon.hop.status'


@dataclass(frozen=True, slots=True)
class EmailStatus:
    kind: Literal['email.status'] = 'email.status'


@dataclass(frozen=True, slots=True)
class PSKReporterStatus:
    kind: Literal['pskreporter.status'] = 'pskreporter.status'


@dataclass(frozen=True, slots=True)
class ChatSend:
    text: str
    kind: Literal['chat.send'] = 'chat.send'


@dataclass(frozen=True, slots=True)
class ChatCall:
    peer: str
    slot: int|None
    auto_qsy: bool
    kind: Literal['chat.call'] = 'chat.call'


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    heard_number: int
    kind: Literal['chat.answer'] = 'chat.answer'


@dataclass(frozen=True, slots=True)
class ChatCQ:
    slot: int|None
    repeat: bool
    kind: Literal['chat.cq'] = 'chat.cq'


@dataclass(frozen=True, slots=True)
class ChatQSY:
    slot: int
    kind: Literal['chat.qsy'] = 'chat.qsy'


@dataclass(frozen=True, slots=True)
class ChatTag:
    name: str
    value: str|None
    kind: Literal['chat.tag'] = 'chat.tag'


@dataclass(frozen=True, slots=True)
class ChatCancelWait:
    kind: Literal['chat.cancel_wait'] = 'chat.cancel_wait'


@dataclass(frozen=True, slots=True)
class ChatClose:
    text: str='73'
    kind: Literal['chat.close'] = 'chat.close'


@dataclass(frozen=True, slots=True)
class ChatExit:
    kind: Literal['chat.exit'] = 'chat.exit'


@dataclass(frozen=True, slots=True)
class BufferClear:
    kind: Literal['buffer.clear'] = 'buffer.clear'


@dataclass(frozen=True, slots=True)
class FollowSet:
    target: int|str|None
    kind: Literal['follow.set'] = 'follow.set'


@dataclass(frozen=True, slots=True)
class BandSet:
    band_m: int
    kind: Literal['band.set'] = 'band.set'


@dataclass(frozen=True, slots=True)
class BandReady:
    ready: bool
    kind: Literal['band.ready'] = 'band.ready'


@dataclass(frozen=True, slots=True)
class BroadcastSelect:
    slot: int|None
    kind: Literal['broadcast.select'] = 'broadcast.select'


@dataclass(frozen=True, slots=True)
class BroadcastSend:
    text: str
    kind: Literal['broadcast.send'] = 'broadcast.send'


@dataclass(frozen=True, slots=True)
class BroadcastClear:
    kind: Literal['broadcast.clear'] = 'broadcast.clear'


@dataclass(frozen=True, slots=True)
class BroadcastConfirm:
    proceed: bool
    kind: Literal['broadcast.confirm'] = 'broadcast.confirm'


@dataclass(frozen=True, slots=True)
class BroadcastOff:
    kind: Literal['broadcast.off'] = 'broadcast.off'


@dataclass(frozen=True, slots=True)
class BeaconConfigure:
    minutes: int|None
    kind: Literal['beacon.configure'] = 'beacon.configure'


@dataclass(frozen=True, slots=True)
class BeaconOnce:
    force: bool
    kind: Literal['beacon.once'] = 'beacon.once'


@dataclass(frozen=True, slots=True)
class BeaconHop:
    bands: tuple[int,...]
    auto: bool
    minutes: int
    kind: Literal['beacon.hop'] = 'beacon.hop'


@dataclass(frozen=True, slots=True)
class AwaySet:
    text: str
    kind: Literal['away.set'] = 'away.set'


@dataclass(frozen=True, slots=True)
class AutoSet:
    on: bool
    kind: Literal['auto.set'] = 'auto.set'


@dataclass(frozen=True, slots=True)
class AwayBack:
    kind: Literal['away.back'] = 'away.back'


@dataclass(frozen=True, slots=True)
class VerboseSet:
    on: bool
    kind: Literal['verbose.set'] = 'verbose.set'


@dataclass(frozen=True, slots=True)
class PSKReporterOff:
    kind: Literal['pskreporter.off'] = 'pskreporter.off'


@dataclass(frozen=True, slots=True)
class InquireHeard:
    kind: Literal['inquire.heard'] = 'inquire.heard'


@dataclass(frozen=True, slots=True)
class VMailCompose:
    subject: str
    kind: Literal['vmail.compose'] = 'vmail.compose'


@dataclass(frozen=True, slots=True)
class EmailCompose:
    recipient: str
    reply_to: str
    subject: str
    kind: Literal['email.compose'] = 'email.compose'


@dataclass(frozen=True, slots=True)
class DraftAppend:
    draft_id: str
    expected_revision: int
    line: str
    kind: Literal['draft.append'] = 'draft.append'


@dataclass(frozen=True, slots=True)
class DraftDrop:
    draft_id: str
    expected_revision: int
    kind: Literal['draft.drop'] = 'draft.drop'


@dataclass(frozen=True, slots=True)
class VMailSend:
    draft_id: str
    expected_revision: int
    kind: Literal['vmail.send'] = 'vmail.send'


@dataclass(frozen=True, slots=True)
class EmailSend:
    draft_id: str
    expected_revision: int
    kind: Literal['email.send'] = 'email.send'


@dataclass(frozen=True, slots=True)
class VMailCollect:
    peer: str
    kind: Literal['vmail.collect'] = 'vmail.collect'


@dataclass(frozen=True, slots=True)
class VMailDismiss:
    peer: str
    kind: Literal['vmail.dismiss'] = 'vmail.dismiss'


@dataclass(frozen=True, slots=True)
class BBSList:
    kind: Literal['bbs.list'] = 'bbs.list'


@dataclass(frozen=True, slots=True)
class BBSGet:
    selection: str
    kind: Literal['bbs.get'] = 'bbs.get'


@dataclass(frozen=True, slots=True)
class BBSRequest:
    selection: str
    kind: Literal['bbs.request'] = 'bbs.request'


@dataclass(frozen=True, slots=True)
class BBSCapture:
    kind: Literal['bbs.capture'] = 'bbs.capture'


@dataclass(frozen=True, slots=True)
class BBSEnter:
    kind: Literal['bbs.enter'] = 'bbs.enter'


@dataclass(frozen=True, slots=True)
class BBSTerm:
    mode: Literal['plain','ansi16']
    width: int
    height: int
    kind: Literal['bbs.term'] = 'bbs.term'


@dataclass(frozen=True, slots=True)
class BBSInput:
    context_id: str
    expected_revision: int
    text: str
    kind: Literal['bbs.input'] = 'bbs.input'


@dataclass(frozen=True, slots=True)
class BBSReload:
    kind: Literal['bbs.reload'] = 'bbs.reload'


OperationalAction = (ChatSend | ChatCall | ChatAnswer | ChatCQ | ChatQSY | ChatTag | ChatCancelWait | ChatClose | ChatExit | BufferClear | FollowSet | BandSet | BandReady | BroadcastSelect | BroadcastSend | BroadcastClear | BroadcastConfirm | BroadcastOff | BeaconConfigure | BeaconOnce | BeaconHop | AwaySet | AutoSet | AwayBack | VerboseSet | PSKReporterOff | InquireHeard | VMailCompose | EmailCompose | DraftAppend | DraftDrop | VMailSend | EmailSend | VMailCollect | VMailDismiss | BBSList | BBSGet | BBSRequest | BBSCapture | BBSEnter | BBSTerm | BBSInput | BBSReload)
Action = (ChatSnapshot | ChatProjection | ChatCheck | ChatOutcome | ChatHelp | HeardList | SlotsList | FilesList | VMailList | VMailRead | VMailPending | BroadcastStatus | BeaconHopStatus | EmailStatus | PSKReporterStatus | ChatSend | ChatCall | ChatAnswer | ChatCQ | ChatQSY | ChatTag | ChatCancelWait | ChatClose | ChatExit | BufferClear | FollowSet | BandSet | BandReady | BroadcastSelect | BroadcastSend | BroadcastClear | BroadcastConfirm | BroadcastOff | BeaconConfigure | BeaconOnce | BeaconHop | AwaySet | AutoSet | AwayBack | VerboseSet | PSKReporterOff | InquireHeard | VMailCompose | EmailCompose | DraftAppend | DraftDrop | VMailSend | EmailSend | VMailCollect | VMailDismiss | BBSList | BBSGet | BBSRequest | BBSCapture | BBSEnter | BBSTerm | BBSInput | BBSReload)
ApplicationResult = (Applied | DraftAccepted | WriteAccepted | ConnectionResult | WaitStopped | LinkClosed | BeaconResult | BroadcastResult | DirectoryResult | FileResult | MailResult | TermResult | ReloadResult)
ReadValue = (Snapshot | Projection | CapabilityCheck | RequestResult | NotFound | HelpText | HeardRows | SlotRows | FileEntries | MailHeaders | MailBody | MailNotices | Broadcast | Hop | EmailState | ReporterStatus)
EventPayload = (StateChanged | RequestChanged | ReadResult | TextRecord | Notice | SlotRecord | TxTextOutcome | TxCredit | BBSPage | ApplicationOutcome | Gap | Lifecycle | DisplayChanged | ResultAvailable)

ACTIONS = {
    'chat.snapshot': (ChatSnapshot, Effect.R),
    'chat.projection': (ChatProjection, Effect.R),
    'chat.check': (ChatCheck, Effect.R),
    'chat.outcome': (ChatOutcome, Effect.R),
    'chat.help': (ChatHelp, Effect.R),
    'heard.list': (HeardList, Effect.R),
    'slots.list': (SlotsList, Effect.R),
    'files.list': (FilesList, Effect.R),
    'vmail.list': (VMailList, Effect.R),
    'vmail.read': (VMailRead, Effect.R),
    'vmail.pending': (VMailPending, Effect.R),
    'broadcast.status': (BroadcastStatus, Effect.R),
    'beacon.hop.status': (BeaconHopStatus, Effect.R),
    'email.status': (EmailStatus, Effect.R),
    'pskreporter.status': (PSKReporterStatus, Effect.R),
    'chat.send': (ChatSend, Effect.X),
    'chat.call': (ChatCall, Effect.X),
    'chat.answer': (ChatAnswer, Effect.X),
    'chat.cq': (ChatCQ, Effect.X),
    'chat.qsy': (ChatQSY, Effect.X),
    'chat.tag': (ChatTag, Effect.X),
    'chat.cancel_wait': (ChatCancelWait, Effect.X),
    'chat.close': (ChatClose, Effect.X),
    'chat.exit': (ChatExit, Effect.X),
    'buffer.clear': (BufferClear, Effect.X),
    'follow.set': (FollowSet, Effect.X),
    'band.set': (BandSet, Effect.X),
    'band.ready': (BandReady, Effect.X),
    'broadcast.select': (BroadcastSelect, Effect.X),
    'broadcast.send': (BroadcastSend, Effect.X),
    'broadcast.off': (BroadcastOff, Effect.X),
    'broadcast.clear': (BroadcastClear, Effect.L),
    'broadcast.confirm': (BroadcastConfirm, Effect.X),
    'beacon.configure': (BeaconConfigure, Effect.X),
    'beacon.once': (BeaconOnce, Effect.X),
    'beacon.hop': (BeaconHop, Effect.X),
    'away.set': (AwaySet, Effect.X),
    'auto.set': (AutoSet, Effect.X),
    'away.back': (AwayBack, Effect.L),
    'verbose.set': (VerboseSet, Effect.L),
    'pskreporter.off': (PSKReporterOff, Effect.L),
    'inquire.heard': (InquireHeard, Effect.X),
    'vmail.compose': (VMailCompose, Effect.L),
    'email.compose': (EmailCompose, Effect.L),
    'draft.append': (DraftAppend, Effect.L),
    'draft.drop': (DraftDrop, Effect.L),
    'vmail.send': (VMailSend, Effect.X),
    'email.send': (EmailSend, Effect.X),
    'vmail.collect': (VMailCollect, Effect.X),
    'vmail.dismiss': (VMailDismiss, Effect.L),
    'bbs.list': (BBSList, Effect.X),
    'bbs.get': (BBSGet, Effect.X),
    'bbs.request': (BBSRequest, Effect.X),
    'bbs.capture': (BBSCapture, Effect.X),
    'bbs.enter': (BBSEnter, Effect.X),
    'bbs.term': (BBSTerm, Effect.X),
    'bbs.input': (BBSInput, Effect.X),
    'bbs.reload': (BBSReload, Effect.L),
}


@dataclass(frozen=True, slots=True)
class ReadPage:
    read_id: str
    cursor: str
    next_cursor: str|None
    value: ReadValue


RECORD_TYPES = {cls.__name__: cls for cls in tuple(globals().values())
                if isinstance(cls, type) and is_dataclass(cls)}


def effect(action: Action) -> Effect:
    cls, effect = ACTIONS.get(action.kind, (None, None))
    if type(action) is not cls:
        raise ValueError('unknown action')
    return effect


def to_wire(value):
    if is_dataclass(value):
        return {'_type': type(value).__name__,
                **{f.name: to_wire(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [to_wire(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError(f'not an immutable chat value: {type(value).__name__}')


def _convert(value, annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        for candidate in args:
            try:
                return _convert(value, candidate)
            except (ValueError, TypeError):
                pass
        raise ValueError(f'value does not match {annotation}')
    if origin is Literal:
        if not any(type(value) is type(a) and value == a for a in args):
            raise ValueError('invalid literal')
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError('expected array')
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0]) for v in value)
        if len(value) != len(args):
            raise ValueError('wrong tuple length')
        return tuple(_convert(v, a) for v, a in zip(value, args))
    if is_dataclass(annotation):
        if not isinstance(value, dict) or value.get('_type') != annotation.__name__:
            raise ValueError('wrong record type')
        names = {f.name for f in fields(annotation)}
        if set(value) != names | {'_type'}:
            raise ValueError('unknown record shape: missing or extra record fields')
        hints = get_type_hints(annotation)
        return annotation(**{k: _convert(value[k], hints[k]) for k in names})
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation is float and type(value) in (float, int):
        import math
        if not math.isfinite(value):
            raise ValueError('non-finite timestamp or measurement')
        return float(value)
    if type(value) is annotation:
        return value
    raise ValueError(f'expected {annotation}')


def from_wire(value):
    if not isinstance(value, dict) or value.get('_type') not in RECORD_TYPES:
        raise ValueError('unknown record')
    return _convert(value, RECORD_TYPES[value['_type']])


def encode(value) -> bytes:
    return json.dumps(to_wire(value), ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def decode(value: bytes | str):
    return from_wire(json.loads(value))


def live(snapshot: Snapshot) -> LiveState:
    p = snapshot.presentation
    return LiveState(snapshot.sampled_at, snapshot.identity, snapshot.link,
                     snapshot.operation, snapshot.tx, snapshot.policy, snapshot.beacon,
                     snapshot.broadcast, snapshot.ownership, p.bbs_context,
                     p.bbs_revision, p.bbs_complete, p.bbs_reconnect_required)


def project(snapshot: Snapshot) -> Projection:
    return Projection(snapshot.controller_id, snapshot.revision, snapshot.through_seq,
                      snapshot.sampled_at, snapshot.utc, live(snapshot), snapshot.application,
                      snapshot.receive, snapshot.capabilities, snapshot.requests,
                      snapshot.omitted_sections)


def initial_snapshot() -> Snapshot:
    return Snapshot(
        1, '', 0, 0, 0., '',
        Identity('', '', 20, None, 14105000, None, None, False, ()),
        Link(0, 'idle', State.MONITORING, None, None, None, None, None, False, False, False, None),
        Operation(Phase.MONITORING, None, 'monitoring', 'Choose an action', None,
                  False, None, False, None, False, False, None),
        Tx(False, None, None, 0, 0, 0, None, False, False, False,
           'no pending text reported (not proof the peer read it)', False, ()),
        Policy(False, False, 'off', None, None, None, 0, None, False, False,
               (('enabled', 'false'),)),
        Beacon(None, None, None, None, None, None, False, 0, None,
               Hop(False, False, (), None, None, None, None, (), ())),
        Broadcast(None, False, None, False, 0, None),
        Application(None, False, None, False, EmailState(False, None, None),
                    BBSState(False, False, False, None, None, False, (80, 24), None, False, None),
                    FileState(False, False, False, False, None, None, None, (), None, 0, None, None),
                    (), ()),
        Receive(None, None, None, (), (), (), (), None, None, None, (), None),
        Ownership('fake', None, None, None, None, 'ready', True), (), (),
        PresentationState((), None, None, 0, True, False), ())


def preserves_draft(action: Action) -> bool:
    return effect(action) is Effect.R or action.kind in (
        'follow.set', 'auto.set', 'away.set', 'away.back', 'verbose.set',
        'pskreporter.off', 'vmail.dismiss', 'bbs.enter', 'bbs.term', 'bbs.input',
        'bbs.reload', 'vmail.compose', 'email.compose', 'draft.append', 'draft.drop',
        'vmail.send', 'email.send')
