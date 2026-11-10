# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import sqlite3
import pytest
from hfmodem.bbs.config import BBSConfig, Limits
from hfmodem.bbs.policy import QuotaManager, ConnectionLease, PolicyDenied, StalePolicy
from hfmodem.bbs.store import Store


class Clock:
    def __init__(self, wall=None):
        self.now = 0.0
        self.wall = wall or datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    def advance(self, seconds):
        self.now += seconds
        self.wall += timedelta(seconds=seconds)


@pytest.fixture
def context(tmp_path):
    clock = Clock()
    with Store(tmp_path) as store:
        player = store.player("K1ABC")
        config = BBSConfig(state_root=tmp_path, limits=Limits(door_seconds_per_day=30))
        quota = QuotaManager(store, config, lambda: clock.now, lambda: clock.wall)
        yield clock, store, player, config, quota


def test_connection_late_identity_fixed_snapshot_and_busy():
    lease = ConnectionLease.start(Limits(), 100, selected=False)
    assert lease.select(Limits(connection_seconds=7200), 200)
    assert lease.deadline == 7300
    with pytest.raises(ValueError):
        lease.select(Limits(), 300)
    lease.set_busy(True, 200)
    assert not lease.idle_expired(2000)
    lease.set_busy(False, 2000)
    assert not lease.idle_expired(2799)
    assert lease.idle_expired(2800)
    lease.accept(2800)
    assert lease.deadline == 7300
    assert lease.expired(7300)
    assert lease.remaining(7400) == 0


def test_reservation_current_next_bounds_and_immutable_cutoff(context):
    clock, store, player, config, quota = context
    lease = quota.admit(player.id, player.callsign, "s", "r")
    extended = quota.renew(lease)
    assert lease.deadline == 10
    assert extended.deadline == 20
    assert quota.renew(extended) == extended
    clock.advance(10)
    assert not quota.authorized(lease)
    assert quota.authorized(extended)
    third = quota.renew(extended)
    assert len(third.reservations) == 2
    assert third.deadline == 30
    clock.advance(20)
    assert not quota.authorized(third)
    with pytest.raises(PolicyDenied):
        quota.renew(third)


def test_release_subsecond_refund_idempotence_and_reconnect(context):
    clock, store, player, config, quota = context
    for _ in range(10):
        lease = quota.admit(player.id, player.callsign, "s", "r")
        quota.renew(lease)
        clock.advance(.01)
        quota.release(lease)
        quota.release(lease)
    status = quota.status(player.id, player.callsign)
    assert .1 <= status.used_seconds < .10002
    assert 29.89998 < status.remaining_seconds <= 29.9


def test_crash_leaves_current_and_prepaid_spent(context):
    clock, store, player, config, quota = context
    lease = quota.admit(player.id, player.callsign, "s", "r")
    quota.renew(lease)
    restarted = QuotaManager(store, config, lambda: clock.now, lambda: clock.wall)
    assert restarted.status(player.id, player.callsign).used_seconds == 20
    assert not restarted.authorized(lease)
    restarted.release(lease)
    assert restarted.status(player.id, player.callsign).used_seconds == 20


def test_zero_policy_and_reload_fence(context):
    clock, store, player, config, quota = context
    lease = quota.admit(player.id, player.callsign, "s", "r")
    banned = replace(config, callsigns={"K1ABC": {"door_seconds_per_day": 0}})
    quota.reload(banned)
    assert not quota.authorized(lease)
    with pytest.raises(StalePolicy):
        quota.renew(lease)
    quota.release(lease)
    with pytest.raises(PolicyDenied):
        quota.admit(player.id, "K1ABC-2", "s2", "r2")


def test_failed_ledger_write_never_authorizes_memory_only(context, monkeypatch):
    clock, store, player, config, quota = context
    def failure(*args):
        raise sqlite3.OperationalError("disk full")
    monkeypatch.setattr(quota, "_reserve", failure)
    with pytest.raises(sqlite3.OperationalError):
        quota.admit(player.id, player.callsign, "s", "r")
    assert not quota._active
    assert quota.status(player.id, player.callsign).used_seconds == 0


def test_expired_interval_cannot_be_renewed_late(context):
    clock, store, player, config, quota = context
    lease = quota.admit(player.id, player.callsign, "s", "r")
    clock.advance(10.1)
    with pytest.raises(PolicyDenied):
        quota.renew(lease)
    assert not quota.authorized(lease)


def test_midnight_splits_prepaid_intervals_and_new_day(context):
    clock, store, player, config, quota = context
    clock.wall = datetime(2026, 1, 1, 23, 59, 55, tzinfo=timezone.utc)
    lease = quota.admit(player.id, player.callsign, "s", "r")
    assert lease.deadline == 5
    lease = quota.renew(lease)
    assert lease.deadline == 15
    assert lease.reservations[0].day != lease.reservations[1].day
    clock.advance(6)
    status = quota.status(player.id, player.callsign)
    assert status.used_seconds == 1
    quota.release(lease)
    rows = list(store.db.execute("SELECT day,charged_us FROM usage ORDER BY day"))
    assert [row[1] for row in rows] == [5_000_000, 1_000_000]


def test_system_sleep_keeps_doors_open(context):
    clock, store, player, config, quota = context
    lease = quota.admit(player.id, player.callsign, "s", "r")
    clock.advance(4)
    quota.release(lease)
    clock.wall += timedelta(hours=8)
    lease = quota.admit(player.id, player.callsign, "s", "r2")
    assert quota.authorized(lease)
    assert lease.reservations[0].day == "UTC:2026-01-01"


@pytest.mark.parametrize("mono", [0.0, 50.0, 1e6])
def test_restart_after_overnight_gap_opens_a_fresh_day(context, mono):
    clock, store, player, config, quota = context
    clock.now = 1000.0
    quota.release(quota.admit(player.id, player.callsign, "s", "r"))
    clock.now, clock.wall = mono, clock.wall + timedelta(days=1)
    restarted = QuotaManager(store, config, lambda: clock.now, lambda: clock.wall)
    lease = restarted.admit(player.id, player.callsign, "s", "r")
    assert restarted.authorized(lease)
    assert lease.reservations[0].day == "UTC:2026-01-02"


def test_upgrade_clears_a_tripped_clock_flag(context):
    clock, store, player, config, quota = context
    with store.transaction() as db:
        db.executemany("INSERT INTO metadata VALUES(?,?)", [("clock_suspended", "1"), ("clock_checkpoint", "[9e9, \"2030-01-01T00:00:00+00:00\"]"), ("wall_highwater", "2030-01-01T00:00:00+00:00")])
    restarted = QuotaManager(store, config, lambda: clock.now, lambda: clock.wall)
    assert restarted.authorized(restarted.admit(player.id, player.callsign, "s", "r"))
    keys = {row[0] for row in store.db.execute("SELECT key FROM metadata")}
    assert not keys & {"clock_suspended", "clock_checkpoint", "wall_highwater"}


def test_wall_reversal_cannot_refill_a_spent_day(context):
    clock, store, player, config, quota = context
    clock.wall = datetime(2026, 1, 1, 23, 59, 0, tzinfo=timezone.utc)
    lease = quota.admit(player.id, player.callsign, "s", "r")
    clock.advance(7)
    quota.release(lease)
    clock.wall += timedelta(minutes=2)
    assert quota.status(player.id, player.callsign).used_seconds == 0
    clock.wall -= timedelta(minutes=2)
    assert quota.status(player.id, player.callsign).used_seconds == 7


def test_timezone_change_refused(context):
    clock, store, player, config, quota = context
    changed = replace(config, quota_timezone="America/Chicago")
    with pytest.raises(ValueError, match="timezone"):
        quota.reload(changed)
    with pytest.raises(ValueError, match="timezone"):
        QuotaManager(store, changed)


def test_alias_merge_adds_usage_and_reservation_ids(context):
    clock, store, player, config, quota = context
    second = store.player("W9SSJ")
    first_lease = quota.admit(player.id, player.callsign, "s1", "r1")
    clock.advance(2.5)
    quota.release(first_lease)
    second_lease = quota.admit(second.id, second.callsign, "s2", "r2")
    with pytest.raises(PolicyDenied):
        quota.merge_players(player.id, second.id)
    clock.advance(3.25)
    quota.release(second_lease)
    ids = {row[0] for row in store.db.execute("SELECT id FROM reservations")}
    quota.merge_players(player.id, second.id)
    assert quota.status(player.id, player.callsign).used_seconds == 5.75
    assert {row[0] for row in store.db.execute("SELECT id FROM reservations")} == ids
    assert store.player("W9SSJ").id == player.id


def test_dst_reset_uses_calendar_midnight(context):
    clock, store, player, config, quota = context
    # Fresh timezone configuration is tested in a separate store below.
    from zoneinfo import ZoneInfo
    local = datetime(2026, 3, 8, 0, tzinfo=ZoneInfo("America/Chicago"))
    quota._policy = (quota.revision, replace(config, quota_timezone="America/Chicago"))
    day, reset = quota._day(local.astimezone(timezone.utc))
    assert (reset.astimezone(timezone.utc) - local.astimezone(timezone.utc)).total_seconds() == 23 * 3600
    local = datetime(2026, 11, 1, 0, tzinfo=ZoneInfo("America/Chicago"))
    _, reset = quota._day(local.astimezone(timezone.utc))
    assert (reset.astimezone(timezone.utc) - local.astimezone(timezone.utc)).total_seconds() == 25 * 3600


def test_policy_reload_during_admission_refunds_unused_reservation(context, monkeypatch):
    clock, store, player, config, quota = context
    reserve = quota._reserve
    def race(*args):
        reservation = reserve(*args)
        clock.advance(2)
        quota.reload(replace(config, limits=Limits(door_seconds_per_day=0)))
        return reservation
    monkeypatch.setattr(quota, "_reserve", race)
    with pytest.raises(StalePolicy):
        quota.admit(player.id, player.callsign, "s", "r")
    assert not quota._active
    assert quota.status(player.id, player.callsign).used_seconds == 0


def test_renewal_completing_after_paid_deadline_revokes_and_refunds_next(context, monkeypatch):
    clock, store, player, config, quota = context
    lease = quota.admit(player.id, player.callsign, "s", "r")
    clock.advance(9)
    reserve = quota._reserve
    def slow(*args):
        reservation = reserve(*args)
        clock.advance(2)
        return reservation
    monkeypatch.setattr(quota, "_reserve", slow)
    with pytest.raises(PolicyDenied, match="previous paid deadline"):
        quota.renew(lease)
    assert not quota.authorized(lease)
    assert quota.status(player.id, player.callsign).used_seconds == 10


def test_crash_alias_merge_adds_outstanding_reservations(context):
    clock, store, player, config, quota = context
    other = store.player("W9SSJ")
    a = quota.admit(player.id, player.callsign, "a", "a")
    quota.renew(a)
    quota.admit(other.id, other.callsign, "b", "b")
    restarted = QuotaManager(store, config, lambda: clock.now, lambda: clock.wall)
    restarted.merge_players(player.id, other.id)
    assert restarted.status(player.id, player.callsign).used_seconds == 30
    with pytest.raises(PolicyDenied):
        restarted.admit(player.id, player.callsign, "c", "c")


def test_explicit_offline_timezone_transition_carries_spent(context):
    clock, store, player, config, quota = context
    lease = quota.admit(player.id, player.callsign, "s", "r")
    from hfmodem.bbs.store import OwnershipError
    with pytest.raises(OwnershipError):
        store.transition_quota_timezone("America/Chicago", clock.wall)
    clock.advance(3)
    quota.release(lease)
    store.transition_quota_timezone("America/Chicago", clock.wall)
    store.transition_quota_timezone("America/Chicago", clock.wall)
    changed = replace(config, quota_timezone="America/Chicago")
    restarted = QuotaManager(store, changed, lambda: clock.now, lambda: clock.wall)
    assert restarted.status(player.id, player.callsign).used_seconds == 3
    assert restarted.status(player.id, player.callsign).remaining_seconds == 27


def test_release_charges_owner_stop_time_not_delayed_worker_cleanup(context):
    clock, store, player, config, quota = context
    lease = quota.admit(player.id, player.callsign, "s", "r")
    quota.renew(lease)
    clock.advance(2.5)
    stopped_at = clock.now
    clock.advance(8)
    quota.release(lease, stopped_at=stopped_at)
    assert quota.status(player.id, player.callsign).used_seconds == 2.5


def test_unused_completed_admission_can_be_refunded_after_cancellation(context):
    clock, store, player, config, quota = context
    lease = quota.admit(player.id, player.callsign, "s", "r")
    clock.advance(2)
    quota.release(lease, stopped_at=lease.start)
    assert quota.status(player.id, player.callsign).used_seconds == 0



def test_notice_busy_cycles_preserve_accumulated_caller_idle():
    lease = ConnectionLease.start(Limits(idle_seconds=10, connection_seconds=100), 0)
    # An irrelevant submission creates a rejection notice, not accepted input.
    lease.set_busy(True, 8)
    lease.set_busy(True, 9)  # repeated owner ticks cannot restart the pause
    lease.set_busy(False, 10)
    assert lease.last_activity == 2
    assert not lease.idle_expired(11)
    lease.set_busy(True, 11)
    lease.set_busy(False, 12)
    lease.set_busy(False, 12.5)  # settled ticks cannot grant more idle time
    assert lease.last_activity == 3
    assert not lease.idle_expired(12.999)
    assert lease.idle_expired(13)
    assert lease.deadline == 100


def test_accepted_input_during_busy_excludes_only_remaining_busy_time():
    lease = ConnectionLease.start(Limits(idle_seconds=10, connection_seconds=100), 0)
    lease.set_busy(True, 8)
    lease.accept(9)
    lease.set_busy(True, 11)
    lease.set_busy(False, 12)
    assert lease.last_activity == 12
    assert not lease.idle_expired(21.999)
    assert lease.idle_expired(22)
    lease.accept(21)
    assert not lease.idle_expired(30.999)
    assert lease.idle_expired(31)
    assert lease.deadline == 100


def test_busy_does_not_extend_hard_deadline_or_accept_expired_input():
    lease = ConnectionLease.start(Limits(idle_seconds=10, connection_seconds=20), 0)
    lease.set_busy(True, 9)
    lease.accept(20)
    assert lease.last_activity == 0
    assert lease.busy_started == 9
    assert lease.expired(20)
    assert lease.remaining(20) == 0
