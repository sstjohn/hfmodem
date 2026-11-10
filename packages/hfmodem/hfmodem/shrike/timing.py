# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Persist the stream owner's sample timing policy without reinterpreting hosts.

Delivered file indices and waveform lengths remain exact. Their placement
against PTT, output and the session clock can be estimated or nominal.
Untagged measured-host artifacts keep their existing shape.
"""


def timing_fields(*records):
    """Inherit persisted provenance, with the most specific record last."""
    fields = ('timestamp_limitation', 'driver_offset_provenance',
              'driver_offset_samples', 'timing_estimated', 'timing_basis')
    result = {key: record[key] for record in records for key in fields if key in record}
    if result.get('timestamp_limitation') or result.get('driver_offset_provenance') == 'nominal':
        result['timing_estimated'] = True
        basis = result.get('timing_basis') or 'host capture clock; not measured RF'
        if 'estimated' not in basis:
            basis += '; estimated sample timing'
        if result.get('driver_offset_provenance') == 'nominal' and 'nominal' not in basis:
            basis += '; nominal driver offset'
        result['timing_basis'] = basis
    return result


def owner_timing(owner):
    """Snapshot the selected stream's policy, never infer a host here."""
    limitation = getattr(owner, 'timestamp_limitation', None)
    provenance = getattr(owner, 'lat_provenance', None)
    if not limitation and provenance != 'nominal':
        return {}
    result = {}
    if limitation:
        result['timestamp_limitation'] = limitation
    if provenance is not None:
        result['driver_offset_provenance'] = provenance
    offset = getattr(owner, 'driver_offset_samples', None)
    if offset is not None:
        result['driver_offset_samples'] = offset
    return timing_fields(result)


def timed_record(record, timing, collections=()):
    """Label an artifact and its independently consumed coordinate rows."""
    result = dict(record)
    timing = timing_fields(timing, record)
    result.update(timing)
    for name in collections:
        result[name] = [timed_record(row, timing) for row in result[name]]
    return result
