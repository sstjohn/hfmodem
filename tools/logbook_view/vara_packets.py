# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Turn overlapping detector evidence into one selectable physical packet.

True multi-frame overs retain their individual decoded frames. Timing/PTT and
classifier evidence is attached for inspection, not listed as extra packets.
"""


def consolidate(events):
    def evidence(e):
        return {k:e.get(k) for k in ('id','kind','status','origin','detail','t0','t1')}
    def attach(target, source):
        target.setdefault('observations',[]).append(evidence(source))
        target['observations'].extend(source.get('observations',[]))
    def overlaps(a,b):
        return max(0,min(a['t1'],b['t1'])-max(a['t0'],b['t0']))
    # Decision frames outrank an approximate over or a bare classification.
    data=sorted([e for e in events if e.get('frame_type')=='data'],
                key=lambda e:(e['status']!='CRC pass',not bool(e.get('symbols')),not bool(e.get('payloads')),e['t0']))
    chosen=[]
    for e in data:
        duplicate=next((c for c in chosen if c.get('direction')==e.get('direction') and
                        overlaps(c,e)>.65*max(.01,min(c['t1']-c['t0'],e['t1']-e['t0'])) and
                        # Multiple independently timed frames must stay separate.
                        (not(c.get('symbols') and e.get('symbols')) or
                         abs(c['t0']-e['t0'])<.1)),None)
        if duplicate:
            attach(duplicate,e)
        else:
            chosen.append(e)
    ptt=[e for e in events if e.get('evidence_type')=='ptt']
    for e in ptt:
        contained=[c for c in chosen if c.get('direction')=='tx' and
                   overlaps(c,e)>.6*max(.01,c['t1']-c['t0'])]
        if contained:
            for c in contained:
                attach(c,e)
                c['transmit_label']=e['kind']
                c['tx_confirmed']=True
                c['tx_waveform']={k:e[k] for k in ('protocol','level','modulation','symbol_rate','carrier_count','packet_kind') if k in e}
                c['burst_t0'],c['burst_t1']=e['t0'],e['t1']
        else:
            chosen.append(e)
    observations=sorted([e for e in events if e.get('frame_type')!='data' and e.get('evidence_type')!='ptt'],
                        key=lambda e:(e['t0']==e['t1'],e['t0']))
    for e in observations:
        if e.get('timestamp_only'):
            chosen.append(e)
            continue
        mid=(e['t0']+e['t1'])/2
        candidates=[c for c in chosen if
                    (e.get('direction') in (None,'observed') or c.get('direction') in (None,'observed') or e['direction']==c['direction']) and
                    (c.get('burst_t0',c['t0'])-.35<=mid<=c.get('burst_t1',c['t1'])+.15 if e['t0']==e['t1'] else
                     overlaps(c,e)>.6*max(.01,min(c['t1']-c['t0'],e['t1']-e['t0'])))]
        if candidates:
            target=min(candidates,key=lambda c:abs((c['t0']+c['t1'])/2-mid))
            attach(target,e)
        else:
            chosen.append(e)
    return sorted(chosen,key=lambda e:(e['t0'],e['id']))


def packet_bytes(data, layer, **fields):
    """Lossless bytes from a checked decoder result, with their protocol layer."""
    data = bytes(data)
    return {'hex': data.hex(), 'length': len(data), 'layer': layer, **fields}


def vara_payloads(events, caller):
    """Retain checked frame bytes and separately expose their DATA payload.

    Tone/OFDM readers retain CRCs; the base reader has already removed them.
    Link setup carries control fields, not an application DATA block.
    """
    from hfmodem.kestrel.arq.phy import vara_payload

    for event in events:
        setup = 'link-setup' in (event.get('transmit_label') or event.get('kind', ''))
        rows = [event] + [row for view in event.get('views', {}).values()
                          for row in view.get('decoded', [])]
        for row in rows:
            if not (row.get('status') or '').startswith('CRC pass'):
                continue
            for block in row.get('payloads') or []:
                raw = bytes.fromhex(block['hex'])
                base = (row.get('kind') or '').startswith('Base data')
                body = raw if base else raw[:-2]
                block['layer'] = 'VARA decoded body' if base else 'VARA decoded frame · includes CRC'
                block['payload_hex'] = None if setup else vara_payload(
                    body, caller=caller, body_len=len(body)).hex()
                block['payload_length'] = (None if setup else len(block['payload_hex']) // 2)
                block['payload_layer'] = 'VARA link setup fields' if setup else 'VARA DATA payload'
