"""Export display labels from the same waveform rows used by the modems."""
import json
import sys
from pathlib import Path

TREE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TREE/'packages/hfmodem'))


def catalog():
    from hfmodem.kestrel.rx import varatone, varaofdm
    from hfmodem.shrike import placement, pactor2
    result={}
    def add(protocol,bw,level,family,modulation,carriers,rate):
        result[f'{protocol}:{bw}:{level}']={
            'protocol':protocol,'bandwidth_hz':bw,'level':level,'family':family,
            'modulation':modulation,'carrier_count':carriers,'symbol_rate':rate}
    for (bw,level),row in varatone.ROWS.items():
        add('VARA',bw,level,'Multicarrier tones','DQPSK',row.nb,varatone.FS/varatone.H)
    for (bw,level),row in varaofdm.ROWS.items():
        mod=f'{2**row.bpc}-QAM' if row.qam else {2:'DQPSK',3:'8-DPSK'}[row.bpc]
        add('VARA',bw,level,'OFDM',mod,row.carriers,varaofdm.FS/varaofdm.SYM)
    for sl,row in placement.SPEED_PATHS.items():
        add('PACTOR-3',0,sl,'Multicarrier','DBPSK' if row.bits_per_cell==1 else 'DQPSK',len(row.tones),100)
    for index,row in enumerate(pactor2.PATHS):
        add('PACTOR-2',0,index+1,'Two-carrier', {1:'DBPSK',2:'DQPSK',3:'8-DPSK',4:'16-DPSK'}[row.bits_per_cell],2,100)
    return result


if __name__=='__main__':
    Path(__file__).with_name('waveforms.json').write_text(json.dumps(catalog(),sort_keys=True,indent=2)+'\n')
