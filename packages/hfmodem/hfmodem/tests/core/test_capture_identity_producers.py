import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import numpy as np
import pytest
from hfmodem.core import capture_identity as ci, capture_producer as producer

ROOT=Path(__file__).resolve().parents[5]
def tool(name):
 spec=importlib.util.spec_from_file_location('capture_test_'+name, ROOT/'tools'/ (name+'.py'))
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module

def arm(tmp_path):
 return ci.allocate_arm(tmp_path, native_root=tmp_path/'native'/'arm')

def delegated(c,role):
 root=c.root(c.record['writers'][role]['root_role'])
 return dict(capture_id=c.capture_id,capture_root=str(root),capture_role=role)

@pytest.mark.parametrize('role',['tap','launcher-witness','console-witness'])
def test_recorder_role_is_exclusive_and_paths_are_prebound(tmp_path,role):
 c=arm(tmp_path);kw=delegated(c,role);path=Path(kw['capture_root'])/'record.wav'
 bundle=producer.prepare(path,roles=(role,),**kw)
 assert bundle.audio.path==path and bundle.container.path==path.with_suffix('.flac')
 assert not path.exists()
 with pytest.raises((ci.IdentityError,FileExistsError)):
  producer.prepare(path,roles=(role,),**kw)

@pytest.mark.parametrize('missing',['capture_id','capture_root','capture_role'])
def test_partial_recorder_delegation_refuses(tmp_path,missing):
 c=arm(tmp_path);kw=delegated(c,'tap');path=Path(kw['capture_root'])/'tap.wav';kw[missing]=None
 with pytest.raises(ci.IdentityError):producer.prepare(path,roles=('tap',),**kw)

@pytest.mark.parametrize('occupied',['.wav','.flac','.json'])
def test_foreign_family_member_refuses_before_claim(tmp_path,occupied):
 c=arm(tmp_path);kw=delegated(c,'tap');path=Path(kw['capture_root'])/'tap.wav';foreign=path.with_suffix(occupied);foreign.write_bytes(b'synthetic-foreign')
 with pytest.raises(ci.IdentityError):producer.prepare(path,roles=('tap',),**kw)
 assert foreign.read_bytes()==b'synthetic-foreign'
 assert not (path.parent/'.capture-writers'/'tap.json').exists()

@pytest.mark.parametrize('fault',['wrong_identity','wrong_role','outside','symlink','ended'])
def test_invalid_recorder_binding_refuses(tmp_path,fault):
 c=arm(tmp_path);kw=delegated(c,'tap');path=Path(kw['capture_root'])/'tap.wav'
 if fault=='wrong_identity':kw['capture_id']='00000000-0000-0000-0000-000000000001'
 elif fault=='wrong_role':kw['capture_role']='modem'
 elif fault=='outside':path=tmp_path/'outside.wav'
 elif fault=='symlink':
  target=tmp_path/'elsewhere';target.mkdir();(path.parent/'sub').symlink_to(target,target_is_directory=True);path=path.parent/'sub'/'tap.wav'
 else:c.end()
 with pytest.raises((ci.IdentityError,FileExistsError)):producer.prepare(path,roles=('tap',),**kw)
 assert not path.exists()

def test_tap_cli_claims_before_record_and_refuses_second_writer(tmp_path,monkeypatch,tools):
 module=tool('txwitness');c=arm(tmp_path);kw=delegated(c,'tap');path=Path(kw['capture_root'])/'tap.wav';calls=[]
 def record(seconds,where,device,*,ownership):
  assert ownership.audio.path==path
  assert (path.parent/'.capture-writers'/'tap.json').is_file()
  calls.append((seconds,where,device));return 0
 monkeypatch.setattr(module,'record',record)
 monkeypatch.setattr(sys,'argv',['txwitness','record','1',str(path),'--device','synthetic','--capture-id',kw['capture_id'],'--capture-root',kw['capture_root'],'--capture-role','tap'])
 assert module.main()==0
 with pytest.raises((ci.IdentityError,FileExistsError)):module.main()
 assert len(calls)==1

@pytest.mark.parametrize('role',['launcher-witness','console-witness'])
def test_witness_claim_precedes_candidate_access(tmp_path,monkeypatch,role,tools):
 module=tool('kiwi_witness');c=arm(tmp_path);kw=delegated(c,role);path=Path(kw['capture_root'])/'witness.wav';calls=[]
 def candidates(*a):
  assert (path.parent/'.capture-writers'/(role+'.json')).is_file();calls.append(1);return []
 monkeypatch.setattr(module,'candidates',candidates)
 inventory=tmp_path/'candidates.csv';inventory.write_text('hostname,port,lat_lon,free_channels,name_location\n')
 args=['--grid','EM12jt','--candidates',str(inventory),'--dial','7100','--out',str(path),'--capture-id',kw['capture_id'],'--capture-root',kw['capture_root'],'--capture-role',role]
 assert module.main(args,client=SimpleNamespace())==1
 assert calls==[1]
 with pytest.raises((ci.IdentityError,FileExistsError)):module.main(args,client=SimpleNamespace())
 assert calls==[1]

@pytest.mark.parametrize('write_fault',[False,True])
def test_witness_owned_waveform_matches_legacy_and_has_honest_versions(tmp_path,monkeypatch,write_fault,tools):
 module=tool('kiwi_witness');c=arm(tmp_path);kw=delegated(c,'launcher-witness');path=Path(kw['capture_root'])/'witness.wav'
 got=SimpleNamespace(host='synthetic',port=0,name='synthetic',mi_us=200,mi_peer=None,mi_mid=None)
 x=np.asarray([.01,-.02,.03,-.04],np.float32)
 closes=[]
 def acquire(*a):return (SimpleNamespace(sample_rate=48000,close=lambda:closes.append(1)),got,[x],time.time(),[])
 monkeypatch.setattr(module,'candidates',lambda *a:[got]);monkeypatch.setattr(module,'acquire',acquire);monkeypatch.setattr(module,'placement',lambda *a:'synthetic placement')
 client=SimpleNamespace()
 inventory=tmp_path/'candidates.csv';inventory.write_text('hostname,port,lat_lon,free_channels,name_location\n')
 legacy=tmp_path/'legacy.wav';assert module.main(['--grid','EM12jt','--candidates',str(inventory),'--dial','7100','--out',str(legacy)],client=client)==0
 if write_fault:
  def refuse(*a):raise OSError('synthetic publication failure')
  monkeypatch.setattr(ci.Artifact,'publish_file',refuse)
 inventory=tmp_path/'candidates.csv';inventory.write_text('hostname,port,lat_lon,free_channels,name_location\n')
 args=['--grid','EM12jt','--candidates',str(inventory),'--dial','7100','--out',str(path),'--capture-id',kw['capture_id'],'--capture-root',kw['capture_root'],'--capture-role','launcher-witness']
 assert module.main(args,client=client)==int(write_fault)
 import json,hashlib
 facts=json.loads(path.with_suffix('.json').read_text())
 assert facts['capture_id']==c.capture_id
 assert facts['capture_state']==('partial' if write_fault else 'finalized')
 assert facts['complete']==(not write_fault)
 assert facts['metadata_version']==1
 assert len(closes)==2
 if write_fault:
  assert not path.exists();assert list(path.parent.glob('.witness.wav.*.partial'))
 else:
  assert path.read_bytes()==legacy.read_bytes()
  assert facts['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
  assert ci.validate_artifact(path)['status']=='bound'
  old=json.loads(legacy.with_suffix('.json').read_text())
  for key in ['seconds','sample_rate','peak','normalised_to','rms']:assert facts[key]==old[key]


@pytest.mark.parametrize('raises',[False,True])
def test_producer_restores_callers_signal_handlers_on_exit(raises):
    import signal
    from hfmodem.core.capture_producer import restore_signals
    before={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM)}
    @restore_signals
    def producer():
        for sig in before:signal.signal(sig,lambda *a:None)
        if raises:raise RuntimeError('synthetic producer refusal')
        return 7
    if raises:
        with pytest.raises(RuntimeError):producer()
    else:assert producer()==7
    assert {s:signal.getsignal(s) for s in before}==before

@pytest.mark.parametrize('end',['clean','failure','unfinished'])
def test_owned_producer_lifecycle_requires_explicit_completion(tmp_path,end):
    from hfmodem.core import capture_producer
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    owner=ci.Recording.reserve(writer,'synthetic.wav',clock_id='synthetic')
    def execute():
        with capture_producer.Lifecycle(owner) as lifecycle:
            if end=='clean':
                from hfmodem.core import wav
                import numpy as np
                wav.write(owner.audio.path,np.zeros(2),48000,ownership=owner)
                lifecycle.complete(written=1)
            elif end=='failure':raise RuntimeError('synthetic producer refusal')
    if end=='failure':
        with pytest.raises(RuntimeError):execute()
    else:execute()
    capture.end();row=capture.lifecycle()['writers']['modem']
    assert row['state']==('finalized' if end=='clean' else 'partial')
    assert capture.lifecycle()['capture_complete']==(end=='clean')
    assert row['written']==(1 if end=='clean' else 0)


@pytest.mark.parametrize('route',['clean','stream-failure','graceful-signal'])
def test_actual_tap_record_fake_stream_preserves_bytes_and_lifetime(tmp_path,monkeypatch,route,tools):
    import importlib.util,sys,signal,wave
    import numpy as np
    from hfmodem.core import devices,wav
    name='capture_actual_tap_pipeline'
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools/txwitness.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    capture=ci.allocate_default(tmp_path);writer=capture.claim('modem')
    owner=ci.Recording.reserve(writer,'tap.wav',clock_id='card')
    sound=SimpleNamespace(InputStream=None,query_devices=None)
    monkeypatch.setitem(sys.modules,'sounddevice',sound)
    samples=np.array([.25,-.5,.125,0],dtype=np.float32)
    callbacks=[]
    class Stream:
        def __init__(self,**kwargs):self.callback=kwargs['callback'];callbacks.append(kwargs)
        def __enter__(self):
            self.callback(samples[:,None],len(samples),None,SimpleNamespace(input_overflow=False))
            if route=='stream-failure':raise RuntimeError('synthetic stream refusal')
            if route=='graceful-signal':raise SystemExit(128+signal.SIGTERM)
            return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(sound,'InputStream',Stream)
    monkeypatch.setattr(sound,'query_devices',lambda:[{'name':'synthetic input'}])
    monkeypatch.setattr(devices,'find_device',lambda *args,**kwargs:0)
    monkeypatch.setattr(module.time,'sleep',lambda *args:None)
    monkeypatch.setattr(wav,'store',lambda *args,**kwargs:{'test_container':'wav'})
    prior={s:signal.getsignal(s) for s in [signal.SIGINT,signal.SIGTERM]}
    if route=='stream-failure':
        with pytest.raises(RuntimeError):module.record(len(samples)/module.FS,str(owner.audio.path),'synthetic',ownership=owner)
    elif route=='graceful-signal':
        with pytest.raises(SystemExit) as got:module.record(len(samples)/module.FS,str(owner.audio.path),'synthetic',ownership=owner)
        assert got.value.code==128+signal.SIGTERM
    else:assert module.record(len(samples)/module.FS,str(owner.audio.path),'synthetic',ownership=owner)==0
    assert {s:signal.getsignal(s) for s in prior}==prior
    capture.end();row=capture.lifecycle()['writers']['modem']
    assert row['written']==1 and row['pending']==0
    assert row['state']==('partial' if route=='stream-failure' else 'finalized')
    assert capture.lifecycle()['capture_complete']==(route!='stream-failure')
    with wave.open(str(owner.audio.path)) as audio:
        assert audio.getnframes()==len(samples)
        assert audio.readframes(len(samples))==np.round(samples*32768).astype('<i2').tobytes()
    assert len(callbacks)==1
    assert ci.validate_artifact(owner.audio.path,clock_id='card')['status']=='bound'
