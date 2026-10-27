// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
// Compile upstream's unchanged sans-I/O station, excluding Linux device modules.
include!(concat!(env!("OUT_DIR"), "/station_path.rs"));

use std::io::{self, BufRead, Read, Write};
use xara_phy::hf::level::{Bandwidth, Record};
use xara_phy::hf::rx::receiver::{Event, Receiver};
use xara_phy::hf::tx::Modulator;
use xara_testkit::hex;
use station::{Modem, Options, Report};
use xara_arq::host::HostEvent;

fn read_samples(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).unwrap();
    assert_eq!(b.len() % 8, 0);
    b.chunks_exact(8).map(|x| f64::from_le_bytes(x.try_into().unwrap())).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a[1] == "inventory" {
        for bw in Bandwidth::ALL {
            for r in bw.levels() {
                let l = bw.level(r);
                println!("{},{},{},{:.1},{:?}", bw.hz(), r, l.frame_bytes(), l.net_bps(), l.waveform);
            }
        }
        return;
    }
    let bw: Bandwidth = a[2].parse().unwrap();
    if a[1] == "station" {
        serve(bw, &a[3], &a[4], a[5] == "true", &a[6]);
        return;
    }
    let rec = Record::new(a[3].parse().unwrap()).unwrap();
    assert!(rec <= bw.top_level());
    match a[1].as_str() {
        "tx" => {
            let frames: Vec<Vec<u8>> = a[4].split(',').map(|s| hex::decode(s).unwrap()).collect();
            for f in &frames { assert_eq!(f.len(), bw.level(rec).frame_bytes() as usize); }
            let refs: Vec<&[u8]> = frames.iter().map(Vec::as_slice).collect();
            let m = Modulator::new(bw);
            let mut rnd = xara_phy::rnd::VbRnd::new();
            let x = m.transmission(rec, &refs, &mut |n| rnd.draw(n));
            let bytes: Vec<u8> = x.iter().flat_map(|v| (v * m.gain(rec, -5.0)).to_le_bytes()).collect();
            std::fs::write(&a[5], bytes).unwrap();
        }
        "rx" => {
            let nframes = a[5].parse().unwrap();
            let mut rx = Receiver::new(bw, &[rec], nframes, None);
            let x = read_samples(&a[4]);
            let mut events: Vec<Event> = x.chunks(480).flat_map(|b| rx.feed(b)).collect();
            events.extend(rx.flush());
            for e in events {
                if let Event::Frame(f) = e {
                    if f.frame.ok { println!("{}", hex::encode(&f.frame.bytes)); }
                }
            }
        }
        _ => panic!("unknown operation"),
    }
}

// Line command followed by binary f64le PCM for STEP. Every reply is u32le
// byte length + UTF-8 event text, followed by equal-length PCM for STEP.
// No sockets, wall-clock timing, audio devices, PTT, or radio access.
fn serve(bw: Bandwidth, call: &str, remote: &str, registered: bool, comp: &str) {
    let mut cfg = xara_arq::hf::session::Config::new(bw, call);
    cfg.host.registered = registered;
    cfg.host.compression = match comp {
        "OFF" => xara_arq::compression::Compression::Off,
        "TEXT" => xara_arq::compression::Compression::Text,
        "FILES" => xara_arq::compression::Compression::Files,
        _ => panic!("bad compression"),
    };
    let mut s = station::hf::Station::new(cfg, Options::default());
    let mut input = io::stdin().lock();
    let mut out = io::stdout().lock();
    loop {
        let mut line = String::new();
        if input.read_line(&mut line).unwrap() == 0 { break; }
        let mut p = line.split_whitespace();
        let mut pcm = Vec::new();
        match p.next().unwrap() {
            "STEP" => {
                let n: usize = p.next().unwrap().parse().unwrap();
                assert!(n > 0 && n <= 4800);
                let mut b = vec![0; n * 8];
                input.read_exact(&mut b).unwrap();
                let x: Vec<f64> = b.chunks_exact(8).map(|v| f64::from_le_bytes(v.try_into().unwrap())).collect();
                let mut y = vec![0.0; n];
                s.process(&x, &mut y);
                pcm = y.iter().flat_map(|v| v.to_le_bytes()).collect();
            }
            "CONNECT" => { assert!(s.connect(call, remote, &[])); }
            "SEND" => { s.send(&hex::decode(p.next().unwrap()).unwrap()); }
            "DISCONNECT" => s.disconnect(),
            _ => panic!("bad command"),
        }
        let mut events = format!("STATE:{:?}\n", s.state());
        while let Some(e) = s.pop() {
            events += &match e {
                Report::Host(HostEvent::Data(d)) => format!("DATA:{}\n", hex::encode(&d)),
                other => format!("{other:?}\n"),
            };
        }
        out.write_all(&(events.len() as u32).to_le_bytes()).unwrap();
        out.write_all(events.as_bytes()).unwrap();
        out.write_all(&pcm).unwrap();
        out.flush().unwrap();
    }
}
