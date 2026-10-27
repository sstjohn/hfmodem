// SPDX-License-Identifier: AGPL-3.0-only
// SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
use std::{env, fs, path::PathBuf};

fn main() {
    let root = env::var("HFMODEM_INTEROP_XARA").expect("setup supplies the reviewed Xara root");
    let module = PathBuf::from(root).join("crates/xara/src/station/mod.rs");
    assert!(module.is_file(), "missing unchanged Xara station module");
    let declaration = format!("#[path = {:?}]\n#[allow(dead_code)]\nmod station;\n", module);
    fs::write(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("station_path.rs"), declaration).unwrap();
    println!("cargo:rerun-if-env-changed=HFMODEM_INTEROP_XARA");
}
