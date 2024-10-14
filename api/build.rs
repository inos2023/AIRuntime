// extern crate cbindgen;
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/*");

    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let target_dir = get_target_dir(&crate_dir);

    cbindgen::generate(crate_dir)
        .expect("Unable to generate bindings")
        .write_to_file(
            target_dir
                .join("cbindgen")
                .join("airuntime")
                .join("airuntime.h"),
        );
}

fn get_target_dir(crate_dir: &str) -> PathBuf {
    match env::var("CARGO_TARGET_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(crate_dir).join("..").join("target"),
    }
}
