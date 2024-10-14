use std::env;

use cmake::Config;

#[allow(unused_must_use)]
fn main() {
    // 从rust代码生成C++代码
    cxx_build::bridge("src/nndevice/ffi.rs");

    // 使用cmake编译
    let dst = Config::new("cxx")
    .define("FROM_CARGO", "ON")
    .define("CARGO_BUILD_TARGET", get_env_value("CARGO_BUILD_TARGET"))
    .build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=cxxbridge-cxx");
    println!("cargo:rustc-link-lib=ai_chip_client");
    println!("cargo:rustc-link-lib=ai_chip_base");
    println!("cargo:rustc-link-lib=glogd");

    println!("cargo:rerun-if-changed=src/nndevice/ffi.rs");
    println!("cargo:rerun-if-changed=cxx/*");
}

fn get_env_value(key: &str) -> String {
    match env::var(key) {
        Ok(d) => d,
        Err(_) => panic!("No {} value", key),
    }
}
