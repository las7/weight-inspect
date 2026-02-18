use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only build ONNX protobuf if the onnx feature is enabled
    if !cfg!(feature = "onnx") {
        return Ok(());
    }

    let out_dir = env::var("OUT_DIR")?;
    let proto_dir = PathBuf::from(&out_dir).join("onnx-proto");
    fs::create_dir_all(&proto_dir)?;

    let proto_path = proto_dir.join("onnx.proto");

    let onnx_proto = include_bytes!("onnx/onnx.proto");
    fs::write(&proto_path, onnx_proto)?;

    #[cfg(feature = "onnx")]
    {
        let mut config = prost_build::Config::new();
        config
            .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
            .out_dir(&proto_dir);

        config.compile_protos(&[&proto_path], &[&proto_dir])?;
    }

    println!("cargo:rerun-if-changed=onnx/onnx.proto");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_ONNX");

    Ok(())
}
