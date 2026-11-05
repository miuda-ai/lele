use lele::model::OnnxModel;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model.onnx> <output_dir> [ModelClassName]",
            args[0]
        );
        std::process::exit(1);
    }
    let model_path = &args[1];
    let output_dir_str = &args[2];
    let class_name = if args.len() >= 4 { &args[3] } else { "Model" };
    let output_dir = Path::new(output_dir_str);
    if !output_dir.exists() {
        std::fs::create_dir_all(output_dir)?;
    }
    println!("Loading model from: {}", model_path);
    let model = OnnxModel::load(model_path)?;
    let graph = model.graph().ok_or("Missing graph")?;
    println!(
        "Extracting weights and generating code for class '{}'...",
        class_name
    );
    let compiler = lele::compiler::Compiler::new()
        .with_name(class_name)
        .with_default_optimizations();
    let result = compiler.compile(graph)?;
    let weights_filename = format!("{}_weights.bin", class_name.to_lowercase());
    let bin_path = output_dir.join(&weights_filename);
    let mut bin_file = BufWriter::new(File::create(&bin_path)?);
    bin_file.write_all(&result.weights)?;
    bin_file.flush()?;
    println!("Wrote weights to {:?}", bin_path);
    let filename = format!("{}.rs", class_name.to_lowercase());
    let rs_path = output_dir.join(&filename);
    let mut rs = BufWriter::new(File::create(&rs_path)?);
    rs.write_all(result.code.as_bytes())?;
    println!("Wrote code to {:?}", rs_path);
    Ok(())
}
