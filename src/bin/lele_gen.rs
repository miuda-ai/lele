use lele::model::OnnxModel;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const USAGE: &str = "Usage: lele_gen <model.onnx> <output_dir> [ModelClassName] \
    [--shape <input>=<d,d,...>] [--dim <name>=<n>] [--value <input>=<v,v,...>]";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional = Vec::new();
    let mut compiler = lele::compiler::Compiler::new().with_default_optimizations();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let flag = arg.as_str();
        if !matches!(flag, "--shape" | "--dim" | "--value") {
            positional.push(arg);
            continue;
        }
        let setting = args
            .next()
            .ok_or_else(|| format!("{flag} takes <name>=<value>"))?;
        let (name, value) = setting
            .split_once('=')
            .ok_or_else(|| format!("{flag} takes <name>=<value>, not {setting}"))?;
        let list = || value.split(',').map(str::trim).filter(|s| !s.is_empty());
        compiler = match flag {
            "--shape" => {
                let dims: Vec<usize> = list().map(str::parse).collect::<Result<_, _>>()?;
                compiler.with_input_shape(name, &dims)
            }
            "--dim" => compiler.with_dim(name, value.trim().parse()?),
            _ => match list().map(str::parse::<i64>).collect::<Result<Vec<_>, _>>() {
                Ok(ints) => compiler.with_input_value(name, ints),
                Err(_) => {
                    let floats: Vec<f64> = list().map(str::parse).collect::<Result<_, _>>()?;
                    compiler.with_input_value(name, floats)
                }
            },
        };
    }
    if !(2..=3).contains(&positional.len()) {
        eprintln!("{USAGE}");
        std::process::exit(1);
    }
    let model_path = &positional[0];
    let output_dir = Path::new(&positional[1]);
    let class_name = positional.get(2).map_or("Model", String::as_str);
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
    let result = compiler.with_name(class_name).compile(graph)?;
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
