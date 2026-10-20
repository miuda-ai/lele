use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Loads a FunASR `am.mvn` file (Kaldi NNet text format):
///
/// ```text
/// <AddShift> 560 560
/// <LearnRateCoef> 0 [ -8.31 -8.60 ... ]
/// <Rescale> 560 560
/// <LearnRateCoef> 0 [ 0.15 0.15 ... ]
/// ```
///
/// `AddShift` stores `-mean` and `Rescale` stores `1/std`, so the official
/// normalization is `(x + addshift) * rescale == (x - mean) / std`.
/// Returns `(mean, std)` ready for `Cmvn::apply_with_stats`.
pub fn load_am_mvn<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, Vec<f32>), std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut section = "";
    let mut addshift: Vec<f32> = Vec::new();
    let mut rescale: Vec<f32> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.contains("<AddShift>") {
            section = "addshift";
        } else if line.contains("<Rescale>") {
            section = "rescale";
        } else if line.contains('[') {
            let values: Vec<f32> = line
                .split('[')
                .nth(1)
                .and_then(|s| s.split(']').next())
                .unwrap_or("")
                .split_whitespace()
                .filter_map(|v| v.parse().ok())
                .collect();
            match section {
                "addshift" => addshift = values,
                "rescale" => rescale = values,
                _ => {}
            }
        }
    }

    if addshift.len() != rescale.len() || addshift.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "am.mvn: AddShift/Rescale sections missing or mismatched",
        ));
    }

    let mean: Vec<f32> = addshift.iter().map(|v| -v).collect();
    let std: Vec<f32> = rescale.iter().map(|v| 1.0 / v).collect();
    Ok((mean, std))
}
