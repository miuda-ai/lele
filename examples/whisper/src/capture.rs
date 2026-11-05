//! Microphone capture through cpal.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, SizedSample, StreamConfig};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};

/// An open input stream. Mono f32 blocks in [-1, 1] arrive on `blocks`, at
/// `sample_rate`; the stream stops when this is dropped.
pub struct Microphone {
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub format: SampleFormat,
    pub blocks: mpsc::Receiver<Vec<f32>>,
    /// Frames captured so far, to tell how far a consumer trails the device.
    pub captured: Arc<AtomicUsize>,
    _stream: cpal::Stream,
}

impl Microphone {
    /// Opens the default input device, or the first whose name contains
    /// `device` (case-insensitively). `wanted_rate` is used if the device (or
    /// the sound server behind it) offers it, preferring mono; otherwise the
    /// device's default rate.
    pub fn open(
        device: Option<&str>,
        wanted_rate: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = match device {
            Some(want) => host
                .input_devices()?
                .find(|d| {
                    d.description()
                        .map(|desc| desc.name().to_lowercase().contains(&want.to_lowercase()))
                        .unwrap_or(false)
                })
                .ok_or_else(|| format!("no input device matching {want:?}"))?,
            None => host
                .default_input_device()
                .ok_or("no default input device")?,
        };
        let (config, format) = pick_config(&device, wanted_rate)?;
        let name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "?".into());

        let (tx, blocks) = mpsc::channel();
        let captured = Arc::new(AtomicUsize::new(0));
        let stream = match format {
            SampleFormat::F32 => start_capture::<f32>(&device, &config, tx, captured.clone())?,
            SampleFormat::I16 => start_capture::<i16>(&device, &config, tx, captured.clone())?,
            SampleFormat::I32 => start_capture::<i32>(&device, &config, tx, captured.clone())?,
            other => return Err(format!("unsupported sample format {other:?}").into()),
        };
        Ok(Self {
            name,
            sample_rate: config.sample_rate,
            channels: config.channels,
            format,
            blocks,
            captured,
            _stream: stream,
        })
    }
}

fn pick_config(
    device: &cpal::Device,
    wanted_rate: u32,
) -> Result<(StreamConfig, SampleFormat), Box<dyn std::error::Error>> {
    let usable =
        |f: SampleFormat| matches!(f, SampleFormat::F32 | SampleFormat::I16 | SampleFormat::I32);
    let mut direct: Vec<_> = device
        .supported_input_configs()?
        .filter(|c| usable(c.sample_format()))
        .filter_map(|c| c.try_with_sample_rate(wanted_rate))
        .collect();
    direct.sort_by_key(|c| c.channels());
    let chosen = match direct.into_iter().next() {
        Some(c) => c,
        None => device.default_input_config()?,
    };
    Ok((chosen.config(), chosen.sample_format()))
}

fn start_capture<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    tx: mpsc::Sender<Vec<f32>>,
    captured: Arc<AtomicUsize>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>>
where
    T: SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = config.channels as usize;
    let stream = device.build_input_stream(
        *config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            captured.fetch_add(data.len() / channels, Ordering::Relaxed);
            let mono = data
                .chunks_exact(channels)
                .map(|frame| {
                    frame
                        .iter()
                        .map(|&s| <f32 as cpal::FromSample<T>>::from_sample_(s))
                        .sum::<f32>()
                        / channels as f32
                })
                .collect();
            let _ = tx.send(mono);
        },
        |err| eprintln!("\ncapture error: {err}"),
        None,
    )?;
    stream.play()?;
    Ok(stream)
}
