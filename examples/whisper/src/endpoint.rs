//! Cuts a stream of audio into utterances by its energy: speech starts when the
//! level stays above the noise floor for a moment, and ends after a pause.
//!
//! The noise floor is followed while nobody speaks, so the thresholds adapt to a
//! quiet room or a noisy one. This is a simple endpointer, not a model: it cannot
//! tell speech from other loud sounds, nor a pause in a sentence from its end.

use std::collections::VecDeque;

pub const SAMPLE_RATE: usize = 16_000;

/// Frames at the start that only measure the noise floor.
const CALIBRATION_FRAMES: usize = 10;

#[derive(Clone, Debug)]
pub struct Config {
    /// Length of the frames the level is measured over.
    pub frame_ms: usize,
    /// Consecutive loud frames that start an utterance.
    pub start_frames: usize,
    /// Silence that ends it.
    pub hangover_ms: usize,
    /// Audio kept from before the speech was noticed.
    pub pre_roll_ms: usize,
    /// Speech an utterance needs to be reported; a shorter one was a click.
    pub min_speech_ms: usize,
    /// The longest utterance: past it, the audio so far is reported and a new one
    /// goes on from there.
    pub max_secs: f64,
    /// How far above the noise floor speech is, in dB.
    pub margin_db: f32,
    /// The quietest the threshold gets, in dB below full scale.
    pub floor_db: f32,
    /// Silence kept at the end of an utterance; the rest of the hangover is cut.
    pub trailing_ms: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            frame_ms: 30,
            start_frames: 3,
            hangover_ms: 700,
            pre_roll_ms: 300,
            min_speech_ms: 250,
            max_secs: 28.0,
            margin_db: 12.0,
            floor_db: -50.0,
            trailing_ms: 200,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Event {
    /// Speech was noticed, `at` seconds in (counting the pre-roll).
    Start { at: f64 },
    /// An utterance is over.
    End {
        at: f64,
        /// Its audio, with the pre-roll and a little silence at the end.
        audio: Vec<f32>,
        /// Whether it was cut for length, with the speech going on.
        forced: bool,
    },
}

pub struct Endpointer {
    config: Config,
    frame: usize,
    partial: Vec<f32>,
    pre_roll: VecDeque<f32>,
    /// The level of the background noise, in dB.
    noise_db: f32,
    level_db: f32,
    frames: usize,
    loud_run: usize,
    in_speech: bool,
    utterance: Vec<f32>,
    speech_ms: usize,
    silence_ms: usize,
    /// Samples seen.
    clock: usize,
}

fn db(frame: &[f32]) -> f32 {
    let mean_square = frame.iter().map(|&x| x * x).sum::<f32>() / frame.len().max(1) as f32;
    10.0 * (mean_square + 1e-12).log10()
}

impl Endpointer {
    pub fn new(config: Config) -> Self {
        Self {
            frame: config.frame_ms * SAMPLE_RATE / 1000,
            config,
            partial: Vec::new(),
            pre_roll: VecDeque::new(),
            noise_db: -70.0,
            level_db: -100.0,
            frames: 0,
            loud_run: 0,
            in_speech: false,
            utterance: Vec::new(),
            speech_ms: 0,
            silence_ms: 0,
            clock: 0,
        }
    }

    /// Seconds of audio seen.
    pub fn time(&self) -> f64 {
        self.clock as f64 / SAMPLE_RATE as f64
    }

    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// The level of the latest frame, in dB below full scale.
    pub fn level_db(&self) -> f32 {
        self.level_db
    }

    /// The level above which a frame is speech.
    pub fn threshold_db(&self) -> f32 {
        (self.noise_db + self.config.margin_db).max(self.config.floor_db)
    }

    /// Feeds 16 kHz mono audio in [-1, 1], passing what it finds to `emit`.
    pub fn push(&mut self, audio: &[f32], emit: &mut dyn FnMut(Event)) {
        self.partial.extend_from_slice(audio);
        let mut start = 0;
        while self.partial.len() - start >= self.frame {
            let frame: Vec<f32> = self.partial[start..start + self.frame].to_vec();
            start += self.frame;
            self.step(&frame, emit);
        }
        self.partial.drain(..start);
    }

    fn step(&mut self, frame: &[f32], emit: &mut dyn FnMut(Event)) {
        let c = self.config.clone();
        self.clock += frame.len();
        self.level_db = db(frame);
        // The noise floor starts as the quietest of the first frames, then follows
        // the level down at once and up slowly (over tens of seconds), so that
        // speech does not raise it much but a steady hiss, in time, does.
        self.frames += 1;
        if self.frames <= CALIBRATION_FRAMES {
            self.noise_db = if self.frames == 1 {
                self.level_db
            } else {
                self.noise_db.min(self.level_db)
            };
        } else if self.level_db < self.noise_db {
            self.noise_db = 0.9 * self.noise_db + 0.1 * self.level_db;
        } else {
            self.noise_db += 0.001 * (self.level_db - self.noise_db);
        }
        let loud = self.frames > CALIBRATION_FRAMES && self.level_db > self.threshold_db();

        if !self.in_speech {
            self.pre_roll.extend(frame);
            let keep = c.pre_roll_ms * SAMPLE_RATE / 1000 + self.frame;
            while self.pre_roll.len() > keep {
                self.pre_roll.pop_front();
            }
            self.loud_run = if loud { self.loud_run + 1 } else { 0 };
            if self.loud_run >= c.start_frames {
                self.in_speech = true;
                self.utterance = self.pre_roll.drain(..).collect();
                self.speech_ms = self.loud_run * c.frame_ms;
                self.silence_ms = 0;
                emit(Event::Start {
                    at: self.time() - self.utterance.len() as f64 / SAMPLE_RATE as f64,
                });
            }
            return;
        }

        self.utterance.extend_from_slice(frame);
        if loud {
            self.silence_ms = 0;
            self.speech_ms += c.frame_ms;
        } else {
            self.silence_ms += c.frame_ms;
        }
        if self.silence_ms >= c.hangover_ms {
            self.finish(false, emit);
        } else if self.utterance.len() as f64 >= c.max_secs * SAMPLE_RATE as f64 {
            self.finish(true, emit);
        }
    }

    fn finish(&mut self, forced: bool, emit: &mut dyn FnMut(Event)) {
        let c = &self.config;
        let mut audio = std::mem::take(&mut self.utterance);
        let cut = self.silence_ms.saturating_sub(c.trailing_ms) * SAMPLE_RATE / 1000;
        audio.truncate(audio.len().saturating_sub(cut));
        if self.speech_ms >= c.min_speech_ms {
            emit(Event::End {
                at: self.time(),
                audio,
                forced,
            });
        }
        self.speech_ms = 0;
        self.silence_ms = 0;
        if forced {
            // The speech goes on: the next utterance is its continuation.
            return;
        }
        self.in_speech = false;
        self.loud_run = 0;
        self.pre_roll.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tone `ms` long at amplitude `amp`, or silence for 0.
    fn tone(ms: usize, amp: f32) -> Vec<f32> {
        (0..ms * SAMPLE_RATE / 1000)
            .map(|i| {
                amp * (2.0 * std::f32::consts::PI * 300.0 * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect()
    }

    /// The events `config` finds in `parts` played in turn, in blocks of 100 ms.
    fn run(config: Config, parts: &[Vec<f32>]) -> Vec<Event> {
        let mut ep = Endpointer::new(config);
        let mut events = Vec::new();
        for part in parts {
            for block in part.chunks(SAMPLE_RATE / 10) {
                ep.push(block, &mut |e| events.push(e));
            }
        }
        events
    }

    fn ends(events: &[Event]) -> Vec<(f64, usize, bool)> {
        events
            .iter()
            .filter_map(|e| match e {
                Event::End { at, audio, forced } => Some((*at, audio.len(), *forced)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn finds_speech_between_silences() {
        let ev = run(
            Config::default(),
            &[tone(1000, 0.0), tone(1000, 0.3), tone(1500, 0.0)],
        );
        let Event::Start { at } = ev[0] else {
            panic!("{ev:?}")
        };
        // The tone starts at 1.0 s and is noticed 3 frames later; the utterance starts
        // with the 330 ms of audio kept before that.
        assert!((0.7..0.8).contains(&at), "started at {at}");
        let [(at, len, forced)] = ends(&ev)[..] else {
            panic!("{ev:?}")
        };
        assert!(!forced);
        // Over 700 ms after the tone stopped, at 2 s.
        assert!((2.6..2.8).contains(&at), "ended at {at}");
        // Pre-roll (300 ms) + 1 s of tone + 200 ms of silence, give or take a frame.
        let secs = len as f64 / SAMPLE_RATE as f64;
        assert!((1.4..1.6).contains(&secs), "{secs} s of audio");
    }

    #[test]
    fn a_pause_shorter_than_the_hangover_is_part_of_the_utterance() {
        let ev = run(
            Config::default(),
            &[
                tone(500, 0.0),
                tone(600, 0.3),
                tone(400, 0.0),
                tone(600, 0.3),
                tone(1200, 0.0),
            ],
        );
        assert_eq!(ends(&ev).len(), 1, "{ev:?}");
    }

    #[test]
    fn a_pause_longer_than_the_hangover_splits_it() {
        let ev = run(
            Config::default(),
            &[
                tone(500, 0.0),
                tone(600, 0.3),
                tone(900, 0.0),
                tone(600, 0.3),
                tone(1200, 0.0),
            ],
        );
        assert_eq!(ends(&ev).len(), 2, "{ev:?}");
    }

    #[test]
    fn ignores_a_click() {
        let ev = run(
            Config::default(),
            &[tone(500, 0.0), tone(100, 0.5), tone(1500, 0.0)],
        );
        assert!(ends(&ev).is_empty(), "{ev:?}");
    }

    #[test]
    fn follows_the_noise_floor() {
        // A steady hiss at -30 dB is above the floor of -50 dB at first, but is
        // not speech once the noise floor has caught up with it; speech over it is.
        let hiss = |ms| tone(ms, 0.04);
        let ev = run(
            Config::default(),
            &[hiss(6000), tone(1000, 0.4), hiss(1500)],
        );
        let utterances = ends(&ev);
        assert_eq!(utterances.len(), 1, "{ev:?}");
        // The utterance is the tone and what is around it, not the hiss before.
        let secs = utterances[0].1 as f64 / SAMPLE_RATE as f64;
        assert!(secs < 2.0, "{secs} s");
    }

    #[test]
    fn cuts_long_speech_and_goes_on() {
        let config = Config {
            max_secs: 2.0,
            ..Config::default()
        };
        let ev = run(config, &[tone(500, 0.0), tone(5000, 0.3), tone(1200, 0.0)]);
        let utterances = ends(&ev);
        assert!(utterances.len() >= 3, "{ev:?}");
        assert!(
            utterances[..utterances.len() - 1].iter().all(|u| u.2),
            "{ev:?}"
        );
        assert!(!utterances.last().unwrap().2, "{ev:?}");
    }
}
