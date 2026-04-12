//! YOLO26n-Seg instance segmentation - WASM module.

mod image;

#[path = "gen/yolo26seg.rs"]
mod yolo26seg;

use lele::tensor::TensorView;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct SegmentProfile {
    preprocess_ms: f64,
    forward_ms: f64,
    postprocess_ms: f64,
    serialize_ms: f64,
    total_ms: f64,
    detections: usize,
    mask_pixels: usize,
    json_bytes: usize,
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct Yolo26NSegEngine {
    model: yolo26seg::Yolo26Seg<'static>,
    workspace: yolo26seg::Yolo26SegWorkspace,
}

#[wasm_bindgen]
impl Yolo26NSegEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8]) -> Result<Yolo26NSegEngine, JsError> {
        let leaked: &'static [u8] = Box::leak(weights.to_vec().into_boxed_slice());
        let model = yolo26seg::Yolo26Seg::new(leaked);
        let workspace = yolo26seg::Yolo26SegWorkspace::new();
        Ok(Yolo26NSegEngine { model, workspace })
    }

    pub fn segment(
        &mut self,
        image_rgba: &[u8],
        width: u32,
        height: u32,
        threshold: f32,
    ) -> Result<String, JsError> {
        let (_, _, json) = self.run_segment_pipeline(image_rgba, width, height, threshold)?;
        Ok(json)
    }

    pub fn segment_profile(
        &mut self,
        image_rgba: &[u8],
        width: u32,
        height: u32,
        threshold: f32,
    ) -> Result<String, JsError> {
        let (_, profile, _) = self.run_segment_pipeline(image_rgba, width, height, threshold)?;
        serde_json::to_string(&profile).map_err(|e| JsError::new(&format!("JSON error: {}", e)))
    }

    pub fn detect(
        &mut self,
        image_rgba: &[u8],
        width: u32,
        height: u32,
        threshold: f32,
    ) -> Result<String, JsError> {
        let seg_json = self.segment(image_rgba, width, height, threshold)?;
        let seg: image::SegmentationOutput = serde_json::from_str(&seg_json)
            .map_err(|e| JsError::new(&format!("JSON parse error: {}", e)))?;
        let json = serde_json::to_string(&seg.detections)
            .map_err(|e| JsError::new(&format!("JSON error: {}", e)))?;
        Ok(json)
    }
}

impl Yolo26NSegEngine {
    fn run_segment_pipeline(
        &mut self,
        image_rgba: &[u8],
        width: u32,
        height: u32,
        threshold: f32,
    ) -> Result<(image::SegmentationOutput, SegmentProfile, String), JsError> {
        let start_total = now_ms();

        let start_preprocess = now_ms();
        let img = image::Image::from_rgba(width as usize, height as usize, image_rgba);
        let input_data = img.preprocess();
        let preprocess_ms = now_ms() - start_preprocess;

        let start_forward = now_ms();
        let input = TensorView::from_owned(input_data, vec![1, 3, 640, 640]);
        let (logits, mask_features) = self
            .model
            .forward_with_workspace(&mut self.workspace, input);
        let logits: TensorView<'static> = logits.to_owned();
        let mask_features: TensorView<'static> = mask_features.to_owned();
        let forward_ms = now_ms() - start_forward;

        let start_postprocess = now_ms();
        let result = image::postprocess_segmentation(
            logits.data.as_ref(),
            mask_features.data.as_ref(),
            width as usize,
            height as usize,
            threshold,
        );
        let postprocess_ms = now_ms() - start_postprocess;

        let start_serialize = now_ms();
        let json = serde_json::to_string(&result)
            .map_err(|e| JsError::new(&format!("JSON error: {}", e)))?;
        let serialize_ms = now_ms() - start_serialize;

        let mask_pixels = result.mask.iter().filter(|&&value| value != 0).count();
        let profile = SegmentProfile {
            preprocess_ms,
            forward_ms,
            postprocess_ms,
            serialize_ms,
            total_ms: now_ms() - start_total,
            detections: result.detections.len(),
            mask_pixels,
            json_bytes: json.len(),
        };

        Ok((result, profile, json))
    }
}
