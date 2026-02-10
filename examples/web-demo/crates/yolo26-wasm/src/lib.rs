//! YOLO26 Object Detection - WASM module.

mod image;

#[path = "gen/yolo26.rs"]
mod yolo26;

use lele::tensor::TensorView;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct Yolo26Engine {
    model: yolo26::Yolo26<'static>,
    workspace: yolo26::Yolo26Workspace,
}

#[wasm_bindgen]
impl Yolo26Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(weights: &[u8]) -> Result<Yolo26Engine, JsError> {
        let leaked: &'static [u8] = Box::leak(weights.to_vec().into_boxed_slice());
        let model = yolo26::Yolo26::new(leaked);
        let workspace = yolo26::Yolo26Workspace::new();
        Ok(Yolo26Engine { model, workspace })
    }

    pub fn detect(
        &mut self,
        image_rgba: &[u8],
        width: u32,
        height: u32,
        threshold: f32,
    ) -> Result<String, JsError> {
        let img = image::Image::from_rgba(width as usize, height as usize, image_rgba);
        let input_data = img.preprocess();

        let input = TensorView::from_owned(input_data, vec![1, 3, 640, 640]);
        let (logits, pred_boxes) = self
            .model
            .forward_with_workspace(&mut self.workspace, input);
        let logits: TensorView<'static> = logits.to_owned();
        let pred_boxes: TensorView<'static> = pred_boxes.to_owned();

        let detections = image::postprocess(
            logits.data.as_ref(),
            pred_boxes.data.as_ref(),
            width as usize,
            height as usize,
            threshold,
        );

        let json = serde_json::to_string(&detections)
            .map_err(|e| JsError::new(&format!("JSON error: {}", e)))?;
        Ok(json)
    }
}
