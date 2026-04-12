use serde::{Deserialize, Serialize};

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::{
    f32x4, f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat, v128, v128_load,
};

pub const COCO_CLASSES: &[&str] = &[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub class_name: String,
    pub score: f32,
    pub bbox: [f32; 4],
    pub mask_coeffs: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationOutput {
    pub detections: Vec<Detection>,
    pub mask_width: usize,
    pub mask_height: usize,
    pub mask: Vec<u8>,
}

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Image {
    pub fn from_rgba(width: usize, height: usize, rgba: &[u8]) -> Self {
        let mut rgb = Vec::with_capacity(width * height * 3);
        for pixel in rgba.chunks_exact(4) {
            rgb.push(pixel[0]);
            rgb.push(pixel[1]);
            rgb.push(pixel[2]);
        }
        Self {
            width,
            height,
            data: rgb,
        }
    }

    pub fn preprocess(&self) -> Vec<f32> {
        const TARGET_SIZE: usize = 640;

        let resized = self.resize(TARGET_SIZE, TARGET_SIZE);
        let mut output = vec![0.0f32; 3 * TARGET_SIZE * TARGET_SIZE];

        for c in 0..3 {
            for h in 0..TARGET_SIZE {
                for w in 0..TARGET_SIZE {
                    let hwc_idx = (h * TARGET_SIZE + w) * 3 + c;
                    let chw_idx = c * TARGET_SIZE * TARGET_SIZE + h * TARGET_SIZE + w;
                    output[chw_idx] = resized.data[hwc_idx] as f32 / 255.0;
                }
            }
        }

        output
    }

    fn resize(&self, new_width: usize, new_height: usize) -> Image {
        let mut data = vec![0u8; new_width * new_height * 3];

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x =
                    ((x as f32 + 0.5) * self.width as f32 / new_width as f32).floor() as usize;
                let src_y =
                    ((y as f32 + 0.5) * self.height as f32 / new_height as f32).floor() as usize;
                let src_x = src_x.min(self.width - 1);
                let src_y = src_y.min(self.height - 1);

                for c in 0..3 {
                    let src_idx = (src_y * self.width + src_x) * 3 + c;
                    let dst_idx = (y * new_width + x) * 3 + c;
                    data[dst_idx] = self.data[src_idx];
                }
            }
        }

        Image {
            width: new_width,
            height: new_height,
            data,
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn mask_dot(mask_coeffs: &[f32], mask_features: &[f32], mask_hw: usize, pixel_index: usize) -> f32 {
    let mut channel = 0;
    let mut acc = unsafe { f32x4_splat(0.0) };

    while channel + 4 <= mask_coeffs.len() {
        let coeffs = unsafe { v128_load(mask_coeffs.as_ptr().add(channel) as *const v128) };
        let features = unsafe {
            f32x4(
                *mask_features.get_unchecked(channel * mask_hw + pixel_index),
                *mask_features.get_unchecked((channel + 1) * mask_hw + pixel_index),
                *mask_features.get_unchecked((channel + 2) * mask_hw + pixel_index),
                *mask_features.get_unchecked((channel + 3) * mask_hw + pixel_index),
            )
        };
        acc = unsafe { f32x4_add(acc, f32x4_mul(coeffs, features)) };
        channel += 4;
    }

    let mut sum = unsafe {
        f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc)
    };
    while channel < mask_coeffs.len() {
        sum += mask_coeffs[channel] * mask_features[channel * mask_hw + pixel_index];
        channel += 1;
    }
    sum
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn mask_dot(mask_coeffs: &[f32], mask_features: &[f32], mask_hw: usize, pixel_index: usize) -> f32 {
    let mut sum = 0.0f32;
    for (channel, coeff) in mask_coeffs.iter().enumerate() {
        sum += coeff * mask_features[channel * mask_hw + pixel_index];
    }
    sum
}

#[inline]
fn bbox_pixel_range(start: f32, end: f32, limit: usize) -> Option<(usize, usize)> {
    if limit == 0 {
        return None;
    }

    let start_idx = start.ceil().clamp(0.0, limit as f32) as usize;
    let end_idx = ((end.floor() as isize) + 1).clamp(0, limit as isize) as usize;
    if start_idx >= end_idx {
        None
    } else {
        Some((start_idx, end_idx))
    }
}

pub fn postprocess_segmentation(
    logits: &[f32],
    mask_features: &[f32],
    img_width: usize,
    img_height: usize,
    threshold: f32,
) -> SegmentationOutput {
    const NUM_QUERIES: usize = 300;
    const MASK_DIM: usize = 32;
    const LOGIT_LEN: usize = 38;
    const SCORE_OFFSET: usize = 4;
    const CLASS_OFFSET: usize = 5;
    const MASK_COEFF_START: usize = 6;

    let num_classes = COCO_CLASSES.len();
    let mask_total = mask_features.len();
    let mask_hw = (mask_total / MASK_DIM).max(1);
    let mask_h = (mask_hw as f32).sqrt() as usize;
    let mask_w = mask_h.max(1);

    let mut detections = Vec::new();
    let scale_x = img_width as f32 / 640.0;
    let scale_y = img_height as f32 / 640.0;

    for i in 0..NUM_QUERIES {
        let base = i * LOGIT_LEN;
        if base + LOGIT_LEN > logits.len() {
            break;
        }

        let score = logits[base + SCORE_OFFSET];
        if score < threshold {
            continue;
        }

        let class_id = (logits[base + CLASS_OFFSET] as usize).min(num_classes - 1);

        let x1_raw = logits[base];
        let y1_raw = logits[base + 1];
        let x2_raw = logits[base + 2];
        let y2_raw = logits[base + 3];
        if x2_raw <= x1_raw || y2_raw <= y1_raw {
            continue;
        }

        let x1 = (x1_raw * scale_x).max(0.0);
        let y1 = (y1_raw * scale_y).max(0.0);
        let x2 = (x2_raw * scale_x).min(img_width as f32);
        let y2 = (y2_raw * scale_y).min(img_height as f32);

        let mut mask_coeffs = Vec::with_capacity(MASK_DIM);
        for j in 0..MASK_DIM {
            mask_coeffs.push(logits[base + MASK_COEFF_START + j]);
        }

        detections.push(Detection {
            class_name: COCO_CLASSES[class_id].to_string(),
            score,
            bbox: [x1, y1, x2, y2],
            mask_coeffs,
        });
    }

    if detections.is_empty() {
        return SegmentationOutput {
            detections,
            mask_width: img_width,
            mask_height: img_height,
            mask: vec![0u8; img_width * img_height],
        };
    }

    let mut mask_img = vec![0u8; img_width * img_height];
    let x_scale = mask_w as f32 / img_width as f32;
    let y_scale = mask_h as f32 / img_height as f32;
    let mut det_mask = vec![0.0f32; mask_h * mask_w];

    for det in &detections {
        det_mask.fill(0.0);

        for y in 0..mask_h {
            for x in 0..mask_w {
                let pixel_index = y * mask_w + x;
                let sum = mask_dot(
                    &det.mask_coeffs,
                    mask_features,
                    mask_h * mask_w,
                    pixel_index,
                );
                det_mask[pixel_index] = sigmoid(sum);
            }
        }

        let Some((x_start, x_end)) = bbox_pixel_range(det.bbox[0], det.bbox[2], img_width) else {
            continue;
        };
        let Some((y_start, y_end)) = bbox_pixel_range(det.bbox[1], det.bbox[3], img_height) else {
            continue;
        };

        for img_y in y_start..y_end {
            for img_x in x_start..x_end {
                let mx = ((img_x as f32 + 0.5) * x_scale).floor() as usize;
                let my = ((img_y as f32 + 0.5) * y_scale).floor() as usize;
                let mx = mx.min(mask_w - 1);
                let my = my.min(mask_h - 1);
                let m = det_mask[my * mask_w + mx];

                if m > 0.5 && (m * det.score) > 0.5 {
                    mask_img[img_y * img_width + img_x] = 255;
                }
            }
        }
    }

    SegmentationOutput {
        detections,
        mask_width: img_width,
        mask_height: img_height,
        mask: mask_img,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_postprocess(
        logits: &[f32],
        mask_features: &[f32],
        img_width: usize,
        img_height: usize,
        threshold: f32,
    ) -> SegmentationOutput {
        const NUM_QUERIES: usize = 300;
        const MASK_DIM: usize = 32;
        const LOGIT_LEN: usize = 38;
        const SCORE_OFFSET: usize = 4;
        const CLASS_OFFSET: usize = 5;
        const MASK_COEFF_START: usize = 6;

        let mask_total = mask_features.len();
        let mask_hw = (mask_total / MASK_DIM).max(1);
        let mask_h = (mask_hw as f32).sqrt() as usize;
        let mask_w = mask_h.max(1);
        let scale_x = img_width as f32 / 640.0;
        let scale_y = img_height as f32 / 640.0;

        let mut detections = Vec::new();
        for i in 0..NUM_QUERIES {
            let base = i * LOGIT_LEN;
            if base + LOGIT_LEN > logits.len() {
                break;
            }
            let score = logits[base + SCORE_OFFSET];
            if score < threshold {
                continue;
            }

            let class_id = (logits[base + CLASS_OFFSET] as usize).min(COCO_CLASSES.len() - 1);
            let x1_raw = logits[base];
            let y1_raw = logits[base + 1];
            let x2_raw = logits[base + 2];
            let y2_raw = logits[base + 3];
            if x2_raw <= x1_raw || y2_raw <= y1_raw {
                continue;
            }

            let mut mask_coeffs = Vec::with_capacity(MASK_DIM);
            for j in 0..MASK_DIM {
                mask_coeffs.push(logits[base + MASK_COEFF_START + j]);
            }

            detections.push(Detection {
                class_name: COCO_CLASSES[class_id].to_string(),
                score,
                bbox: [
                    (x1_raw * scale_x).max(0.0),
                    (y1_raw * scale_y).max(0.0),
                    (x2_raw * scale_x).min(img_width as f32),
                    (y2_raw * scale_y).min(img_height as f32),
                ],
                mask_coeffs,
            });
        }

        let mut mask_img = vec![0u8; img_width * img_height];
        let x_scale = mask_w as f32 / img_width as f32;
        let y_scale = mask_h as f32 / img_height as f32;

        for det in &detections {
            let mut det_mask = vec![0.0f32; mask_h * mask_w];
            for y in 0..mask_h {
                for x in 0..mask_w {
                    let pixel_index = y * mask_w + x;
                    let sum = mask_dot(
                        &det.mask_coeffs,
                        mask_features,
                        mask_h * mask_w,
                        pixel_index,
                    );
                    det_mask[pixel_index] = sigmoid(sum);
                }
            }

            for img_y in 0..img_height {
                for img_x in 0..img_width {
                    let mx = ((img_x as f32 + 0.5) * x_scale).floor() as usize;
                    let my = ((img_y as f32 + 0.5) * y_scale).floor() as usize;
                    let mx = mx.min(mask_w - 1);
                    let my = my.min(mask_h - 1);
                    let m = det_mask[my * mask_w + mx];
                    let in_bbox = img_x as f32 >= det.bbox[0]
                        && img_x as f32 <= det.bbox[2]
                        && img_y as f32 >= det.bbox[1]
                        && img_y as f32 <= det.bbox[3];
                    if in_bbox && m > 0.5 && (m * det.score) > 0.5 {
                        mask_img[img_y * img_width + img_x] = 255;
                    }
                }
            }
        }

        SegmentationOutput {
            detections,
            mask_width: img_width,
            mask_height: img_height,
            mask: mask_img,
        }
    }

    #[test]
    fn resize_uses_pil_nearest_neighbor_mapping() {
        let image = Image {
            width: 2,
            height: 2,
            data: vec![10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42],
        };

        let resized = image.resize(4, 4);
        assert_eq!(resized.width, 4);
        assert_eq!(resized.height, 4);

        let pixel = |x: usize, y: usize| -> [u8; 3] {
            let idx = (y * resized.width + x) * 3;
            [
                resized.data[idx],
                resized.data[idx + 1],
                resized.data[idx + 2],
            ]
        };

        assert_eq!(pixel(0, 0), [10, 11, 12]);
        assert_eq!(pixel(3, 0), [20, 21, 22]);
        assert_eq!(pixel(0, 3), [30, 31, 32]);
        assert_eq!(pixel(3, 3), [40, 41, 42]);
        assert_eq!(pixel(1, 1), [10, 11, 12]);
        assert_eq!(pixel(2, 1), [20, 21, 22]);
    }

    #[test]
    fn postprocess_returns_scaled_bbox_and_mask() {
        let mut logits = vec![0.0f32; 300 * 38];
        logits[0] = 100.0;
        logits[1] = 50.0;
        logits[2] = 200.0;
        logits[3] = 150.0;
        logits[4] = 0.9;
        logits[5] = 1.0;
        logits[6] = 1.0;

        let mut mask_features = vec![0.0f32; 32 * 2 * 2];
        mask_features[0] = 10.0;
        mask_features[1] = -10.0;
        mask_features[2] = -10.0;
        mask_features[3] = -10.0;

        let output = postprocess_segmentation(&logits, &mask_features, 640, 640, 0.3);
        assert_eq!(output.detections.len(), 1);

        let detection = &output.detections[0];
        assert_eq!(detection.class_name, "bicycle");
        assert!((detection.score - 0.9).abs() < 1e-6);
        assert_eq!(detection.bbox, [100.0, 50.0, 200.0, 150.0]);

        let non_zero = output.mask.iter().filter(|&&value| value != 0).count();
        assert!(non_zero > 0);
        assert_eq!(output.mask[0], 0);

        let inside_bbox_idx = 60 * 640 + 120;
        assert_eq!(output.mask[inside_bbox_idx], 255);

        let outside_bbox_idx = 60 * 640 + 250;
        assert_eq!(output.mask[outside_bbox_idx], 0);
    }

    #[test]
    fn postprocess_filters_invalid_boxes_and_low_scores() {
        let mut logits = vec![0.0f32; 300 * 38];
        logits[0] = 10.0;
        logits[1] = 10.0;
        logits[2] = 5.0;
        logits[3] = 15.0;
        logits[4] = 0.95;

        logits[38] = 0.0;
        logits[39] = 0.0;
        logits[40] = 20.0;
        logits[41] = 20.0;
        logits[42] = 0.2;

        let mask_features = vec![0.0f32; 32 * 2 * 2];
        let output = postprocess_segmentation(&logits, &mask_features, 640, 640, 0.3);

        assert!(output.detections.is_empty());
        assert!(output.mask.iter().all(|&value| value == 0));
    }

    #[test]
    fn optimized_postprocess_matches_reference() {
        let mut logits = vec![0.0f32; 300 * 38];

        logits[0] = 120.2;
        logits[1] = 80.1;
        logits[2] = 260.8;
        logits[3] = 240.9;
        logits[4] = 0.85;
        logits[5] = 0.0;
        logits[6] = 0.8;
        logits[7] = -0.2;
        logits[8] = 0.5;

        let second = 38;
        logits[second] = 20.0;
        logits[second + 1] = 20.0;
        logits[second + 2] = 80.0;
        logits[second + 3] = 90.0;
        logits[second + 4] = 0.72;
        logits[second + 5] = 2.0;
        logits[second + 6] = -0.3;
        logits[second + 7] = 1.1;

        let mut mask_features = vec![0.0f32; 32 * 4 * 4];
        for channel in 0..32 {
            for idx in 0..16 {
                mask_features[channel * 16 + idx] =
                    ((channel as f32 * 0.17) + idx as f32 * 0.11) - 1.2;
            }
        }

        let optimized = postprocess_segmentation(&logits, &mask_features, 320, 256, 0.3);
        let reference = reference_postprocess(&logits, &mask_features, 320, 256, 0.3);

        assert_eq!(optimized.detections.len(), reference.detections.len());
        assert_eq!(optimized.mask_width, reference.mask_width);
        assert_eq!(optimized.mask_height, reference.mask_height);
        assert_eq!(optimized.mask, reference.mask);

        for (lhs, rhs) in optimized.detections.iter().zip(reference.detections.iter()) {
            assert_eq!(lhs.class_name, rhs.class_name);
            assert!((lhs.score - rhs.score).abs() < 1e-6);
            assert_eq!(lhs.bbox, rhs.bbox);
            assert_eq!(lhs.mask_coeffs, rhs.mask_coeffs);
        }
    }
}
