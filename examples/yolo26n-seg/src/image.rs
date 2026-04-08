/// Image preprocessing and postprocessing for YOLO26 Segmentation.
use std::path::Path;

/// Default COCO 80 class names.
pub const COCO_CLASSES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

/// Detection result from YOLO26 Segmentation.
#[derive(Debug, Clone)]
pub struct Detection {
    pub class_name: String,
    pub score: f32,
    /// Bounding box in [x1, y1, x2, y2] format (pixel coordinates)
    pub bbox: [f32; 4],
    /// Mask coefficients (32 values)
    pub mask_coeffs: Vec<f32>,
}

/// Simple image struct for preprocessing.
#[derive(Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    /// RGB data in HWC format
    pub data: Vec<u8>,
}

impl Image {
    /// Load image from file (supports JPEG, PNG, etc.).
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let img = ::image::open(path.as_ref())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let rgb = img.to_rgb8();
        let width = rgb.width() as usize;
        let height = rgb.height() as usize;
        let data = rgb.into_raw();
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Preprocess image for YOLO26: resize to 640x640, convert to CHW, normalize to [0,1].
    /// Returns a Vec<f32> in NCHW format (1, 3, 640, 640).
    pub fn preprocess(&self) -> Vec<f32> {
        const TARGET_SIZE: usize = 640;

        // Simple bilinear resize (naive implementation)
        let resized = self.resize(TARGET_SIZE, TARGET_SIZE);

        // Convert HWC u8 -> CHW f32 [0,1]
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

    /// Nearest-neighbor resize matching PIL/Pillow's NEAREST mapping.
    /// PIL uses center-pixel formula: src = floor((dst + 0.5) * src_size / dst_size)
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

    #[cfg(target_arch = "wasm32")]
    /// Create image from raw RGB data.
    pub fn from_rgb(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height * 3);
        Self {
            width,
            height,
            data,
        }
    }
}

/// Post-process YOLO26 Segmentation outputs.
/// logits: [1, 300, 38] - 4(bbox) + 1(score) + 1(class_id) + 32(mask_coeffs)
/// mask_features: [1, 32, H, W] - 32 channels mask feature map (dynamic size)
/// Returns detections with masks and the combined mask image.
pub fn postprocess_segmentation(
    logits: &[f32],
    mask_features: &[f32],
    img_width: usize,
    img_height: usize,
    threshold: f32,
    class_names: Option<&[&str]>,
) -> (Vec<Detection>, Vec<u8>) {
    const NUM_QUERIES: usize = 300;
    const MASK_DIM: usize = 32;
    let classes = class_names.unwrap_or(COCO_CLASSES);
    let num_classes = classes.len();

    // Infer mask dimensions from mask_features length
    // mask_features: [1, 32, H, W] -> total = 32 * H * W
    let mask_total = mask_features.len();
    let mask_hw = mask_total / MASK_DIM;
    let mask_h = (mask_hw as f32).sqrt() as usize;
    let mask_w = mask_h;

    let mut detections = Vec::new();

    // Output format: [x1, y1, x2, y2, score, class_id, mask_coeffs(32)] = 38
    // Note: coordinates are already in pixel space (relative to 640x640 input)
    const LOGIT_LEN: usize = 38;
    const SCORE_OFFSET: usize = 4;
    const CLASS_OFFSET: usize = 5;
    const MASK_COEFF_START: usize = 6;

    // Scale factors from 640x640 to original image size
    let scale_x = img_width as f32 / 640.0;
    let scale_y = img_height as f32 / 640.0;

    for i in 0..NUM_QUERIES {
        let bbox_offset = i * LOGIT_LEN;

        // Get score (already sigmoid applied)
        let score = logits[bbox_offset + SCORE_OFFSET];

        if score < threshold {
            continue;
        }

        // Get class_id
        let class_id = logits[bbox_offset + CLASS_OFFSET] as usize;
        let class_id = class_id.min(num_classes - 1);

        // Get bbox - coordinates are [x1, y1, x2, y2] in pixel space (relative to 640x640)
        let x1_raw = logits[bbox_offset];
        let y1_raw = logits[bbox_offset + 1];
        let x2_raw = logits[bbox_offset + 2];
        let y2_raw = logits[bbox_offset + 3];

        // Skip invalid boxes (negative or zero area)
        if x2_raw <= x1_raw || y2_raw <= y1_raw {
            continue;
        }

        // Convert from 640x640 coordinates to original image coordinates
        let x1 = (x1_raw * scale_x).max(0.0);
        let y1 = (y1_raw * scale_y).max(0.0);
        let x2 = (x2_raw * scale_x).min(img_width as f32);
        let y2 = (y2_raw * scale_y).min(img_height as f32);

        // Get mask coefficients
        let mut mask_coeffs = Vec::with_capacity(MASK_DIM);
        for j in 0..MASK_DIM {
            mask_coeffs.push(logits[bbox_offset + MASK_COEFF_START + j]);
        }

        detections.push(Detection {
            class_name: classes[class_id].to_string(),
            score: score,
            bbox: [x1, y1, x2, y2],
            mask_coeffs,
        });
    }

    // If no detections, return empty
    if detections.is_empty() {
        return (detections, vec![0u8; img_width * img_height]);
    }

    // Generate mask for each detection
    // mask_features: [32, H, W]
    // For each detection: compute sigmoid of dot product of mask_coeffs and mask_features
    let mut mask_img = vec![0u8; img_width * img_height];

    // Scale factors from mask_size to img dimensions
    let scale_x = mask_w as f32 / img_width as f32;
    let scale_y = mask_h as f32 / img_height as f32;

    for det in &detections {
        // Compute mask: for each pixel in mask, compute dot product
        let mut det_mask = vec![0.0f32; mask_h * mask_w];

        for y in 0..mask_h {
            for x in 0..mask_w {
                let mut sum = 0.0f32;
                for c in 0..MASK_DIM {
                    let idx = c * mask_h * mask_w + y * mask_w + x;
                    sum += det.mask_coeffs[c] * mask_features[idx];
                }
                // Sigmoid activation
                det_mask[y * mask_w + x] = 1.0 / (1.0 + (-sum).exp());
            }
        }

        // Scale mask to image size and combine
        for img_y in 0..img_height {
            for img_x in 0..img_width {
                // Map img coords to mask coords (nearest neighbor)
                let mask_x = ((img_x as f32 + 0.5) * scale_x).floor() as usize;
                let mask_y = ((img_y as f32 + 0.5) * scale_y).floor() as usize;
                let mask_x = mask_x.min(mask_w - 1);
                let mask_y = mask_y.min(mask_h - 1);

                let mask_val = det_mask[mask_y * mask_w + mask_x];

                // Check if this pixel is inside the bbox
                let in_bbox = img_x as f32 >= det.bbox[0]
                    && img_x as f32 <= det.bbox[2]
                    && img_y as f32 >= det.bbox[1]
                    && img_y as f32 <= det.bbox[3];

                // Combine masks (take max)
                if in_bbox && mask_val > 0.5 {
                    let idx = img_y * img_width + img_x;
                    // Use detection score as mask threshold
                    if mask_val * det.score > 0.5 {
                        mask_img[idx] = 255;
                    }
                }
            }
        }
    }

    (detections, mask_img)
}
