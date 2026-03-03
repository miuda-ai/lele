/// Image preprocessing and postprocessing for YOLO26 Segmentation.
use std::path::Path;

/// COCO class names (80 classes)
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
/// logits: [1, 300, 38] - 4(bbox) + 1(score) + 32(mask_coeffs) + 1(class)
/// mask_features: [1, 32, 160, 160] - 32 channels mask feature map
/// Returns detections with masks and the combined mask image.
pub fn postprocess_segmentation(
    logits: &[f32],
    mask_features: &[f32],
    img_width: usize,
    img_height: usize,
    threshold: f32,
) -> (Vec<Detection>, Vec<u8>) {
    const NUM_QUERIES: usize = 300;
    const NUM_CLASSES: usize = 80;
    const MASK_DIM: usize = 32;
    const MASK_H: usize = 160;
    const MASK_W: usize = 160;

    let mut detections = Vec::new();

    // Output format: [bbox(4), score(1), mask_coeffs(32)] = 37 (class missing due to Mod op unimplemented)
    const LOGIT_LEN: usize = 37;
    const MASK_COEFF_START: usize = 5;

    for i in 0..NUM_QUERIES {
        let bbox_offset = i * LOGIT_LEN;
        let score_offset = bbox_offset + 4;

        // Get score
        let score = logits[score_offset];
        let score_sigmoid = 1.0 / (1.0 + (-score).exp());

        if score_sigmoid < threshold {
            continue;
        }

        // Class is unknown due to Mod op not implemented, default to 0
        let class = 0;

        // Get bbox (normalized cx, cy, w, h)
        let cx = logits[bbox_offset];
        let cy = logits[bbox_offset + 1];
        let w = logits[bbox_offset + 2];
        let h = logits[bbox_offset + 3];

        // Skip invalid boxes
        if w <= 0.0 || h <= 0.0 {
            continue;
        }

        // Convert to pixel coordinates [x1, y1, x2, y2]
        let x1 = ((cx - w / 2.0) * img_width as f32).max(0.0);
        let y1 = ((cy - h / 2.0) * img_height as f32).max(0.0);
        let x2 = ((cx + w / 2.0) * img_width as f32).min(img_width as f32);
        let y2 = ((cy + h / 2.0) * img_height as f32).min(img_height as f32);

        // Get mask coefficients
        let mut mask_coeffs = Vec::with_capacity(MASK_DIM);
        for j in 0..MASK_DIM {
            mask_coeffs.push(logits[bbox_offset + MASK_COEFF_START + j]);
        }

        detections.push(Detection {
            class_name: COCO_CLASSES[class].to_string(),
            score: score_sigmoid,
            bbox: [x1, y1, x2, y2],
            mask_coeffs,
        });
    }

    // If no detections, return empty
    if detections.is_empty() {
        return (detections, vec![0u8; img_width * img_height]);
    }

    // Generate mask for each detection
    // mask_features: [32, 160, 160]
    // For each detection: compute sigmoid of dot product of mask_coeffs and mask_features
    let mut mask_img = vec![0u8; img_width * img_height];

    // Scale factors from 160x160 to img dimensions
    let scale_x = MASK_W as f32 / img_width as f32;
    let scale_y = MASK_H as f32 / img_height as f32;

    for det in &detections {
        // Compute mask: for each pixel in 160x160, compute dot product
        let mut det_mask = vec![0.0f32; MASK_H * MASK_W];

        for y in 0..MASK_H {
            for x in 0..MASK_W {
                let mut sum = 0.0f32;
                for c in 0..MASK_DIM {
                    let idx = c * MASK_H * MASK_W + y * MASK_W + x;
                    sum += det.mask_coeffs[c] * mask_features[idx];
                }
                // Sigmoid activation
                det_mask[y * MASK_W + x] = 1.0 / (1.0 + (-sum).exp());
            }
        }

        // Scale mask to image size and combine
        for img_y in 0..img_height {
            for img_x in 0..img_width {
                // Map img coords to mask coords (nearest neighbor)
                let mask_x = ((img_x as f32 + 0.5) * scale_x).floor() as usize;
                let mask_y = ((img_y as f32 + 0.5) * scale_y).floor() as usize;
                let mask_x = mask_x.min(MASK_W - 1);
                let mask_y = mask_y.min(MASK_H - 1);

                let mask_val = det_mask[mask_y * MASK_W + mask_x];

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
