/// Image preprocessing and postprocessing for YOLO26.
use std::path::Path;

/// COCO class names (80 classes)
pub const COCO_CLASSES: &[&str] = &[
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
];

/// Detection result from YOLO26.
#[derive(Debug, Clone)]
pub struct Detection {
    pub class_id: usize,
    pub class_name: String,
    pub score: f32,
    /// Bounding box in [x1, y1, x2, y2] format (pixel coordinates)
    pub bbox: [f32; 4],
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
        Ok(Self { width, height, data })
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
                let src_x = ((x as f32 + 0.5) * self.width as f32 / new_width as f32).floor() as usize;
                let src_y = ((y as f32 + 0.5) * self.height as f32 / new_height as f32).floor() as usize;
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
    
    /// Create image from raw RGB data.
    pub fn from_rgb(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), width * height * 3);
        Self { width, height, data }
    }
}

/// Post-process YOLO26 DETR-style outputs (logits + pred_boxes).
/// logits: [1, 300, 80]  raw logits
/// pred_boxes: [1, 300, 4]  normalized [cx, cy, w, h] in [0,1]
/// Returns detections filtered by threshold.
pub fn postprocess(
    logits: &[f32],
    pred_boxes: &[f32],
    img_width: usize,
    img_height: usize,
    threshold: f32,
) -> Vec<Detection> {
    const NUM_QUERIES: usize = 300;
    const NUM_CLASSES: usize = 80;
    
    let mut detections = Vec::new();
    
    for i in 0..NUM_QUERIES {
        // Find max class score after sigmoid
        let logit_offset = i * NUM_CLASSES;
        let mut max_score = f32::NEG_INFINITY;
        let mut max_class = 0;
        
        for c in 0..NUM_CLASSES {
            let logit = logits[logit_offset + c];
            let score = 1.0 / (1.0 + (-logit).exp()); // sigmoid
            if score > max_score {
                max_score = score;
                max_class = c;
            }
        }
        
        if max_score < threshold {
            continue;
        }
        
        // Get bbox (normalized cx, cy, w, h)
        let box_offset = i * 4;
        let cx = pred_boxes[box_offset];
        let cy = pred_boxes[box_offset + 1];
        let w = pred_boxes[box_offset + 2];
        let h = pred_boxes[box_offset + 3];
        
        // Convert to pixel coordinates [x1, y1, x2, y2]
        let x1 = ((cx - w / 2.0) * img_width as f32).max(0.0);
        let y1 = ((cy - h / 2.0) * img_height as f32).max(0.0);
        let x2 = ((cx + w / 2.0) * img_width as f32).min(img_width as f32);
        let y2 = ((cy + h / 2.0) * img_height as f32).min(img_height as f32);
        
        detections.push(Detection {
            class_id: max_class,
            class_name: COCO_CLASSES[max_class].to_string(),
            score: max_score,
            bbox: [x1, y1, x2, y2],
        });
    }
    
    detections
}
