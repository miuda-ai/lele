/// Image processing for PP-OCRv6: detection + recognition preprocessing/postprocessing.
use std::path::Path;

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>, // RGB HWC
}

impl Image {
    pub fn load<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let img = ::image::open(path.as_ref())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let rgb = img.to_rgb8();
        Ok(Self {
            width: rgb.width() as usize,
            height: rgb.height() as usize,
            data: rgb.into_raw(),
        })
    }
}

/// Bilinear resize matching OpenCV (half-pixel centers). Returns RGB HWC u8.
fn bilinear_resize(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h {
        return src.to_vec();
    }
    let mut dst = vec![0u8; dst_w * dst_h * 3];
    let scale_x = src_w as f64 / dst_w as f64;
    let scale_y = src_h as f64 / dst_h as f64;

    for dy in 0..dst_h {
        let src_y_f = (dy as f64 + 0.5) * scale_y - 0.5;
        let sy0_i = src_y_f.floor();
        let fy = src_y_f - sy0_i;
        let sy0 = sy0_i.clamp(0.0, (src_h - 1) as f64) as usize;
        let sy1 = (sy0 + 1).min(src_h - 1);

        for dx in 0..dst_w {
            let src_x_f = (dx as f64 + 0.5) * scale_x - 0.5;
            let sx0_i = src_x_f.floor();
            let fx = src_x_f - sx0_i;
            let sx0 = sx0_i.clamp(0.0, (src_w - 1) as f64) as usize;
            let sx1 = (sx0 + 1).min(src_w - 1);

            let i00 = (sy0 * src_w + sx0) * 3;
            let i01 = (sy0 * src_w + sx1) * 3;
            let i10 = (sy1 * src_w + sx0) * 3;
            let i11 = (sy1 * src_w + sx1) * 3;
            let di = (dy * dst_w + dx) * 3;

            for c in 0..3 {
                let val = (1.0 - fx) * (1.0 - fy) * src[i00 + c] as f64
                    + fx * (1.0 - fy) * src[i01 + c] as f64
                    + (1.0 - fx) * fy * src[i10 + c] as f64
                    + fx * fy * src[i11 + c] as f64;
                dst[di + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    dst
}

pub struct DetInput {
    pub chw: Vec<f32>,
    pub resized_h: usize,
    pub resized_w: usize,
    pub ratio_h: f32,
    pub ratio_w: f32,
}

/// PP-OCRv6 detection preprocessing.
/// limit_side_len=960, round to 32, ImageNet normalize, BGR CHW.
pub fn det_preprocess(img: &Image, limit_side_len: usize) -> DetInput {
    let src_h = img.height;
    let src_w = img.width;
    let ratio = if src_h.max(src_w) > limit_side_len {
        limit_side_len as f64 / src_h.max(src_w) as f64
    } else {
        1.0
    };
    let mut new_h = (src_h as f64 * ratio).round() as usize;
    let mut new_w = (src_w as f64 * ratio).round() as usize;
    new_h = ((new_h + 31) / 32) * 32;
    new_w = ((new_w + 31) / 32) * 32;
    new_h = new_h.max(32);
    new_w = new_w.max(32);

    let resized = bilinear_resize(&img.data, src_w, src_h, new_w, new_h);
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    let mut chw = vec![0f32; 3 * new_h * new_w];
    for h in 0..new_h {
        for w in 0..new_w {
            let idx = (h * new_w + w) * 3;
            let b = resized[idx + 2] as f32 / 255.0;
            let g = resized[idx + 1] as f32 / 255.0;
            let r = resized[idx] as f32 / 255.0;
            chw[h * new_w + w] = (b - mean[0]) / std[0];
            chw[new_h * new_w + h * new_w + w] = (g - mean[1]) / std[1];
            chw[2 * new_h * new_w + h * new_w + w] = (r - mean[2]) / std[2];
        }
    }

    DetInput {
        chw,
        resized_h: new_h,
        resized_w: new_w,
        ratio_h: src_h as f32 / new_h as f32,
        ratio_w: src_w as f32 / new_w as f32,
    }
}

/// Connected component labeling (8-connectivity) via iterative flood fill.
fn connected_components(binary: &[u8], w: usize, h: usize) -> (Vec<i32>, i32) {
    let mut labels = vec![0i32; w * h];
    let mut stack: Vec<usize> = Vec::with_capacity(8192);
    let mut next_label = 1i32;

    for start in 0..w * h {
        if binary[start] != 0 && labels[start] == 0 {
            stack.clear();
            stack.push(start);
            labels[start] = next_label;
            while let Some(idx) = stack.pop() {
                let x = idx % w;
                let y = idx / w;
                // 8-connectivity
                for &dxy in &[
                    (-1i32, -1i32), (0, -1), (1, -1),
                    (-1, 0), (1, 0),
                    (-1, 1), (0, 1), (1, 1),
                ] {
                    let nx = x as i32 + dxy.0;
                    let ny = y as i32 + dxy.1;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let n = ny as usize * w + nx as usize;
                        if binary[n] != 0 && labels[n] == 0 {
                            labels[n] = next_label;
                            stack.push(n);
                        }
                    }
                }
            }
            next_label += 1;
        }
    }
    (labels, next_label - 1)
}

/// Convex hull via Andrew's monotone chain. Returns CCW hull.
fn convex_hull(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }
    let mut pts: Vec<(f64, f64)> = points.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap()));
    pts.dedup();
    let n = pts.len();
    if n < 3 {
        return pts;
    }

    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| -> f64 {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };

    let mut hull = Vec::with_capacity(2 * n);
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// Minimum area bounding rectangle via rotating calipers on convex hull.
/// Returns 4 corner points (clockwise from top-left).
fn min_area_rect(points: &[(f64, f64)]) -> [(f64, f64); 4] {
    let hull = convex_hull(points);
    if hull.is_empty() {
        return [(0.0, 0.0); 4];
    }
    if hull.len() < 3 {
        let p0 = hull[0];
        let p1 = hull.get(1).copied().unwrap_or(p0);
        return [p0, p1, p1, p0];
    }

    let mut best_area = f64::MAX;
    let mut best_cx = 0.0;
    let mut best_cy = 0.0;
    let mut best_w = 0.0;
    let mut best_h = 0.0;
    let mut best_ux = 1.0;
    let mut best_uy = 0.0;

    for i in 0..hull.len() {
        let j = (i + 1) % hull.len();
        let dx = hull[j].0 - hull[i].0;
        let dy = hull[j].1 - hull[i].1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-10 {
            continue;
        }
        let ux = dx / len;
        let uy = dy / len;
        let vx = -uy;
        let vy = ux;

        let mut min_u = f64::MAX;
        let mut max_u = f64::MIN;
        let mut min_v = f64::MAX;
        let mut max_v = f64::MIN;
        for &p in &hull {
            let du = (p.0 - hull[i].0) * ux + (p.1 - hull[i].1) * uy;
            let dv = (p.0 - hull[i].0) * vx + (p.1 - hull[i].1) * vy;
            min_u = min_u.min(du);
            max_u = max_u.max(du);
            min_v = min_v.min(dv);
            max_v = max_v.max(dv);
        }
        let w = max_u - min_u;
        let h = max_v - min_v;
        let area = w * h;
        if area < best_area {
            best_area = area;
            best_w = w;
            best_h = h;
            let cu = (min_u + max_u) / 2.0;
            let cv = (min_v + max_v) / 2.0;
            best_cx = hull[i].0 + cu * ux + cv * vx;
            best_cy = hull[i].1 + cu * uy + cv * vy;
            best_ux = ux;
            best_uy = uy;
        }
    }

    let vx = -best_uy;
    let vy = best_ux;
    let hw = best_w / 2.0;
    let hh = best_h / 2.0;
    [
        (best_cx - hw * best_ux - hh * vx, best_cy - hw * best_uy - hh * vy),
        (best_cx + hw * best_ux - hh * vx, best_cy + hw * best_uy - hh * vy),
        (best_cx + hw * best_ux + hh * vx, best_cy + hw * best_uy + hh * vy),
        (best_cx - hw * best_ux + hh * vx, best_cy - hw * best_uy + hh * vy),
    ]
}

/// Expand a 4-point box outward by offset distance (simplified unclip).
fn expand_box(box_pts: &[(f64, f64); 4], distance: f64) -> [(f64, f64); 4] {
    let cx: f64 = box_pts.iter().map(|p| p.0).sum::<f64>() / 4.0;
    let cy: f64 = box_pts.iter().map(|p| p.1).sum::<f64>() / 4.0;
    let mut result = [(0.0f64, 0.0); 4];

    for i in 0..4 {
        let prev = (i + 3) % 4;
        let next = (i + 1) % 4;

        // Normal of edge prev->i, pointing outward
        let e1x = box_pts[i].0 - box_pts[prev].0;
        let e1y = box_pts[i].1 - box_pts[prev].1;
        let l1 = (e1x * e1x + e1y * e1y).sqrt().max(1e-10);
        let mut n1x = e1y / l1;
        let mut n1y = -e1x / l1;
        if (n1x * (box_pts[i].0 - cx) + n1y * (box_pts[i].1 - cy)) < 0.0 {
            n1x = -n1x; n1y = -n1y;
        }

        // Normal of edge i->next
        let e2x = box_pts[next].0 - box_pts[i].0;
        let e2y = box_pts[next].1 - box_pts[i].1;
        let l2 = (e2x * e2x + e2y * e2y).sqrt().max(1e-10);
        let mut n2x = e2y / l2;
        let mut n2y = -e2x / l2;
        if (n2x * (box_pts[i].0 - cx) + n2y * (box_pts[i].1 - cy)) < 0.0 {
            n2x = -n2x; n2y = -n2y;
        }

        // Intersection of offset lines
        let p1 = (box_pts[prev].0 + n1x * distance, box_pts[prev].1 + n1y * distance);
        let p2 = (box_pts[i].0 + n1x * distance, box_pts[i].1 + n1y * distance);
        let p3 = (box_pts[i].0 + n2x * distance, box_pts[i].1 + n2y * distance);
        let p4 = (box_pts[next].0 + n2x * distance, box_pts[next].1 + n2y * distance);

        let d = (p1.0 - p2.0) * (p3.1 - p4.1) - (p1.1 - p2.1) * (p3.0 - p4.0);
        if d.abs() < 1e-10 {
            result[i] = p2;
        } else {
            let t = ((p1.0 - p3.0) * (p3.1 - p4.1) - (p1.1 - p3.1) * (p3.0 - p4.0)) / d;
            result[i] = (p1.0 + t * (p2.0 - p1.0), p1.1 + t * (p2.1 - p1.1));
        }
    }
    result
}

/// DB postprocess: find text boxes from segmentation map.
pub fn det_postprocess(
    pred: &[f32],
    out_h: usize,
    out_w: usize,
    ratio_h: f32,
    ratio_w: f32,
    src_h: usize,
    src_w: usize,
    thresh: f32,
    box_thresh: f32,
    unclip_ratio: f64,
    max_candidates: usize,
    min_size: usize,
) -> Vec<[[f32; 2]; 4]> {
    let mut binary = vec![0u8; out_h * out_w];
    for i in 0..out_h * out_w {
        if pred[i] > thresh {
            binary[i] = 1;
        }
    }

    let (labels, num_labels) = connected_components(&binary, out_w, out_h);
    let mut boxes = Vec::new();

    for label in 1..=num_labels {
        if boxes.len() >= max_candidates {
            break;
        }

        let mut pts: Vec<(f64, f64)> = Vec::new();
        let mut min_x = usize::MAX;
        let mut max_x = 0;
        let mut score_sum = 0.0f64;
        let mut count = 0usize;

        for idx in 0..out_h * out_w {
            if labels[idx] == label {
                let x = idx % out_w;
                pts.push((x as f64, (idx / out_w) as f64));
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                score_sum += pred[idx] as f64;
                count += 1;
            }
        }
        if count == 0 || (max_x - min_x + 1) < 2 {
            continue;
        }

        let score = score_sum / count as f64;
        if score < box_thresh as f64 {
            continue;
        }

        let rect = min_area_rect(&pts);
        let rw = ((rect[1].0 - rect[0].0).powi(2) + (rect[1].1 - rect[0].1).powi(2)).sqrt();
        let rh = ((rect[2].0 - rect[1].0).powi(2) + (rect[2].1 - rect[1].1).powi(2)).sqrt();
        if rw.min(rh) < min_size as f64 {
            continue;
        }

        // Compute area/perimeter for unclip distance
        let area = rw * rh;
        let perimeter = 2.0 * (rw + rh);
        let distance = if perimeter > 0.0 { area / perimeter * unclip_ratio } else { 0.0 };

        let expanded = expand_box(&rect, distance);
        let final_rect = min_area_rect(&expanded.to_vec());

        let mut box_pts = [[0f32; 2]; 4];
        for i in 0..4 {
            box_pts[i][0] = (final_rect[i].0 * ratio_w as f64) as f32;
            box_pts[i][1] = (final_rect[i].1 * ratio_h as f64) as f32;
            box_pts[i][0] = box_pts[i][0].clamp(0.0, src_w as f32);
            box_pts[i][1] = box_pts[i][1].clamp(0.0, src_h as f32);
        }
        boxes.push(box_pts);
    }

    // Sort top-to-bottom then left-to-right (round y to nearest 10px for line grouping)
    boxes.sort_by(|a, b| {
        let ay = ((a[0][1] + a[1][1] + a[2][1] + a[3][1]) * 0.25 / 10.0).round() as i64;
        let by = ((b[0][1] + b[1][1] + b[2][1] + b[3][1]) * 0.25 / 10.0).round() as i64;
        let ax = ((a[0][0] + a[1][0] + a[2][0] + a[3][0]) * 0.25).round() as i64;
        let bx = ((b[0][0] + b[1][0] + b[2][0] + b[3][0]) * 0.25).round() as i64;
        ay.cmp(&by).then(ax.cmp(&bx))
    });

    boxes
}

/// Crop axis-aligned region from image.
pub fn crop_text_region(
    img_data: &[u8],
    img_w: usize,
    box_pts: &[[f32; 2]; 4],
) -> (Vec<u8>, usize, usize) {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for p in box_pts {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    let x0 = min_x.floor().max(0.0) as usize;
    let y0 = min_y.floor().max(0.0) as usize;
    let x1 = max_x.ceil().min(img_w as f32) as usize;
    let y1 = (max_y.ceil()) as usize;
    let crop_w = x1 - x0;
    let crop_h = y1 - y0;
    if crop_w == 0 || crop_h == 0 {
        return (vec![], 0, 0);
    }
    let mut data = vec![0u8; crop_w * crop_h * 3];
    for y in 0..crop_h {
        for x in 0..crop_w {
            let si = ((y0 + y) * img_w + (x0 + x)) * 3;
            let di = (y * crop_w + x) * 3;
            data[di..di + 3].copy_from_slice(&img_data[si..si + 3]);
        }
    }
    (data, crop_w, crop_h)
}

/// REC preprocessing: resize to height=imgH, dynamic width, BGR CHW normalized [-1,1].
pub fn rec_preprocess(region: &[u8], rw: usize, rh: usize, img_h: usize) -> (Vec<f32>, usize) {
    if rw == 0 || rh == 0 {
        return (vec![], 0);
    }
    let ratio = rw as f64 / rh as f64;
    let mut new_w = (img_h as f64 * ratio).round() as usize;
    new_w = ((new_w + 2) / 4) * 4;
    new_w = new_w.max(160);

    let resized = bilinear_resize(region, rw, rh, new_w, img_h);
    let mut chw = vec![0f32; 3 * img_h * new_w];
    for h in 0..img_h {
        for w in 0..new_w {
            let idx = (h * new_w + w) * 3;
            // RGB input -> BGR CHW, normalize [-1,1]
            chw[h * new_w + w] = resized[idx + 2] as f32 / 127.5 - 1.0; // B
            chw[img_h * new_w + h * new_w + w] = resized[idx + 1] as f32 / 127.5 - 1.0; // G
            chw[2 * img_h * new_w + h * new_w + w] = resized[idx] as f32 / 127.5 - 1.0; // R
        }
    }
    (chw, new_w)
}

/// CTC decode: argmax, deduplicate, skip blanks.
pub fn rec_postprocess(output: &[f32], seq_len: usize, num_classes: usize, dict: &[&str]) -> (String, f32) {
    let mut prev: i32 = -1;
    let mut text = String::new();
    let mut conf_sum = 0.0f32;
    let mut conf_n = 0usize;

    for t in 0..seq_len {
        let off = t * num_classes;
        let row = &output[off..off + num_classes];
        let (best_idx, best_val) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &v)| (i, v))
            .unwrap_or((0, 0.0));

        if best_idx != 0 && best_idx != prev as usize {
            if best_idx > 0 && best_idx - 1 < dict.len() {
                text.push_str(dict[best_idx - 1]);
            }
            conf_sum += best_val;
            conf_n += 1;
        }
        prev = best_idx as i32;
    }

    (text, if conf_n > 0 { conf_sum / conf_n as f32 } else { 0.0 })
}
