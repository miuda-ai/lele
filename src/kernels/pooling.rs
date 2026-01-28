pub fn adaptive_avg_pool1d(
    input: &[f32],
    output: &mut [f32],
    channels: usize,
    input_len: usize,
    output_len: usize,
) {
    assert_eq!(input.len(), channels * input_len);
    assert_eq!(output.len(), channels * output_len);
    for c in 0..channels {
        let in_offset = c * input_len;
        let out_offset = c * output_len;
        for i in 0..output_len {
            let start_idx_raw = (i * input_len) / output_len;
            let end_idx_raw = ((i + 1) * input_len).div_ceil(output_len);
            let start_idx = start_idx_raw.min(input_len);
            let end_idx = end_idx_raw.min(input_len);
            let kernel_len = end_idx - start_idx;
            if kernel_len == 0 {
                output[out_offset + i] = 0.0;
                continue;
            }
            let mut sum = 0.0;
            for k in start_idx..end_idx {
                sum += input[in_offset + k];
            }
            output[out_offset + i] = sum / kernel_len as f32;
        }
    }
}
