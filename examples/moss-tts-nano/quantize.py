#!/usr/bin/env python3
"""Weight-only int8 quantization for the MOSS-TTS-Nano ONNX models.

Produces the `DynamicQuantizeLinear` + `MatMulInteger` shape that lele lowers to
its int8 GEMM kernels (`linear_quantized_arm` on aarch64). MatMul/Gemm/Conv
weights shrink ~4x; embeddings (Gather), attention (Einsum) and sampling stay
f32.

Requires onnx + onnxruntime (`pip install onnx onnxruntime`).

Usage:
    quantize.py <input.onnx> <output.onnx> [--per-channel]

Before quantizing, `expose_weights` folds the Identity / constant-Transpose
wrappers sitting between the initializers and the MatMuls. Without that, the
quantizer only sees the handful of matmuls whose weight is a direct
initializer. It deliberately does *not* use `onnxruntime.quantization.preprocess`:
that runs ORT's own optimizer, which also drops no-op int64 casts. lele keeps
TopK indices as f32 internally and depends on one such cast, so the aggressively
optimized graph no longer type-checks.
"""

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic


def _subgraphs(graph):
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield attr.g
            elif attr.type == onnx.AttributeProto.GRAPHS:
                yield from attr.graphs


def _all_graphs(graph):
    yield graph
    for sub in _subgraphs(graph):
        yield from _all_graphs(sub)


def _rename(graph, old, new):
    """Rewrite every use of `old` to `new` in this graph and its subgraphs."""
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
    for out in graph.output:
        if out.name == old:
            out.name = new
    for sub in _subgraphs(graph):
        _rename(sub, old, new)


def _expose_weights(root):
    """Fold Identity and constant Transpose so the matmul weights are direct
    initializers. Purely structural: no dtype is changed, unlike ORT's own
    optimizer pass."""
    for graph in _all_graphs(root):
        inits = {i.name: i for i in graph.initializer}
        graph_outputs = {o.name for o in graph.output}

        changed = True
        while changed:
            changed = False

            for node in list(graph.node):
                if node.op_type != "Identity" or node.output[0] in graph_outputs:
                    continue
                _rename(graph, node.output[0], node.input[0])
                graph.node.remove(node)
                changed = True

            for node in list(graph.node):
                if node.op_type != "Transpose" or node.input[0] not in inits:
                    continue
                if node.output[0] in graph_outputs:
                    continue
                perm = next((list(a.ints) for a in node.attribute if a.name == "perm"), None)
                arr = numpy_helper.to_array(inits[node.input[0]])
                folded = np.ascontiguousarray(np.transpose(arr, perm))
                new = numpy_helper.from_array(folded, node.output[0])
                graph.initializer.append(new)
                inits[node.output[0]] = new
                graph.node.remove(node)
                changed = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--per-channel", action="store_true")
    args = parser.parse_args()

    model = onnx.load(args.input)
    _expose_weights(model.graph)

    inits = {i.name for i in model.graph.initializer}
    direct = sum(
        1 for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in inits
    )
    total = sum(1 for n in model.graph.node if n.op_type == "MatMul")
    print(f"[quantize] {os.path.basename(args.input)}: MatMul {total}, weight-initializer {direct}")

    # MatMul/Gemm only. The codec's grouped Conv lowers to a ConvInteger path
    # lele does not implement, and conv is under 2 MB there anyway.
    quantize_dynamic(
        model,
        args.output,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=args.per_channel,
        weight_type=QuantType.QUInt8,
    )
    print(f"[quantize] wrote {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
