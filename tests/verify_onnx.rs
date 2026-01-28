use lele::model::onnx_proto::{GraphProto, ModelProto, NodeProto};
use prost::Message;

#[test]
fn test_onnx_roundtrip() {
    let mut model = ModelProto::default();
    model.producer_name = "lele-test".to_string();
    model.ir_version = 8;

    let mut graph = GraphProto::default();
    graph.name = "test-graph".to_string();

    let mut node = NodeProto::default();
    node.name = "node1".to_string();
    node.op_type = "Conv".to_string();
    node.input.push("X".to_string());
    node.output.push("Y".to_string());

    graph.node.push(node);
    model.graph = Some(graph);

    // Serialize
    let mut buf = Vec::new();
    model.encode(&mut buf).unwrap();

    // Deserialize
    let decoded = ModelProto::decode(&buf[..]).unwrap();

    assert_eq!(decoded.producer_name, "lele-test");
    assert_eq!(decoded.ir_version, 8);
    let g = decoded.graph.unwrap();
    assert_eq!(g.name, "test-graph");
    assert_eq!(g.node.len(), 1);
    assert_eq!(g.node[0].op_type, "Conv");
}
