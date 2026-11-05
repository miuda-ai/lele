fn main() -> lele_build::Result {
    lele_build::build_model("../../../../examples/yolo26n-seg/model.toml", "src/gen")
}
