fn main() -> lele_build::Result {
    lele_build::build_model("../../../../examples/yolo26/model.toml", "src/gen")
}
