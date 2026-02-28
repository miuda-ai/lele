fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/gen/yolo26.rs");
    println!("cargo:rerun-if-changed=src/gen/yolo26_weights.bin");
}
