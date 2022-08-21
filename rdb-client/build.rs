fn main() {
    #[cfg(all(feature = "bincode", feature = "json"))]
    compile_error!("feature \"bincode\" and feature \"json\" cannot be enabled at the time time")
}
