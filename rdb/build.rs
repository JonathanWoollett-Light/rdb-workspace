fn main() {
    #[cfg(not(target_os = "linux"))]
    compile_error!("This library only supports linux");
    #[cfg(all(feature = "bincode", feature = "json"))]
    compile_error!("feature \"bincode\" and feature \"json\" cannot be enabled at the time time")
}
