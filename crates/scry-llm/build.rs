fn main() {
    // When the `blas` feature is enabled, link against the system CBLAS library.
    #[cfg(feature = "blas")]
    {
        println!("cargo:rustc-link-lib=dylib=cblas");
    }
}
