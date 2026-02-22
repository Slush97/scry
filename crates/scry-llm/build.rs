fn main() {
    // When the `dnnl` feature is enabled, link against Intel oneDNN.
    #[cfg(feature = "dnnl")]
    {
        println!("cargo:rustc-link-lib=dylib=dnnl");
    }

    // When the `blas` feature is enabled, link against the system CBLAS library.
    #[cfg(feature = "blas")]
    {
        println!("cargo:rustc-link-lib=dylib=cblas");
    }
}
