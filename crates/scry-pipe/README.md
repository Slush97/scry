# scry-pipe

Cross-language feature engineering pipeline compiler.

Define ML feature pipelines once in Python, compile to standalone
zero-dependency Rust or WASM binaries with all fitted parameters baked in.

## Features

- JSON-serializable pipeline IR with all fitted parameters
- Runtime execution engine (`PipelineEngine`)
- Rust code generation from pipeline definitions (feature `codegen`)
- Transforms: StandardScale, MinMaxScale, RobustScale, Clip, Log1p, Impute, LabelEncode, OneHotEncode, BinDiscretize, Polynomial
- Exact numerical parity between training (Python) and serving (Rust/WASM)

## Quick Start

```rust
use scry_pipe::{PipelineDef, PipelineEngine};

let pipeline = PipelineDef::from_json(json_str)?;
let engine = PipelineEngine::new(&pipeline);

let input = vec![35.0, 50000.0, 2.0];
let output = engine.transform(&input)?;
```

## License

MIT OR Apache-2.0
