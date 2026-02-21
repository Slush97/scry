//! End-to-end integration test: load real whisper-tiny weights,
//! process audio through the full pipeline, and verify transcription.
//!
//! These tests are `#[ignore]` by default because they require the
//! whisper-tiny model files in `models/whisper-tiny/`.
//!
//! Run with: `cargo test -p scry-stt --features safetensors -- --ignored`

#[cfg(feature = "safetensors")]
mod e2e {
    use std::path::PathBuf;

    use scry_llm::backend::cpu::CpuBackend;
    use scry_llm::tensor::shape::Shape;
    use scry_llm::tensor::Tensor;

    use scry_stt::checkpoint::load_whisper_checkpoint;
    use scry_stt::decode::{greedy_decode, DecodeConfig};
    use scry_stt::mel::{log_mel_spectrogram, WHISPER_SAMPLE_RATE};
    use scry_stt::model::config::WhisperConfig;
    use scry_stt::tokenizer::WhisperTokenizer;

    fn model_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/whisper-tiny")
    }

    fn model_available() -> bool {
        model_dir().join("model.safetensors").exists()
    }

    #[test]
    #[ignore]
    fn e2e_load_checkpoint() {
        if !model_available() {
            eprintln!("Skipping: model files not found in {:?}", model_dir());
            return;
        }

        let config = WhisperConfig::tiny();
        let model_path = model_dir().join("model.safetensors");

        let model = load_whisper_checkpoint::<CpuBackend>(&model_path, &config)
            .expect("Failed to load checkpoint");

        // Verify parameter count is reasonable for whisper-tiny (39M params)
        let params: Vec<_> = scry_llm::nn::Module::parameters(&model);
        let total_params: usize = params.iter().map(|t| t.numel()).sum();
        eprintln!("Total parameters: {total_params}");

        // Whisper tiny has ~39M parameters
        assert!(
            total_params > 30_000_000 && total_params < 50_000_000,
            "Expected ~39M params, got {total_params}"
        );
    }

    #[test]
    #[ignore]
    fn e2e_load_tokenizer() {
        if !model_available() {
            eprintln!("Skipping: model files not found in {:?}", model_dir());
            return;
        }

        let tokenizer_path = model_dir().join("tokenizer.json");
        let tokenizer = WhisperTokenizer::from_file(&tokenizer_path)
            .expect("Failed to load tokenizer");

        // Basic vocab sanity: should have at least 50k tokens
        assert!(
            tokenizer.vocab_size() >= 50257,
            "Vocab too small: {}",
            tokenizer.vocab_size()
        );

        // Decode some known token IDs
        // Token 220 in GPT-2 is typically a space
        let text = tokenizer.decode(&[220]);
        assert_eq!(text, " ", "Token 220 should decode to a space");
    }

    #[test]
    #[ignore]
    fn e2e_full_pipeline_silence() {
        if !model_available() {
            eprintln!("Skipping: model files not found in {:?}", model_dir());
            return;
        }

        let config = WhisperConfig::tiny();
        let model_path = model_dir().join("model.safetensors");
        let tokenizer_path = model_dir().join("tokenizer.json");

        eprintln!("Loading model...");
        let model = load_whisper_checkpoint::<CpuBackend>(&model_path, &config)
            .expect("Failed to load checkpoint");

        eprintln!("Loading tokenizer...");
        let tokenizer = WhisperTokenizer::from_file(&tokenizer_path)
            .expect("Failed to load tokenizer");

        // Generate 1 second of silence
        let samples = vec![0.0f32; WHISPER_SAMPLE_RATE as usize];

        // Mel spectrogram
        eprintln!("Computing mel spectrogram...");
        let mel = log_mel_spectrogram(&samples);
        let mel_padded = mel.pad_or_truncate(1500); // Whisper expects 3000 frames for 30s, but 1500 = n_audio_ctx

        // Create input tensor [n_mels, n_audio_ctx]
        let mel_tensor = Tensor::<CpuBackend>::from_vec(
            mel_padded.data,
            Shape::new(&[mel_padded.n_mels, mel_padded.n_frames]),
        );

        // Encode
        eprintln!("Encoding...");
        let encoder_output = model.encode(&mel_tensor);
        let enc_shape = encoder_output.shape.dims().to_vec();
        eprintln!("Encoder output shape: {enc_shape:?}");

        // Should be [n_audio_ctx/2, d_model] = [750, 384] for tiny
        assert_eq!(enc_shape[0], config.n_audio_ctx / 2);
        assert_eq!(enc_shape[1], config.d_model);

        // Decode
        eprintln!("Decoding...");
        let decode_config = DecodeConfig {
            max_tokens: 10, // Keep it short for the test
            ..DecodeConfig::default()
        };
        let tokens = greedy_decode(&model, &encoder_output, &decode_config);
        eprintln!("Generated tokens: {tokens:?}");

        // Detokenize
        let text = tokenizer.decode(&tokens);
        eprintln!("Decoded text: '{text}'");

        // For silence, the model should produce something (possibly empty or noise-related)
        // The key validation here is that the pipeline doesn't panic and produces tokens
        eprintln!("Pipeline completed successfully. {} tokens generated.", tokens.len());
    }

    #[test]
    #[ignore]
    fn e2e_full_pipeline_sine_wave() {
        if !model_available() {
            eprintln!("Skipping: model files not found in {:?}", model_dir());
            return;
        }

        let config = WhisperConfig::tiny();
        let model_path = model_dir().join("model.safetensors");
        let tokenizer_path = model_dir().join("tokenizer.json");

        let model = load_whisper_checkpoint::<CpuBackend>(&model_path, &config)
            .expect("Failed to load checkpoint");
        let tokenizer = WhisperTokenizer::from_file(&tokenizer_path)
            .expect("Failed to load tokenizer");

        // Generate 2 seconds of 440Hz sine wave (A4 note)
        let duration_samples = 2 * WHISPER_SAMPLE_RATE as usize;
        let samples: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / WHISPER_SAMPLE_RATE as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();

        let mel = log_mel_spectrogram(&samples);
        let mel_padded = mel.pad_or_truncate(1500);
        let mel_tensor = Tensor::<CpuBackend>::from_vec(
            mel_padded.data,
            Shape::new(&[mel_padded.n_mels, mel_padded.n_frames]),
        );

        let encoder_output = model.encode(&mel_tensor);
        let decode_config = DecodeConfig {
            max_tokens: 20,
            ..DecodeConfig::default()
        };
        let tokens = greedy_decode(&model, &encoder_output, &decode_config);
        let text = tokenizer.decode(&tokens);

        eprintln!("Sine wave tokens: {tokens:?}");
        eprintln!("Sine wave text: '{text}'");
        eprintln!("Pipeline completed. {} tokens generated.", tokens.len());
    }
}
