//! Fuzz target: Text vectorizers.
//!
//! Interprets fuzz bytes as UTF-8 documents and exercises CountVectorizer
//! and TfidfVectorizer. Tests empty docs, single-char docs, all-same content.

#![no_main]

use libfuzzer_sys::fuzz_target;
use scry_learn::text::{CountVectorizer, TfidfVectorizer};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let dispatch = data[0] % 2;

    // Interpret remaining bytes as UTF-8 text, split into documents on newline.
    let text = String::from_utf8_lossy(&data[1..]);
    let documents: Vec<&str> = text.split('\n').collect();

    if documents.is_empty() {
        return;
    }

    match dispatch {
        0 => {
            // CountVectorizer
            let mut cv = CountVectorizer::new();
            cv.fit(&documents);
            if cv.is_fitted() {
                let _ = cv.transform(&documents);
                let _ = cv.get_feature_names();
                let _ = cv.n_features();
            }
        }
        _ => {
            // TfidfVectorizer
            let mut tv = TfidfVectorizer::new();
            let _ = tv.fit_transform(&documents);
            // After fit_transform, always fitted — exercise transform.
            let _ = tv.transform(&documents);
            let _ = tv.get_feature_names();
        }
    }
});
