use std::collections::HashMap;
use std::path::Path;

/// GPT-2 byte-level BPE tokenizer.
///
/// Implements the same algorithm as the original OpenAI GPT-2 tokenizer:
/// byte-to-unicode mapping, pre-tokenization regex, then iterative BPE merges.
pub struct BpeTokenizer {
    encoder: HashMap<String, usize>,
    decoder: HashMap<usize, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
}

/// Build the GPT-2 `bytes_to_unicode()` mapping.
///
/// This creates a bijection from all 256 byte values to Unicode characters,
/// avoiding control characters and whitespace that cause issues in BPE.
fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    // printable ASCII ranges
    bs.extend(b'!'..=b'~');
    bs.extend(0xa1u8..=0xac);
    bs.extend(0xaeu8..=0xff);

    let mut cs: Vec<u32> = bs.iter().map(|&b| u32::from(b)).collect();
    let mut n: u32 = 0;
    for b in 0u16..=255 {
        let b = b as u8;
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::new();
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).expect("invalid unicode codepoint in bytes_to_unicode");
        encoder[b as usize] = ch;
        decoder.insert(ch, b);
    }

    (encoder, decoder)
}

/// GPT-2 pre-tokenization: split text into chunks that BPE processes independently.
///
/// Matches the pattern: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
///
/// Hand-rolled to avoid a `regex` dependency.
fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;

    while i < n {
        // Try contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if chars[i] == '\'' && i + 1 < n {
            let next = chars[i + 1].to_ascii_lowercase();
            match next {
                's' | 't' | 'm' | 'd' => {
                    tokens.push(chars[i..i + 2].iter().collect());
                    i += 2;
                    continue;
                }
                'r' if i + 2 < n && chars[i + 2].to_ascii_lowercase() == 'e' => {
                    tokens.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
                'v' if i + 2 < n && chars[i + 2].to_ascii_lowercase() == 'e' => {
                    tokens.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
                'l' if i + 2 < n && chars[i + 2].to_ascii_lowercase() == 'l' => {
                    tokens.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
                _ => {}
            }
        }

        // ` ?\p{L}+` — optional space then letters
        if (chars[i] == ' ' && i + 1 < n && chars[i + 1].is_alphabetic())
            || chars[i].is_alphabetic()
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && chars[i].is_alphabetic() {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // ` ?\p{N}+` — optional space then digits
        if (chars[i] == ' ' && i + 1 < n && chars[i + 1].is_ascii_digit())
            || chars[i].is_ascii_digit()
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n && chars[i].is_ascii_digit() {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+` — optional space then non-alnum-non-whitespace
        if (chars[i] == ' '
            && i + 1 < n
            && !chars[i + 1].is_whitespace()
            && !chars[i + 1].is_alphabetic()
            && !chars[i + 1].is_ascii_digit())
            || (!chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_ascii_digit())
        {
            let start = i;
            if chars[i] == ' ' {
                i += 1;
            }
            while i < n
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_ascii_digit()
            {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // `\s+(?!\S)|\s+` — whitespace (trailing whitespace without non-ws lookahead)
        if chars[i].is_whitespace() {
            let start = i;
            while i < n && chars[i].is_whitespace() {
                i += 1;
            }
            // If there's a non-whitespace char following, keep last ws char separate
            // (it'll be consumed as the optional space in the next token)
            if i < n && (i - start) > 1 {
                // Emit all but last whitespace
                tokens.push(chars[start..i - 1].iter().collect());
                i -= 1; // re-process the last whitespace
            } else {
                tokens.push(chars[start..i].iter().collect());
            }
            continue;
        }

        // Fallback: single character
        tokens.push(chars[i].to_string());
        i += 1;
    }

    tokens
}

impl BpeTokenizer {
    /// Load a GPT-2 tokenizer from `vocab.json` and `merges.txt` files.
    ///
    /// # Errors
    ///
    /// Returns an error if files cannot be read or parsed.
    pub fn from_files(vocab_path: &Path, merges_path: &Path) -> Result<Self, String> {
        let vocab_data = std::fs::read_to_string(vocab_path)
            .map_err(|e| format!("failed to read vocab file: {e}"))?;
        let merges_data = std::fs::read_to_string(merges_path)
            .map_err(|e| format!("failed to read merges file: {e}"))?;

        let encoder: HashMap<String, usize> = serde_json::from_str(&vocab_data)
            .map_err(|e| format!("failed to parse vocab.json: {e}"))?;

        let decoder: HashMap<usize, String> = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        let mut bpe_ranks = HashMap::new();
        for (rank, line) in merges_data.lines().enumerate() {
            // Skip header line (starts with #)
            if line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split(' ').collect();
            if parts.len() == 2 {
                let actual_rank = if merges_data.lines().next().map_or(false, |l| l.starts_with('#')) {
                    rank - 1
                } else {
                    rank
                };
                bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), actual_rank);
            }
        }

        let (byte_encoder, byte_decoder) = bytes_to_unicode();

        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut token_ids = Vec::new();

        for chunk in pre_tokenize(text) {
            // Convert bytes to unicode representation
            let bpe_input: String = chunk.bytes().map(|b| self.byte_encoder[b as usize]).collect();

            // Run BPE on this word
            let bpe_tokens = self.bpe(&bpe_input);

            for tok in bpe_tokens {
                if let Some(&id) = self.encoder.get(&tok) {
                    token_ids.push(id);
                }
            }
        }

        token_ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let text: String = tokens
            .iter()
            .filter_map(|&id| self.decoder.get(&id))
            .cloned()
            .collect();

        // Convert from unicode representation back to raw bytes, then interpret as UTF-8
        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Vocabulary size (typically 50257 for GPT-2).
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// The BPE merge loop for a single word (already converted to unicode chars).
    fn bpe(&self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return Vec::new();
        }

        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

        if word.len() == 1 {
            return word;
        }

        loop {
            // Find the pair with the lowest rank
            let mut best_pair = None;
            let mut best_rank = usize::MAX;

            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(pair);
                    }
                }
            }

            let Some((first, second)) = best_pair else {
                break;
            };

            // Merge all occurrences of this pair
            let merged = format!("{first}{second}");
            let mut new_word = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
                    new_word.push(merged.clone());
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;

            if word.len() == 1 {
                break;
            }
        }

        word
    }
}
