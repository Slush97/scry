// SPDX-License-Identifier: MIT OR Apache-2.0
//! Text utilities for chart label overflow handling.
//!
//! Provides word-boundary wrapping and truncation-with-ellipsis for chart
//! titles and axis labels that exceed their available pixel width.

/// Wrap `text` at word boundaries so no line exceeds `max_width_px`.
///
/// `char_width` is the average character width in pixels (e.g. from
/// [`char_width_for_size`](crate::layout::char_width_for_size)).
///
/// Returns one or more lines. If a single word is wider than `max_width_px`,
/// it occupies its own line un-broken (hard truncation is left to
/// [`ellipsize`]).
///
/// # Examples
///
/// ```
/// use scry_chart::text_utils::wrap_text;
///
/// let lines = wrap_text("Hello World", 50.0, 6.5);
/// assert_eq!(lines, vec!["Hello", "World"]);
///
/// let lines = wrap_text("Hi", 50.0, 6.5);
/// assert_eq!(lines, vec!["Hi"]);
/// ```
#[must_use]
pub fn wrap_text(text: &str, max_width_px: f32, char_width: f32) -> Vec<String> {
    if text.is_empty() || max_width_px <= 0.0 || char_width <= 0.0 {
        return vec![text.to_string()];
    }

    let max_chars = (max_width_px / char_width).floor().max(1.0) as usize;
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in words {
        if current_line.is_empty() {
            // First word on the line — always add it, even if too long.
            current_line.push_str(word);
        } else if current_line.len() + 1 + word.len() <= max_chars {
            // Fits on the current line.
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            // Doesn't fit — start a new line.
            lines.push(current_line);
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    lines
}

/// Truncate `text` with `…` if it exceeds `max_width_px`.
///
/// `char_width` is the average character width in pixels.
///
/// # Examples
///
/// ```
/// use scry_chart::text_utils::ellipsize;
///
/// let s = ellipsize("A very long label", 50.0, 6.5);
/// assert!(s.ends_with('…'));
/// assert!(s.chars().count() <= 8); // ≈ 50/6.5 = 7.69
///
/// let s = ellipsize("Short", 100.0, 6.5);
/// assert_eq!(s, "Short");
/// ```
#[must_use]
pub fn ellipsize(text: &str, max_width_px: f32, char_width: f32) -> String {
    if text.is_empty() || max_width_px <= 0.0 || char_width <= 0.0 {
        return text.to_string();
    }

    let max_chars = (max_width_px / char_width).floor() as usize;
    let text_chars: usize = text.chars().count();

    if text_chars <= max_chars {
        return text.to_string();
    }

    if max_chars <= 1 {
        return "…".to_string();
    }

    // Keep max_chars - 1 characters and append ellipsis.
    let truncated: String = text.chars().take(max_chars - 1).collect();
    format!("{truncated}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_short_text_unchanged() {
        let lines = wrap_text("Hi", 100.0, 6.5);
        assert_eq!(lines, vec!["Hi"]);
    }

    #[test]
    fn wrap_splits_at_word_boundary() {
        let lines = wrap_text("Hello World Foo", 60.0, 6.5);
        // 60/6.5 ≈ 9 chars max per line
        assert_eq!(lines, vec!["Hello", "World Foo"]);
    }

    #[test]
    fn wrap_empty_string() {
        let lines = wrap_text("", 100.0, 6.5);
        assert_eq!(lines, vec![""]);
    }

    #[test]
    fn wrap_single_long_word() {
        let lines = wrap_text("Supercalifragilistic", 40.0, 6.5);
        // Can't break the word — it goes on its own line.
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "Supercalifragilistic");
    }

    #[test]
    fn ellipsize_short_unchanged() {
        assert_eq!(ellipsize("Hi", 100.0, 6.5), "Hi");
    }

    #[test]
    fn ellipsize_truncates_with_ellipsis() {
        let s = ellipsize("A very long label text", 50.0, 6.5);
        assert!(s.ends_with('…'));
        assert!(s.chars().count() <= 8);
    }

    #[test]
    fn ellipsize_empty_string() {
        assert_eq!(ellipsize("", 50.0, 6.5), "");
    }

    #[test]
    fn ellipsize_max_one_char() {
        assert_eq!(ellipsize("Hello", 6.5, 6.5), "…");
    }

    #[test]
    fn ellipsize_exact_fit() {
        // "Hello" = 5 chars, max = 5 chars → no truncation
        assert_eq!(ellipsize("Hello", 32.5, 6.5), "Hello");
    }
}
