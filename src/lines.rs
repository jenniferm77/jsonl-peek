//! Streaming line reader.
//!
//! Everything in this crate is single pass: a JSONL training set is routinely
//! larger than RAM, so lines are read into one reusable buffer and handed out
//! by reference. The reader normalises the things that break naive `split('\n')`
//! pipelines: CRLF endings, a UTF-8 BOM on the first line and a final line
//! without a trailing newline.

use std::io::{self, BufRead};

/// One line of input.
#[derive(Debug)]
pub struct Line<'a> {
    /// 1-based line number.
    pub number: u64,
    /// Line content with the BOM and the end-of-line marker removed.
    pub bytes: &'a [u8],
    /// Bytes consumed from the file for this line, including the line ending.
    pub raw_len: usize,
}

impl Line<'_> {
    /// True when the line has no content at all.
    pub fn is_blank(&self) -> bool {
        self.bytes.iter().all(|b| b.is_ascii_whitespace())
    }
}

/// Reads a stream one line at a time using a single reusable buffer.
#[derive(Debug)]
pub struct LineReader<R: BufRead> {
    inner: R,
    buf: Vec<u8>,
    number: u64,
    bytes_read: u64,
    at_start: bool,
}

impl<R: BufRead> LineReader<R> {
    /// Wraps a buffered reader.
    pub fn new(inner: R) -> Self {
        LineReader {
            inner,
            buf: Vec::with_capacity(8 * 1024),
            number: 0,
            bytes_read: 0,
            at_start: true,
        }
    }

    /// Total number of bytes consumed so far, line endings included.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Number of lines returned so far.
    pub fn lines_read(&self) -> u64 {
        self.number
    }

    /// Reads the next line, or `Ok(None)` at end of input.
    pub fn next_line(&mut self) -> io::Result<Option<Line<'_>>> {
        self.buf.clear();
        let raw_len = self.inner.read_until(b'\n', &mut self.buf)?;
        if raw_len == 0 {
            return Ok(None);
        }
        self.bytes_read += raw_len as u64;
        self.number += 1;

        let mut start = 0usize;
        if self.at_start {
            self.at_start = false;
            if self.buf.starts_with(&[0xEF, 0xBB, 0xBF]) {
                start = 3;
            }
        }
        let mut end = self.buf.len();
        if end > start && self.buf[end - 1] == b'\n' {
            end -= 1;
            if end > start && self.buf[end - 1] == b'\r' {
                end -= 1;
            }
        }

        Ok(Some(Line {
            number: self.number,
            bytes: &self.buf[start..end],
            raw_len,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(input: &[u8]) -> Vec<(u64, String, usize)> {
        let mut reader = LineReader::new(input);
        let mut out = Vec::new();
        while let Some(line) = reader.next_line().unwrap() {
            out.push((
                line.number,
                String::from_utf8_lossy(line.bytes).into_owned(),
                line.raw_len,
            ));
        }
        assert_eq!(reader.bytes_read(), input.len() as u64);
        assert_eq!(reader.lines_read(), out.len() as u64);
        out
    }

    #[test]
    fn splits_unix_lines() {
        let got = collect(b"a\nbb\nccc\n");
        assert_eq!(
            got,
            vec![
                (1, "a".to_string(), 2),
                (2, "bb".to_string(), 3),
                (3, "ccc".to_string(), 4)
            ]
        );
    }

    #[test]
    fn handles_crlf_and_missing_final_newline() {
        let got = collect(b"a\r\nb");
        assert_eq!(got, vec![(1, "a".to_string(), 3), (2, "b".to_string(), 1)]);
    }

    #[test]
    fn strips_a_leading_bom_only_once() {
        let got = collect("\u{feff}first\nsecond\n".as_bytes());
        assert_eq!(got[0].1, "first");
        assert_eq!(got[1].1, "second");
        // The BOM still counts towards the bytes consumed from the file.
        assert_eq!(got[0].2, 9);
    }

    #[test]
    fn empty_input_yields_no_lines() {
        assert!(collect(b"").is_empty());
        // A file that is a single newline still has one (blank) line.
        assert_eq!(collect(b"\n"), vec![(1, String::new(), 1)]);
    }

    #[test]
    fn detects_blank_lines() {
        let mut reader = LineReader::new(&b"   \n{}\n"[..]);
        assert!(reader.next_line().unwrap().unwrap().is_blank());
        assert!(!reader.next_line().unwrap().unwrap().is_blank());
        assert!(reader.next_line().unwrap().is_none());
    }

    #[test]
    fn a_bom_only_line_is_empty_not_negative() {
        let got = collect("\u{feff}\nx\n".as_bytes());
        assert_eq!(got, vec![(1, String::new(), 4), (2, "x".to_string(), 2)]);
    }
}
