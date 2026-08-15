//! A small, dependency-free JSON reader.
//!
//! The parser is a hand written recursive descent parser over `&str`. It is
//! strict: it follows RFC 8259 and rejects the usual "almost JSON" mistakes
//! that show up in scraped training data (trailing commas, single quotes,
//! `NaN`, leading zeros, unescaped control characters, lone surrogates).
//!
//! Every failure carries the byte offset inside the input, which is what makes
//! it possible to point at the exact column of a broken line in a JSONL file.
//!
//! ```
//! use jsonl_peek::json::{parse, Value};
//!
//! let v = parse(r#"{"role":"user","tokens":[1,2,3]}"#).unwrap();
//! assert_eq!(v.get("role").and_then(Value::as_str), Some("user"));
//! assert_eq!(parse("{,}").unwrap_err().offset, 1);
//! ```

use std::fmt;

/// Maximum nesting depth accepted by [`parse`].
///
/// Deeply nested input is rejected instead of blowing the native stack.
pub const MAX_DEPTH: usize = 128;

/// A parsed JSON value.
///
/// JSON has a single numeric type, but telling integers from floats is useful
/// when profiling datasets, so the two are kept apart. A number without a
/// fraction or exponent that fits in an `i64` becomes [`Value::Int`], anything
/// else becomes [`Value::Float`].
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `null`
    Null,
    /// `true` / `false`
    Bool(bool),
    /// An integral number that fits in an `i64`.
    Int(i64),
    /// Any other number.
    Float(f64),
    /// A string, with escape sequences already decoded.
    Str(String),
    /// An array, in source order.
    Array(Vec<Value>),
    /// An object. Member order is preserved and duplicate keys are kept.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// Short type name used in reports: `null`, `bool`, `int`, `float`,
    /// `string`, `array` or `object`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }

    /// Looks up an object member by name, returning the first match.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Borrows the value as a string slice, if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the value as an `i64`, if it is an integral number.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns the value as an `f64`, widening integers.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(n) => Some(*n as f64),
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the boolean payload, if any.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Borrows the elements of an array.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    /// Borrows the members of an object.
    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(fields) => Some(fields.as_slice()),
            _ => None,
        }
    }

    /// True for [`Value::Null`].
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Serialises back to compact JSON.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    /// Appends the compact JSON encoding of this value to `out`.
    pub fn write_json(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            Value::Int(n) => out.push_str(&n.to_string()),
            Value::Float(f) => {
                if f.fract() == 0.0 && f.abs() < 1e16 {
                    out.push_str(&format!("{:.1}", f));
                } else {
                    out.push_str(&f.to_string());
                }
            }
            Value::Str(s) => escape_into(s, out),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write_json(out);
                }
                out.push(']');
            }
            Value::Object(fields) => {
                out.push('{');
                for (i, (name, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    escape_into(name, out);
                    out.push(':');
                    value.write_json(out);
                }
                out.push('}');
            }
        }
    }
}

const HEX: [u8; 16] = *b"0123456789abcdef";

/// Appends `s` to `out` as a quoted, escaped JSON string.
pub fn escape_into(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str("\\u");
                let n = c as u32;
                for shift in [12, 8, 4, 0] {
                    out.push(HEX[((n >> shift) & 0xF) as usize] as char);
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// What went wrong while parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// Input ended in the middle of a value.
    UnexpectedEof,
    /// A byte that cannot start a value.
    UnexpectedByte(u8),
    /// Something that looked like `true`, `false` or `null` but was not.
    InvalidLiteral,
    /// A malformed number, e.g. `1.`, `.5`, `1e`.
    InvalidNumber,
    /// A number the `f64` type cannot represent, e.g. `1e400`.
    NumberOutOfRange,
    /// An unknown `\x` escape inside a string.
    InvalidEscape(u8),
    /// A `\u` escape with fewer than four hex digits.
    InvalidUnicodeEscape,
    /// A surrogate half without its partner.
    UnpairedSurrogate,
    /// A raw control character inside a string; it must be escaped.
    ControlCharacter(u8),
    /// An object member without its `:`.
    ExpectedColon,
    /// An object that continued with something other than `,` or `}`.
    ExpectedCommaOrCloseBrace,
    /// An array that continued with something other than `,` or `]`.
    ExpectedCommaOrCloseBracket,
    /// An object member that did not start with a quoted key.
    ExpectedKey,
    /// Non-whitespace bytes after the top level value.
    TrailingData,
    /// Nesting deeper than [`MAX_DEPTH`].
    DepthLimitExceeded,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::UnexpectedEof => f.write_str("unexpected end of input"),
            ErrorKind::UnexpectedByte(b) => {
                f.write_str("unexpected ")?;
                f.write_str(&describe_byte(*b))
            }
            ErrorKind::InvalidLiteral => f.write_str("invalid literal, expected true/false/null"),
            ErrorKind::InvalidNumber => f.write_str("invalid number"),
            ErrorKind::NumberOutOfRange => f.write_str("number out of range for f64"),
            ErrorKind::InvalidEscape(b) => {
                f.write_str("invalid escape sequence \\")?;
                f.write_str(&describe_byte(*b))
            }
            ErrorKind::InvalidUnicodeEscape => f.write_str("invalid \\u escape"),
            ErrorKind::UnpairedSurrogate => f.write_str("unpaired UTF-16 surrogate"),
            ErrorKind::ControlCharacter(b) => {
                f.write_str("unescaped control character ")?;
                f.write_str(&describe_byte(*b))
            }
            ErrorKind::ExpectedColon => f.write_str("expected ':' after object key"),
            ErrorKind::ExpectedCommaOrCloseBrace => f.write_str("expected ',' or '}' in object"),
            ErrorKind::ExpectedCommaOrCloseBracket => f.write_str("expected ',' or ']' in array"),
            ErrorKind::ExpectedKey => f.write_str("expected a quoted object key"),
            ErrorKind::TrailingData => f.write_str("trailing data after JSON value"),
            ErrorKind::DepthLimitExceeded => f.write_str("nesting too deep"),
        }
    }
}

fn describe_byte(b: u8) -> String {
    if b.is_ascii_graphic() {
        let mut s = String::from("'");
        s.push(b as char);
        s.push('\'');
        s
    } else {
        let mut s = String::from("byte 0x");
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xF) as usize] as char);
        s
    }
}

/// A parse failure together with the byte offset where it was detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Zero based byte offset into the parsed input.
    pub offset: usize,
    /// What went wrong.
    pub kind: ErrorKind,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.offset, self.kind)
    }
}

impl std::error::Error for ParseError {}

/// Parses a complete JSON document.
///
/// Leading and trailing whitespace is allowed; anything else after the value is
/// reported as [`ErrorKind::TrailingData`].
pub fn parse(input: &str) -> Result<Value, ParseError> {
    let mut p = Parser {
        src: input,
        b: input.as_bytes(),
        i: 0,
        depth: 0,
    };
    p.skip_ws();
    let value = p.parse_value()?;
    p.skip_ws();
    if p.i != p.b.len() {
        return Err(p.err(ErrorKind::TrailingData));
    }
    Ok(value)
}

struct Parser<'a> {
    src: &'a str,
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, kind: ErrorKind) -> ParseError {
        ParseError {
            offset: self.i,
            kind,
        }
    }

    fn err_at(&self, offset: usize, kind: ErrorKind) -> ParseError {
        ParseError { offset, kind }
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') = self.peek() {
            self.i += 1;
        }
    }

    fn enter(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err(ErrorKind::DepthLimitExceeded));
        }
        Ok(())
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        let c = self.peek().ok_or_else(|| self.err(ErrorKind::UnexpectedEof))?;
        match c {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => Ok(Value::Str(self.parse_string()?)),
            b't' => self.parse_literal("true", Value::Bool(true)),
            b'f' => self.parse_literal("false", Value::Bool(false)),
            b'n' => self.parse_literal("null", Value::Null),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(self.err(ErrorKind::UnexpectedByte(other))),
        }
    }

    fn parse_literal(&mut self, word: &str, value: Value) -> Result<Value, ParseError> {
        let end = self.i + word.len();
        if end <= self.b.len() && &self.b[self.i..end] == word.as_bytes() {
            self.i = end;
            Ok(value)
        } else {
            Err(self.err(ErrorKind::InvalidLiteral))
        }
    }

    fn eat_digits(&mut self) -> usize {
        let start = self.i;
        while matches!(self.peek(), Some(d) if d.is_ascii_digit()) {
            self.i += 1;
        }
        self.i - start
    }

    fn parse_number(&mut self) -> Result<Value, ParseError> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            // A leading zero may not be followed by more digits. We simply stop
            // here; the caller then reports the stray digit as trailing data or
            // as a missing separator, which points at the right column.
            Some(b'0') => self.i += 1,
            Some(d) if d.is_ascii_digit() => {
                self.eat_digits();
            }
            Some(_) => return Err(self.err_at(start, ErrorKind::InvalidNumber)),
            None => return Err(self.err(ErrorKind::UnexpectedEof)),
        }

        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.i += 1;
            if self.eat_digits() == 0 {
                return Err(self.err(ErrorKind::InvalidNumber));
            }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.i += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.i += 1;
            }
            if self.eat_digits() == 0 {
                return Err(self.err(ErrorKind::InvalidNumber));
            }
        }

        let raw = &self.src[start..self.i];
        if !is_float {
            if let Ok(n) = raw.parse::<i64>() {
                return Ok(Value::Int(n));
            }
        }
        match raw.parse::<f64>() {
            Ok(f) if f.is_finite() => Ok(Value::Float(f)),
            _ => Err(self.err_at(start, ErrorKind::NumberOutOfRange)),
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        // Caller guarantees the current byte is '"'.
        self.i += 1;
        let mut chunk_start = self.i;
        let mut decoded: Option<String> = None;
        loop {
            let c = self.peek().ok_or_else(|| self.err(ErrorKind::UnexpectedEof))?;
            match c {
                b'"' => {
                    // Slicing is safe: we only ever stop on ASCII bytes, so
                    // `chunk_start..i` never splits a multi-byte character.
                    let tail = &self.src[chunk_start..self.i];
                    self.i += 1;
                    return Ok(match decoded {
                        Some(mut s) => {
                            s.push_str(tail);
                            s
                        }
                        None => tail.to_string(),
                    });
                }
                b'\\' => {
                    let buf = decoded.get_or_insert_with(String::new);
                    buf.push_str(&self.src[chunk_start..self.i]);
                    self.i += 1;
                    let ch = self.parse_escape()?;
                    buf.push(ch);
                    chunk_start = self.i;
                }
                0x00..=0x1F => return Err(self.err(ErrorKind::ControlCharacter(c))),
                _ => self.i += 1,
            }
        }
    }

    fn parse_escape(&mut self) -> Result<char, ParseError> {
        let esc_at = self.i - 1;
        let c = self.peek().ok_or_else(|| self.err(ErrorKind::UnexpectedEof))?;
        self.i += 1;
        let ch = match c {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let hi = self.parse_hex4()?;
                let cp = if (0xD800..=0xDBFF).contains(&hi) {
                    if self.peek() != Some(b'\\') || self.b.get(self.i + 1) != Some(&b'u') {
                        return Err(self.err_at(esc_at, ErrorKind::UnpairedSurrogate));
                    }
                    self.i += 2;
                    let lo = self.parse_hex4()?;
                    if !(0xDC00..=0xDFFF).contains(&lo) {
                        return Err(self.err_at(esc_at, ErrorKind::UnpairedSurrogate));
                    }
                    0x1_0000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                } else if (0xDC00..=0xDFFF).contains(&hi) {
                    return Err(self.err_at(esc_at, ErrorKind::UnpairedSurrogate));
                } else {
                    hi
                };
                char::from_u32(cp).ok_or_else(|| self.err_at(esc_at, ErrorKind::InvalidUnicodeEscape))?
            }
            other => return Err(self.err_at(esc_at, ErrorKind::InvalidEscape(other))),
        };
        Ok(ch)
    }

    fn parse_hex4(&mut self) -> Result<u32, ParseError> {
        if self.i + 4 > self.b.len() {
            return Err(self.err_at(self.b.len(), ErrorKind::UnexpectedEof));
        }
        let mut v = 0u32;
        for offset in 0..4 {
            let d = self.b[self.i + offset];
            let digit = match d {
                b'0'..=b'9' => u32::from(d - b'0'),
                b'a'..=b'f' => u32::from(d - b'a') + 10,
                b'A'..=b'F' => u32::from(d - b'A') + 10,
                _ => return Err(self.err_at(self.i + offset, ErrorKind::InvalidUnicodeEscape)),
            };
            v = v * 16 + digit;
        }
        self.i += 4;
        Ok(v)
    }

    fn parse_object(&mut self) -> Result<Value, ParseError> {
        self.enter()?;
        self.i += 1; // '{'
        let mut fields: Vec<(String, Value)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Object(fields));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(match self.peek() {
                    Some(_) => self.err(ErrorKind::ExpectedKey),
                    None => self.err(ErrorKind::UnexpectedEof),
                });
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(match self.peek() {
                    Some(_) => self.err(ErrorKind::ExpectedColon),
                    None => self.err(ErrorKind::UnexpectedEof),
                });
            }
            self.i += 1;
            self.skip_ws();
            let value = self.parse_value()?;
            fields.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Value::Object(fields));
                }
                Some(_) => return Err(self.err(ErrorKind::ExpectedCommaOrCloseBrace)),
                None => return Err(self.err(ErrorKind::UnexpectedEof)),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, ParseError> {
        self.enter()?;
        self.i += 1; // '['
        let mut items: Vec<Value> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    self.depth -= 1;
                    return Ok(Value::Array(items));
                }
                Some(_) => return Err(self.err(ErrorKind::ExpectedCommaOrCloseBracket)),
                None => return Err(self.err(ErrorKind::UnexpectedEof)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(input: &str) -> ParseError {
        parse(input).unwrap_err()
    }

    #[test]
    fn parses_scalars() {
        assert_eq!(parse("null").unwrap(), Value::Null);
        assert_eq!(parse(" true ").unwrap(), Value::Bool(true));
        assert_eq!(parse("false").unwrap(), Value::Bool(false));
        assert_eq!(parse("0").unwrap(), Value::Int(0));
        assert_eq!(parse("-42").unwrap(), Value::Int(-42));
        assert_eq!(parse("1.5").unwrap(), Value::Float(1.5));
        assert_eq!(parse("1e3").unwrap(), Value::Float(1000.0));
        assert_eq!(parse("-0.0").unwrap(), Value::Float(-0.0));
    }

    #[test]
    fn integers_beyond_i64_become_floats() {
        // 2^64 has no i64 representation, so it degrades to f64 rather than
        // failing outright.
        let v = parse("18446744073709551616").unwrap();
        assert_eq!(v.type_name(), "float");
        assert_eq!(v.as_f64(), Some(18446744073709551616.0));
        assert_eq!(parse("9223372036854775807").unwrap(), Value::Int(i64::MAX));
    }

    #[test]
    fn rejects_malformed_numbers() {
        assert_eq!(err("01").kind, ErrorKind::TrailingData);
        assert_eq!(err("1.").kind, ErrorKind::InvalidNumber);
        assert_eq!(err("-").kind, ErrorKind::UnexpectedEof);
        assert_eq!(err("1e").kind, ErrorKind::InvalidNumber);
        assert_eq!(err("1e+").kind, ErrorKind::InvalidNumber);
        assert_eq!(err("+1").kind, ErrorKind::UnexpectedByte(b'+'));
        assert_eq!(err("1e400").kind, ErrorKind::NumberOutOfRange);
        assert_eq!(err("NaN").kind, ErrorKind::UnexpectedByte(b'N'));
    }

    #[test]
    fn decodes_string_escapes() {
        let v = parse(r#""a\/b\\c\"d\n\t\r\b\f""#).unwrap();
        assert_eq!(v.as_str(), Some("a/b\\c\"d\n\t\r\u{8}\u{c}"));
        assert_eq!(parse(r#""é""#).unwrap().as_str(), Some("é"));
        // Surrogate pair for U+1F600.
        assert_eq!(parse(r#""😀""#).unwrap().as_str(), Some("😀"));
    }

    #[test]
    fn rejects_bad_strings() {
        assert_eq!(err(r#""\x""#).kind, ErrorKind::InvalidEscape(b'x'));
        assert_eq!(err(r#""\u12x4""#).kind, ErrorKind::InvalidUnicodeEscape);
        assert_eq!(err(r#""\u12""#).kind, ErrorKind::UnexpectedEof);
        assert_eq!(err(r#""\ud83d""#).kind, ErrorKind::UnpairedSurrogate);
        assert_eq!(err(r#""\udc00\ud83d""#).kind, ErrorKind::UnpairedSurrogate);
        assert_eq!(err("\"raw\tTab\"").kind, ErrorKind::ControlCharacter(b'\t'));
        assert_eq!(err("\"unterminated").kind, ErrorKind::UnexpectedEof);
    }

    #[test]
    fn multibyte_strings_slice_correctly() {
        let v = parse("\"héllo→\\n世界\"").unwrap();
        assert_eq!(v.as_str(), Some("héllo→\n世界"));
    }

    #[test]
    fn parses_nested_containers() {
        let v = parse(r#"{"a":[1,{"b":null}],"c":{}}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_array().unwrap().len(), 2);
        assert!(v.get("a").unwrap().as_array().unwrap()[1]
            .get("b")
            .unwrap()
            .is_null());
        assert_eq!(v.get("c").unwrap().as_object().unwrap().len(), 0);
        assert_eq!(v.get("missing"), None);
    }

    #[test]
    fn reports_offsets() {
        let e = err(r#"{"a":1,}"#);
        assert_eq!(e.kind, ErrorKind::ExpectedKey);
        assert_eq!(e.offset, 7);
        let e = err(r#"{"a" 1}"#);
        assert_eq!(e.kind, ErrorKind::ExpectedColon);
        assert_eq!(e.offset, 5);
        let e = err("[1 2]");
        assert_eq!(e.kind, ErrorKind::ExpectedCommaOrCloseBracket);
        assert_eq!(e.offset, 3);
        let e = err("{} {}");
        assert_eq!(e.kind, ErrorKind::TrailingData);
        assert_eq!(e.offset, 3);
    }

    #[test]
    fn rejects_common_near_json() {
        assert_eq!(err("{'a':1}").kind, ErrorKind::ExpectedKey);
        assert_eq!(err("[1,]").kind, ErrorKind::UnexpectedByte(b']'));
        assert_eq!(err("{a:1}").kind, ErrorKind::ExpectedKey);
        assert_eq!(err("").kind, ErrorKind::UnexpectedEof);
        assert_eq!(err("   ").kind, ErrorKind::UnexpectedEof);
    }

    #[test]
    fn enforces_depth_limit() {
        let deep = "[".repeat(MAX_DEPTH + 1);
        assert_eq!(err(&deep).kind, ErrorKind::DepthLimitExceeded);
        let ok = format!("{}{}", "[".repeat(MAX_DEPTH), "]".repeat(MAX_DEPTH));
        assert!(parse(&ok).is_ok());
    }

    #[test]
    fn round_trips_through_writer() {
        let src = r#"{"s":"a\"b\n","i":-3,"f":2.5,"w":1.0,"a":[true,false,null],"o":{}}"#;
        let v = parse(src).unwrap();
        let out = v.to_json();
        assert_eq!(out, src);
        assert_eq!(parse(&out).unwrap(), v);
    }

    #[test]
    fn escapes_control_characters_on_write() {
        let v = Value::Str("\u{1}\u{1f}ok".to_string());
        assert_eq!(v.to_json(), "\"\\u0001\\u001fok\"");
        assert_eq!(parse(&v.to_json()).unwrap(), v);
    }

    #[test]
    fn duplicate_keys_are_preserved() {
        let v = parse(r#"{"a":1,"a":2}"#).unwrap();
        assert_eq!(v.as_object().unwrap().len(), 2);
        assert_eq!(v.get("a"), Some(&Value::Int(1)));
    }

    #[test]
    fn error_display_mentions_offset() {
        let e = err("[1 2]");
        assert_eq!(e.to_string(), "at byte 3: expected ',' or ']' in array");
    }
}
