//! Field paths: the `meta.source` / `messages[].role` selector syntax.
//!
//! `stats --field` and `schema` both need to name a location inside a JSON
//! record without pulling in a full query language. A path is a dotted chain
//! of object keys, each optionally followed by one or more `[...]` array
//! selectors: `[N]` for a fixed index (negative counts from the end), or `[]`
//! to fan out over every element. A path may also start with a bare `[N]`
//! for records that are themselves arrays.
//!
//! ```
//! use jsonl_peek::path::FieldPath;
//! use jsonl_peek::json::parse;
//!
//! let path = FieldPath::parse("messages[].role").unwrap();
//! let record = parse(r#"{"messages":[{"role":"user"},{"role":"assistant"}]}"#).unwrap();
//! let roles: Vec<&str> = path.resolve(&record).iter().filter_map(|v| v.as_str()).collect();
//! assert_eq!(roles, vec!["user", "assistant"]);
//! ```

use std::fmt;

use crate::json::Value;

/// One step of a [`FieldPath`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// An object member, by name.
    Key(String),
    /// A single array element. Negative indices count from the end, so `-1`
    /// is the last element.
    Index(i64),
    /// Every element of an array.
    All,
}

/// A parsed field path, e.g. `messages[].role`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldPath {
    raw: String,
    segments: Vec<Segment>,
}

impl FieldPath {
    /// Parses a field path in the syntax documented on the module.
    pub fn parse(path: &str) -> Result<FieldPath, PathError> {
        if path.is_empty() {
            return Err(PathError::new("field path is empty"));
        }
        let mut segments = Vec::new();
        for part in path.split('.') {
            parse_part(path, part, &mut segments)?;
        }
        Ok(FieldPath {
            raw: path.to_string(),
            segments,
        })
    }

    /// The path text this was parsed from.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// The parsed steps, in order.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// Selects every value inside `root` that this path names.
    ///
    /// A `Key` segment that misses, or an `Index`/`All` segment applied to
    /// something other than an array, simply drops that branch instead of
    /// erroring: most records in a dataset only partially match a path, and
    /// that is exactly what `present in N of M records` is meant to count.
    pub fn resolve<'v>(&self, root: &'v Value) -> Vec<&'v Value> {
        let mut current = vec![root];
        for segment in &self.segments {
            let mut next = Vec::new();
            for value in current {
                match segment {
                    Segment::Key(key) => {
                        if let Some(found) = value.get(key) {
                            next.push(found);
                        }
                    }
                    Segment::Index(index) => {
                        if let Some(item) = index_array(value, *index) {
                            next.push(item);
                        }
                    }
                    Segment::All => {
                        if let Some(items) = value.as_array() {
                            next.extend(items.iter());
                        }
                    }
                }
            }
            current = next;
        }
        current
    }
}

impl fmt::Display for FieldPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.raw)
    }
}

fn index_array(value: &Value, index: i64) -> Option<&Value> {
    let items = value.as_array()?;
    let real = if index < 0 {
        items.len().checked_sub(index.unsigned_abs() as usize)?
    } else {
        index as usize
    };
    items.get(real)
}

/// One `[...]` group's worth of parsing, applied to `part` (a single
/// dot-separated segment of the whole path).
fn parse_part(full: &str, part: &str, segments: &mut Vec<Segment>) -> Result<(), PathError> {
    let key_end = part.find('[').unwrap_or(part.len());
    let key = &part[..key_end];
    if key.is_empty() {
        if !part.starts_with('[') {
            return Err(PathError::new(format!("empty segment in '{}'", full)));
        }
    } else {
        if key.contains(']') {
            return Err(PathError::new(format!("unexpected ']' in '{}'", full)));
        }
        segments.push(Segment::Key(key.to_string()));
    }

    let bytes = part.as_bytes();
    let mut i = key_end;
    while i < bytes.len() {
        if bytes[i] != b'[' {
            return Err(PathError::new(format!("expected '[' in '{}'", full)));
        }
        let close = part[i + 1..]
            .find(']')
            .map(|p| p + i + 1)
            .ok_or_else(|| PathError::new(format!("unterminated '[' in '{}'", full)))?;
        let inner = &part[i + 1..close];
        if inner.is_empty() {
            segments.push(Segment::All);
        } else {
            let index: i64 = inner
                .parse()
                .map_err(|_| PathError::new(format!("invalid array index '{}' in '{}'", inner, full)))?;
            segments.push(Segment::Index(index));
        }
        i = close + 1;
    }
    Ok(())
}

/// Why a field path failed to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathError {
    message: String,
}

impl PathError {
    fn new(message: impl Into<String>) -> Self {
        PathError {
            message: message.into(),
        }
    }
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PathError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse as parse_json;

    #[test]
    fn parses_a_bare_key() {
        let p = FieldPath::parse("role").unwrap();
        assert_eq!(p.segments(), &[Segment::Key("role".to_string())]);
        assert_eq!(p.as_str(), "role");
    }

    #[test]
    fn parses_a_dotted_path() {
        let p = FieldPath::parse("meta.source").unwrap();
        assert_eq!(
            p.segments(),
            &[Segment::Key("meta".to_string()), Segment::Key("source".to_string())]
        );
    }

    #[test]
    fn parses_a_fixed_index() {
        let p = FieldPath::parse("messages[0].content").unwrap();
        assert_eq!(
            p.segments(),
            &[
                Segment::Key("messages".to_string()),
                Segment::Index(0),
                Segment::Key("content".to_string()),
            ]
        );
    }

    #[test]
    fn parses_a_negative_index() {
        let p = FieldPath::parse("messages[-1].content").unwrap();
        assert_eq!(
            p.segments(),
            &[
                Segment::Key("messages".to_string()),
                Segment::Index(-1),
                Segment::Key("content".to_string()),
            ]
        );
    }

    #[test]
    fn parses_a_wildcard() {
        let p = FieldPath::parse("messages[].role").unwrap();
        assert_eq!(
            p.segments(),
            &[
                Segment::Key("messages".to_string()),
                Segment::All,
                Segment::Key("role".to_string()),
            ]
        );
    }

    #[test]
    fn parses_a_leading_index() {
        let p = FieldPath::parse("[0].id").unwrap();
        assert_eq!(p.segments(), &[Segment::Index(0), Segment::Key("id".to_string())]);
    }

    #[test]
    fn rejects_empty_and_malformed_paths() {
        assert!(FieldPath::parse("").is_err());
        assert!(FieldPath::parse("a..b").is_err());
        assert!(FieldPath::parse(".a").is_err());
        assert!(FieldPath::parse("a[").is_err());
        assert!(FieldPath::parse("a[x]").is_err());
        assert!(FieldPath::parse("a]b").is_err());
    }

    #[test]
    fn resolves_a_simple_key() {
        let record = parse_json(r#"{"role":"user"}"#).unwrap();
        let p = FieldPath::parse("role").unwrap();
        assert_eq!(p.resolve(&record), vec![&Value::Str("user".to_string())]);
    }

    #[test]
    fn resolves_a_missing_key_to_nothing() {
        let record = parse_json(r#"{"role":"user"}"#).unwrap();
        let p = FieldPath::parse("meta.source").unwrap();
        assert!(p.resolve(&record).is_empty());
    }

    #[test]
    fn resolves_a_wildcard_across_an_array() {
        let record =
            parse_json(r#"{"messages":[{"role":"user"},{"role":"assistant"},{"role":"user"}]}"#).unwrap();
        let p = FieldPath::parse("messages[].role").unwrap();
        let roles: Vec<&str> = p.resolve(&record).iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "user"]);
    }

    #[test]
    fn resolves_positive_and_negative_indices() {
        let record = parse_json(r#"{"messages":[{"role":"user"},{"role":"assistant"}]}"#).unwrap();
        let first = FieldPath::parse("messages[0].role").unwrap();
        let last = FieldPath::parse("messages[-1].role").unwrap();
        assert_eq!(first.resolve(&record)[0].as_str(), Some("user"));
        assert_eq!(last.resolve(&record)[0].as_str(), Some("assistant"));
    }

    #[test]
    fn out_of_range_index_resolves_to_nothing() {
        let record = parse_json(r#"{"messages":[{"role":"user"}]}"#).unwrap();
        let p = FieldPath::parse("messages[5].role").unwrap();
        assert!(p.resolve(&record).is_empty());
        let p = FieldPath::parse("messages[-9].role").unwrap();
        assert!(p.resolve(&record).is_empty());
    }

    #[test]
    fn indexing_a_non_array_resolves_to_nothing() {
        let record = parse_json(r#"{"role":"user"}"#).unwrap();
        let p = FieldPath::parse("role[0]").unwrap();
        assert!(p.resolve(&record).is_empty());
        let p = FieldPath::parse("role[]").unwrap();
        assert!(p.resolve(&record).is_empty());
    }

    #[test]
    fn resolves_records_that_are_themselves_arrays() {
        let record = parse_json(r#"[{"id":1},{"id":2}]"#).unwrap();
        let p = FieldPath::parse("[1].id").unwrap();
        assert_eq!(p.resolve(&record), vec![&Value::Int(2)]);
    }

    #[test]
    fn display_round_trips_the_original_text() {
        let p = FieldPath::parse("messages[].role").unwrap();
        assert_eq!(p.to_string(), "messages[].role");
    }
}
