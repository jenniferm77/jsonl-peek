//! File level statistics: counts, top level key profiling, line length
//! distribution and `--field` value distributions.
//!
//! [`Stats::from_reader`] makes one streaming pass over a JSONL source and
//! returns everything the `stats` command reports: how many lines parsed,
//! what the top level keys and their types look like, an approximate
//! distribution of line lengths, and - for any [`FieldPath`]s passed in -
//! how often each is present and what values it takes.
//!
//! ```
//! use jsonl_peek::{FieldPath, Stats, StatsOptions};
//!
//! let data = b"{\"messages\":[{\"role\":\"user\"},{\"role\":\"assistant\"}]}\n";
//! let options = StatsOptions {
//!     fields: vec![FieldPath::parse("messages[].role").unwrap()],
//!     ..StatsOptions::default()
//! };
//! let stats = Stats::from_reader(&data[..], options).unwrap();
//! assert_eq!(stats.valid, 1);
//! assert_eq!(
//!     stats.fields[0].top(2),
//!     vec![("\"assistant\"", 1), ("\"user\"", 1)],
//! );
//! ```

use std::collections::HashMap;
use std::io::{self, BufRead, Write};

use crate::hist::Histogram;
use crate::json::{self, Value};
use crate::lines::LineReader;
use crate::path::FieldPath;

/// Key table cap: past this many distinct top level keys, further keys are
/// dropped instead of growing the table without bound.
const MAX_KEYS: usize = 512;

/// Per-field value table cap: past this many distinct values, a field keeps
/// counting `values`/`present` but stops remembering individual values.
const MAX_FIELD_VALUES: usize = 10_000;

/// Options controlling a [`Stats::from_reader`] pass.
#[derive(Debug, Clone)]
pub struct StatsOptions {
    /// Field paths to profile with `--field`.
    pub fields: Vec<FieldPath>,
    /// Distinct values kept and reported per field, most common first.
    pub top: usize,
    /// Broken lines recorded in `issues` before further ones are dropped
    /// (the `invalid` count keeps growing regardless).
    pub max_errors: usize,
}

impl Default for StatsOptions {
    fn default() -> Self {
        StatsOptions {
            fields: Vec::new(),
            top: 10,
            max_errors: 10,
        }
    }
}

/// One broken line: where it is and why it did not parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issue {
    /// 1-based line number.
    pub line: u64,
    /// 1-based byte column inside the line where the problem was found.
    pub column: usize,
    /// Human readable reason, e.g. `"unexpected ','"`.
    pub reason: String,
}

/// How often one top level object key showed up, and with what types.
#[derive(Debug, Clone)]
pub struct KeyStat {
    key: String,
    count: u64,
    types: Vec<(&'static str, u64)>,
}

impl KeyStat {
    /// The key name.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Number of top level objects that had this key.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Value types seen under this key, in first-seen order.
    pub fn types(&self) -> &[(&'static str, u64)] {
        &self.types
    }

    /// Share of top level objects that had this key, in `[0, 1]`.
    pub fn rate(&self, object_records: u64) -> f64 {
        if object_records == 0 {
            0.0
        } else {
            self.count as f64 / object_records as f64
        }
    }
}

/// Value distribution for one `--field` path.
#[derive(Debug, Clone)]
pub struct FieldStats {
    path: FieldPath,
    present: u64,
    values: u64,
    types: Vec<(&'static str, u64)>,
    counts: HashMap<String, u64>,
    truncated: bool,
}

impl FieldStats {
    fn new(path: FieldPath) -> Self {
        FieldStats {
            path,
            present: 0,
            values: 0,
            types: Vec::new(),
            counts: HashMap::new(),
            truncated: false,
        }
    }

    fn record(&mut self, root: &Value) {
        let matches = self.path.resolve(root);
        if matches.is_empty() {
            return;
        }
        self.present += 1;
        for value in matches {
            self.values += 1;
            record_type(&mut self.types, value.type_name());
            let key = value.to_json();
            match self.counts.get_mut(&key) {
                Some(count) => *count += 1,
                None if self.counts.len() < MAX_FIELD_VALUES => {
                    self.counts.insert(key, 1);
                }
                None => self.truncated = true,
            }
        }
    }

    /// The path this profiles.
    pub fn path(&self) -> &FieldPath {
        &self.path
    }

    /// Number of records containing at least one match for the path.
    pub fn present(&self) -> u64 {
        self.present
    }

    /// Total number of matched values, across all records (more than
    /// `present` for a wildcard path that fans out over an array).
    pub fn values(&self) -> u64 {
        self.values
    }

    /// Value types seen at this path, in first-seen order.
    pub fn types(&self) -> &[(&'static str, u64)] {
        &self.types
    }

    /// Number of distinct values seen (a lower bound once `truncated`).
    pub fn distinct(&self) -> usize {
        self.counts.len()
    }

    /// True once the value table hit [`MAX_FIELD_VALUES`] and stopped
    /// tracking new distinct values.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Share of records containing this field, in `[0, 1]`.
    pub fn present_rate(&self, valid: u64) -> f64 {
        if valid == 0 {
            0.0
        } else {
            self.present as f64 / valid as f64
        }
    }

    /// The `n` most common values, each rendered as compact JSON (so a
    /// string value reads as `"web"`), most common first and ties broken
    /// alphabetically for a stable order.
    pub fn top(&self, n: usize) -> Vec<(&str, u64)> {
        let mut items: Vec<(&str, u64)> = self.counts.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        items.truncate(n);
        items
    }
}

/// Result of one streaming pass over a JSONL source.
#[derive(Debug, Clone)]
pub struct Stats {
    /// Total lines read, blank and non-blank.
    pub lines: u64,
    /// Lines that were blank (whitespace only) and so were skipped.
    pub blank: u64,
    /// Lines that parsed as a JSON value.
    pub valid: u64,
    /// Non-blank lines that were not valid UTF-8 or not valid JSON.
    pub invalid: u64,
    /// Total bytes across non-blank lines, line endings excluded.
    pub bytes: u64,
    /// Valid records whose top level value was an object; the denominator
    /// for [`KeyStat::rate`].
    pub object_records: u64,
    /// Top level value types, in first-seen order.
    pub top_level: Vec<(&'static str, u64)>,
    /// Distribution of line lengths in bytes.
    pub line_length: Histogram,
    /// Top level object keys, in first-seen order.
    pub keys: Vec<KeyStat>,
    /// True once the key table hit [`MAX_KEYS`] and stopped tracking new
    /// keys.
    pub keys_truncated: bool,
    /// One entry per field passed in [`StatsOptions::fields`], same order.
    pub fields: Vec<FieldStats>,
    /// Broken lines, most recent last, capped at
    /// [`StatsOptions::max_errors`].
    pub issues: Vec<Issue>,
}

impl Stats {
    /// Makes one streaming pass over `reader`, in the newline-delimited JSON
    /// format the rest of the crate assumes.
    pub fn from_reader<R: BufRead>(reader: R, options: StatsOptions) -> io::Result<Stats> {
        let mut line_reader = LineReader::new(reader);
        let mut stats = Stats {
            lines: 0,
            blank: 0,
            valid: 0,
            invalid: 0,
            bytes: 0,
            object_records: 0,
            top_level: Vec::new(),
            line_length: Histogram::new(),
            keys: Vec::new(),
            keys_truncated: false,
            fields: options.fields.iter().cloned().map(FieldStats::new).collect(),
            issues: Vec::new(),
        };

        while let Some(line) = line_reader.next_line()? {
            stats.lines += 1;
            if line.is_blank() {
                stats.blank += 1;
                continue;
            }
            stats.bytes += line.bytes.len() as u64;
            stats.line_length.record(line.bytes.len() as u64);

            let text = match std::str::from_utf8(line.bytes) {
                Ok(text) => text,
                Err(e) => {
                    stats.invalid += 1;
                    if stats.issues.len() < options.max_errors {
                        stats.issues.push(Issue {
                            line: line.number,
                            column: e.valid_up_to() + 1,
                            reason: "invalid UTF-8".to_string(),
                        });
                    }
                    continue;
                }
            };

            match json::parse(text) {
                Ok(value) => {
                    stats.valid += 1;
                    record_type(&mut stats.top_level, value.type_name());
                    if let Some(members) = value.as_object() {
                        stats.object_records += 1;
                        record_object_keys(&mut stats.keys, &mut stats.keys_truncated, members);
                    }
                    for field in &mut stats.fields {
                        field.record(&value);
                    }
                }
                Err(e) => {
                    stats.invalid += 1;
                    if stats.issues.len() < options.max_errors {
                        stats.issues.push(Issue {
                            line: line.number,
                            column: e.offset + 1,
                            reason: e.kind.to_string(),
                        });
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Writes the human readable report the `stats` command prints.
    ///
    /// `source` is the label shown on the `file` line (a path, or `-` for
    /// stdin); `top` is how many distinct values to show per field.
    pub fn write_report<W: Write>(&self, out: &mut W, source: &str, top: usize) -> io::Result<()> {
        writeln!(out, "file    {}", source)?;
        writeln!(
            out,
            "lines   {}   blank {}   invalid {}   valid {}",
            with_commas(self.lines),
            self.blank,
            self.invalid,
            with_commas(self.valid),
        )?;
        writeln!(
            out,
            "bytes   {}   ({:.1} KiB)",
            with_commas(self.bytes),
            self.bytes as f64 / 1024.0,
        )?;
        writeln!(out, "top level  {}", format_type_counts(&self.top_level))?;
        writeln!(out)?;

        writeln!(out, "line length in bytes")?;
        writeln!(
            out,
            "  min {}   p50 {}   p90 {}   p99 {}   max {}   mean {:.1}",
            self.line_length.min().unwrap_or(0),
            self.line_length.quantile(0.5).unwrap_or(0),
            self.line_length.quantile(0.9).unwrap_or(0),
            self.line_length.quantile(0.99).unwrap_or(0),
            self.line_length.max().unwrap_or(0),
            self.line_length.mean().unwrap_or(0.0),
        )?;
        writeln!(out)?;

        writeln!(out, "top level keys over {} objects", with_commas(self.object_records))?;
        writeln!(out, "  key                          count     rate  types")?;
        for key in &self.keys {
            writeln!(
                out,
                "  {:<28} {:>8}  {:>5.1}%  {}",
                key.key,
                with_commas(key.count),
                key.rate(self.object_records) * 100.0,
                format_type_counts(&key.types),
            )?;
        }
        if self.keys_truncated {
            writeln!(out, "  ... key table truncated at {} keys", MAX_KEYS)?;
        }

        for field in &self.fields {
            writeln!(out)?;
            writeln!(out, "field {}", field.path)?;
            writeln!(
                out,
                "  present in {} of {} records ({:.1}%), {} values, types {}",
                with_commas(field.present),
                with_commas(self.valid),
                field.present_rate(self.valid) * 100.0,
                with_commas(field.values),
                format_type_counts(&field.types),
            )?;
            writeln!(
                out,
                "  {} distinct values{}",
                field.distinct(),
                if field.truncated { " (truncated)" } else { "" },
            )?;
            for (value, count) in field.top(top) {
                writeln!(
                    out,
                    "  {:>8}  {:.1}%  {}",
                    with_commas(count),
                    safe_rate(count, field.values),
                    value,
                )?;
            }
        }

        if !self.issues.is_empty() || self.invalid > 0 {
            writeln!(out)?;
            writeln!(
                out,
                "invalid lines ({} total, showing {})",
                self.invalid,
                self.issues.len(),
            )?;
            for issue in &self.issues {
                writeln!(out, "  line {} col {}: {}", issue.line, issue.column, issue.reason)?;
            }
        }

        Ok(())
    }

    /// The machine-readable form of the report, for `--json`.
    pub fn to_json(&self) -> Value {
        Value::Object(vec![
            ("lines".to_string(), Value::Int(self.lines as i64)),
            ("blank".to_string(), Value::Int(self.blank as i64)),
            ("valid".to_string(), Value::Int(self.valid as i64)),
            ("invalid".to_string(), Value::Int(self.invalid as i64)),
            ("bytes".to_string(), Value::Int(self.bytes as i64)),
            ("top_level".to_string(), type_counts_json(&self.top_level)),
            ("line_length".to_string(), self.line_length_json()),
            ("keys".to_string(), self.keys_json()),
            ("fields".to_string(), self.fields_json()),
            ("issues".to_string(), self.issues_json()),
        ])
    }

    fn line_length_json(&self) -> Value {
        Value::Object(vec![
            ("min".to_string(), Value::Int(self.line_length.min().unwrap_or(0) as i64)),
            (
                "p50".to_string(),
                Value::Int(self.line_length.quantile(0.5).unwrap_or(0) as i64),
            ),
            (
                "p90".to_string(),
                Value::Int(self.line_length.quantile(0.9).unwrap_or(0) as i64),
            ),
            (
                "p99".to_string(),
                Value::Int(self.line_length.quantile(0.99).unwrap_or(0) as i64),
            ),
            ("max".to_string(), Value::Int(self.line_length.max().unwrap_or(0) as i64)),
            ("mean".to_string(), Value::Float(self.line_length.mean().unwrap_or(0.0))),
        ])
    }

    fn keys_json(&self) -> Value {
        Value::Array(
            self.keys
                .iter()
                .map(|k| {
                    Value::Object(vec![
                        ("key".to_string(), Value::Str(k.key.clone())),
                        ("count".to_string(), Value::Int(k.count as i64)),
                        ("rate".to_string(), Value::Float(k.rate(self.object_records))),
                        ("types".to_string(), type_counts_json(&k.types)),
                    ])
                })
                .collect(),
        )
    }

    fn fields_json(&self) -> Value {
        Value::Array(
            self.fields
                .iter()
                .map(|field| {
                    Value::Object(vec![
                        ("path".to_string(), Value::Str(field.path.to_string())),
                        ("present".to_string(), Value::Int(field.present as i64)),
                        ("values".to_string(), Value::Int(field.values as i64)),
                        ("types".to_string(), type_counts_json(&field.types)),
                        ("distinct".to_string(), Value::Int(field.distinct() as i64)),
                        ("truncated".to_string(), Value::Bool(field.truncated)),
                    ])
                })
                .collect(),
        )
    }

    fn issues_json(&self) -> Value {
        Value::Array(
            self.issues
                .iter()
                .map(|issue| {
                    Value::Object(vec![
                        ("line".to_string(), Value::Int(issue.line as i64)),
                        ("column".to_string(), Value::Int(issue.column as i64)),
                        ("reason".to_string(), Value::Str(issue.reason.clone())),
                    ])
                })
                .collect(),
        )
    }
}

fn record_type(types: &mut Vec<(&'static str, u64)>, type_name: &'static str) {
    match types.iter_mut().find(|(name, _)| *name == type_name) {
        Some(entry) => entry.1 += 1,
        None => types.push((type_name, 1)),
    }
}

fn record_object_keys(keys: &mut Vec<KeyStat>, truncated: &mut bool, members: &[(String, Value)]) {
    // A record with a duplicate key should only count once per key, so track
    // which keys this record has already contributed.
    let mut seen: Vec<&str> = Vec::new();
    for (key, value) in members {
        if seen.contains(&key.as_str()) {
            continue;
        }
        seen.push(key.as_str());
        record_key(keys, truncated, key, value.type_name());
    }
}

fn record_key(keys: &mut Vec<KeyStat>, truncated: &mut bool, key: &str, type_name: &'static str) {
    if let Some(stat) = keys.iter_mut().find(|k| k.key == key) {
        stat.count += 1;
        record_type(&mut stat.types, type_name);
        return;
    }
    if keys.len() >= MAX_KEYS {
        *truncated = true;
        return;
    }
    let mut stat = KeyStat {
        key: key.to_string(),
        count: 1,
        types: Vec::new(),
    };
    record_type(&mut stat.types, type_name);
    keys.push(stat);
}

fn type_counts_json(types: &[(&'static str, u64)]) -> Value {
    Value::Object(
        types
            .iter()
            .map(|(name, count)| (name.to_string(), Value::Int(*count as i64)))
            .collect(),
    )
}

fn format_type_counts(types: &[(&'static str, u64)]) -> String {
    types
        .iter()
        .map(|(name, count)| format!("{}:{}", name, with_commas(*count)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn safe_rate(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}

fn with_commas(n: u64) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let mut out = String::with_capacity(bytes.len() + bytes.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats_for(text: &str, options: StatsOptions) -> Stats {
        Stats::from_reader(text.as_bytes(), options).unwrap()
    }

    #[test]
    fn default_options_match_the_documented_defaults() {
        let options = StatsOptions::default();
        assert!(options.fields.is_empty());
        assert_eq!(options.top, 10);
        assert_eq!(options.max_errors, 10);
    }

    #[test]
    fn counts_blank_and_invalid_lines() {
        let text = "{\"a\":1}\n\n   \n{bad}\n{\"a\":2}\n";
        let s = stats_for(text, StatsOptions::default());
        assert_eq!(s.lines, 5);
        assert_eq!(s.blank, 2);
        assert_eq!(s.valid, 2);
        assert_eq!(s.invalid, 1);
        assert_eq!(s.issues.len(), 1);
        assert_eq!(s.issues[0].line, 4);
    }

    #[test]
    fn issue_column_is_one_based() {
        let s = stats_for("{\"a\":1,}\n", StatsOptions::default());
        assert_eq!(s.issues.len(), 1);
        assert_eq!(s.issues[0].line, 1);
        assert_eq!(s.issues[0].column, 8);
        assert_eq!(s.issues[0].reason, "expected a quoted object key");
    }

    #[test]
    fn caps_the_issue_list_but_keeps_the_total_invalid_count() {
        let text = "{bad1}\n{bad2}\n{bad3}\n";
        let options = StatsOptions {
            max_errors: 1,
            ..StatsOptions::default()
        };
        let s = stats_for(text, options);
        assert_eq!(s.invalid, 3);
        assert_eq!(s.issues.len(), 1);
        assert_eq!(s.issues[0].line, 1);
    }

    #[test]
    fn reports_invalid_utf8_as_an_invalid_line() {
        let mut data = Vec::new();
        data.extend_from_slice(b"{\"a\":1}\n");
        data.extend_from_slice(&[0x80, 0x81, b'\n']);
        let s = Stats::from_reader(&data[..], StatsOptions::default()).unwrap();
        assert_eq!(s.valid, 1);
        assert_eq!(s.invalid, 1);
        assert_eq!(s.issues[0].line, 2);
        assert_eq!(s.issues[0].column, 1);
        assert_eq!(s.issues[0].reason, "invalid UTF-8");
    }

    #[test]
    fn records_top_level_types_and_key_rates() {
        let text = "{\"id\":1,\"tags\":[\"a\"]}\n{\"id\":2}\n[1,2,3]\n";
        let s = stats_for(text, StatsOptions::default());
        assert_eq!(s.valid, 3);
        assert_eq!(s.object_records, 2);
        assert_eq!(s.top_level, vec![("object", 2), ("array", 1)]);

        let id = s.keys.iter().find(|k| k.key() == "id").unwrap();
        assert_eq!(id.count(), 2);
        assert_eq!(id.rate(s.object_records), 1.0);
        assert_eq!(id.types(), &[("int", 2)]);

        let tags = s.keys.iter().find(|k| k.key() == "tags").unwrap();
        assert_eq!(tags.count(), 1);
        assert_eq!(tags.rate(s.object_records), 0.5);
    }

    #[test]
    fn a_key_with_mixed_types_records_both() {
        let text = "{\"v\":1}\n{\"v\":\"x\"}\n";
        let s = stats_for(text, StatsOptions::default());
        let v = s.keys.iter().find(|k| k.key() == "v").unwrap();
        assert_eq!(v.count(), 2);
        assert_eq!(v.types(), &[("int", 1), ("string", 1)]);
    }

    #[test]
    fn a_duplicate_key_in_one_object_counts_once() {
        let text = "{\"a\":1,\"a\":2}\n";
        let s = stats_for(text, StatsOptions::default());
        let a = s.keys.iter().find(|k| k.key() == "a").unwrap();
        assert_eq!(a.count(), 1);
    }

    #[test]
    fn profiles_a_field_across_a_wildcard() {
        let text = "{\"messages\":[{\"role\":\"user\"},{\"role\":\"assistant\"},{\"role\":\"user\"}]}\n\
                     {\"messages\":[]}\n\
                     {\"other\":true}\n";
        let options = StatsOptions {
            fields: vec![FieldPath::parse("messages[].role").unwrap()],
            ..StatsOptions::default()
        };
        let s = stats_for(text, options);
        let field = &s.fields[0];
        assert_eq!(field.present(), 1);
        assert_eq!(field.values(), 3);
        assert_eq!(field.types(), &[("string", 3)]);
        assert_eq!(field.present_rate(s.valid), 1.0 / 3.0);
        assert_eq!(
            field.top(10),
            vec![("\"user\"", 2), ("\"assistant\"", 1)],
        );
    }

    #[test]
    fn field_top_breaks_ties_alphabetically() {
        let text = "{\"v\":\"b\"}\n{\"v\":\"a\"}\n";
        let options = StatsOptions {
            fields: vec![FieldPath::parse("v").unwrap()],
            ..StatsOptions::default()
        };
        let s = stats_for(text, options);
        assert_eq!(s.fields[0].top(5), vec![("\"a\"", 1), ("\"b\"", 1)]);
    }

    #[test]
    fn truncates_the_key_table_past_the_cap() {
        let mut line = String::from("{");
        for i in 0..(MAX_KEYS + 1) {
            if i > 0 {
                line.push(',');
            }
            line.push_str(&format!("\"k{}\":1", i));
        }
        line.push_str("}\n");
        let s = stats_for(&line, StatsOptions::default());
        assert_eq!(s.keys.len(), MAX_KEYS);
        assert!(s.keys_truncated);
    }

    #[test]
    fn truncates_the_field_value_table_past_the_cap() {
        let mut text = String::new();
        for i in 0..(MAX_FIELD_VALUES + 1) {
            text.push_str(&format!("{{\"v\":\"{}\"}}\n", i));
        }
        let options = StatsOptions {
            fields: vec![FieldPath::parse("v").unwrap()],
            ..StatsOptions::default()
        };
        let s = stats_for(&text, options);
        let field = &s.fields[0];
        assert_eq!(field.values(), (MAX_FIELD_VALUES + 1) as u64);
        assert_eq!(field.distinct(), MAX_FIELD_VALUES);
        assert!(field.truncated());
    }

    #[test]
    fn write_report_includes_the_expected_sections() {
        let text = "{\"id\":1,\"meta\":{\"source\":\"web\"}}\n{bad}\n";
        let options = StatsOptions {
            fields: vec![FieldPath::parse("meta.source").unwrap()],
            ..StatsOptions::default()
        };
        let s = stats_for(text, options);
        let mut out = Vec::new();
        s.write_report(&mut out, "sample.jsonl", 10).unwrap();
        let report = String::from_utf8(out).unwrap();
        assert!(report.contains("file    sample.jsonl"));
        assert!(report.contains("lines   2   blank 0   invalid 1   valid 1"));
        assert!(report.contains("top level keys over 1 objects"));
        assert!(report.contains("field meta.source"));
        assert!(report.contains("\"web\""));
        assert!(report.contains("invalid lines (1 total, showing 1)"));
    }

    #[test]
    fn to_json_reports_line_length_and_issue_details() {
        let text = "{\"a\":1}\n{bad}\n";
        let s = stats_for(text, StatsOptions::default());
        let json = s.to_json();
        assert_eq!(json.get("lines").and_then(Value::as_i64), Some(2));
        assert_eq!(json.get("valid").and_then(Value::as_i64), Some(1));
        assert_eq!(json.get("invalid").and_then(Value::as_i64), Some(1));

        let line_length = json.get("line_length").unwrap();
        assert_eq!(line_length.get("min").and_then(Value::as_i64), Some(5));
        assert_eq!(line_length.get("max").and_then(Value::as_i64), Some(7));

        let issues = json.get("issues").unwrap().as_array().unwrap();
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].get("line").and_then(Value::as_i64), Some(2));
    }

    #[test]
    fn empty_input_reports_zero_everything() {
        let s = stats_for("", StatsOptions::default());
        assert_eq!(s.lines, 0);
        assert_eq!(s.valid, 0);
        assert!(s.line_length.min().is_none());

        let mut out = Vec::new();
        s.write_report(&mut out, "-", 10).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("lines   0"));
    }
}
