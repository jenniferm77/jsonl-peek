//! Schema discovery: which paths exist across a JSONL file, how often, and
//! with what types.
//!
//! [`Schema::from_reader`] walks every record up to a bounded depth and
//! reports, for each path it finds along the way (`meta.source`,
//! `messages[].role`, and so on), how many records contain it and what value
//! types show up there. An object key adds one level; so does fanning out
//! into an array with `[]`. This is what backs the `schema` command.
//!
//! ```
//! use jsonl_peek::{Schema, SchemaOptions};
//!
//! let data = b"{\"id\":1,\"tags\":[\"a\",\"b\"]}\n{\"id\":2}\n";
//! let schema = Schema::from_reader(&data[..], SchemaOptions::default()).unwrap();
//! let tags = schema.paths.iter().find(|p| p.path() == "tags[]").unwrap();
//! assert_eq!(tags.present(), 1);
//! assert_eq!(tags.count(), 2);
//! ```

use std::collections::HashSet;
use std::io::{self, BufRead, Write};

use crate::json::{self, Value};
use crate::lines::LineReader;

/// Path table cap: past this many distinct paths, further ones are dropped
/// instead of growing the table without bound.
const MAX_PATHS: usize = 2_000;

/// Options controlling a [`Schema::from_reader`] pass.
#[derive(Debug, Clone)]
pub struct SchemaOptions {
    /// Levels to descend. Each object key and each `[]` array fan-out counts
    /// as one level; a path deeper than this is dropped instead of walked.
    pub depth: usize,
    /// Hide paths present in fewer than this share of records, in `[0, 1]`.
    pub min_rate: f64,
}

impl Default for SchemaOptions {
    fn default() -> Self {
        SchemaOptions {
            depth: 3,
            min_rate: 0.0,
        }
    }
}

/// How often one path showed up, and with what types.
#[derive(Debug, Clone)]
pub struct PathStat {
    path: String,
    present: u64,
    count: u64,
    types: Vec<(&'static str, u64)>,
}

impl PathStat {
    /// The path text, e.g. `messages[].role`.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Number of records containing at least one value at this path.
    pub fn present(&self) -> u64 {
        self.present
    }

    /// Total number of values seen at this path, across all records (more
    /// than `present` for a path that fans out over an array).
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Value types seen at this path, in first-seen order.
    pub fn types(&self) -> &[(&'static str, u64)] {
        &self.types
    }

    /// Share of records containing this path, in `[0, 1]`.
    pub fn rate(&self, records: u64) -> f64 {
        if records == 0 {
            0.0
        } else {
            self.present as f64 / records as f64
        }
    }
}

/// Result of one streaming pass over a JSONL source.
#[derive(Debug, Clone)]
pub struct Schema {
    /// Levels descended, as passed in [`SchemaOptions::depth`].
    pub depth: usize,
    /// Records that parsed and were walked.
    pub records: u64,
    /// Non-blank lines that did not parse, and so contributed no paths.
    /// Blank lines are skipped too but are not counted here.
    pub skipped: u64,
    /// Paths found, in alphabetical order.
    pub paths: Vec<PathStat>,
    /// True once the path table hit [`MAX_PATHS`] and stopped tracking new
    /// paths.
    pub truncated: bool,
}

impl Schema {
    /// Makes one streaming pass over `reader`, walking every record up to
    /// `options.depth` levels deep and recording every path found.
    pub fn from_reader<R: BufRead>(reader: R, options: SchemaOptions) -> io::Result<Schema> {
        let mut line_reader = LineReader::new(reader);
        let mut table: Vec<PathStat> = Vec::new();
        let mut truncated = false;
        let mut records = 0u64;
        let mut skipped = 0u64;

        while let Some(line) = line_reader.next_line()? {
            if line.is_blank() {
                continue;
            }
            let text = match std::str::from_utf8(line.bytes) {
                Ok(text) => text,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            let value = match json::parse(text) {
                Ok(value) => value,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            records += 1;

            let mut seen = HashSet::new();
            match &value {
                Value::Object(members) => {
                    walk_object(members, "", 0, options.depth, &mut seen, &mut table, &mut truncated)
                }
                Value::Array(items) => walk_array(items, "", 0, options.depth, &mut seen, &mut table, &mut truncated),
                _ => {}
            }
            for path in &seen {
                if let Some(stat) = table.iter_mut().find(|p| &p.path == path) {
                    stat.present += 1;
                }
            }
        }

        let mut paths: Vec<PathStat> = table.into_iter().filter(|p| p.rate(records) >= options.min_rate).collect();
        paths.sort_by(|a, b| a.path.cmp(&b.path));

        Ok(Schema {
            depth: options.depth,
            records,
            skipped,
            paths,
            truncated,
        })
    }

    /// Writes the human readable report the `schema` command prints.
    pub fn write_report<W: Write>(&self, out: &mut W) -> io::Result<()> {
        writeln!(out, "{} records, depth {}", self.records, self.depth)?;
        writeln!(out)?;
        writeln!(out, "  {:<30} {:>6}  {}", "path", "rate", "types")?;
        for path in &self.paths {
            writeln!(
                out,
                "  {:<30} {:>5.1}%  {}",
                path.path,
                path.rate(self.records) * 100.0,
                format_type_counts(&path.types),
            )?;
        }
        if self.truncated {
            writeln!(out, "  ... path table truncated at {} paths", MAX_PATHS)?;
        }

        if self.skipped > 0 {
            writeln!(out)?;
            writeln!(out, "{} unparseable lines skipped", self.skipped)?;
        }

        Ok(())
    }

    /// The machine-readable form of the report, for `--json`.
    pub fn to_json(&self) -> Value {
        Value::Object(vec![
            ("records".to_string(), Value::Int(self.records as i64)),
            ("skipped".to_string(), Value::Int(self.skipped as i64)),
            ("depth".to_string(), Value::Int(self.depth as i64)),
            ("truncated".to_string(), Value::Bool(self.truncated)),
            (
                "paths".to_string(),
                Value::Array(
                    self.paths
                        .iter()
                        .map(|p| {
                            Value::Object(vec![
                                ("path".to_string(), Value::Str(p.path.clone())),
                                ("present".to_string(), Value::Int(p.present as i64)),
                                ("count".to_string(), Value::Int(p.count as i64)),
                                ("rate".to_string(), Value::Float(p.rate(self.records))),
                                ("types".to_string(), type_counts_json(&p.types)),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

fn record_occurrence(table: &mut Vec<PathStat>, truncated: &mut bool, path: &str, type_name: &'static str) {
    if let Some(stat) = table.iter_mut().find(|p| p.path == path) {
        stat.count += 1;
        record_type(&mut stat.types, type_name);
        return;
    }
    if table.len() >= MAX_PATHS {
        *truncated = true;
        return;
    }
    let mut stat = PathStat {
        path: path.to_string(),
        present: 0,
        count: 1,
        types: Vec::new(),
    };
    record_type(&mut stat.types, type_name);
    table.push(stat);
}

fn record_type(types: &mut Vec<(&'static str, u64)>, type_name: &'static str) {
    match types.iter_mut().find(|(name, _)| *name == type_name) {
        Some(entry) => entry.1 += 1,
        None => types.push((type_name, 1)),
    }
}

/// Visits every member of an object at `prefix`, recording each one that is
/// within `max_depth` levels and recursing into the ones that are objects or
/// arrays and still have depth budget left.
fn walk_object(
    members: &[(String, Value)],
    prefix: &str,
    depth: usize,
    max_depth: usize,
    seen: &mut HashSet<String>,
    table: &mut Vec<PathStat>,
    truncated: &mut bool,
) {
    for (key, value) in members {
        let child_depth = depth + 1;
        if child_depth > max_depth {
            continue;
        }
        let child_path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{}.{}", prefix, key)
        };
        record_occurrence(table, truncated, &child_path, value.type_name());
        seen.insert(child_path.clone());
        if child_depth < max_depth {
            match value {
                Value::Object(inner) => walk_object(inner, &child_path, child_depth, max_depth, seen, table, truncated),
                Value::Array(items) => walk_array(items, &child_path, child_depth, max_depth, seen, table, truncated),
                _ => {}
            }
        }
    }
}

/// Visits every element of an array at `prefix` under the synthetic
/// `prefix[]` path, the same way `walk_object` visits object members.
fn walk_array(
    items: &[Value],
    prefix: &str,
    depth: usize,
    max_depth: usize,
    seen: &mut HashSet<String>,
    table: &mut Vec<PathStat>,
    truncated: &mut bool,
) {
    let child_depth = depth + 1;
    if child_depth > max_depth {
        return;
    }
    let child_path = format!("{}[]", prefix);
    for item in items {
        record_occurrence(table, truncated, &child_path, item.type_name());
        seen.insert(child_path.clone());
        if child_depth < max_depth {
            match item {
                Value::Object(inner) => walk_object(inner, &child_path, child_depth, max_depth, seen, table, truncated),
                Value::Array(inner) => walk_array(inner, &child_path, child_depth, max_depth, seen, table, truncated),
                _ => {}
            }
        }
    }
}

fn type_counts_json(types: &[(&'static str, u64)]) -> Value {
    Value::Object(types.iter().map(|(name, count)| (name.to_string(), Value::Int(*count as i64))).collect())
}

fn format_type_counts(types: &[(&'static str, u64)]) -> String {
    types.iter().map(|(name, count)| format!("{}:{}", name, count)).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_for(text: &str, options: SchemaOptions) -> Schema {
        Schema::from_reader(text.as_bytes(), options).unwrap()
    }

    fn find<'a>(schema: &'a Schema, path: &str) -> &'a PathStat {
        schema.paths.iter().find(|p| p.path() == path).unwrap_or_else(|| panic!("no path {}", path))
    }

    #[test]
    fn default_options_match_the_documented_defaults() {
        let options = SchemaOptions::default();
        assert_eq!(options.depth, 3);
        assert_eq!(options.min_rate, 0.0);
    }

    #[test]
    fn top_level_keys_are_depth_one() {
        let text = "{\"id\":1,\"tags\":[\"a\"]}\n{\"id\":2}\n";
        let s = schema_for(text, SchemaOptions { depth: 1, ..SchemaOptions::default() });
        assert_eq!(s.records, 2);
        let id = find(&s, "id");
        assert_eq!(id.present(), 2);
        assert_eq!(id.count(), 2);
        assert_eq!(id.rate(s.records), 1.0);
        let tags = find(&s, "tags");
        assert_eq!(tags.present(), 1);
        assert_eq!(tags.types(), &[("array", 1)]);
        assert!(s.paths.iter().all(|p| !p.path().contains('[')));
    }

    #[test]
    fn depth_limits_how_far_the_walk_descends() {
        let text = "{\"messages\":[{\"role\":\"user\"}]}\n";
        let shallow = schema_for(text, SchemaOptions { depth: 2, ..SchemaOptions::default() });
        assert!(shallow.paths.iter().any(|p| p.path() == "messages[]"));
        assert!(!shallow.paths.iter().any(|p| p.path() == "messages[].role"));

        let deep = schema_for(text, SchemaOptions { depth: 3, ..SchemaOptions::default() });
        assert!(deep.paths.iter().any(|p| p.path() == "messages[].role"));
    }

    #[test]
    fn array_fan_out_counts_every_element_but_marks_presence_once() {
        let text = "{\"messages\":[{\"role\":\"user\"},{\"role\":\"assistant\"},{\"role\":\"user\"}]}\n\
                     {\"messages\":[]}\n\
                     {\"other\":true}\n";
        let s = schema_for(text, SchemaOptions::default());
        let role = find(&s, "messages[].role");
        assert_eq!(role.present(), 1);
        assert_eq!(role.count(), 3);
        assert_eq!(role.rate(s.records), 1.0 / 3.0);
    }

    #[test]
    fn a_path_with_mixed_types_records_both() {
        let text = "{\"v\":1}\n{\"v\":\"x\"}\n";
        let s = schema_for(text, SchemaOptions::default());
        let v = find(&s, "v");
        assert_eq!(v.count(), 2);
        assert_eq!(v.types(), &[("int", 1), ("string", 1)]);
    }

    #[test]
    fn blank_and_unparseable_lines_are_skipped_not_counted_as_records() {
        let text = "{\"a\":1}\n\n{bad}\n{\"a\":2}\n";
        let s = schema_for(text, SchemaOptions::default());
        assert_eq!(s.records, 2);
        assert_eq!(s.skipped, 1);
    }

    #[test]
    fn min_rate_hides_infrequent_paths() {
        let text = "{\"a\":1}\n{\"a\":1,\"b\":1}\n{\"a\":1}\n{\"a\":1}\n";
        let s = schema_for(text, SchemaOptions { min_rate: 0.5, ..SchemaOptions::default() });
        assert!(s.paths.iter().any(|p| p.path() == "a"));
        assert!(!s.paths.iter().any(|p| p.path() == "b"));
    }

    #[test]
    fn paths_are_reported_in_alphabetical_order() {
        let text = "{\"z\":1,\"a\":1,\"m\":1}\n";
        let s = schema_for(text, SchemaOptions::default());
        let names: Vec<&str> = s.paths.iter().map(|p| p.path()).collect();
        assert_eq!(names, vec!["a", "m", "z"]);
    }

    #[test]
    fn records_that_are_themselves_arrays_use_a_leading_fan_out() {
        let text = "[{\"id\":1},{\"id\":2}]\n";
        let s = schema_for(text, SchemaOptions::default());
        let elements = find(&s, "[]");
        assert_eq!(elements.count(), 2);
        let id = find(&s, "[].id");
        assert_eq!(id.count(), 2);
    }

    #[test]
    fn truncates_the_path_table_past_the_cap() {
        let mut line = String::from("{");
        for i in 0..(MAX_PATHS + 1) {
            if i > 0 {
                line.push(',');
            }
            line.push_str(&format!("\"k{}\":1", i));
        }
        line.push_str("}\n");
        let s = schema_for(&line, SchemaOptions::default());
        assert_eq!(s.paths.len(), MAX_PATHS);
        assert!(s.truncated);
    }

    #[test]
    fn write_report_includes_the_expected_sections() {
        let text = "{\"id\":1,\"meta\":{\"source\":\"web\"}}\n{bad}\n";
        let s = schema_for(text, SchemaOptions::default());
        let mut out = Vec::new();
        s.write_report(&mut out).unwrap();
        let report = String::from_utf8(out).unwrap();
        assert!(report.contains("1 records, depth 3"));
        assert!(report.contains("meta.source"));
        assert!(report.contains("string:1"));
        assert!(report.contains("1 unparseable lines skipped"));
    }

    #[test]
    fn to_json_reports_path_details() {
        let text = "{\"id\":1}\n{\"id\":2}\n";
        let s = schema_for(text, SchemaOptions::default());
        let json = s.to_json();
        assert_eq!(json.get("records").and_then(Value::as_i64), Some(2));
        let paths = json.get("paths").unwrap().as_array().unwrap();
        let id = paths.iter().find(|p| p.get("path").and_then(Value::as_str) == Some("id")).unwrap();
        assert_eq!(id.get("present").and_then(Value::as_i64), Some(2));
        assert_eq!(id.get("rate").and_then(Value::as_f64), Some(1.0));
    }

    #[test]
    fn empty_input_reports_zero_records_and_no_paths() {
        let s = schema_for("", SchemaOptions::default());
        assert_eq!(s.records, 0);
        assert!(s.paths.is_empty());
        let mut out = Vec::new();
        s.write_report(&mut out).unwrap();
        assert!(String::from_utf8(out).unwrap().contains("0 records, depth 3"));
    }
}
