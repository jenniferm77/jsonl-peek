//! Library backing the `jsonl-peek` binary.
//!
//! Every module here is single pass and holds no third-party dependencies.
//! See each module's doc comment for what it does; [`json`], [`lines`],
//! [`hist`] and [`rng`] are the primitives, [`path`] is the field selector
//! that [`stats`] is built on, and [`schema`] walks a record's whole shape
//! instead of one path at a time.

pub mod hist;
pub mod json;
pub mod lines;
pub mod path;
pub mod rng;
pub mod schema;
pub mod stats;

pub use path::{FieldPath, PathError, Segment};
pub use schema::{PathStat, Schema, SchemaOptions};
pub use stats::{FieldStats, Issue, KeyStat, Stats, StatsOptions};
