# jsonl-peek

Quick health checks for JSONL (newline-delimited JSON) datasets, in one
streaming pass and with a bounded memory footprint.

Fine-tuning and pretraining corpora ship as multi-gigabyte `.jsonl` files. Before
spending GPU hours on one you want to know: how many records are there, how long
are they, which keys actually exist, which lines are silently broken, and what
the value distribution of `meta.source` or `messages[].role` looks like. Doing
that with `jq` means parsing the whole file into memory; doing it with `wc -l`
tells you almost nothing.

`jsonl-peek` reads the file once, never holds more than one line at a time, and
answers all of it.

- **Zero dependencies.** The JSON parser, the percentile histogram and the random
  number generator are all in this repository. `cargo build` works offline and
  the entire dependency tree is auditable by reading `src/`.
- **Unbiased sampling** of a stream whose length you do not know in advance
  (reservoir sampling, seedable for reproducibility).
- **Strict parsing.** Trailing commas, single quotes, `NaN`, lone UTF-16
  surrogates and raw control characters are reported with a line and column,
  not silently accepted.

## Install

```sh
git clone https://github.com/jenniferm77/jsonl-peek.git
cd jsonl-peek
cargo install --path .
```

Or run it straight out of the checkout:

```sh
cargo run --release -- stats data.jsonl
```

Requires Rust 1.70 or newer. There is nothing to configure and nothing to
download.

## Usage

### `stats` - the file health check

```sh
$ jsonl-peek stats sample.jsonl
file    sample.jsonl
lines          2,003   blank 1   invalid 2   valid 2,000
bytes        936,782   (914.8 KiB)
top level  object:2,000

line length in bytes
  min 25   p50 472   p90 624   p99 720   max 791   mean 466.9

top level keys over 2,000 objects
  key                          count     rate  types
  id                            2,000  100.0%  int:2,000
  messages                      2,000  100.0%  array:2,000
  meta                          2,000  100.0%  object:2,000
  tags                            285   14.2%  array:285

invalid lines (2 total, showing 2)
  line 501 col 23: unexpected ','
  line 1201 col 40: unexpected end of input
```

The last block is usually the reason you ran the tool: it gives you the exact
line and byte column of every record your training pipeline would choke on.

### `--field` - value distributions

```sh
$ jsonl-peek stats --field meta.source --field 'messages[].role' --top 4 sample.jsonl
...
field meta.source
  present in 2,000 of 2,000 records (100.0%), 2,000 values, types string:2,000
  4 distinct values
           503   25.1%  "web"
           501   25.1%  "code"
           499   24.9%  "forum"
           497   24.9%  "books"

field messages[].role
  present in 2,000 of 2,000 records (100.0%), 4,666 values, types string:4,666
  3 distinct values
         2,000   42.9%  "assistant"
         2,000   42.9%  "user"
           666   14.3%  "system"
```

### `schema` - what is actually in there

```sh
$ jsonl-peek schema --depth 3 sample.jsonl
2000 records, depth 3

  path                                      rate  types
  id                                      100.0%  int:2000
  messages                                100.0%  array:2000
  messages[]                              100.0%  object:4666
  messages[].content                      100.0%  string:4666
  messages[].role                         100.0%  string:4666
  meta                                    100.0%  object:2000
  meta.quality                            100.0%  float:2000
  meta.source                             100.0%  string:2000
  tags                                     14.2%  array:285
  tags[]                                   14.2%  string:570

2 unparseable lines skipped
```

`rate` is the share of records containing the path; `types` counts occurrences.
A key that shows two types (`int:1990 string:10`) is exactly the kind of thing
that breaks a loader three hours into a run.

### `head` and `sample`

```sh
$ jsonl-peek head -n 2 sample.jsonl            # first 2 lines
$ jsonl-peek sample -n 100 sample.jsonl        # 100 uniformly random lines
$ jsonl-peek sample -n 100 --seed 42 data.jsonl > subset.jsonl
```

`sample` is a true uniform sample of the whole file, not of its first N lines. It
holds only the selected lines in memory, so sampling 100 records out of a 40 GB
file costs a single sequential read. Blank lines are skipped and the output keeps
the original file order. Pass `--seed` to make the selection reproducible.

### Pipes and JSON output

Every command reads standard input when no file is given, and `stats` and
`schema` take `--json` for machine-readable output:

```sh
$ zstdcat shard-0000.jsonl.zst | jsonl-peek stats --json - | jq .line_length
{
  "min": 25,
  "p50": 472,
  "p90": 624,
  "p99": 720,
  "max": 791,
  "mean": 466.9226
}
```

## CLI reference

```
jsonl-peek head   [-n N] [FILE]
jsonl-peek sample [-n N] [--seed S] [FILE]
jsonl-peek stats  [--field PATH]... [--top N] [--max-errors N] [--json] [FILE]
jsonl-peek schema [--depth N] [--min-rate R] [--json] [FILE]
```

| Option | Commands | Meaning |
| --- | --- | --- |
| `-n N` | head, sample | number of lines (default 10) |
| `--seed S` | sample | seed the sampler for a reproducible subset |
| `--field PATH` | stats | profile a field; repeatable |
| `--top N` | stats | distinct values listed per field (default 10) |
| `--max-errors N` | stats | broken lines shown (default 10) |
| `--depth N` | schema | levels to descend (default 3) |
| `--min-rate R` | schema | hide paths present in fewer than R of the records |
| `--json` | stats, schema | machine-readable output |

Exit status is `0` on success, `1` on a runtime error (missing file, unreadable
input) and `2` on a usage error.

### Field path syntax

| Path | Selects |
| --- | --- |
| `role` | the top level member `role` |
| `meta.source` | a member of a nested object |
| `messages[0].content` | the first array element |
| `messages[-1].content` | the last array element |
| `messages[].role` | every array element |
| `[0].id` | records that are themselves arrays |

## Library

The binary is a thin shell over the library, which is usable on its own:

```rust
use jsonl_peek::{FieldPath, Stats, StatsOptions};

let file = std::fs::File::open("data.jsonl")?;
let options = StatsOptions {
    fields: vec![FieldPath::parse("messages[].role")?],
    ..StatsOptions::default()
};
let stats = Stats::from_reader(std::io::BufReader::new(file), options)?;

println!("{} of {} lines parsed", stats.valid, stats.lines);
for issue in &stats.issues {
    println!("line {} col {}: {}", issue.line, issue.column, issue.reason);
}
for (value, count) in stats.fields[0].top(5) {
    println!("{:>8}  {}", count, value);
}
```

| Module | What it holds |
| --- | --- |
| `jsonl_peek::json` | `parse`, `Value`, `ParseError` - the strict JSON reader |
| `jsonl_peek::lines` | `LineReader`, a reusable-buffer line splitter |
| `jsonl_peek::hist` | `Histogram`, log-bucketed approximate percentiles |
| `jsonl_peek::rng` | `SplitMix64`, `Reservoir` |
| `jsonl_peek::path` | `FieldPath` |
| `jsonl_peek::stats` | `Stats`, `StatsOptions` |
| `jsonl_peek::schema` | `Schema`, `SchemaOptions` |

The library name is fixed to `jsonl_peek` regardless of what the package is
called, so `use jsonl_peek::...` always works.

## Scope and limits

What it does:

- streams the input, one line at a time, with a reusable buffer
- handles CRLF endings, a UTF-8 BOM and a missing final newline
- distinguishes integers from floats when reporting types
- caps its own memory: the key table (512 keys), the per-field value table
  (10,000 distinct values), the schema path table (2,000 paths) and the error
  list (10 entries) all stop growing, and say so in the report when they do

What it deliberately does not do:

- **No query language.** It profiles fields, it does not filter or transform.
  Use `jq` for that.
- **Percentiles are approximate.** Line lengths go into a log-bucketed
  histogram, so quantiles carry up to ~3% relative error above 16 bytes.
  `min`, `max`, `mean` and all counts are exact.
- **Numbers beyond `i64`** (or with a fraction or exponent) are held as `f64`,
  so a 19-digit integer id loses precision when re-serialised.
- **No compression.** Pipe through `zstdcat` / `zcat` / `gzip -dc`.
- **No parallelism.** One thread, one sequential read.
- **Records must be UTF-8.** A line that is not is counted as invalid and
  reported with the byte offset where decoding failed.

## Test

```sh
cargo test
```

Covers the JSON parser against a table of accepted and rejected documents
(including the near-JSON that real datasets are full of), the histogram's error
bound, the statistical properties of the reservoir sampler, and the CLI itself
end to end against a file on disk.

```sh
cargo clippy --all-targets
```

## License

MIT. See [LICENSE](LICENSE).
