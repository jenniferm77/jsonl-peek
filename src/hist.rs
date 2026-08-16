//! A fixed-memory histogram for approximate quantiles.
//!
//! Percentiles over a multi-gigabyte file cannot be computed by keeping every
//! sample around. This histogram buckets values on a log scale with 16
//! sub-buckets per octave, which bounds the relative error of any reported
//! quantile to about 3% (values below 16 are counted exactly), while using a
//! few kilobytes no matter how many samples arrive.

/// Number of sub-buckets per power of two. 16 gives ~3% relative error.
const SUB_BITS: u32 = 4;
const SUB_COUNT: u64 = 1 << SUB_BITS;

/// Streaming histogram over `u64` values.
#[derive(Debug, Clone, Default)]
pub struct Histogram {
    buckets: Vec<u64>,
    count: u64,
    sum: u128,
    min: u64,
    max: u64,
}

impl Histogram {
    /// Creates an empty histogram.
    pub fn new() -> Self {
        Histogram {
            buckets: Vec::new(),
            count: 0,
            sum: 0,
            min: u64::MAX,
            max: 0,
        }
    }

    /// Records one observation.
    pub fn record(&mut self, value: u64) {
        let idx = Self::bucket_of(value);
        if idx >= self.buckets.len() {
            self.buckets.resize(idx + 1, 0);
        }
        self.buckets[idx] += 1;
        self.count += 1;
        self.sum += u128::from(value);
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Number of recorded observations.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Smallest observed value, or `None` when empty.
    pub fn min(&self) -> Option<u64> {
        (self.count > 0).then_some(self.min)
    }

    /// Largest observed value, or `None` when empty.
    pub fn max(&self) -> Option<u64> {
        (self.count > 0).then_some(self.max)
    }

    /// Exact sum of all observations.
    pub fn sum(&self) -> u128 {
        self.sum
    }

    /// Exact arithmetic mean, or `None` when empty.
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum as f64 / self.count as f64)
    }

    /// Approximate quantile for `q` in `[0, 1]`.
    ///
    /// The result is the midpoint of the bucket holding the `q`-th observation,
    /// clamped to the exact minimum and maximum. `q <= 0` returns the minimum
    /// and `q >= 1` returns the maximum exactly.
    pub fn quantile(&self, q: f64) -> Option<u64> {
        if self.count == 0 {
            return None;
        }
        if q <= 0.0 {
            return Some(self.min);
        }
        if q >= 1.0 {
            return Some(self.max);
        }
        // Rank of the wanted observation, 1-based.
        let rank = (q * self.count as f64).ceil().max(1.0) as u64;
        let mut seen = 0u64;
        for (idx, &n) in self.buckets.iter().enumerate() {
            if n == 0 {
                continue;
            }
            seen += n;
            if seen >= rank {
                let (lo, hi) = Self::bucket_range(idx);
                let mid = lo + (hi - lo) / 2;
                return Some(mid.clamp(self.min, self.max));
            }
        }
        Some(self.max)
    }

    /// Bucket index for a value. Monotonically non-decreasing in `value`.
    fn bucket_of(value: u64) -> usize {
        if value < SUB_COUNT {
            return value as usize;
        }
        let msb = 63 - value.leading_zeros(); // position of the highest set bit
        let sub = (value >> (msb - SUB_BITS)) & (SUB_COUNT - 1);
        let octave = u64::from(msb - SUB_BITS) + 1;
        (octave * SUB_COUNT + sub) as usize
    }

    /// Half-open `[lo, hi)` value range covered by a bucket index.
    ///
    /// The upper bound saturates at `u64::MAX` for the very last bucket.
    fn bucket_range(idx: usize) -> (u64, u64) {
        let idx = idx as u64;
        if idx < SUB_COUNT {
            return (idx, idx + 1);
        }
        let octave = idx / SUB_COUNT;
        let sub = idx % SUB_COUNT;
        let msb = octave + u64::from(SUB_BITS) - 1;
        let width = 1u64 << (msb - u64::from(SUB_BITS));
        let lo = (SUB_COUNT + sub) * width;
        (lo, lo.saturating_add(width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_histogram_reports_nothing() {
        let h = Histogram::new();
        assert_eq!(h.count(), 0);
        assert_eq!(h.min(), None);
        assert_eq!(h.max(), None);
        assert_eq!(h.mean(), None);
        assert_eq!(h.quantile(0.5), None);
    }

    #[test]
    fn small_values_are_exact() {
        let mut h = Histogram::new();
        for v in 0..16u64 {
            h.record(v);
        }
        assert_eq!(h.count(), 16);
        assert_eq!(h.min(), Some(0));
        assert_eq!(h.max(), Some(15));
        assert_eq!(h.quantile(0.0), Some(0));
        assert_eq!(h.quantile(1.0), Some(15));
        assert_eq!(h.quantile(0.5), Some(7));
    }

    #[test]
    fn bucket_indices_are_monotonic_and_consistent() {
        let mut previous = 0usize;
        for v in 0..5000u64 {
            let idx = Histogram::bucket_of(v);
            assert!(idx >= previous, "bucket index went backwards at {}", v);
            previous = idx;
            let (lo, hi) = Histogram::bucket_range(idx);
            assert!(lo <= v && v < hi, "value {} not inside bucket {:?}", v, (lo, hi));
        }
    }

    #[test]
    fn extreme_values_do_not_panic() {
        let mut h = Histogram::new();
        h.record(0);
        h.record(u64::MAX);
        let idx = Histogram::bucket_of(u64::MAX);
        let (lo, hi) = Histogram::bucket_range(idx);
        assert!(lo < u64::MAX, "last bucket should still start below the top");
        assert_eq!(hi, u64::MAX, "the top bucket saturates instead of overflowing");
        assert_eq!(h.max(), Some(u64::MAX));
        assert_eq!(h.quantile(1.0), Some(u64::MAX));
        assert_eq!(h.quantile(0.0), Some(0));
    }

    #[test]
    fn quantiles_stay_within_three_percent() {
        let mut h = Histogram::new();
        for v in 1..=100_000u64 {
            h.record(v);
        }
        assert_eq!(h.count(), 100_000);
        assert_eq!(h.mean(), Some(50_000.5));
        for &(q, expected) in &[(0.5, 50_000.0), (0.9, 90_000.0), (0.99, 99_000.0)] {
            let got = h.quantile(q).unwrap() as f64;
            let error = (got - expected).abs() / expected;
            assert!(error < 0.03, "q={} got={} expected={}", q, got, expected);
        }
        assert_eq!(h.quantile(1.0), Some(100_000));
    }

    #[test]
    fn single_value_repeated() {
        let mut h = Histogram::new();
        for _ in 0..1000 {
            h.record(4096);
        }
        assert_eq!(h.quantile(0.5), Some(4096));
        assert_eq!(h.quantile(0.99), Some(4096));
        assert_eq!(h.mean(), Some(4096.0));
    }
}
