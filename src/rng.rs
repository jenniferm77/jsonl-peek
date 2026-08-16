//! Deterministic pseudo randomness plus single-pass reservoir sampling.
//!
//! `sample` needs an unbiased random subset of a file it can only read once and
//! cannot hold in memory. Algorithm R does exactly that: keep the first `k`
//! records, then for record number `n` (1-based) keep it with probability
//! `k/n`, evicting a uniformly chosen incumbent.
//!
//! The generator is SplitMix64, which is tiny, has no dependencies and is
//! seeded explicitly so that two runs over the same file produce the same
//! sample.

/// SplitMix64, the mixing function used to seed xoshiro generators.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator from an explicit seed.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Returns the next 64 pseudo random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a uniformly distributed value in `[0, n)`.
    ///
    /// Uses rejection sampling, so there is no modulo bias.
    ///
    /// # Panics
    /// Panics if `n` is zero.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "upper bound must be positive");
        // Largest multiple of n that fits; values at or above it are rejected.
        let limit = u64::MAX - (u64::MAX % n);
        loop {
            let x = self.next_u64();
            if x < limit {
                return x % n;
            }
        }
    }
}

/// A fixed-capacity uniform sample of a stream of unknown length.
#[derive(Debug)]
pub struct Reservoir<T> {
    capacity: usize,
    seen: u64,
    items: Vec<(u64, T)>,
    rng: SplitMix64,
}

impl<T> Reservoir<T> {
    /// Creates a reservoir holding at most `capacity` items.
    pub fn new(capacity: usize, seed: u64) -> Self {
        Reservoir {
            capacity,
            seen: 0,
            items: Vec::with_capacity(capacity.min(4096)),
            rng: SplitMix64::new(seed),
        }
    }

    /// Offers the item identified by `index` to the reservoir.
    ///
    /// `make` is only called when the item is actually retained, so callers can
    /// avoid allocating for the overwhelming majority of records that get
    /// dropped.
    pub fn offer<F: FnOnce() -> T>(&mut self, index: u64, make: F) {
        self.seen += 1;
        if self.capacity == 0 {
            return;
        }
        if self.items.len() < self.capacity {
            self.items.push((index, make()));
            return;
        }
        let slot = self.rng.below(self.seen);
        if (slot as usize) < self.capacity {
            self.items[slot as usize] = (index, make());
        }
    }

    /// How many items were offered so far.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// How many items are currently retained.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// True when nothing is retained.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Consumes the reservoir, returning the retained items ordered by the
    /// index they were offered with.
    pub fn into_sorted(mut self) -> Vec<(u64, T)> {
        self.items.sort_by_key(|(index, _)| *index);
        self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_deterministic_and_seed_sensitive() {
        let a: Vec<u64> = (0..5).map(|_| SplitMix64::new(7).next_u64()).collect();
        assert!(a.windows(2).all(|w| w[0] == w[1]), "same seed, same first draw");

        let mut r1 = SplitMix64::new(42);
        let mut r2 = SplitMix64::new(42);
        let mut r3 = SplitMix64::new(43);
        let s1: Vec<u64> = (0..8).map(|_| r1.next_u64()).collect();
        let s2: Vec<u64> = (0..8).map(|_| r2.next_u64()).collect();
        let s3: Vec<u64> = (0..8).map(|_| r3.next_u64()).collect();
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
        assert_eq!(s1.len(), s1.iter().collect::<std::collections::HashSet<_>>().len());
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = SplitMix64::new(1);
        let mut hits = [0u32; 6];
        for _ in 0..60_000 {
            let v = rng.below(6);
            assert!(v < 6);
            hits[v as usize] += 1;
        }
        // With 60k draws over 6 buckets, every bucket should be near 10k.
        for (face, &n) in hits.iter().enumerate() {
            assert!(
                (8_500..11_500).contains(&n),
                "face {} came up {} times, distribution looks broken",
                face,
                n
            );
        }
        assert_eq!(rng.below(1), 0);
    }

    #[test]
    fn reservoir_keeps_everything_when_stream_is_short() {
        let mut r = Reservoir::new(10, 99);
        for i in 0..4u64 {
            r.offer(i, || i * 2);
        }
        assert_eq!(r.seen(), 4);
        assert_eq!(r.len(), 4);
        assert!(!r.is_empty());
        assert_eq!(r.into_sorted(), vec![(0, 0), (1, 2), (2, 4), (3, 6)]);
    }

    #[test]
    fn reservoir_is_capped_sorted_and_reproducible() {
        let run = |seed: u64| -> Vec<u64> {
            let mut r = Reservoir::new(5, seed);
            for i in 0..1000u64 {
                r.offer(i, || i);
            }
            r.into_sorted().into_iter().map(|(_, v)| v).collect()
        };
        let a = run(2024);
        assert_eq!(a.len(), 5);
        assert!(a.windows(2).all(|w| w[0] < w[1]), "output must be in file order");
        assert_eq!(a, run(2024), "same seed must reproduce the sample");
        assert_ne!(a, run(2025));
    }

    #[test]
    fn reservoir_is_unbiased() {
        // Every one of the 20 positions should be picked roughly equally often
        // when sampling 2 out of 20, over many independent runs.
        let mut counts = [0u32; 20];
        for seed in 0..4000u64 {
            let mut r = Reservoir::new(2, seed);
            for i in 0..20u64 {
                r.offer(i, || i);
            }
            for (idx, _) in r.into_sorted() {
                counts[idx as usize] += 1;
            }
        }
        let expected = 4000.0 * 2.0 / 20.0; // 400
        for (idx, &n) in counts.iter().enumerate() {
            let error = (f64::from(n) - expected).abs() / expected;
            assert!(error < 0.25, "position {} picked {} times, expected ~{}", idx, n, expected);
        }
    }

    #[test]
    fn zero_capacity_reservoir_is_inert() {
        let mut r: Reservoir<u64> = Reservoir::new(0, 5);
        for i in 0..100u64 {
            r.offer(i, || panic!("must never materialise an item"));
        }
        assert_eq!(r.seen(), 100);
        assert!(r.is_empty());
        assert!(r.into_sorted().is_empty());
    }
}
