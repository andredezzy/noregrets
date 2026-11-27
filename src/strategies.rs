//! The blueprint's strategies, keyed by (bucket, history) bytes as [`crate::cfr::make_key`] builds
//! them, held by history: the infosets of one betting history sit together, their buckets
//! ascending beside their probabilities, so a history's bytes are kept once instead of once per
//! bucket. r3's 273M infosets average 18.6 key bytes and 2.4 actions: one record per key took ~45
//! bytes an infoset (12.3 GB). They fall in 57.8M histories, 4.7 infosets each: by history r3 holds
//! 5.9 GB resident after its load (peak 6.6 GB), ~22 bytes an infoset, of which 40 are each
//! history's [`Head`]. The probabilities are the f32s stored, bit for bit: bpcheck read all 273M
//! back from r3's file (2026-11-27).
//!
//! A map read from a file or collected is held by history at once. Inserts into it go to a table
//! of their own, which lookups read first, until [`StrategyMap::freeze`] folds them in.

use crate::record::{Arena, Record};
use hashbrown::HashTable;
use serde::{Deserialize, Serialize};

/// Strategy store keyed by (bucket, history) bytes. It serializes as a map of keys to
/// probabilities (len + entries), so every .bin file loads as it is.
pub struct StrategyMap {
    groups: Groups,
    loose: Loose,
    /// Keys in `loose` that `groups` lacks.
    added: usize,
}

impl StrategyMap {
    pub fn with_capacity(n: usize) -> Self {
        StrategyMap { groups: Groups::default(), loose: Loose::with_capacity(n), added: 0 }
    }

    pub fn len(&self) -> usize {
        self.groups.len() + self.added
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, key: &[u8]) -> Option<&[f32]> {
        if !self.loose.is_empty() {
            if let Some(probs) = self.loose.get(key) {
                return Some(probs);
            }
        }
        let (bucket, hist) = split(key)?;
        self.groups.get(bucket, hist)
    }

    /// `key`'s probabilities, in place of any it had.
    pub fn insert(&mut self, key: &[u8], probs: &[f32]) {
        let fresh = self.loose.insert(key, probs);
        if fresh && split(key).is_none_or(|(bucket, hist)| self.groups.get(bucket, hist).is_none()) {
            self.added += 1;
        }
    }

    /// Folds every insert since the last freeze into the infosets held by history.
    pub fn freeze(&mut self) {
        if self.loose.iter().all(|(key, _)| split(key).is_none()) {
            return;
        }
        let mut build = Builder::with_capacity(self.len());
        let mut short = Loose::with_capacity(0);
        for (key, probs) in self.loose.iter() {
            match split(key) {
                Some(_) => build.push(key, probs),
                None => {
                    short.insert(key, probs);
                }
            }
        }
        for (key, probs) in self.groups.iter() {
            if self.loose.get(&key).is_none() {
                build.push(&key, probs);
            }
        }
        self.added = short.len();
        self.loose = short;
        self.groups = build.finish();
    }

    /// Every infoset, in no set order.
    pub fn iter(&self) -> impl Iterator<Item = (Vec<u8>, &[f32])> {
        let shadowed = |key: &[u8]| !self.loose.is_empty() && self.loose.get(key).is_some();
        self.loose
            .iter()
            .map(|(key, probs)| (key.to_vec(), probs))
            .chain(self.groups.iter().filter(move |(key, _)| !shadowed(key.as_slice())))
    }

    pub fn keys(&self) -> impl Iterator<Item = Vec<u8>> + '_ {
        self.iter().map(|(key, _)| key)
    }

    pub fn values(&self) -> impl Iterator<Item = &[f32]> {
        self.iter().map(|(_, probs)| probs)
    }
}

/// A key's bucket and history; None for a key too short to hold a bucket.
fn split(key: &[u8]) -> Option<(u16, &[u8])> {
    let (bucket, hist) = key.split_at_checked(2)?;
    Some((u16::from_le_bytes([bucket[0], bucket[1]]), hist))
}

fn key_of(bucket: u16, hist: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(2 + hist.len());
    key.extend_from_slice(&bucket.to_le_bytes());
    key.extend_from_slice(hist);
    key
}

impl Default for StrategyMap {
    fn default() -> Self {
        StrategyMap::with_capacity(0)
    }
}

impl Clone for StrategyMap {
    fn clone(&self) -> Self {
        let mut loose = Loose::with_capacity(self.loose.len());
        for (key, probs) in self.loose.iter() {
            loose.insert(key, probs);
        }
        StrategyMap { groups: self.groups.clone(), loose, added: self.added }
    }
}

impl PartialEq for StrategyMap {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(key, probs)| other.get(&key) == Some(probs))
    }
}

impl std::fmt::Debug for StrategyMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StrategyMap({} infosets)", self.len())
    }
}

impl FromIterator<(Vec<u8>, Vec<f32>)> for StrategyMap {
    fn from_iter<I: IntoIterator<Item = (Vec<u8>, Vec<f32>)>>(entries: I) -> Self {
        let mut map = StrategyMap::default();
        for (key, probs) in entries {
            map.insert(&key, &probs);
        }
        map.freeze();
        map
    }
}

impl Serialize for StrategyMap {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(self.len()))?;
        for (key, probs) in self.iter() {
            map.serialize_entry(&key[..], probs)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for StrategyMap {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Entries;
        impl<'de> serde::de::Visitor<'de> for Entries {
            type Value = StrategyMap;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a map of infoset keys to probabilities")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut entries: A) -> Result<StrategyMap, A::Error> {
                let mut build = Builder::with_capacity(entries.size_hint().unwrap_or(0));
                let mut short = Loose::with_capacity(0);
                while let Some((key, probs)) = entries.next_entry::<Vec<u8>, Vec<f32>>()? {
                    match split(&key) {
                        Some(_) => build.push(&key, &probs),
                        None => {
                            short.insert(&key, &probs);
                        }
                    }
                }
                Ok(StrategyMap { groups: build.finish(), added: short.len(), loose: short })
            }
        }
        d.deserialize_map(Entries)
    }
}

/// One history's infosets in [`Groups`].
#[derive(Clone, Copy)]
struct Head {
    /// The history's bytes from here in `Groups::hists`.
    hist: u64,
    /// Its infosets' buckets from here in `Groups::buckets`, ascending.
    first: u64,
    /// Their probabilities from here in `Groups::floats`, in their buckets' order.
    floats: u64,
    /// With `actions` 0, each infoset's first probability after `floats`, then one past the
    /// last, from here in `Groups::starts`.
    starts: u64,
    count: u32,
    hist_len: u16,
    /// Every infoset's action count, or 0 when they differ.
    actions: u8,
}

/// The infosets held by history.
#[derive(Clone, Default)]
struct Groups {
    /// Indexes into `heads`, by history.
    table: HashTable<u32>,
    heads: Vec<Head>,
    hists: Vec<u8>,
    buckets: Vec<u16>,
    floats: Vec<f32>,
    starts: Vec<u32>,
    hasher: ahash::RandomState,
}

impl Groups {
    fn len(&self) -> usize {
        self.buckets.len()
    }

    fn hist(&self, h: &Head) -> &[u8] {
        &self.hists[h.hist as usize..h.hist as usize + h.hist_len as usize]
    }

    fn head(&self, hist: &[u8]) -> Option<&Head> {
        let hash = self.hasher.hash_one(hist);
        self.table.find(hash, |&i| self.hist(&self.heads[i as usize]) == hist).map(|&i| &self.heads[i as usize])
    }

    /// The probabilities of `h`'s `i`th infoset.
    fn probs(&self, h: &Head, i: usize) -> &[f32] {
        let (from, to) = if h.actions > 0 {
            let n = h.actions as usize;
            (i * n, (i + 1) * n)
        } else {
            let s = h.starts as usize + i;
            (self.starts[s] as usize, self.starts[s + 1] as usize)
        };
        &self.floats[h.floats as usize + from..h.floats as usize + to]
    }

    fn get(&self, bucket: u16, hist: &[u8]) -> Option<&[f32]> {
        let h = self.head(hist)?;
        let buckets = &self.buckets[h.first as usize..h.first as usize + h.count as usize];
        let i = buckets.binary_search(&bucket).ok()?;
        Some(self.probs(h, i))
    }

    fn iter(&self) -> impl Iterator<Item = (Vec<u8>, &[f32])> {
        self.heads.iter().flat_map(move |h| {
            (0..h.count as usize).map(move |i| (key_of(self.buckets[h.first as usize + i], self.hist(h)), self.probs(h, i)))
        })
    }
}

/// Collects infosets in any order and lays them out by history. Each key is pushed once.
struct Builder {
    table: HashTable<u32>,
    /// Each history's bytes in `hists`: (from, length), in the order they first came.
    spans: Vec<(u64, u16)>,
    hists: Vec<u8>,
    hasher: ahash::RandomState,
    /// Every infoset as it came: its history's index in `spans`, bucket and probabilities.
    ids: Vec<u32>,
    buckets: Vec<u16>,
    actions: Vec<u8>,
    floats: Vec<f32>,
}

impl Builder {
    fn with_capacity(n: usize) -> Self {
        Builder {
            table: HashTable::new(),
            spans: Vec::new(),
            hists: Vec::new(),
            hasher: ahash::RandomState::new(),
            ids: Vec::with_capacity(n),
            buckets: Vec::with_capacity(n),
            actions: Vec::with_capacity(n),
            floats: Vec::new(),
        }
    }

    /// `key` must hold a bucket ([`split`]).
    fn push(&mut self, key: &[u8], probs: &[f32]) {
        let (bucket, hist) = split(key).expect("a key with a bucket");
        assert!(hist.len() <= u16::MAX as usize && probs.len() <= u8::MAX as usize);
        let Builder { table, spans, hists, hasher, .. } = self;
        let span = |spans: &[(u64, u16)], i: u32| {
            let (from, len) = spans[i as usize];
            from as usize..from as usize + len as usize
        };
        let hash = hasher.hash_one(hist);
        let found = table.find(hash, |&i| &hists[span(spans, i)] == hist).copied();
        let id = match found {
            Some(i) => i,
            None => {
                let i = u32::try_from(spans.len()).expect("fewer than 2^32 histories");
                spans.push((hists.len() as u64, hist.len() as u16));
                hists.extend_from_slice(hist);
                table.insert_unique(hash, i, |&j| hasher.hash_one(&hists[span(spans, j)]));
                i
            }
        };
        self.ids.push(id);
        self.buckets.push(bucket);
        self.actions.push(probs.len() as u8);
        self.floats.extend_from_slice(probs);
    }

    fn finish(self) -> Groups {
        let Builder { table, spans, hists, hasher, ids, buckets, actions, floats } = self;
        const UNSET: u16 = u16::MAX;
        const MIXED: u16 = u16::MAX - 1;
        let g = spans.len();
        let mut count = vec![0u32; g];
        let mut n_floats = vec![0u64; g];
        let mut acts = vec![UNSET; g];
        for (i, &id) in ids.iter().enumerate() {
            let k = id as usize;
            count[k] += 1;
            n_floats[k] += u64::from(actions[i]);
            let a = u16::from(actions[i]);
            acts[k] = if acts[k] == UNSET || acts[k] == a { a } else { MIXED };
        }
        let mut heads = Vec::with_capacity(g);
        let (mut first, mut at_float, mut at_start) = (0u64, 0u64, 0u64);
        for k in 0..g {
            // WARNING: an infoset with no actions has action count 0, which a head reads as "they
            // differ": its history keeps per-infoset starts.
            let uniform = acts[k] != MIXED && acts[k] != 0;
            heads.push(Head {
                hist: spans[k].0,
                first,
                floats: at_float,
                starts: at_start,
                count: count[k],
                hist_len: spans[k].1,
                actions: if uniform { acts[k] as u8 } else { 0 },
            });
            first += u64::from(count[k]);
            at_float += n_floats[k];
            if !uniform {
                at_start += u64::from(count[k]) + 1;
            }
        }
        drop((count, acts));
        let mut out_buckets = vec![0u16; ids.len()];
        let mut out_floats = vec![0f32; floats.len()];
        let mut starts = vec![0u32; at_start as usize];
        let mut placed = vec![0u32; g];
        let mut placed_floats = vec![0u64; g];
        let mut from = 0usize;
        for (i, &id) in ids.iter().enumerate() {
            let (k, n) = (id as usize, actions[i] as usize);
            let h = &heads[k];
            out_buckets[(h.first + u64::from(placed[k])) as usize] = buckets[i];
            let to = (h.floats + placed_floats[k]) as usize;
            out_floats[to..to + n].copy_from_slice(&floats[from..from + n]);
            if h.actions == 0 {
                starts[(h.starts + u64::from(placed[k])) as usize] = placed_floats[k] as u32;
            }
            placed[k] += 1;
            placed_floats[k] += n as u64;
            from += n;
        }
        for (k, h) in heads.iter().enumerate() {
            if h.actions == 0 {
                starts[(h.starts + u64::from(h.count)) as usize] = n_floats[k] as u32;
            }
        }
        drop((ids, buckets, actions, floats, placed, placed_floats, n_floats));
        let mut groups = Groups { table, heads, hists, buckets: out_buckets, floats: out_floats, starts, hasher };
        groups.sort_buckets();
        groups
    }
}

impl Groups {
    /// Orders each history's infosets by bucket, their probabilities with them.
    fn sort_buckets(&mut self) {
        let Groups { heads, buckets, floats, starts, .. } = self;
        let (mut order, mut sorted_buckets, mut sorted_floats, mut sorted_starts) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for h in heads.iter() {
            let (b0, n) = (h.first as usize, h.count as usize);
            let bs = &mut buckets[b0..b0 + n];
            if bs.windows(2).all(|w| w[0] < w[1]) {
                continue;
            }
            order.clear();
            order.extend(0..n);
            order.sort_unstable_by_key(|&i| bs[i]);
            assert!(order.windows(2).all(|w| bs[w[0]] < bs[w[1]]), "a key pushed twice");
            sorted_buckets.clear();
            sorted_buckets.extend(order.iter().map(|&i| bs[i]));
            bs.copy_from_slice(&sorted_buckets);
            let f0 = h.floats as usize;
            sorted_floats.clear();
            if h.actions > 0 {
                let a = h.actions as usize;
                for &i in &order {
                    sorted_floats.extend_from_slice(&floats[f0 + i * a..f0 + (i + 1) * a]);
                }
            } else {
                let s0 = h.starts as usize;
                sorted_starts.clear();
                for &i in &order {
                    sorted_starts.push(sorted_floats.len() as u32);
                    sorted_floats.extend_from_slice(&floats[f0 + starts[s0 + i] as usize..f0 + starts[s0 + i + 1] as usize]);
                }
                starts[s0..s0 + n].copy_from_slice(&sorted_starts);
            }
            floats[f0..f0 + sorted_floats.len()].copy_from_slice(&sorted_floats);
        }
    }
}

/// Infosets one record each in an arena, behind thin pointers: the inserts a map takes after
/// it was laid out by history.
struct Loose {
    table: HashTable<Record<4>>,
    arena: Arena,
    hasher: ahash::RandomState,
}

impl Loose {
    fn with_capacity(n: usize) -> Self {
        Loose { table: HashTable::with_capacity(n), arena: Arena::new(), hasher: ahash::RandomState::new() }
    }

    fn len(&self) -> usize {
        self.table.len()
    }

    fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    fn get(&self, key: &[u8]) -> Option<&[f32]> {
        let hash = self.hasher.hash_one(key);
        self.table.find(hash, |r| r.key() == key).map(|r| r.floats())
    }

    /// `key`'s probabilities, in place of any it had; true when `key` was not there.
    fn insert(&mut self, key: &[u8], probs: &[f32]) -> bool {
        let Loose { table, arena, hasher } = self;
        let hash = hasher.hash_one(key);
        let mut fresh = Record::new_in(arena, key, probs.len());
        fresh.floats_mut().copy_from_slice(probs);
        match table.find_mut(hash, |r| r.key() == key) {
            Some(r) => {
                *r = fresh;
                false
            }
            None => {
                table.insert_unique(hash, fresh, |r| hasher.hash_one(r.key()));
                true
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = (&[u8], &[f32])> {
        self.table.iter().map(|r| (r.key(), r.floats()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::make_key;

    fn entries() -> Vec<(Vec<u8>, Vec<f32>)> {
        let mut out = Vec::new();
        for h in 0..40u8 {
            // the history's first byte tells it apart from every other one
            let hist: Vec<u8> = std::iter::once(h).chain((0..h % 7).map(|t| t.wrapping_mul(h))).collect();
            for bucket in [700u16, 3, 256, 0, 65_535, 41] {
                if (bucket as u32 + h as u32) % 3 == 0 {
                    continue;
                }
                // one history in five has infosets of different action counts
                let n = if h % 5 == 0 { 1 + (bucket % 4) as usize } else { 3 };
                let probs: Vec<f32> = (0..n).map(|a| (a as f32 + 1.0) / (bucket as f32 + h as f32 + 7.0)).collect();
                out.push((make_key(bucket, &hist).into_vec(), probs));
            }
        }
        out
    }

    #[test]
    fn a_map_laid_out_by_history_returns_every_infoset_bit_for_bit() {
        let all = entries();
        let map: StrategyMap = all.clone().into_iter().collect();
        assert!(map.loose.is_empty());
        assert_eq!(map.len(), all.len());
        for (key, probs) in &all {
            assert_eq!(map.get(key), Some(&probs[..]), "key {key:?}");
        }
        assert_eq!(map.get(&make_key(9, &[1, 2, 3])), None);
        assert_eq!(map.iter().count(), all.len());
        assert!(map.iter().all(|(key, probs)| all.iter().any(|(k, p)| *k == key && p[..] == *probs)));
    }

    #[test]
    fn a_map_written_and_read_back_is_the_same_and_keeps_the_old_file_layout() {
        let all = entries();
        let map: StrategyMap = all.clone().into_iter().collect();
        let bytes = bincode::serialize(&map).unwrap();
        let back: StrategyMap = bincode::deserialize(&bytes).unwrap();
        assert_eq!(back, map);
        // what the one-record-per-key map wrote: len, then (key, probabilities) entries
        let as_map: std::collections::HashMap<Vec<u8>, Vec<f32>> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(as_map.len(), all.len());
        let old = bincode::serialize(&all.iter().cloned().collect::<std::collections::BTreeMap<_, _>>()).unwrap();
        let read: StrategyMap = bincode::deserialize(&old).unwrap();
        assert_eq!(read, map);
    }

    #[test]
    fn inserts_after_the_layout_shadow_it_until_a_freeze_folds_them_in() {
        let all = entries();
        let mut map: StrategyMap = all.clone().into_iter().collect();
        let (key, _) = &all[5];
        map.insert(key, &[0.5, 0.5]);
        map.insert(&make_key(12, &[9, 9]), &[1.0]);
        map.insert(&[4u8], &[0.25, 0.75]);
        assert_eq!(map.len(), all.len() + 2);
        assert_eq!(map.get(key), Some(&[0.5, 0.5][..]));
        assert_eq!(map.get(&[4u8]), Some(&[0.25, 0.75][..]));
        let before = map.clone();
        map.freeze();
        assert_eq!(map.len(), all.len() + 2);
        assert_eq!(map, before);
        assert_eq!(map.get(key), Some(&[0.5, 0.5][..]));
        assert_eq!(map.get(&make_key(12, &[9, 9])), Some(&[1.0][..]));
        assert_eq!(map.get(&[4u8]), Some(&[0.25, 0.75][..]));
        assert_eq!(map.loose.len(), 1, "a key too short for a bucket stays loose");
    }

    #[test]
    fn a_map_filled_by_inserts_alone_answers_like_one_laid_out() {
        let all = entries();
        let mut map = StrategyMap::with_capacity(all.len());
        for (key, probs) in &all {
            map.insert(key, probs);
        }
        let laid: StrategyMap = all.into_iter().collect();
        assert_eq!(map, laid);
        map.freeze();
        assert_eq!(map, laid);
    }
}
