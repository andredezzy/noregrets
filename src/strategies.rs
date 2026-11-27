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
//!
//! [`StrategyMap::write_mapped`] writes that layout as a file and [`StrategyMap::map`] reads it in
//! place: every process that maps one file shares one copy in memory, the system's file cache, and
//! a map is ready at once instead of after a load of minutes.

use crate::record::{Arena, Record};
use hashbrown::HashTable;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(not(target_endian = "little"))]
compile_error!("a mapped strategies file is little-endian, as the machines that write and read it are");

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

/// One history's infosets in [`Groups`]. Its bytes are a mapped file's own, field by field.
#[derive(Clone, Copy)]
#[repr(C)]
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
    pad: u8,
}

const _: () = assert!(std::mem::size_of::<Head>() == 40);

/// An array the groups own, or one inside the mapped file they keep open.
enum Store<T: 'static> {
    Owned(Vec<T>),
    /// It lives as long as `Groups::map`: the groups never hand it out longer than they live.
    Mapped(&'static [T]),
}

impl<T> std::ops::Deref for Store<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Store::Owned(v) => v,
            Store::Mapped(s) => s,
        }
    }
}

impl<T: Clone> Clone for Store<T> {
    fn clone(&self) -> Self {
        match self {
            Store::Owned(v) => Store::Owned(v.clone()),
            Store::Mapped(s) => Store::Mapped(s),
        }
    }
}

impl<T> Default for Store<T> {
    fn default() -> Self {
        Store::Owned(Vec::new())
    }
}

/// A history's place in the index: the same in every process and on every machine, unlike a seeded
/// hasher's.
fn fixed_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // splitmix64's finalizer: histories one token apart land far apart
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

fn slot(hash: u64, n: usize) -> usize {
    ((u128::from(hash) * n as u128) >> 64) as usize
}

/// The index of `heads`: a head's number plus one at its history's slot, or the next free one
/// after it; 0 is free. A quarter of the slots stay free.
fn index_of(heads: &[Head], hists: &[u8]) -> Vec<u32> {
    let n = heads.len() + heads.len() / 3 + 1;
    let mut index = vec![0u32; n];
    for (k, h) in heads.iter().enumerate() {
        let hist = &hists[h.hist as usize..h.hist as usize + h.hist_len as usize];
        let mut s = slot(fixed_hash(hist), n);
        while index[s] != 0 {
            s = if s + 1 == n { 0 } else { s + 1 };
        }
        index[s] = u32::try_from(k + 1).expect("fewer than 2^32 - 1 histories");
    }
    index
}

/// The infosets held by history.
#[derive(Clone, Default)]
struct Groups {
    /// [`index_of`] the heads.
    index: Store<u32>,
    heads: Store<Head>,
    hists: Store<u8>,
    buckets: Store<u16>,
    floats: Store<f32>,
    starts: Store<u32>,
    /// The file the stores are mapped from, when they are.
    map: Option<Arc<memmap2::Mmap>>,
}

impl Groups {
    fn len(&self) -> usize {
        self.buckets.len()
    }

    fn hist(&self, h: &Head) -> &[u8] {
        &self.hists[h.hist as usize..h.hist as usize + h.hist_len as usize]
    }

    fn head(&self, hist: &[u8]) -> Option<&Head> {
        let n = self.index.len();
        if n == 0 {
            return None;
        }
        let mut s = slot(fixed_hash(hist), n);
        loop {
            let v = self.index[s];
            if v == 0 {
                return None;
            }
            let h = &self.heads[(v - 1) as usize];
            if self.hist(h) == hist {
                return Some(h);
            }
            s = if s + 1 == n { 0 } else { s + 1 };
        }
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
                pad: 0,
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
        drop((ids, buckets, actions, floats, placed, placed_floats, n_floats, table, hasher));
        let (mut buckets, mut floats) = (out_buckets, out_floats);
        sort_buckets(&heads, &mut buckets, &mut floats, &mut starts);
        let index = index_of(&heads, &hists);
        Groups {
            index: Store::Owned(index),
            heads: Store::Owned(heads),
            hists: Store::Owned(hists),
            buckets: Store::Owned(buckets),
            floats: Store::Owned(floats),
            starts: Store::Owned(starts),
            map: None,
        }
    }
}

/// Orders each history's infosets by bucket, their probabilities with them.
fn sort_buckets(heads: &[Head], buckets: &mut [u16], floats: &mut [f32], starts: &mut [u32]) {
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

/// A mapped strategies file's first bytes.
const MAGIC: [u8; 8] = *b"NRSTRAT1";
/// The magic, then the six stores' lengths, their seven offsets (the tail's last) and the tail's length, as u64s.
const HEADER: usize = 8 + 8 * 14;

fn as_bytes<T>(s: &[T]) -> &[u8] {
    // SAFETY: every store holds plain integers and floats (a head has no padding: its size is asserted), so each
    // byte of the slice is initialized.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

impl StrategyMap {
    /// Whether the file at `path` is one [`StrategyMap::write_mapped`] wrote.
    pub fn is_mapped_file(path: &str) -> std::io::Result<bool> {
        use std::io::Read;
        let mut magic = [0u8; 8];
        let mut f = std::fs::File::open(path)?;
        Ok(f.read_exact(&mut magic).is_ok() && magic == MAGIC)
    }

    /// Writes the map as a file [`StrategyMap::map`] reads in place, `tail` after it. Every insert
    /// must be frozen in: a key too short for a bucket has no place in it.
    pub fn write_mapped(&self, w: &mut impl std::io::Write, tail: &[u8]) -> std::io::Result<()> {
        if !self.loose.is_empty() {
            return Err(std::io::Error::other("inserts not frozen in, or keys too short for a bucket"));
        }
        let g = &self.groups;
        let stores: [(&[u8], usize); 6] = [
            (as_bytes(&g.index), g.index.len()),
            (as_bytes(&g.heads), g.heads.len()),
            (as_bytes(&g.hists), g.hists.len()),
            (as_bytes(&g.buckets), g.buckets.len()),
            (as_bytes(&g.floats), g.floats.len()),
            (as_bytes(&g.starts), g.starts.len()),
        ];
        let mut offsets = [0u64; 7];
        let mut at = HEADER as u64;
        for (i, (bytes, _)) in stores.iter().enumerate() {
            offsets[i] = at;
            at = (at + bytes.len() as u64).div_ceil(8) * 8;
        }
        offsets[6] = at;
        w.write_all(&MAGIC)?;
        for (_, len) in &stores {
            w.write_all(&(*len as u64).to_le_bytes())?;
        }
        for o in offsets {
            w.write_all(&o.to_le_bytes())?;
        }
        w.write_all(&(tail.len() as u64).to_le_bytes())?;
        for (i, (bytes, _)) in stores.iter().enumerate() {
            w.write_all(bytes)?;
            let end = offsets[i] + bytes.len() as u64;
            w.write_all(&vec![0u8; (offsets[i + 1] - end) as usize])?;
        }
        w.write_all(tail)
    }

    /// The map in the file at `path`, read in place, and the tail written after it. A file that is
    /// not one [`StrategyMap::write_mapped`] wrote, or is cut short, is an error.
    pub fn map(path: &str) -> std::io::Result<(StrategyMap, Vec<u8>)> {
        let bad = |what: &str| std::io::Error::other(format!("{path}: not a mapped strategies file ({what})"));
        let file = std::fs::File::open(path)?;
        // SAFETY: the file is only read; a blueprint file is written once, then served.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        if map.len() < HEADER || map[..8] != MAGIC {
            return Err(bad("magic"));
        }
        let word = |i: usize| u64::from_le_bytes(map[8 + 8 * i..16 + 8 * i].try_into().expect("8 bytes"));
        let lens: Vec<usize> = (0..6).map(|i| word(i) as usize).collect();
        let offsets: Vec<usize> = (6..13).map(|i| word(i) as usize).collect();
        let tail_len = word(13) as usize;
        let sizes = [4, std::mem::size_of::<Head>(), 1, 2, 4, 4];
        for i in 0..6 {
            if offsets[i] % 8 != 0 || offsets[i].checked_add(lens[i] * sizes[i]).is_none_or(|end| end > map.len()) {
                return Err(bad("a store past the end"));
            }
        }
        if offsets[6].checked_add(tail_len).is_none_or(|end| end > map.len()) {
            return Err(bad("the tail past the end"));
        }
        let base = map.as_ptr();
        // SAFETY: each store lies inside the mapping (checked above), starts 8-byte aligned on a page-aligned base, and
        // holds types every bit pattern of which is valid; `Groups::map` keeps the mapping alive as long as the stores.
        unsafe fn store<T>(base: *const u8, at: usize, len: usize) -> Store<T> {
            Store::Mapped(std::slice::from_raw_parts(base.add(at) as *const T, len))
        }
        let tail = map[offsets[6]..offsets[6] + tail_len].to_vec();
        let groups = unsafe {
            Groups {
                index: store(base, offsets[0], lens[0]),
                heads: store(base, offsets[1], lens[1]),
                hists: store(base, offsets[2], lens[2]),
                buckets: store(base, offsets[3], lens[3]),
                floats: store(base, offsets[4], lens[4]),
                starts: store(base, offsets[5], lens[5]),
                map: Some(Arc::new(map)),
            }
        };
        if groups.heads.iter().any(|h| (h.first + u64::from(h.count)) as usize > groups.buckets.len()) {
            return Err(bad("a history past its buckets"));
        }
        Ok((StrategyMap { groups, loose: Loose::with_capacity(0), added: 0 }, tail))
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
    fn a_mapped_file_answers_every_infoset_bit_for_bit_and_keeps_its_tail() {
        let all = entries();
        let map: StrategyMap = all.clone().into_iter().collect();
        let path = std::env::temp_dir().join(format!("strategies-{}.map", std::process::id()));
        let path = path.to_str().unwrap();
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        map.write_mapped(&mut f, b"the blueprint's other fields").unwrap();
        drop(f);
        assert!(StrategyMap::is_mapped_file(path).unwrap());
        let (mapped, tail) = StrategyMap::map(path).unwrap();
        assert_eq!(tail, b"the blueprint's other fields");
        assert!(mapped.groups.map.is_some());
        assert_eq!(mapped.len(), all.len());
        for (key, probs) in &all {
            let got = mapped.get(key).unwrap();
            assert!(got.iter().zip(probs).all(|(a, b)| a.to_bits() == b.to_bits()) && got.len() == probs.len());
        }
        assert_eq!(mapped.get(&make_key(9, &[1, 2, 3])), None);
        assert_eq!(mapped, map);
        // an insert into a mapped map shadows it, and a freeze lays both out in memory
        let mut grown = mapped.clone();
        grown.insert(&make_key(12, &[9, 9]), &[1.0]);
        assert_eq!(grown.get(&make_key(12, &[9, 9])), Some(&[1.0][..]));
        grown.freeze();
        assert!(grown.groups.map.is_none());
        assert_eq!(grown.len(), all.len() + 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_file_cut_short_or_of_another_kind_is_refused() {
        let map: StrategyMap = entries().into_iter().collect();
        let mut bytes = Vec::new();
        map.write_mapped(&mut bytes, b"").unwrap();
        let path = std::env::temp_dir().join(format!("strategies-cut-{}.map", std::process::id()));
        let path = path.to_str().unwrap();
        std::fs::write(path, &bytes[..bytes.len() / 2]).unwrap();
        assert!(StrategyMap::map(path).is_err());
        std::fs::write(path, b"not a strategies file at all, only words").unwrap();
        assert!(!StrategyMap::is_mapped_file(path).unwrap());
        assert!(StrategyMap::map(path).is_err());
        std::fs::remove_file(path).unwrap();
        let mut loose = StrategyMap::default();
        loose.insert(&[4u8], &[1.0]);
        assert!(loose.write_mapped(&mut Vec::new(), b"").is_err());
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
