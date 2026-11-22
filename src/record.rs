//! One infoset's key and per-action values packed in a few words of an [`Arena`], behind a thin
//! pointer: a table of hundreds of millions of infosets pays 8 bytes a slot and each record's
//! exact size. The 172M-infoset blueprint as one mimalloc allocation per record peaked at 23.7 GB
//! on load, 8.7 GB under the system allocator.
//! Layout, in 4-byte words: the key's length (u16) and the action count (u8) in the first word,
//! the key padded to whole words, then `V` bytes of values per action.

use std::ptr::NonNull;

/// Zeroed words handed out in order from large chunks, all freed with the arena.
pub(crate) struct Arena {
    chunks: Vec<(NonNull<u32>, usize)>,
    used: usize,
}

// An Arena owns its chunks, as the Box<[u32]>s they were made from do.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

/// Words in an arena's first chunk (1 KiB); each next chunk doubles, up to `MAX_CHUNK_WORDS`
/// (1 MiB). A search's trainer fills a few KiB in each of its 1,024 shards, and a fixed 16 MiB
/// chunk held up to 16 GB per search: a simulation of four searches at once reached 58 GB.
const FIRST_CHUNK_WORDS: usize = 1 << 8;
const MAX_CHUNK_WORDS: usize = 1 << 18;

impl Arena {
    pub fn new() -> Self {
        Arena { chunks: Vec::new(), used: 0 }
    }

    fn alloc(&mut self, words: usize) -> NonNull<u32> {
        let room = self.chunks.last().map_or(0, |&(_, len)| len - self.used);
        if room < words {
            let next = self.chunks.last().map_or(FIRST_CHUNK_WORDS, |&(_, len)| (2 * len).min(MAX_CHUNK_WORDS));
            let len = next.max(words);
            let chunk = Box::into_raw(vec![0u32; len].into_boxed_slice()) as *mut u32;
            self.chunks.push((NonNull::new(chunk).expect("Box is never null"), len));
            self.used = 0;
        }
        let (chunk, _) = *self.chunks.last().expect("a chunk is there");
        // SAFETY: `used + words` stays within the chunk's length, checked above.
        let p = unsafe { chunk.as_ptr().add(self.used) };
        self.used += words;
        NonNull::new(p).expect("inside a live chunk")
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        for &(chunk, len) in &self.chunks {
            // SAFETY: each chunk came from `Box::into_raw` of a `Box<[u32]>` this long.
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(chunk.as_ptr(), len)) });
        }
    }
}

/// A record in an arena. It must not be used after its arena drops: a table keeps its records
/// and their arena side by side.
pub(crate) struct Record<const V: usize>(NonNull<u32>);

// A Record is a handle into its arena, whose owner gives out `&mut` only through `&mut` to itself.
unsafe impl<const V: usize> Send for Record<V> {}
unsafe impl<const V: usize> Sync for Record<V> {}

impl<const V: usize> Record<V> {
    fn words(key_len: usize, actions: usize) -> usize {
        1 + key_len.div_ceil(4) + (V * actions).div_ceil(4)
    }

    /// `key`'s record in `arena`, every value byte zero.
    pub fn new_in(arena: &mut Arena, key: &[u8], actions: usize) -> Self {
        assert!(key.len() <= u16::MAX as usize && actions <= u8::MAX as usize);
        let words = arena.alloc(Self::words(key.len(), actions));
        // SAFETY: the first of the zeroed words just handed out.
        unsafe { *words.as_ptr() = key.len() as u32 | (actions as u32) << 16 };
        let mut record = Record(words);
        record.bytes_mut()[4..4 + key.len()].copy_from_slice(key);
        record
    }

    /// (key length, action count).
    fn header(&self) -> (usize, usize) {
        // SAFETY: the first word of a record in a live arena.
        let w = unsafe { *self.0.as_ptr() };
        ((w & 0xFFFF) as usize, (w >> 16 & 0xFF) as usize)
    }

    fn len_words(&self) -> usize {
        let (key_len, actions) = self.header();
        Self::words(key_len, actions)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the arena holds `len_words` words here, read off the record's own header.
        unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, 4 * self.len_words()) }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        let len = 4 * self.len_words();
        // SAFETY: as in `bytes`, and `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(self.0.as_ptr() as *mut u8, len) }
    }

    pub fn key(&self) -> &[u8] {
        let (key_len, _) = self.header();
        &self.bytes()[4..4 + key_len]
    }

    pub fn actions(&self) -> usize {
        self.header().1
    }

    fn values_range(&self) -> std::ops::Range<usize> {
        let (key_len, actions) = self.header();
        let at = 4 + 4 * key_len.div_ceil(4);
        at..at + V * actions
    }

    /// The values, `V` bytes per action, starting on a 4-byte boundary.
    pub fn values(&self) -> &[u8] {
        let range = self.values_range();
        &self.bytes()[range]
    }

    pub fn values_mut(&mut self) -> &mut [u8] {
        let range = self.values_range();
        &mut self.bytes_mut()[range]
    }
}

impl Record<4> {
    /// The values as one f32 per action.
    pub fn floats(&self) -> &[f32] {
        let v = self.values();
        // SAFETY: the values start on a 4-byte boundary of a [u32] chunk and hold 4 bytes per
        // action; every bit pattern is an f32.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4) }
    }

    pub fn floats_mut(&mut self) -> &mut [f32] {
        let v = self.values_mut();
        // SAFETY: as in `floats`, and `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut f32, v.len() / 4) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_keeps_its_key_and_values_for_every_key_length() {
        for key_len in 0..9 {
            let key: Vec<u8> = (0..key_len as u8).collect();
            let mut arena = Arena::new();
            let mut r = Record::<4>::new_in(&mut arena, &key, 3);
            assert_eq!(r.key(), &key[..]);
            assert_eq!(r.floats(), &[0.0, 0.0, 0.0]);
            r.floats_mut().copy_from_slice(&[0.25, 0.5, 0.25]);
            assert_eq!(r.floats(), &[0.25, 0.5, 0.25]);
            assert_eq!(r.key(), &key[..]);
            let r = Record::<12>::new_in(&mut arena, &key, 5);
            assert_eq!((r.key(), r.actions(), r.values().len()), (&key[..], 5, 60));
        }
    }
}
