//! A concurrent hashmap using a sharding strategy.
//!
//! # Examples
//! ```
//! use tokio::runtime::Runtime;
//! use std::sync::Arc;
//! use whirlwind::ShardMap;
//!
//! let rt = Runtime::new().unwrap();
//! let map = Arc::new(ShardMap::new());
//! rt.block_on(async {
//!    map.insert("foo", "bar").await;
//!    assert_eq!(map.len().await, 1);
//!    assert_eq!(map.contains_key(&"foo").await, true);
//!    assert_eq!(map.contains_key(&"bar").await, false);
//!
//!    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
//!    assert_eq!(map.remove(&"foo").await, Some("bar"));
//! });
//! ```
use std::{
    hash::{BuildHasher, RandomState},
    sync::{Arc, OnceLock},
};

use crossbeam_utils::CachePadded;
use hashbrown::hash_table::Entry;

#[cfg(feature = "stream")]
use async_stream::stream;
#[cfg(feature = "stream")]
use futures::{
    pin_mut,
    stream::{self, Stream},
    StreamExt,
};

use crate::{
    mapref::{MapRef, MapRefMut},
    shard::Shard,
};

struct Inner<K, V, S = RandomState> {
    shards: Box<[CachePadded<Shard<K, V>>]>,
    hasher: S,
    shift: usize,
}

impl<K, V, S> std::ops::Deref for Inner<K, V, S> {
    type Target = Box<[CachePadded<Shard<K, V>>]>;

    fn deref(&self) -> &Self::Target {
        &self.shards
    }
}

impl<K, V, S> std::ops::DerefMut for Inner<K, V, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shards
    }
}

/// A concurrent hashmap using a sharding strategy.
///
/// # Examples
/// ```
/// use tokio::runtime::Runtime;
/// use std::sync::Arc;
/// use whirlwind::ShardMap;
///
/// let rt = Runtime::new().unwrap();
/// let map = Arc::new(ShardMap::new());
/// rt.block_on(async {
///    map.insert("foo", "bar").await;
///    assert_eq!(map.len().await, 1);
///    assert_eq!(map.contains_key(&"foo").await, true);
///    assert_eq!(map.contains_key(&"bar").await, false);
///
///    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
///    assert_eq!(map.remove(&"foo").await, Some("bar"));
/// });
/// ```
pub struct ShardMap<K, V, S = std::hash::RandomState> {
    inner: Arc<Inner<K, V, S>>,
}

impl<K, V, H> Clone for ShardMap<K, V, H> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<K, V> Default for ShardMap<K, V, RandomState>
where
    K: Eq + std::hash::Hash + 'static,
    V: 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn calculate_shard_count() -> usize {
    (std::thread::available_parallelism().map_or(1, usize::from) * 4).next_power_of_two()
}

#[inline(always)]
fn shard_count() -> usize {
    static SHARD_COUNT: OnceLock<usize> = OnceLock::new();
    *SHARD_COUNT.get_or_init(calculate_shard_count)
}

impl<K, V> ShardMap<K, V, RandomState>
where
    K: Eq + std::hash::Hash + 'static,
    V: 'static,
{
    /// Creates a new `ShardMap` with the default hasher.
    pub fn new() -> Self {
        Self::with_shards(shard_count())
    }

    /// Creates a new `ShardMap` with the default hasher and `shards` shards.
    pub fn with_shards(shards: usize) -> Self {
        Self::with_shards_and_hasher(shards, RandomState::new())
    }

    /// Creates a new `ShardMap` with the default hasher and space for at least `cap` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }

    /// Creates a new `ShardMap` with the default hasher, `shards` shards, and space for at least `cap` elements.
    pub fn with_shards_and_capacity(shards: usize, cap: usize) -> Self {
        Self::with_shards_and_capacity_and_hasher(shards, cap, RandomState::new())
    }
}

fn ptr_size_bits() -> usize {
    std::mem::size_of::<*const ()>() * 8
}

impl<K, V, S: BuildHasher> ShardMap<K, V, S>
where
    K: Eq + std::hash::Hash + 'static,
    V: 'static,
{
    /// Creates a new `ShardMap` with the provided hasher `S`.
    pub fn with_hasher(hasher: S) -> Self {
        Self::with_shards_and_hasher(shard_count(), hasher)
    }

    /// Creates a new `ShardMap` with the provided hasher `S` and space for at least `cap` elements.
    pub fn with_capacity_and_hasher(cap: usize, hasher: S) -> Self {
        Self::with_shards_and_capacity_and_hasher(shard_count(), cap, hasher)
    }

    /// Creates a new `ShardMap` with the provided hasher `S` and `shards` shards.
    pub fn with_shards_and_hasher(shards: usize, hasher: S) -> Self {
        Self::with_shards_and_capacity_and_hasher(shards, 4, hasher)
    }

    /// Creates a new `ShardMap` with the provided hasher `S`, `shards` shards, and space for at
    /// least `cap` elements.
    pub fn with_shards_and_capacity_and_hasher(shards: usize, mut cap: usize, hasher: S) -> Self {
        debug_assert!(shards > 1);
        debug_assert!(shards.is_power_of_two());

        let shift = ptr_size_bits() - (shards.trailing_zeros() as usize);

        if cap != 0 {
            cap = (cap + (shards - 1)) & !(shards - 1);
        }
        let shard_capacity = cap / shards;

        let shards = std::iter::repeat_n((), shards)
            .map(|_| CachePadded::new(Shard::with_capacity(shard_capacity)))
            .collect();

        Self {
            inner: Arc::new(Inner {
                shards,
                shift,
                hasher,
            }),
        }
    }

    #[inline]
    fn shard_for_hash(&self, hash: usize) -> usize {
        // 7 high bits for the HashBrown simd tag
        (hash << 7) >> self.inner.shift
    }

    #[inline]
    fn shard(&self, key: &K) -> (&CachePadded<Shard<K, V>>, u64) {
        let hash = self.inner.hasher.hash_one(key);

        let shard_idx = self.shard_for_hash(hash as usize);

        (unsafe { self.inner.shards.get_unchecked(shard_idx) }, hash)
    }

    /// Inserts a key-value pair into the map. If the key already exists, the value is updated and
    /// the old value is returned.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    /// rt.block_on(async {
    ///     map.insert("foo", "bar").await;
    ///
    ///     assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    /// });
    /// ```
    pub async fn insert(&self, key: K, value: V) -> Option<V> {
        let (shard, hash) = self.shard(&key);
        let mut writer = shard.write().await;

        let (old, slot) = match writer.entry(
            hash,
            |(k, _)| k == &key,
            |(k, _)| self.inner.hasher.hash_one(k),
        ) {
            Entry::Occupied(entry) => {
                let ((_, old), slot) = entry.remove();
                (Some(old), slot)
            }
            Entry::Vacant(slot) => (None, slot),
        };

        slot.insert((key, value));

        old
    }

    /// Returns a reference to the value associated with the key.
    /// If the key is not in the map, `None` is returned.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::{ShardMap, mapref::MapRef};
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    ///
    /// rt.block_on(async {
    ///     map.insert("foo", "bar").await;
    ///
    ///     // `get` returns a `MapRef` which holds a read lock on the shard.
    ///     let entry: MapRef<'_, _, _> = map.get(&"foo").await.unwrap();
    ///
    ///     assert_eq!(entry.value(), &"bar");
    /// });
    /// ```
    pub async fn get<'a>(&'a self, key: &'a K) -> Option<MapRef<'a, K, V>> {
        let (shard, hash) = self.shard(key);
        let reader = shard.read().await;

        if let Some((k, v)) = reader.find(hash, |(k, _)| k == key) {
            let (k, v) = (k as *const K, v as *const V);
            // SAFETY: The key and value are guaranteed to be valid for the lifetime of the reader.
            unsafe { Some(MapRef::new(reader, &*k, &*v)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the value associated with the key.
    /// If the key is not in the map, `None` is returned.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::{ShardMap, mapref::MapRefMut};
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    ///
    /// rt.block_on(async {
    ///     map.insert("foo", "bar").await;
    ///
    ///     // `get_mut` returns a `MapRefMut` which holds a write lock on the shard.
    ///     let mut entry: MapRefMut<'_, _, _> = map.get_mut(&"foo").await.unwrap();
    ///     *entry.value_mut() = "baz";
    ///
    ///     assert_eq!(entry.value(), &"baz");
    ///     drop(entry);
    ///
    ///     assert_eq!(map.get(&"foo").await.unwrap().value(), &"baz");
    /// });
    /// ```
    pub async fn get_mut<'a>(&'a self, key: &'a K) -> Option<MapRefMut<'a, K, V>> {
        let (shard, hash) = self.shard(key);
        let mut writer = shard.write().await;

        if let Some((k, v)) = writer.find_mut(hash, |(k, _)| k == key) {
            let (k, v) = (k as *const K, v as *mut V);
            // SAFETY: The key and value are guaranteed to be valid for the lifetime of the writer.
            unsafe { Some(MapRefMut::new(writer, &*k, &mut *v)) }
        } else {
            None
        }
    }

    /// Returns `true` if the map contains the key.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    ///
    /// rt.block_on(async {
    ///     map.insert("foo", "bar").await;
    ///
    ///     assert_eq!(map.contains_key(&"foo").await, true);
    ///
    ///     assert_eq!(map.contains_key(&"bar").await, false);
    /// });
    /// ```
    pub async fn contains_key(&self, key: &K) -> bool {
        let (shard, hash) = self.shard(key);

        let reader = shard.read().await;

        reader.find(hash, |(k, _)| k == key).is_some()
    }

    /// Removes a key from the map and returns the value associated with the key.
    /// If the key is not in the map, `None` is returned.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    ///
    /// rt.block_on(async {
    ///     map.insert("foo", "bar").await;
    ///
    ///     assert_eq!(map.contains_key(&"foo").await, true);
    ///
    ///     let value = map.remove(&"foo").await;
    ///
    ///     assert_eq!(value, Some("bar"));
    ///
    ///     assert_eq!(map.contains_key(&"foo").await, false);
    /// });
    /// ```
    pub async fn remove(&self, key: &K) -> Option<V> {
        let (shard, hash) = self.shard(key);

        match shard.write().await.find_entry(hash, |(k, _)| k == key) {
            Ok(occupied) => {
                let ((_, v), _) = occupied.remove();
                Some(v)
            }
            _ => None,
        }
    }

    /// Returns the number of elements in the map.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    ///
    /// rt.block_on(async {
    ///     map.insert("foo", "bar").await;
    ///     assert_eq!(map.len().await, 1);
    ///     map.insert("foo2", "bar2").await;
    ///     assert_eq!(map.len().await, 2);
    /// });
    /// ```
    pub async fn len(&self) -> usize {
        let mut sum = 0;
        for shard in self.inner.iter() {
            sum += shard.read().await.len();
        }
        sum
    }

    /// Returns `true` if the map is empty.
    ///
    /// This is equivalent to `map.len().await == 0`.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    /// rt.block_on(async {
    ///    assert_eq!(map.is_empty().await, true);
    ///
    ///    map.insert("foo", "bar").await;
    ///    assert_eq!(map.is_empty().await, false);
    ///
    ///    map.remove(&"foo").await;
    ///    assert_eq!(map.is_empty().await, true);
    /// });
    ///
    /// ```
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Clears the map, removing all key-value pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    ///
    /// rt.block_on(async {
    ///    map.insert("foo", "bar").await;
    ///    map.insert("baz", "qux").await;
    ///
    ///    assert_eq!(map.len().await, 2);
    ///
    ///    map.clear().await;
    ///
    ///    assert_eq!(map.is_empty().await, true);
    /// });
    pub async fn clear(&self) {
        for shard in self.inner.iter() {
            shard.write().await.clear();
        }
    }

    #[cfg(feature = "stream")]
    /// Stream over all shards in the map.
    ///
    /// Each item is a `ShardRead` that *holds a read-lock* on that shard while you iterate it
    /// synchronously. Writes to **other shards** continue to proceed concurrently.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use futures::{pin_mut, StreamExt};
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    /// rt.block_on(async {
    ///     map.insert(1, "a").await;
    ///     map.insert(2, "b").await;
    ///
    ///     let shards = map.stream_shards();
    ///     pin_mut!(shards); // Stream is not Unpin
    ///
    ///     let mut seen = 0;
    ///     while let Some(sh) = shards.next().await {
    ///         for (_k, _v) in sh.iter() {
    ///             seen += 1;
    ///         }
    ///     }
    ///     assert_eq!(seen, 2);
    /// });
    /// ```
    pub fn stream_shards(&self) -> impl Stream<Item = ShardRead<'_, K, V>> + '_ {
        let total = self.inner.len();

        stream::unfold(0usize, move |mut idx| async move {
            if idx >= total {
                return None;
            }

            // SAFETY: idx is checked against total above.
            let shard = unsafe { self.inner.get_unchecked(idx) };

            let guard = shard.read().await;

            idx += 1;
            Some((ShardRead { guard }, idx))
        })
    }

    #[cfg(feature = "stream")]
    /// Flattened stream of **owned** `(K, V)` items.
    ///
    /// Locks one shard at a time, snapshots (clones) its entries into a `Vec`, drops the lock,
    /// then yields items. This allows concurrent writes to other shards.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use futures::{pin_mut, StreamExt};
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    /// rt.block_on(async {
    ///     map.insert(1, "a".to_string()).await;
    ///     map.insert(2, "b".to_string()).await;
    ///
    ///     let s = map.stream_owned();
    ///
    ///     let mut items = Vec::new();
    ///     while let Some((k, v)) = s.next().await {
    ///         items.push((k, v));
    ///     }
    ///     items.sort_by_key(|(k, _)| *k);
    ///     assert_eq!(items, vec![(1, "a".into()), (2, "b".into())]);
    /// });
    /// ```
    pub fn stream_owned(&self) -> impl Stream<Item = (K, V)> + '_
    where
        K: Clone,
        V: Clone,
    {
        let shard_stream = self.stream_shards();

        stream! {
            pin_mut!(shard_stream);

            while let Some(shard) = shard_stream.next().await {
                let items: Vec<(K, V)> =
                    shard.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                drop(shard);
                for item in items {
                    yield item;
                }
            }
        }
    }

    #[cfg(feature = "stream")]
    /// Collect all entries into a `Vec<(K, V)>` by cloning.
    ///
    /// Iterates shard-by-shard, cloning items under a read lock, then releasing the lock
    /// before pushing into the result.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    /// rt.block_on(async {
    ///     map.insert(1, "a".to_string()).await;
    ///     map.insert(2, "b".to_string()).await;
    ///
    ///     let mut items = map.entries().await;
    ///     items.sort_by_key(|(k, _)| *k);
    ///     assert_eq!(items, vec![(1, "a".into()), (2, "b".into())]);
    /// });
    /// ```
    pub async fn entries(&self) -> Vec<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        self.stream_owned().collect::<Vec<(K, V)>>().await
    }

    #[cfg(feature = "stream")]
    /// Collect all keys into a `Vec<K>` by cloning.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    /// rt.block_on(async {
    ///     map.insert(10, "x").await;
    ///     map.insert(20, "y").await;
    ///
    ///     let mut ks = map.keys().await;
    ///     ks.sort();
    ///     assert_eq!(ks, vec![10, 20]);
    /// });
    /// ```
    pub async fn keys(&self) -> Vec<K>
    where
        K: Clone,
    {
        let shard_stream = self.stream_shards();
        pin_mut!(shard_stream);
        let mut keys = Vec::new();
        while let Some(shard) = shard_stream.next().await {
            let mut shard_keys: Vec<K> = shard.keys().cloned().collect();
            drop(shard);
            keys.append(&mut shard_keys);
        }
        keys
    }

    #[cfg(feature = "stream")]
    /// Collect all values into a `Vec<V>` by cloning.
    ///
    /// # Example
    /// ```
    /// use tokio::runtime::Runtime;
    /// use std::sync::Arc;
    /// use whirlwind::ShardMap;
    ///
    /// let rt = Runtime::new().unwrap();
    /// let map = Arc::new(ShardMap::new());
    /// rt.block_on(async {
    ///     map.insert(1, "a".to_string()).await;
    ///     map.insert(2, "b".to_string()).await;
    ///
    ///     let mut vs = map.values().await;
    ///     vs.sort();
    ///     assert_eq!(vs, vec!["a".to_string(), "b".to_string()]);
    /// });
    /// ```
    pub async fn values(&self) -> Vec<V>
    where
        V: Clone,
    {
        let shard_stream = self.stream_shards();
        pin_mut!(shard_stream);
        let mut values = Vec::new();
        while let Some(shard) = shard_stream.next().await {
            let mut shard_values: Vec<V> = shard.values().cloned().collect();
            drop(shard);
            values.append(&mut shard_values);
        }
        values
    }
}

#[cfg(feature = "stream")]
pub struct ShardRead<'a, K, V> {
    guard: crate::shard::ShardReader<'a, K, V>,
}

#[cfg(feature = "stream")]
impl<'a, K, V> ShardRead<'a, K, V>
where
    K: Eq + std::hash::Hash + 'static,
    V: 'static,
{
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.guard.iter().map(|(k, v)| (k, v))
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.guard.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.guard.iter().map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.guard.len()
    }

    pub fn is_empty(&self) -> bool {
        self.guard.len() == 0
    }
}
