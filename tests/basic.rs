use hashbrown::hash_table::Entry;
use whirlwind::*;

#[tokio::test]
async fn test_shardmap() {
    let map = ShardMap::new();
    map.insert("foo", "bar").await;
    assert_eq!(map.len().await, 1);
    assert_eq!(map.contains_key(&"foo").await, true);
    assert_eq!(map.contains_key(&"bar").await, false);
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    assert!(map.get(&"bar").await.is_none());
    assert_eq!(map.remove(&"foo").await, Some("bar"));
    assert_eq!(map.len().await, 0);
    assert_eq!(map.contains_key(&"foo").await, false);
}

#[tokio::test]
async fn test_shardmap_clone() {
    let map = ShardMap::new();
    map.insert("foo", "bar").await;
    let map2 = map.clone();
    assert_eq!(map2.len().await, 1);
    assert_eq!(map2.contains_key(&"foo").await, true);
    assert_eq!(map2.contains_key(&"bar").await, false);
    assert_eq!(map2.get(&"foo").await.unwrap().value(), &"bar");
    assert!(map2.get(&"bar").await.is_none());
    assert_eq!(map2.remove(&"foo").await, Some("bar"));
    assert_eq!(map2.len().await, 0);
    assert_eq!(map2.contains_key(&"foo").await, false);
}

#[tokio::test]
async fn test_shardmap_shards() {
    let map = ShardMap::with_shards(4);
    map.insert("foo", "bar").await;
    assert_eq!(map.len().await, 1);
    assert_eq!(map.contains_key(&"foo").await, true);
    assert_eq!(map.contains_key(&"bar").await, false);
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    assert!(map.get(&"bar").await.is_none());
    assert_eq!(map.remove(&"foo").await, Some("bar"));
    assert_eq!(map.len().await, 0);
    assert_eq!(map.contains_key(&"foo").await, false);
}

#[tokio::test]
async fn test_shardmap_len() {
    let map = ShardMap::new();
    map.insert("foo", "bar").await;
    assert_eq!(map.len().await, 1);
    map.insert("foo2", "bar2").await;
    assert_eq!(map.len().await, 2);
    map.remove(&"foo").await;
    assert_eq!(map.len().await, 1);
    map.remove(&"foo2").await;
    assert_eq!(map.len().await, 0);
}

#[tokio::test]
async fn test_shardmap_is_empty() {
    let map = ShardMap::new();
    assert_eq!(map.is_empty().await, true);
    map.insert("foo", "bar").await;
    assert_eq!(map.is_empty().await, false);
    map.remove(&"foo").await;
    assert_eq!(map.is_empty().await, true);
}

#[tokio::test]
async fn test_shardmap_with_entry() {
    let map = ShardMap::new();

    // Test with vacant entry
    let result = map
        .with_entry("foo", |key, entry| {
            assert_eq!(key, "foo");
            match entry {
                Entry::Vacant(slot) => {
                    slot.insert((key, "bar"));
                    Some("inserted")
                }
                Entry::Occupied(_) => None,
            }
        })
        .await;
    assert_eq!(result, Some("inserted"));
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    assert_eq!(map.len().await, 1);

    // Test with occupied entry
    let result = map
        .with_entry("foo", |key, entry| {
            assert_eq!(key, "foo");
            assert!(
                matches!(entry, Entry::Occupied(_)),
                "Expected Occupied entry for existing key"
            );
            match entry {
                Entry::Occupied(entry) => {
                    let ((_, value), _) = entry.remove();
                    Some(value)
                }
                Entry::Vacant(_) => None,
            }
        })
        .await;
    assert_eq!(result, Some("bar"));
    assert_eq!(map.contains_key(&"foo").await, false);
    assert_eq!(map.len().await, 0);
}

#[tokio::test]
async fn test_shardmap_compute() {
    let map = ShardMap::new();

    // Test inserting new key-value pair
    map.compute("foo", |current| {
        assert_eq!(current, None);
        Some("bar")
    })
    .await;
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    assert_eq!(map.len().await, 1);

    // Test updating existing key
    map.compute("foo", |current| {
        assert_eq!(current, Some("bar"));
        Some("baz")
    })
    .await;
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"baz");
    assert_eq!(map.len().await, 1);

    // Test removing key by returning None
    map.compute("foo", |current| {
        assert_eq!(current, Some("baz"));
        None
    })
    .await;
    assert_eq!(map.contains_key(&"foo").await, false);
    assert_eq!(map.len().await, 0);

    // Test no insertion when returning None for vacant entry
    map.compute("foo", |current| {
        assert_eq!(current, None);
        None
    })
    .await;
    assert_eq!(map.contains_key(&"foo").await, false);
    assert_eq!(map.len().await, 0);
}

#[tokio::test]
async fn test_shardmap_compute_if_absent() {
    let map = ShardMap::new();

    // Test inserting new key-value pair
    let inserted = map.compute_if_absent("foo", || "bar").await;
    assert_eq!(inserted, true);
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    assert_eq!(map.len().await, 1);

    // Test no insertion when key exists
    let inserted = map.compute_if_absent("foo", || "baz").await;
    assert_eq!(inserted, false);
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    assert_eq!(map.len().await, 1);

    // Test inserting another new key
    let inserted = map.compute_if_absent("foo2", || "bar2").await;
    assert_eq!(inserted, true);
    assert_eq!(map.get(&"foo2").await.unwrap().value(), &"bar2");
    assert_eq!(map.len().await, 2);
}
