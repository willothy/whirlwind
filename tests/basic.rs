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
async fn test_shardmap_iter() {
    let map = ShardMap::new();
    map.insert("foo", "bar").await;
    map.insert("baz", "qux").await;

    let mut pairs = Vec::new();
    for (key, value) in map.iter().await {
        pairs.push((*key, *value));
    }

    assert_eq!(pairs.len(), 2);
    assert!(pairs.contains(&(&"foo", &"bar")));
    assert!(pairs.contains(&(&"baz", &"qux")));
    // Verify map state is unchanged
    assert_eq!(map.len().await, 2);
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    assert_eq!(map.get(&"baz").await.unwrap().value(), &"qux");
}

#[tokio::test]
async fn test_shardmap_iter_empty() {
    let map = ShardMap::new();
    let pairs: Vec<(&&str, &&str)> = map.iter().await.collect();
    assert_eq!(pairs.len(), 0);
    assert_eq!(map.len().await, 0);
}

#[tokio::test]
async fn test_shardmap_keys() {
    let map = ShardMap::new();
    map.insert("foo", "bar").await;
    map.insert("baz", "qux").await;

    let mut keys = Vec::new();
    for key in map.keys().await {
        keys.push(*key);
    }

    assert_eq!(keys.len(), 2);
    assert!(keys.contains(&"foo"));
    assert!(keys.contains(&"baz"));
    // Verify map state is unchanged
    assert_eq!(map.len().await, 2);
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    assert_eq!(map.get(&"baz").await.unwrap().value(), &"qux");
}

#[tokio::test]
async fn test_shardmap_values() {
    let map = ShardMap::new();
    map.insert("foo", "bar").await;
    map.insert("baz", "qux").await;

    let mut values = Vec::new();
    for value in map.values().await {
        values.push(*value);
    }

    assert_eq!(values.len(), 2);
    assert!(values.contains(&"bar"));
    assert!(values.contains(&"qux"));
    // Verify map state is unchanged
    assert_eq!(map.len().await, 2);
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"bar");
    assert_eq!(map.get(&"baz").await.unwrap().value(), &"qux");
}

#[tokio::test]
async fn test_shardmap_iter_mut() {
    let map = ShardMap::new();
    map.insert("foo", "bar").await;
    map.insert("baz", "qux").await;

    for (key, value) in map.iter_mut().await {
        if *key == "foo" {
            *value = "updated";
        }
    }

    assert_eq!(map.len().await, 2);
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"updated");
    assert_eq!(map.get(&"baz").await.unwrap().value(), &"qux");
}

#[tokio::test]
async fn test_shardmap_values_mut() {
    let map = ShardMap::new();
    map.insert("foo", "bar").await;
    map.insert("baz", "qux").await;

    for value in map.values_mut().await {
        if *value == "bar" {
            *value = "updated";
        }
    }

    assert_eq!(map.len().await, 2);
    assert_eq!(map.get(&"foo").await.unwrap().value(), &"updated");
    assert_eq!(map.get(&"baz").await.unwrap().value(), &"qux");
}

#[tokio::test]
async fn test_shardmap_iter_mut_empty() {
    let map = ShardMap::new();
    let pairs: Vec<(&&str, &mut &str)> = map.iter_mut().await.collect();
    assert_eq!(pairs.len(), 0);
    assert_eq!(map.len().await, 0);
}

#[tokio::test]
async fn test_shardset_iter() {
    let set = ShardSet::new();
    set.insert("foo").await;
    set.insert("bar").await;

    let mut items = Vec::new();
    for item in set.iter().await {
        items.push(*item);
    }

    assert_eq!(items.len(), 2);
    assert!(items.contains(&"foo"));
    assert!(items.contains(&"bar"));
    // Verify set state is unchanged
    assert_eq!(set.len().await, 2);
    assert!(set.contains(&"foo").await);
    assert!(set.contains(&"bar").await);
}