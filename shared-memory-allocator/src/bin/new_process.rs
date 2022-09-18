use shared_memory_allocator::bindings::shared_memory_exists;
use shared_memory_allocator::SharedAllocator;

fn main() {
    let arg = std::env::args().nth(1);
    dbg!(&arg);
    let key_str = arg.unwrap();
    dbg!(&key_str);
    let key = key_str.parse::<i32>().unwrap();
    dbg!(key);

    assert!(shared_memory_exists(key).unwrap());

    let smd_map = SharedAllocator::shared_memory_description_map();
    assert!(smd_map.lock().unwrap().is_empty());

    let allocator = SharedAllocator::new_process(key).unwrap();

    assert_eq!(smd_map.lock().unwrap().len(), 1);
    assert!(shared_memory_exists(key).unwrap());

    drop(allocator);

    assert!(smd_map.lock().unwrap().is_empty());
    assert!(shared_memory_exists(key).unwrap());
}
