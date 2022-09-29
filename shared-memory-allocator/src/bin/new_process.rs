use shared_memory_allocator::bindings::shared_memory_exists;
use shared_memory_allocator::SharedAllocator;

fn main() {
    // Get key from command line arguments
    let arg = std::env::args().nth(1);
    dbg!(&arg);
    let key_str = arg.unwrap();
    dbg!(&key_str);
    let key = key_str.parse::<i32>().unwrap();
    dbg!(key);

    let shared_memory_description_map = SharedAllocator::shared_memory_description_map();

    // Check memory exists
    assert!(shared_memory_exists(key).unwrap());
    // Check memory not attached
    assert!(!shared_memory_description_map
        .lock()
        .unwrap()
        .contains_key(&key));

    let allocator = SharedAllocator::new_process(key).unwrap();

    // Check memory exists
    assert!(shared_memory_exists(key).unwrap());
    // Check memory attached
    assert!(shared_memory_description_map
        .lock()
        .unwrap()
        .contains_key(&key));

    drop(allocator);

    // Check memory exists
    assert!(shared_memory_exists(key).unwrap());
    // Check memory not attached
    assert!(!shared_memory_description_map
        .lock()
        .unwrap()
        .contains_key(&key));
}
