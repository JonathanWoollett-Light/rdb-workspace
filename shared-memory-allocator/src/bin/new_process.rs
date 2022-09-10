use shared_memory_allocator::SharedAllocator;
const NEW_PROCESS_KEY: i32 = 240434;
fn main() {
    assert!(
        shared_memory_allocator::bindings::shared_memory_allocated(NEW_PROCESS_KEY)
            .unwrap()
            .is_some()
    );

    let shared_memory_description_map = SharedAllocator::shared_memory_description_map();
    assert!(shared_memory_description_map.lock().unwrap().is_empty());

    let _allocator = SharedAllocator::new_process(NEW_PROCESS_KEY).unwrap();

    assert!(
        shared_memory_allocator::bindings::shared_memory_allocated(NEW_PROCESS_KEY)
            .unwrap()
            .is_some()
    );
}
