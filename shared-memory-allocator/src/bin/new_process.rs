use const_format::concatcp;
use shared_memory_allocator::SharedAllocator;
const DIR: &str = "/tmp/";
const TAG: &str = "__rdb_";
const TEST_FILE_PREFIX: &str = concatcp!(DIR, TAG);
const SHMID: &str = concatcp!(TEST_FILE_PREFIX, "new_process");
fn main() {
    assert!(std::path::Path::new(SHMID).exists(), "1");

    let shared_memory_description_map = SharedAllocator::shared_memory_description_map();
    assert!(shared_memory_description_map.lock().expect("2").is_empty());

    let allocator = unsafe { SharedAllocator::new_process(SHMID).expect("3") };
    assert_eq!(shared_memory_description_map.lock().expect("4").len(), 1);

    drop(allocator);
    assert!(std::path::Path::new(SHMID).exists(), "5");
}
