#![feature(allocator_api)]
#![feature(vec_into_raw_parts)]
#![feature(adt_const_params)]
#![feature(unchecked_math)]
#![feature(ptr_to_from_bits)]
#![allow(incomplete_features)]
#![warn(clippy::all, clippy::restriction, clippy::pedantic)]
#![allow(
    clippy::blanket_clippy_restriction_lints,
    clippy::implicit_return,
    clippy::std_instead_of_core,
    clippy::std_instead_of_alloc,
    clippy::unseparated_literal_suffix,
    clippy::exhaustive_enums,
    clippy::missing_inline_in_public_items
)]

//! An extremely unsafe experiment in writing a custom allocator to use linux shared memory.

use std::alloc::{AllocError, Layout};
use std::io::{Read, Write};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use log::{error, info, trace};
use log_derive::logfn;

/// The description of shared memory stored within the shared memory.
#[derive(Debug)]
struct InMemoryDescription {
    /// The number of processes currently attached to this shared memory.
    count: AtomicU8,
    /// The capacity of the shared memory (the maximum possible `length`).
    capacity: usize,
    /// The currently used amount of the shared memory.
    length: RwLock<usize>,
}

use std::sync::RwLock;

/// Wrappers around `libc` shared memory functions.
mod bindings;

/// An allocator implementing [`std::alloc::Allocator`] which allocates items in linux shared
/// memory.
///
/// Constructing multiple allocators with the same `shmid_path` will use the same shared memory.
///
/// After constructing the first allocator of a given `shmid_path` constructing new allocators
/// with the same `shmid_path` is the same as cloning the original allocator.
///
/// At the moment, both allocation and deallocation of memory functions like pushing and
/// removing element from a vector (e.g. very inefficient), this is something which will be
/// improved moving forward.
#[derive(Debug)]
pub struct SharedAllocator(usize);

/// A description of shared memory stored in a map in attached processes.
///
/// 0. `shmid_path`.
/// 1. `shmid`.
/// 2. Attached address.
/// 3. Count of existing shared allocators for this shared memory.
type SharedMemoryData = Vec<(String, i32, usize, u8)>;

/// Error type for [`SharedAllocator::new`].
#[derive(Debug, thiserror::Error)]
pub enum SharedAllocatorNewError {
    /// Failed to acquire lock on shared memory description map.
    #[error("Failed to acquire lock on shared memory description map.")]
    SharedMemoryDescriptionMapLock,
    /// Memory size given was too big to also contain [`InMemoryDescription`].
    #[error("Memory size given was too big to also contain [`InMemoryDescription`].")]
    TooBig,
    /// You cannot create any more shared memory allocators to this shared memory in this process
    /// until some have been dropped (the maximum of 255 has been reached).
    #[error(
        "You cannot create any more shared memory allocators to this shared memory in this \
         process until some have been dropped (the maximum of 255 has been reached)."
    )]
    TooMany,
    /// Failed to allocate shared memory.
    #[error("Failed to allocate shared memory.")]
    Allocate(i32),
    /// Failed to create shared memory id file.
    #[error("Failed to create shared memory id file: {0}")]
    ShmidFileCreate(std::io::Error),
    /// Failed to write shared memory id file.
    #[error("Failed to write shared memory id file: {0}")]
    ShmidFileWrite(std::io::Error),
    /// Failed to open shared memory id file.
    #[error("Failed to open shared memory id file: {0}")]
    ShmidFileOpen(std::io::Error),
    /// Failed to read shared memory id file.
    #[error("Failed to read shared memory id file: {0}")]
    ShmidFileRead(std::io::Error),
    /// Failed to attach shared memory.
    #[error("Failed to attach shared memory: {0}")]
    Attach(i32),
}
/// Error type for [`SharedAllocator::new_memory`].
#[derive(Debug, thiserror::Error)]
pub enum SharedAllocatorNewMemoryError {
    /// Failed to acquire lock on shared memory description map.
    #[error("Failed to acquire lock on shared memory description map.")]
    SharedMemoryDescriptionMapLock,
    /// Memory size given was too big to also contain [`InMemoryDescription`].
    #[error("Memory size given was too big to also contain [`InMemoryDescription`].")]
    TooBig,
    /// Failed to allocate shared memory.
    #[error("Failed to allocate shared memory.")]
    Allocate(i32),
    /// Failed to create shared memory id file.
    #[error("Failed to create shared memory id file: {0}")]
    ShmidFileCreate(std::io::Error),
    /// Failed to write shared memory id file.
    #[error("Failed to write shared memory id file: {0}")]
    ShmidFileWrite(std::io::Error),
    /// Failed to attach shared memory.
    #[error("Failed to attach shared memory: {0}")]
    Attach(i32),
}
/// Error type for [`SharedAllocator::new_process`].
#[derive(Debug, thiserror::Error)]
pub enum SharedAllocatorNewProcessError {
    /// Failed to acquire lock on shared memory description map.
    #[error("Failed to acquire lock on shared memory description map.")]
    SharedMemoryDescriptionMapLock,
    /// Failed to open shared memory id file.
    #[error("Failed to open shared memory id file: {0}")]
    ShmidFileOpen(std::io::Error),
    /// Failed to read shared memory id file.
    #[error("Failed to read shared memory id file: {0}")]
    ShmidFileRead(std::io::Error),
    /// Failed to allocate shared memory.
    #[error("Failed to allocate shared memory: {0}")]
    Allocate(i32),
    /// Failed to attach shared memory.
    #[error("Failed to attach shared memory: {0}")]
    Attach(i32),
}

impl SharedAllocator {
    /// Since rust doesn't support static members on structs, we do this.
    #[must_use]
    #[inline]
    #[logfn(Trace)]
    pub fn shared_memory_description_map() -> Arc<Mutex<SharedMemoryData>> {
        // Counts instance non-dropped shared allocators in this process pointing to the same shared
        // memory.
        // (file, id, ptr, count)
        /// Static memory holding structures used to store the description of all shared memory
        /// attached to this process.
        static mut SHARED_MEMORY_DESCRIPTION_MAP: MaybeUninit<Arc<Mutex<SharedMemoryData>>> =
            MaybeUninit::uninit();
        /// Static used to ensure initialization of `SHARED_MEMORY_DESCRIPTION_MAP` once.
        static INIT_SHARED_MEMORY_DESCRIPTION_MAP: std::sync::Once = std::sync::Once::new();
        INIT_SHARED_MEMORY_DESCRIPTION_MAP.call_once(|| {
            trace!("initializing shared memory description map");
            // SAFETY:
            // The write only occurs once, as it is guarded by `INIT`.
            unsafe {
                SHARED_MEMORY_DESCRIPTION_MAP.write(Arc::new(Mutex::new(Vec::new())));
            }
            trace!("initialized shared memory description map");
        });
        // SAFETY:
        // `SHARED_MEMORY_DESCRIPTION_MAP` is initialized in this function, immediately above this
        // line.
        Arc::clone(unsafe { SHARED_MEMORY_DESCRIPTION_MAP.assume_init_ref() })
    }

    /// Returns address of the shared memory in the process memory (`self.0`).
    #[must_use]
    #[logfn(Trace)]
    pub fn address(&self) -> usize {
        self.0
    }

    /// Constructs a shared memory allocator.
    ///
    /// Storing the shared memory id in the file `shmid_path`.
    ///
    /// # Safety
    ///
    /// TODO
    ///
    /// # Errors
    ///
    /// For a whole lot of reasons. This is not a production ready library, it is a toy, treat it as
    /// such.
    #[logfn(Trace)]
    pub unsafe fn new(shmid_path: &str, size: usize) -> Result<Self, SharedAllocatorNewError> {
        info!("SharedAllocator::new");
        // Acquire reference to map of all shared allocators in this process
        let map = Self::shared_memory_description_map();
        // We cannot return `PoisonError`, thus we must discard information.
        #[allow(clippy::map_err_ignore)]
        let mut guard = map
            .lock()
            .map_err(|_| SharedAllocatorNewError::SharedMemoryDescriptionMapLock)?;

        // If shared allocator exists in this process for this shared memory
        #[allow(clippy::pattern_type_mismatch)]
        if let Some((_, _, shared_memory_ptr, count)) =
            guard.iter_mut().find(|(p, ..)| p == shmid_path)
        {
            *count = count
                .checked_add(1)
                .ok_or(SharedAllocatorNewError::TooMany)?;
            // Return allocator
            Ok(Self(*shared_memory_ptr))
        }
        // If this is first shared allocator in this process for this shared memory
        else {
            // Check if this is first allocator for this shared memory
            let first = !std::path::Path::new(shmid_path).exists();
            info!("first allocator: {}", first);
            let full_size = size
                .checked_add(std::mem::size_of::<InMemoryDescription>())
                .ok_or(SharedAllocatorNewError::TooBig)?;

            // If the shared memory id file doesn't exist, this is the first process to use this
            // shared memory. Thus we must allocate the shared memory.
            let shmid = if first {
                // Allocate shared memory
                let shmid = bindings::allocate_shared_memory(full_size)
                    .map_err(SharedAllocatorNewError::Allocate)?;
                // We simply save the shared memory id to a file for now
                let mut shmid_file = std::fs::File::create(shmid_path)
                    .map_err(SharedAllocatorNewError::ShmidFileCreate)?;
                shmid_file
                    .write_all(&shmid.to_ne_bytes())
                    .map_err(SharedAllocatorNewError::ShmidFileWrite)?;
                shmid
            } else {
                // Gets shared memory id
                let mut shmid_file = std::fs::File::open(shmid_path)
                    .map_err(SharedAllocatorNewError::ShmidFileOpen)?;
                let mut shmid_bytes = [0; 4];
                shmid_file
                    .read_exact(&mut shmid_bytes)
                    .map_err(SharedAllocatorNewError::ShmidFileRead)?;
                i32::from_ne_bytes(shmid_bytes)
            };
            info!("shmid: {}", shmid);

            // Attach shared memory
            let shared_mem_ptr =
                bindings::attach_shared_memory(shmid).map_err(SharedAllocatorNewError::Attach)?;

            // If first allocator for this shared memory, create memory description, else increment
            // process count
            if first {
                // Create in-memory memory description
                std::ptr::write(
                    shared_mem_ptr.cast::<InMemoryDescription>(),
                    InMemoryDescription {
                        count: AtomicU8::new(1),
                        capacity: size,
                        length: RwLock::new(0usize),
                    },
                );
            } else {
                // Increment process count
                (*shared_mem_ptr.cast::<InMemoryDescription>())
                    .count
                    .fetch_add(1, Ordering::SeqCst);
            }

            // Update allocator map and drop guard
            guard.push((String::from(shmid_path), shmid, shared_mem_ptr.to_bits(), 1));
            // REturn allocator
            Ok(Self(shared_mem_ptr.to_bits()))
        }
    }

    /// Constructs a shared memory allocator for shared memory (identified by `shmid_path`) which
    /// does not currently exist.
    ///
    /// Storing the shared memory id in the file `shmid_path`.
    ///
    /// This is slightly more efficient and explicit version of [`SharedAllocator::new`] when you
    /// know the precondition is met.
    ///
    /// # Safety
    ///
    /// When used for already allocated shared memory.
    ///
    /// # Errors
    ///
    /// For a whole lot of reasons. This is not a production ready library, it is a toy, treat it as
    /// such.
    #[logfn(Trace)]
    pub unsafe fn new_memory(
        shmid_path: &str,
        size: usize,
    ) -> Result<Self, SharedAllocatorNewMemoryError> {
        info!("SharedAllocator::new_memory");
        // Acquire reference to map of all shared allocators in this process
        let map = Self::shared_memory_description_map();
        // We cannot return `PoisonError`, thus we must discard information.
        #[allow(clippy::map_err_ignore)]
        let mut guard = map
            .lock()
            .map_err(|_| SharedAllocatorNewMemoryError::SharedMemoryDescriptionMapLock)?;

        // Presume 1st allocator for this shared memory
        debug_assert!(!std::path::Path::new(shmid_path).exists());
        let full_size = size
            .checked_add(std::mem::size_of::<InMemoryDescription>())
            .ok_or(SharedAllocatorNewMemoryError::TooBig)?;

        // Allocate shared memory
        let shmid = {
            let shmid = bindings::allocate_shared_memory(full_size)
                .map_err(SharedAllocatorNewMemoryError::Allocate)?;
            // We simply save the shared memory id to a file for now
            let mut shmid_file = std::fs::File::create(shmid_path)
                .map_err(SharedAllocatorNewMemoryError::ShmidFileCreate)?;
            shmid_file
                .write_all(&shmid.to_ne_bytes())
                .map_err(SharedAllocatorNewMemoryError::ShmidFileWrite)?;
            shmid
        };
        info!("shmid: {}", shmid);

        // Attach shared memory
        let shared_mem_ptr =
            bindings::attach_shared_memory(shmid).map_err(SharedAllocatorNewMemoryError::Attach)?;
        info!("shared memory pointer: {:?}", shared_mem_ptr);

        // Create in-memory memory description
        std::ptr::write(
            shared_mem_ptr.cast::<InMemoryDescription>(),
            InMemoryDescription {
                count: AtomicU8::new(1),
                capacity: size,
                length: RwLock::new(0usize),
            },
        );

        // Update allocator map and drop guard
        guard.push((String::from(shmid_path), shmid, shared_mem_ptr.to_bits(), 1));
        trace!("updated allocator map and dropped guard");

        // panic!("got here?");

        // Return allocator
        Ok(Self(shared_mem_ptr.to_bits()))
    }

    /// Constructs a shared memory allocator for existing shared memory (identified by `shmid_path`)
    /// which is not currently attached to this process.
    ///
    /// This is slightly more efficient and explicit version of [`SharedAllocator::new`] when you
    /// know the precondition is met.
    ///
    /// # Safety
    ///
    /// When used for shared memory not allocated or shared memory already attached to this process.
    ///
    /// # Errors
    ///
    /// For a whole lot of reasons. This is not a production ready library, it is a toy, treat it as
    /// such.
    #[logfn(Trace)]
    pub unsafe fn new_process(shmid_path: &str) -> Result<Self, SharedAllocatorNewProcessError> {
        info!("SharedAllocator::new_process");
        // Acquire reference to map of all shared allocators in this process
        let map = Self::shared_memory_description_map();
        // We cannot return `PoisonError`, thus we must discard information.
        #[allow(clippy::map_err_ignore)]
        let mut guard = map
            .lock()
            .map_err(|_| SharedAllocatorNewProcessError::SharedMemoryDescriptionMapLock)?;

        // Presume not 1st allocator for this shared memory
        debug_assert!(std::path::Path::new(shmid_path).exists());
        let shmid = {
            let mut shmid_file = std::fs::File::open(shmid_path)
                .map_err(SharedAllocatorNewProcessError::ShmidFileOpen)?;
            let mut shmid_bytes = [0; 4];
            shmid_file
                .read_exact(&mut shmid_bytes)
                .map_err(SharedAllocatorNewProcessError::ShmidFileRead)?;
            i32::from_ne_bytes(shmid_bytes)
        };
        info!("shmid: {}", shmid);

        // Attach shared memory
        let shared_mem_ptr = bindings::attach_shared_memory(shmid)
            .map_err(SharedAllocatorNewProcessError::Attach)?;

        // Increment process count (presume this allocator is not the first allocator for this
        // shared memory)
        (*shared_mem_ptr.cast::<InMemoryDescription>())
            .count
            .fetch_add(1, Ordering::SeqCst);

        // Update allocator map and drop guard
        guard.push((String::from(shmid_path), shmid, shared_mem_ptr.to_bits(), 1));
        // Return allocator
        Ok(Self(shared_mem_ptr.to_bits()))
    }
}
impl Drop for SharedAllocator {
    // We cannot return a result here, and we do not want to use [`Result::unchecked_unwrap`], so we
    // allow `expect`s here.
    #[allow(clippy::expect_used, clippy::pattern_type_mismatch)]
    #[logfn(Trace)]
    fn drop(&mut self) {
        let map = Self::shared_memory_description_map();

        let mut guard = map
            .lock()
            .expect("Failed to lock shared memory description");

        let (index, shared_memory_description) = guard
            .iter_mut()
            .enumerate()
            .find(|(_, (_, _, ptr, _))| *ptr == self.0)
            .expect("Failed to find shared memory description");
        trace!(
            "(shmid_path, shmid, _address, count): {:?}",
            shared_memory_description
        );
        let (shmid_path, shmid, _address, count) = shared_memory_description;

        // Decrement number of allocators in this process attached to this shared memory
        *count = count
            .checked_sub(1)
            .expect("Failed to decrement count of allocators");
        // If last allocator in this process attached to this shared memory
        let last_process_allocator = *count == 0;
        trace!("last_process_allocator: {}", last_process_allocator);

        if last_process_allocator {
            // Gets reference to shared memory description.
            // SAFETY:
            // Always safe.
            let description = unsafe { &*<*mut InMemoryDescription>::from_bits(self.0) };
            trace!("description: {:?}", description);

            // Decrement number of processes attached to this shared memory.
            let last_global_allocator = description.count.fetch_sub(1, Ordering::SeqCst) == 1;
            trace!("last_global_allocator: {}", last_global_allocator);

            // Detach shared memory
            // SAFETY:
            // TODO: Write more here.
            unsafe {
                bindings::detach_shared_memory(<*const libc::c_void>::from_bits(self.0))
                    .expect("Failed to detach shared memory");
            }

            // If last process attached to this shared memory.
            if last_global_allocator {
                // De-allocate shared memory
                // SAFETY:
                // TODO: Write more here.
                unsafe {
                    bindings::deallocate_shared_memory(*shmid)
                        .expect("Failed to deallocate shared memory");
                }

                // Since the second process closes last this one deletes the file
                info!("deleting shmid file: {}", shmid_path);
                std::fs::remove_file(shmid_path).expect("Failed to remove shmid file");
            }

            // Drops shared memory description.
            // Since this was last allocator in this process we no longer need a description for
            // this shared memory.
            guard.remove(index);
        }
    }
}
// SAFETY:
// Always safe.
unsafe impl std::alloc::Allocator for SharedAllocator {
    #[logfn(Trace)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // To allocate memory we need a write lock to the length
        // SAFETY:
        // Always safe.
        let description = unsafe { &*<*mut InMemoryDescription>::from_bits(self.0) };
        let mut guard = description.length.write().map_err(|err| {
            error!("allocation description lock error: {}", err);
            AllocError
        })?;
        let length = &mut *guard;

        let remaining_capacity = description
            .capacity
            .checked_sub(*length)
            .ok_or(AllocError)?;
        if remaining_capacity > layout.size() {
            // Alloc memory
            let ptr = <*mut u8>::from_bits(*length);
            // SAFETY:
            // This is always safe.
            let non_null_ptr = unsafe {
                NonNull::new_unchecked(std::ptr::slice_from_raw_parts_mut(ptr, layout.size()))
            };
            // Increase allocated length
            // SAFETY:
            // `remaining_capacity > layout.size()` therefore `length + layout.size() < usize::MAX`
            *length = unsafe { length.unchecked_add(layout.size()) };
            // Return pointer
            Ok(non_null_ptr)
        } else {
            Err(AllocError)
        }
    }

    #[allow(clippy::significant_drop_in_scrutinee)]
    #[logfn(Trace)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // To de-allocate memory we need a write lock to the length
        let description = &*<*mut InMemoryDescription>::from_bits(self.0);
        // TODO Would it be safe to use [`Result::unwrap_unchecked`] here?
        #[allow(clippy::unwrap_used)]
        let mut guard = description.length.write().unwrap();
        let length = &mut *guard;

        assert!(*length > layout.size());

        // Shifts all data after the deallocated item down the memory space by the size of the
        // deallocated item.
        let end = ptr.as_ptr().add(layout.size());
        // SAFETY:
        // We assert `*length > layout.size()` therefore this is safe.
        let following_len = unsafe { length.unchecked_sub(end.to_bits()) };
        let mem_after = std::ptr::slice_from_raw_parts_mut(end, following_len);
        #[allow(clippy::cast_ptr_alignment)]
        std::ptr::replace(ptr.as_ptr().cast::<&[u8]>(), &*mem_after);
        // Decreases the length
        // SAFETY:
        // We assert `*length > layout.size()` therefore this is safe.
        *length = unsafe { length.unchecked_sub(layout.size()) };
    }
}

#[cfg(test)]
#[allow(clippy::print_stdout)]
mod tests {
    use std::path::Path;

    use const_format::concatcp;
    use sequential_test::sequential;

    use super::*;

    // Create link to cargo binary e.g. `sudo ln -s /home/$USER/.cargo/bin/cargo /usr/bin/cargo`

    // const KB:usize = 1024;
    // const MB:usize = 1024*KB;
    // const GB:usize = 1024 * MB;
    // use quote::quote;
    // use std::process::Command;
    // use std::fs::{File, OpenOptions};
    // use std::io::Write;
    // const TEST_PREFIX:&str = "__rbd_test_";
    static INIT_LOGGER: std::sync::Once = std::sync::Once::new();

    const DIR: &str = "/tmp/";
    const TAG: &str = "__rdb_";
    const TEST_FILE_PREFIX: &str = concatcp!(DIR, TAG);
    const CARGO_BIN: &str = "/home/jonathan/.cargo/bin/cargo";

    // We need to run tests sequentially as they rely on checking the shared memory description
    // attached to this process. This description would not have a deterministic state if tests
    // were ran in parallel.

    // Tests creating and dropping an allocator with shared memory not previously existing.
    #[test]
    #[sequential]
    fn new_memory() {
        // Init logger
        INIT_LOGGER.call_once(|| {
            simple_logger::SimpleLogger::new().init().unwrap();
        });
        // Test
        const SHMID: &str = concatcp!(TEST_FILE_PREFIX, "new_memory");
        assert!(!Path::new(SHMID).exists(), "{}", SHMID);

        let shared_memory_description_map = SharedAllocator::shared_memory_description_map();
        assert!(shared_memory_description_map.lock().unwrap().is_empty());

        let shared_memory = unsafe { SharedAllocator::new_memory(SHMID, 1024 * 1024).unwrap() };
        assert!(Path::new(SHMID).exists());
        assert_eq!(shared_memory_description_map.lock().unwrap().len(), 1);

        drop(shared_memory);
        assert!(!Path::new(SHMID).exists());
        assert!(shared_memory_description_map.lock().unwrap().is_empty());
    }
    // Tests creating and dropping an allocator with shared memory previously created by a different
    // process.
    #[test]
    #[sequential]
    fn new_process() {
        // Init logger
        INIT_LOGGER.call_once(|| {
            simple_logger::SimpleLogger::new().init().unwrap();
        });
        // Test
        const SHMID: &str = concatcp!(TEST_FILE_PREFIX, "new_process");
        assert!(!Path::new(SHMID).exists(), "{}", SHMID);

        let shared_memory_description_map = SharedAllocator::shared_memory_description_map();
        assert!(shared_memory_description_map.lock().unwrap().is_empty());

        let shared_memory = unsafe { SharedAllocator::new_memory(SHMID, 1024 * 1024).unwrap() };
        assert!(Path::new(SHMID).exists());
        assert_eq!(shared_memory_description_map.lock().unwrap().len(), 1);

        // Create new process with shared memory allocator point to the same shared memory.
        let output = std::process::Command::new(CARGO_BIN)
            .args(["run", "--bin", "new_process"])
            .output()
            .unwrap();
        // Assert exit okay
        assert_eq!(output.status.code(), Some(0));

        // Check outputs
        let stdout = std::str::from_utf8(&output.stdout).unwrap();
        println!("stdout: \"\n{}\"", stdout);
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        println!("stderr: \"\n{}\"", stderr);

        // Drop memory
        drop(shared_memory);
        assert!(!Path::new(SHMID).exists());
        assert!(shared_memory_description_map.lock().unwrap().is_empty());
    }
    // Chain processes which each push 1 value to a vector stored in the shared memory.
    #[test]
    #[sequential]
    fn chain() {}
}
