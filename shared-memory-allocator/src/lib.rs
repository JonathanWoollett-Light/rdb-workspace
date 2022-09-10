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
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use log::{error, info, trace};
use log_derive::logfn;

/// The description of shared memory stored within the shared memory.
#[derive(Debug)]
struct InMemoryDescription {
    /// The currently used amount of the shared memory.
    length: RwLock<usize>,
}

use std::sync::RwLock;

/// Wrappers around `libc` shared memory functions.
pub mod bindings;

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
    /// Failed to schedule de-allocation of memory.
    #[error("Failed to schedule de-allocation of memory: {0}")]
    Deallocate(i32),
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
    /// Failed to schedule de-allocation of memory.
    #[error("Failed to schedule de-allocation of memory: {0}")]
    Deallocate(i32),
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

/// An allocator implementing [`std::alloc::Allocator`] which allocates items in linux shared
/// memory.
///
/// Constructing multiple allocators with the same `key` will use the same shared memory.
///
/// After constructing the first allocator of a given `key` constructing new allocators
/// with the same `key` is the same as cloning the original allocator.
///
/// At the moment, both allocation and deallocation of memory, functions like pushing and
/// removing element from a vector (e.g. very inefficient), this is something which will be
/// improved moving forward.
///
/// Once a shared allocator is constructed this shared memory is attached for the whole lifetime of
/// the process.Shared memory can be detached earlier by calling `SharedAllocator::detach`, using
/// shared allocators which referred to this shared memory after detaching it, is undefined
/// behavior.
#[derive(Debug, Clone)]
pub struct SharedAllocator {
    /// Shared memory key
    key: i32,
    /// Shared memory shmid
    shmid: i32,
    /// Address of start of shared memory
    addr: usize,
}

impl SharedAllocator {
    /// Returns the address of the shared memory.
    #[must_use]
    #[inline]
    pub fn address(&self) -> usize {
        self.addr
    }

    /// Returns the key of the shared memory.
    #[must_use]
    #[inline]
    pub fn key(&self) -> i32 {
        self.key
    }

    /// Since rust doesn't support static members on structs, we do this.
    #[must_use]
    #[inline]
    #[logfn(Trace)]
    pub fn shared_memory_description_map() -> Arc<Mutex<HashMap<i32, (i32, usize)>>> {
        // Counts instance non-dropped shared allocators in this process pointing to the same shared
        // memory.
        // <key, (shmid, address)>
        /// Static memory holding structures used to store the description of all shared memory
        /// attached to this process.
        static mut SHARED_MEMORY_DESCRIPTION_MAP: MaybeUninit<
            Arc<Mutex<HashMap<i32, (i32, usize)>>>,
        > = MaybeUninit::uninit();
        /// Static used to ensure initialization of `SHARED_MEMORY_DESCRIPTION_MAP` once.
        static INIT_SHARED_MEMORY_DESCRIPTION_MAP: std::sync::Once = std::sync::Once::new();
        INIT_SHARED_MEMORY_DESCRIPTION_MAP.call_once(|| {
            trace!("initializing shared memory description map");
            // SAFETY:
            // The write only occurs once, as it is guarded by `INIT`.
            unsafe {
                SHARED_MEMORY_DESCRIPTION_MAP.write(Arc::new(Mutex::new(HashMap::new())));
            }
            trace!("initialized shared memory description map");
        });
        // SAFETY:
        // `SHARED_MEMORY_DESCRIPTION_MAP` is initialized in this function, immediately above this
        // line.
        Arc::clone(unsafe { SHARED_MEMORY_DESCRIPTION_MAP.assume_init_ref() })
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
    pub fn new(key: i32, size: usize) -> Result<Self, SharedAllocatorNewError> {
        info!("SharedAllocator::new");

        let map = Self::shared_memory_description_map();
        #[allow(clippy::map_err_ignore)]
        let mut guard = map
            .lock()
            .map_err(|_| SharedAllocatorNewError::SharedMemoryDescriptionMapLock)?;

        // If shared allocator exists in this process for this shared memory
        #[allow(clippy::map_err_ignore)]
        if let Some(&(shmid, addr)) = guard.get(&key) {
            Ok(Self { key, shmid, addr })
        }
        // If this is first shared allocator in this process for this shared memory
        else {
            // Check if this is first allocator for this shared memory
            #[allow(clippy::unwrap_used)]
            let check = bindings::shared_memory_allocated(key).unwrap();

            let shmid = match check {
                Some(shmid) => shmid,
                // If the shared memory does not exist, this is the first process to use this
                // shared memory. Thus we must allocate the shared memory.
                None => {
                    let full_size = size
                        .checked_add(std::mem::size_of::<InMemoryDescription>())
                        .ok_or(SharedAllocatorNewError::TooBig)?;
                    // Allocate shared memory
                    bindings::allocate_shared_memory(Some(key), full_size)
                        .map_err(SharedAllocatorNewError::Allocate)?
                }
            };

            // Attach shared memory
            let shared_mem_ptr =
                bindings::attach_shared_memory(shmid).map_err(SharedAllocatorNewError::Attach)?;

            // If first allocator for this shared memory, create memory description.
            if check.is_none() {
                // Create in-memory memory description
                unsafe { std::ptr::write(
                    shared_mem_ptr.cast::<InMemoryDescription>(),
                    InMemoryDescription {
                        length: RwLock::new(0usize),
                    },
                ) };

                // Schedule de-allocation of shared memory.
                //
                // We do this here as the documentation notes:
                // > Mark the segment to be destroyed. The segment will only actually be destroyed after the
                // > last process detaches
                // Thus we can do this immediately after attaching the shared memory to this process.
                //
                // SAFETY:
                // TODO: Write more here.
                bindings::deallocate_shared_memory(shmid).map_err(SharedAllocatorNewError::Deallocate)?;
            }

            // Update allocator map and drop guard
            guard.insert(key, (shmid, shared_mem_ptr.to_bits()));
            // REturn allocator
            Ok(Self {
                key,
                shmid,
                addr: shared_mem_ptr.to_bits(),
            })
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
    pub fn new_memory(key: i32, size: usize) -> Result<Self, SharedAllocatorNewMemoryError> {
        // Acquire reference to map of all shared allocators in this process
        let map = Self::shared_memory_description_map();
        // We cannot return `PoisonError`, thus we must discard information.
        #[allow(clippy::map_err_ignore)]
        let mut guard = map
            .lock()
            .map_err(|_| SharedAllocatorNewMemoryError::SharedMemoryDescriptionMapLock)?;

        // Presume 1st allocator for this shared memory
        let full_size = size
            .checked_add(std::mem::size_of::<InMemoryDescription>())
            .ok_or(SharedAllocatorNewMemoryError::TooBig)?;

        // Allocate shared memory
        let shmid = bindings::allocate_shared_memory(Some(key), full_size)
            .map_err(SharedAllocatorNewMemoryError::Allocate)?;

        // Attach shared memory
        let shared_mem_ptr =
            bindings::attach_shared_memory(shmid).map_err(SharedAllocatorNewMemoryError::Attach)?;
        info!("shared memory pointer: {:?}", shared_mem_ptr);

        // Create in-memory memory description
        unsafe {
            std::ptr::write(
                shared_mem_ptr.cast::<InMemoryDescription>(),
                InMemoryDescription {
                    length: RwLock::new(0usize),
                },
            )
        };

        // Update allocator map and drop guard
        guard.insert(shmid, (shmid, shared_mem_ptr.to_bits()));
        trace!("updated allocator map and dropped guard");

        // Schedule de-allocation of shared memory.
        //
        // We do this here as the documentation notes:
        // > Mark the segment to be destroyed. The segment will only actually be destroyed after the
        // > last process detaches
        // Thus we can do this immediately after attaching the shared memory to this process.
        //
        // SAFETY:
        // TODO: Write more here.
        bindings::deallocate_shared_memory(shmid)
            .map_err(SharedAllocatorNewMemoryError::Deallocate)?;

        // Return allocator
        Ok(Self {
            key,
            shmid,
            addr: shared_mem_ptr.to_bits(),
        })
    }

    /// Detaches shared memory referred to by `Self`.
    #[logfn(Trace)]
    pub fn detach(self) {
        bindings::detach_shared_memory(<*mut libc::c_void>::from_bits(self.addr)).unwrap();
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
    pub fn new_process(key: i32) -> Result<Self, SharedAllocatorNewProcessError> {
        let map = Self::shared_memory_description_map();
        // We cannot return `PoisonError`, thus we must discard information.
        #[allow(clippy::map_err_ignore)]
        let mut guard = map
            .lock()
            .map_err(|_| SharedAllocatorNewProcessError::SharedMemoryDescriptionMapLock)?;

        // Presume not 1st allocator for this shared memory
        debug_assert!(guard.get(&key).is_none());

        let shmid = bindings::shared_memory_allocated(key)
            .unwrap()
            .expect("shared memory not allocated");
        // Attach shared memory
        let shared_mem_ptr = bindings::attach_shared_memory(shmid)
            .map_err(SharedAllocatorNewProcessError::Attach)?;

        // Update allocator map and drop guard
        guard.insert(key, (shmid, shared_mem_ptr.to_bits()));
        // Return allocator
        Ok(Self {
            key,
            shmid,
            addr: shared_mem_ptr.to_bits(),
        })
    }
}
#[cfg(doc)]
impl Drop for SharedAllocator {
    #[logfn(Trace)]
    fn drop(&mut self) {
        // We do not need to detach the shared memory manually as the documentation notes:
        // > Upon _exit(2) all attached shared memory segments are detached from the process.
        // And we want memory to be attached for the whole lifetime of the process.
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
        let description = unsafe { &*<*mut InMemoryDescription>::from_bits(self.addr) };
        let mut guard = description.length.write().map_err(|err| {
            error!("allocation description lock error: {}", err);
            AllocError
        })?;
        let length = &mut *guard;

        // SAFETY:
        // todo
        let capacity = bindings::shared_memory_capacity(self.shmid).map_err(|err| {
            error!("failed to get shared memory capacity: {}", err);
            AllocError
        })?;
        let remaining_capacity = capacity.checked_sub(*length).ok_or(AllocError)?;
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
        let description = &*<*mut InMemoryDescription>::from_bits(self.addr);
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
    use std::sync::atomic::{AtomicI32, Ordering};

    use sequential_test::sequential;

    use super::*;

    fn unique_key() -> i32 {
        const BASE_KEY: i32 = 453845;
        static BASE: AtomicI32 = AtomicI32::new(BASE_KEY);
        BASE.fetch_add(1, Ordering::SeqCst)
    }

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
    fn init_logger() {
        INIT_LOGGER.call_once(|| {
            simple_logger::SimpleLogger::new().init().unwrap();
        });
    }
    const CARGO_BIN: &str = "/home/jonathan/.cargo/bin/cargo";

    // We need to run tests sequentially as they rely on checking the shared memory description
    // attached to this process. This description would not have a deterministic state if tests
    // were ran in parallel.

    // Tests creating and dropping an allocator with shared memory not previously existing.
    #[test]
    #[sequential]
    fn new_memory() {
        // Init logger
        init_logger();

        let key = unique_key();

        // Check if shared memory allocated under `key`.
        assert!(bindings::shared_memory_allocated(key).unwrap().is_none());

        let shared_memory_description_map = SharedAllocator::shared_memory_description_map();
        assert!(shared_memory_description_map.lock().unwrap().is_empty());

        let _shared_memory = SharedAllocator::new_memory(key, 1024 * 1024).unwrap();
    }
    // Tests creating and dropping an allocator with shared memory previously created by a different
    // process.
    #[test]
    #[sequential]
    fn new_process() {
        // Init logger
        init_logger();
        // Test
        const NEW_PROCESS_KEY: i32 = 240434;
        assert!(bindings::shared_memory_allocated(NEW_PROCESS_KEY)
            .unwrap()
            .is_none());

        let shared_memory_description_map = SharedAllocator::shared_memory_description_map();
        assert!(shared_memory_description_map.lock().unwrap().is_empty());

        let _shared_memory = SharedAllocator::new_memory(NEW_PROCESS_KEY, 1024 * 1024).unwrap();
        assert!(bindings::shared_memory_allocated(NEW_PROCESS_KEY)
            .unwrap()
            .is_some());

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
    }
    const SIZE: usize = 1024 * 1024;
    #[test]
    #[sequential]
    fn base() {
        // Init logger
        init_logger();

        let key = unique_key();

        let shmid = bindings::allocate_shared_memory(Some(key), SIZE).unwrap();
        // let shmid = unsafe {
        //     let shmid = libc::shmget(KEY, SIZE, libc::IPC_CREAT);
        //     dbg!(*libc::__errno_location());
        //     shmid
        // };

        let description = bindings::shared_memory_description(shmid).unwrap();
        // let description = unsafe {
        //     let mut description: MaybeUninit<libc::shmid_ds> = MaybeUninit::uninit();
        //     let result = libc::shmctl(shmid, libc::IPC_STAT, description.as_mut_ptr());
        //     dbg!(result);
        //     dbg!(*libc::__errno_location());
        //     description.assume_init_read();
        // };
        assert_eq!(description.shm_perm.__key, key);
        assert_eq!(description.shm_segsz, SIZE);

        let ptr = bindings::attach_shared_memory(shmid).unwrap();
        bindings::detach_shared_memory(ptr).unwrap();

        bindings::deallocate_shared_memory(shmid).unwrap();
    }
    #[test]
    #[sequential]
    fn exists() {
        // Init logger
        init_logger();

        let key = unique_key();

        // Check if shared memory allocated under `key`.
        assert!(bindings::shared_memory_allocated(key).unwrap().is_none());

        // Allocates shared memory under `key`.
        let shmid = bindings::allocate_shared_memory(Some(key), SIZE).unwrap();

        // Check if shared memory allocated under `key`.
        assert!(bindings::shared_memory_allocated(key).unwrap().is_some());

        // Deallocates shared memory bound to `shmid`.
        bindings::deallocate_shared_memory(shmid).unwrap();

        // Check if shared memory allocated under `key`.
        assert!(bindings::shared_memory_allocated(key).unwrap().is_none());
    }
}
