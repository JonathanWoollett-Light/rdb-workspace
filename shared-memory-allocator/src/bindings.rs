#![allow(
    clippy::restriction,
    clippy::missing_errors_doc,
    clippy::missing_safety_doc
)]
use std::mem::MaybeUninit;

use log::trace;
use log_derive::{logfn, logfn_inputs};

/// Returns capacity of shared memory.
pub fn shared_memory_capacity(shmid: i32) -> Result<usize, i32> {
    shared_memory_description(shmid).map(|x| x.shm_segsz)
}

/// Returns whether the shared memory with a given key exists.
pub fn shared_memory_exists(key: i32) -> Result<bool, i32> {
    shared_memory_id(key).map(|x| x.is_some())
}

/// Returns the shared memory id for a given key.
pub fn shared_memory_id(key: i32) -> Result<Option<i32>, i32> {
    let res = unsafe { libc::shmget(key, Default::default(), libc::IPC_EXCL) };
    let errno = errno();
    trace!("res: {}", res);
    match (res, errno) {
        // > No segment exists for the given key, and IPC_CREAT was not specified.
        (-1i32, libc::ENOENT) => Ok(None),
        (-1i32, _) => Err(errno),
        (res, _) => Ok(Some(res)),
    }
}
/// Allocate shared memory.
///
/// <https://linux.die.net/man/2/shmget>
#[logfn(Trace)]
#[logfn_inputs(Trace)]
pub fn allocate_shared_memory(key: Option<i32>, size: usize) -> Result<i32, i32> {
    #[allow(clippy::cast_possible_wrap)]
    const PERMISSIONS: i32 = (libc::S_IRGRP
        | libc::S_IROTH
        | libc::S_IRUSR
        | libc::S_IWGRP
        | libc::S_IWOTH
        | libc::S_IWUSR) as i32;
    debug_assert!(key != Some(libc::IPC_PRIVATE));
    let allocated_shmid = unsafe {
        libc::shmget(
            key.unwrap_or(libc::IPC_PRIVATE),
            size,
            libc::IPC_CREAT | PERMISSIONS,
        )
    };
    trace!("allocated_shmid: {allocated_shmid}");
    match allocated_shmid {
        -1i32 => Err(errno()),
        _ => Ok(allocated_shmid),
    }
}
/// Gets the `shmid_ds` structure via <https://linux.die.net/man/2/shmctl>.
#[logfn(Trace)]
#[logfn_inputs(Trace)]
pub fn shared_memory_description(shmid: i32) -> Result<libc::shmid_ds, i32> {
    let mut description: MaybeUninit<libc::shmid_ds> = MaybeUninit::uninit();
    let res = unsafe { libc::shmctl(shmid, libc::IPC_STAT, description.as_mut_ptr()) };
    match res {
        0i32 => Ok(unsafe { description.assume_init_read() }),
        -1i32 => Err(errno()),
        // The documentation specifies:
        // > Other operations return 0 on success.
        // > On error, -1 is returned, and errno is set appropriately.
        // Therefore this is safe.
        #[allow(clippy::unreachable)]
        _ => unreachable!(),
    }
}
/// Attach shared memory to this process. If memory is already attached, does nothing.
///
/// <https://linux.die.net/man/2/shmat>
#[logfn(Trace)]
#[logfn_inputs(Trace)]
pub fn attach_shared_memory(shmid: i32) -> Result<*mut libc::c_void, i32> {
    let shared_mem_ptr = unsafe { libc::shmat(shmid, std::ptr::null(), 0) };
    trace!("shared_mem_ptr: {shared_mem_ptr:?}");

    #[allow(clippy::as_conversions)]
    if shared_mem_ptr as isize == -1 {
        Err(errno())
    } else {
        Ok(shared_mem_ptr)
    }
}
/// Detach shared memory from this process.
///
/// <https://linux.die.net/man/2/shmdt>
#[logfn(Trace)]
#[logfn_inputs(Trace)]
pub fn detach_shared_memory(ptr: *const libc::c_void) -> Result<(), i32> {
    let rtn = unsafe { libc::shmdt(ptr) };
    match rtn {
        0i32 => Ok(()),
        -1i32 => Err(errno()),
        // The documentation specifies:
        // > On success shmdt() returns 0; on error -1 is returned
        // Therefore this is safe.
        #[allow(clippy::unreachable)]
        _ => unreachable!(),
    }
}
/// Deallocate shared memory.
///
/// <https://linux.die.net/man/2/shmctl>
#[logfn(Trace)]
#[logfn_inputs(Trace)]
pub fn deallocate_shared_memory(shmid: i32) -> Result<(), i32> {
    let rtn = unsafe { libc::shmctl(shmid, libc::IPC_RMID, std::ptr::null_mut()) };
    match rtn {
        0i32 => Ok(()),
        -1i32 => Err(errno()),
        // The documentation specifies:
        // > A successful IPC_INFO or SHM_INFO operation returns the index of the highest used 
        // > entry in the kernel's internal array recording information about all shared memory segments.
        // > ...
        // > A successful SHM_STAT operation returns the identifier of the shared memory segment 
        // > whose index was given in shmid. Other operations return 0 on success.
        // > On error, -1 is returned, and errno is set appropriately.
        // Therefore this is safe.
        #[allow(clippy::unreachable)]
        _ => unreachable!(),
    }
}
/// Returns [`errno`](https://man7.org/linux/man-pages/man3/errno.3.html).
#[logfn(Trace)]
pub fn errno() -> i32 {
    // SAFETY:
    // Always safe.
    unsafe { *libc::__errno_location() }
}
/// Prints the error to stdout.
#[allow(dead_code)]
pub fn perror() {
    #[allow(clippy::expect_used)]
    let string = std::ffi::CString::new("Error").expect("Failed to write error to stdout.");
    // SAFETY:
    // Always safe.
    unsafe { libc::perror(string.as_ptr()) };
}
