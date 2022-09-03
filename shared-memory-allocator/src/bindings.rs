use libc::c_void;
use log::trace;
use log_derive::logfn;
/// Allocate shared memory
///
/// <https://linux.die.net/man/2/shmget>
#[logfn(Trace)]
pub unsafe fn allocate_shared_memory(size: usize) -> Result<i32, i32> {
    let shmid = libc::shmget(libc::IPC_PRIVATE, size, libc::IPC_CREAT);
    trace!("shmid: {shmid}");
    // trace!("errno(): {}",errno());
    match shmid {
        -1i32 => Err(errno()),
        _ => Ok(shmid),
    }
}

/// Attach shared memory to this process.
///
/// <https://linux.die.net/man/2/shmat>
#[logfn(Trace)]
pub unsafe fn attach_shared_memory(id: i32) -> Result<*mut c_void, i32> {
    let shared_mem_ptr = libc::shmat(id, std::ptr::null(), 0);
    trace!("shared_mem_ptr: {shared_mem_ptr:?}");
    // trace!("errno(): {}",errno());
    // TODO Is this right?
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
pub unsafe fn detach_shared_memory(ptr: *const libc::c_void) -> Result<(), i32> {
    let rtn = libc::shmdt(ptr);
    // trace!("errno(): {}",errno());
    match rtn {
        0i32 => Ok(()),
        -1i32 => Err(errno()),
        #[cfg(debug_assertions)]
        #[allow(clippy::unreachable)]
        _ => unreachable!(),
        // The documentation specifies:
        // > On success shmdt() returns 0; on error -1 is returned
        // Therefore this is safe.
        #[cfg(not(debug_assertions))]
        _ => std::hint::unreachable_unchecked(),
    }
}
/// Deallocate shared memory.
///
/// <https://linux.die.net/man/2/shmctl>
#[logfn(Trace)]
pub unsafe fn deallocate_shared_memory(id: i32) -> Result<(), i32> {
    let rtn = libc::shmctl(id, libc::IPC_RMID, std::ptr::null_mut());
    // trace!("errno(): {}",errno());
    match rtn {
        0i32 => Ok(()),
        -1i32 => Err(errno()),
        #[cfg(debug_assertions)]
        #[allow(clippy::unreachable)]
        _ => unreachable!(),
        // The documentation specifies:
        // > A successful IPC_INFO or SHM_INFO operation returns the index of the highest used 
        // > entry in the kernel's internal array recording information about all shared memory segments.
        // > ...
        // > A successful SHM_STAT operation returns the identifier of the shared memory segment 
        // > whose index was given in shmid. Other operations return 0 on success.
        // > On error, -1 is returned, and errno is set appropriately.
        // Therefore this is safe.
        #[cfg(not(debug_assertions))]
        _ => std::hint::unreachable_unchecked(),
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
