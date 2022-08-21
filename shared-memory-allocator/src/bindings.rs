use std::hint::unreachable_unchecked;

use libc::c_void;
/// Allocate shared memory
///
/// <https://linux.die.net/man/2/shmget>
pub unsafe fn allocate_shared_memory(size: usize) -> Result<i32, i32> {
    let shmid = libc::shmget(libc::IPC_PRIVATE, size, libc::IPC_CREAT);
    match shmid {
        -1i32 => Err(errno()),
        _ => Ok(shmid),
    }
}

/// Attach shared memory to this process.
///
/// <https://linux.die.net/man/2/shmat>
pub unsafe fn attach_shared_memory(id: i32) -> Result<*mut c_void, i32> {
    let shared_mem_ptr = libc::shmat(id, std::ptr::null(), 0);
    #[allow(clippy::as_conversions)]
    match shared_mem_ptr as isize {
        0 => Ok(shared_mem_ptr),
        -1 => Err(errno()),
        // The documentation specifies:
        // > On success shmat() returns the address of the attached shared memory segment; on error
        // > (void *) -1 is returned, and errno is set to indicate the cause of the error.
        // Therefore this is safe.
        _ => unreachable_unchecked(),
    }
}
/// Detach shared memory from this process.
///
/// <https://linux.die.net/man/2/shmdt>
pub unsafe fn detach_shared_memory(ptr: *const libc::c_void) -> Result<(), i32> {
    let rtn = libc::shmdt(ptr);
    match rtn {
        0i32 => Ok(()),
        -1i32 => Err(errno()),
        // The documentation specifies:
        // > On success shmdt() returns 0; on error -1 is returned
        // Therefore this is safe.
        _ => unreachable_unchecked(),
    }
}
/// Deallocate shared memory.
///
/// <https://linux.die.net/man/2/shmctl>
pub unsafe fn deallocate_shared_memory(id: i32) -> Result<(), i32> {
    let rtn = libc::shmctl(id, libc::IPC_RMID, std::ptr::null_mut());
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
        _ => unreachable_unchecked(),
    }
}
/// Returns [`errno`](https://man7.org/linux/man-pages/man3/errno.3.html).
pub fn errno() -> i32 {
    // SAFETY:
    // Always safe.
    unsafe { *libc::__errno_location() }
}
// fn reset_err() {
//     unsafe { *libc::__errno_location() = 0 };
// }

// fn check_err() -> Result<(),i32> {
//     let errno = unsafe { libc::__errno_location() };
//     let errno = unsafe { *errno };
//     if errno != 0 {
//         // let string = std::ffi::CString::new("message").unwrap();
//         // unsafe { libc::perror(string.as_ptr()) };
//         // panic!("Error occured, error code: {errno}");
//         Err(errno)
//     }
//     else {
//         Ok(())
//     }
// }
