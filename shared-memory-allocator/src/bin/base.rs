use std::mem::MaybeUninit;

fn main() {
    const KEY: i32 = 453845;
    const SIZE: usize = 1024 * 1024;
    unsafe {
        let permissions = (libc::S_IRGRP
            | libc::S_IROTH
            | libc::S_IRUSR
            | libc::S_IWGRP
            | libc::S_IWOTH
            | libc::S_IWUSR) as i32;
        let shmid = libc::shmget(KEY, SIZE, libc::IPC_CREAT | permissions);
        dbg!(shmid);
        dbg!(*libc::__errno_location());

        let mut description: MaybeUninit<libc::shmid_ds> = MaybeUninit::uninit();
        let result = libc::shmctl(shmid, libc::IPC_STAT, description.as_mut_ptr());
        dbg!(description.assume_init_read());
        dbg!(result);
        dbg!(*libc::__errno_location());
    }
}
