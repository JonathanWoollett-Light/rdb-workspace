fn overwrite(s: &str) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(s)
}

#[allow(clippy::unwrap_used)]
#[test]
fn special_get_test() {
    let key = shared_memory_allocator::new_key();
    let addr = "127.0.0.1:8484";
    let old_process_socket: &str = "/tmp/get_test_ops";
    let new_process_socket: &str = "/tmp/get_test_nps";

    // Start server
    let mut server_process = std::process::Command::new("cargo")
        .args([
            "run",
            "--",
            "-a",
            addr,
            "-n",
            new_process_socket,
            "-o",
            old_process_socket,
            "-l",
            "Trace",
            "-s",
            &key.to_string(),
        ])
        .stdout(overwrite("/tmp/stdout.txt").unwrap())
        .stderr(overwrite("/tmp/stderr.txt").unwrap())
        .spawn()
        .unwrap();

    // Wait for process to have started
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Connect client
    let mut client = rdb::Client::new(addr).unwrap();

    // Stop server
    {
        // Interrupt server (`ctrl+c`)
        // SAFETY:
        // This call should always be safe.
        unsafe {
            let pid = i32::try_from(server_process.id()).unwrap();
            libc::kill(pid, libc::SIGINT);
        }
        let res = server_process.wait();
        dbg!(&res);
        assert!(matches!(res, Ok(ok) if ok.success()));
    }

    // Wait for process to have started
    std::thread::sleep(std::time::Duration::from_secs(3));

    // Call function with client
    let res = client.raw_get(());

    panic!("res: {:?}", res);
}
