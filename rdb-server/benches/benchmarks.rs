use criterion::{criterion_group, criterion_main, Criterion};
// const CARGO_BIN: &str = "/home/jonathan/.cargo/bin/cargo";
// const SERVER_BINARY:&str = "/home/jonathan/Projects/rdb-workspace/target/debug/rdb-server";

pub fn switch_benchmark(c: &mut Criterion) {
    //     let mut predecessor = std::process::Command::new("sudo")
    //         .arg(SERVER_BINARY)
    //         .spawn()
    //         .unwrap();
    //     let mut successor = None;

    //     for

    //     c.bench_function("switching", |b| b.iter(|| {
    //         successor = Some(std::process::Command::new("sudo")
    //             .arg(SERVER_BINARY)
    //             .spawn()
    //             .unwrap());
    //         predecessor.wait().unwrap();
    //         predecessor = successor.take().unwrap();
    //     }));

    //     // Interrupt server (`ctrl+c`)
    //     unsafe {
    //         libc::kill(predecessor.id() as i32,libc::SIGINT);
    //     }
}

criterion_group!(benches, switch_benchmark);
criterion_main!(benches);
