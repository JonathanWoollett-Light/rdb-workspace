use criterion::{black_box, criterion_group, criterion_main, Criterion};
const CARGO_BIN: &str = "/home/jonathan/.cargo/bin/cargo";

pub fn switch_benchmark(c: &mut Criterion) {
    // let child = std::process::Command::new(CARGO_BIN)
    //     .args(["run", "--bin", "rbd-server"])
    //     .spawn()
    //     .unwrap();

    // c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));

    // // Interrupt server (`ctrl+c`)
    // unsafe {
    //     libc::kill(child.id() as i32,libc::SIGINT);
    // }
}

criterion_group!(benches, switch_benchmark);
criterion_main!(benches);
