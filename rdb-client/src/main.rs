#![warn(clippy::pedantic)]
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use indicatif::ProgressBar;
use log::{debug, info};
use rand::Rng;

const ADDRESS: &str = "127.0.0.1:8080";
const TIMEOUT: u64 = 100;

fn handle_client(requests: u64, bar: &ProgressBar) {
    let mut stream = TcpStream::connect(ADDRESS).unwrap();
    let mut rng = rand::thread_rng();
    for _ in 0..requests {
        let x: u32 = rng.gen_range(0..3);

        info!("x: {}", x);

        let buf = x.to_ne_bytes();

        let write_res = stream.write(&buf);

        debug!("write_res (1): {:?}", &write_res);

        write_res.unwrap();

        let mut rec_buf = [0; 4];
        let read_res = stream.read(&mut rec_buf);

        debug!("read_res (1): {:?}", &read_res);

        if let Ok(4) = read_res {
            info!("ok");
        }
        // If not okay we presume server process transfer and act as such
        else {
            info!("server transition");

            // Re-connect to stream
            std::thread::sleep(Duration::from_millis(TIMEOUT));
            // TODO Can we use `TcpStream::connect_timeout` here so we don't always need to wait
            // full time?
            stream = TcpStream::connect(&ADDRESS).unwrap();

            // Retry request
            let write_res = stream.write(&buf);

            debug!("write_res (2): {:?}", &write_res);

            write_res.unwrap();

            let read_res = stream.read(&mut rec_buf);

            debug!("read_res (2): {:?}", &read_res);

            assert!(read_res.is_ok());
            assert_eq!(read_res.unwrap(), 4);
        }

        let y = u32::from_ne_bytes(rec_buf);

        info!("y: {}", &y);

        match x {
            0 => assert_eq!(y, 2 * 12),
            1 => assert_eq!(y, 3 * 12),
            2 => assert_eq!(y, 4 * 12),
            _ => unreachable!(),
        };

        #[cfg(debug_assertions)]
        std::thread::sleep(Duration::from_secs(1));

        bar.inc(1);
    }
    bar.finish();
}

fn main() {
    const REQUESTS_RANGE: std::ops::Range<u64> = 5_000_000..10_000_000;
    const CLIENTS: usize = 6;

    let now = Instant::now();
    simple_logger::init_with_level(log::Level::Warn).unwrap();
    info!("started");

    let multibar = indicatif::MultiProgress::new();
    let mut rng = rand::thread_rng();

    let handles = (0..CLIENTS)
        .map(|_| {
            let requests = rng.gen_range(REQUESTS_RANGE);
            let bar = multibar.add(indicatif::ProgressBar::new(requests));
            std::thread::spawn(move || handle_client(requests, &bar))
        })
        .collect::<Vec<_>>();
    // Join all threads
    for handle in handles {
        handle.join().unwrap();
    }
    multibar.clear().unwrap();
    println!("live for: {:?}", now.elapsed());
}
