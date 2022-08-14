#![warn(clippy::pedantic)]
use std::io::{Read, Write};
use std::mem::size_of;
use std::net::TcpStream;
use std::time::{Duration, Instant};

use indicatif::ProgressBar;
use log::{debug, info, trace};
use log_derive::logfn;
use rand::Rng;
use serde::Serialize;

const ADDRESS: &str = "127.0.0.1:8080";
/// We allow the client to hang for up to 100ms awiating the server transfer process.
///
/// The vast majority of transfers are <1ms, this is just very cautious.
const SERVER_TRANSFER_TIMEOUT: u64 = 100;

type CommunicationResult<T> = Result<T, CommunicationErr>;
#[derive(Debug)]
enum CommunicationErr {
    /// In attempt to write the input, we discovered the write half of the stream has been
    /// shutdown.
    Write(std::io::Error),
    /// In attempting to read the result, we discovered the read half of the stream has been
    /// shutdown.
    Read(EndOfStreamErr),
}
impl From<std::io::Error> for CommunicationErr {
    fn from(io_err: std::io::Error) -> Self {
        info!("io_err: {:?}", io_err);
        Self::Write(io_err)
    }
}
impl From<EndOfStreamErr> for CommunicationErr {
    fn from(eos_err: EndOfStreamErr) -> Self {
        info!("eos_err: {:?}", eos_err);
        Self::Read(EndOfStreamErr)
    }
}
type GetReturn = Result<u16, ()>;
type SetReturn = Result<(), ()>;
type SumReturn = Result<u16, ()>;
struct Client(TcpStream);
impl Client {
    pub fn new(address: impl std::net::ToSocketAddrs) -> Self {
        Self(TcpStream::connect(address).unwrap())
    }

    fn reconnect(&mut self) {
        // Wait for server transfer to complete
        std::thread::sleep(Duration::from_millis(SERVER_TRANSFER_TIMEOUT));
        // TODO Can we use `TcpStream::connect_SERVER_TRANSFER_TIMEOUT` here so we don't always
        // need to wait full time?
        // Re-connect to stream
        self.0 = TcpStream::connect(&ADDRESS).unwrap();
    }

    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn get(&mut self) -> GetReturn {
        match self.raw_get() {
            Ok(res) => res,
            Err(_) => {
                // Presume error as a result of server transfer
                self.reconnect();
                // Re-rerun query
                self.raw_get().unwrap()
            }
        }
    }

    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn set(&mut self, x: u16) -> SetReturn {
        match self.raw_set(x) {
            Ok(res) => res,
            Err(_) => {
                // Presume error as a result of server transfer
                self.reconnect();
                // Re-rerun query
                self.raw_set(x).unwrap()
            }
        }
    }

    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn sum(&mut self, set: Vec<u16>) -> SumReturn {
        match self.raw_sum(set.clone()) {
            Ok(res) => res,
            Err(_) => {
                // Presume error as a result of server transfer
                self.reconnect();
                // Re-rerun query
                self.raw_sum(set).unwrap()
            }
        }
    }

    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn raw_get(&mut self) -> CommunicationResult<GetReturn> {
        // Write
        write_frame(&mut self.0, 0, ())?;
        // Read
        let serialized = read_frame(&mut self.0)?;
        Ok(bincode::deserialize(&serialized).unwrap())
    }

    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn raw_set(&mut self, x: u16) -> CommunicationResult<SetReturn> {
        // Write
        write_frame(&mut self.0, 1, x)?;
        // Read
        let serialized = read_frame(&mut self.0)?;
        Ok(bincode::deserialize(&serialized).unwrap())
    }

    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn raw_sum(&mut self,set:Vec<u16>) -> CommunicationResult<SumReturn> {
        // Write
        write_frame(&mut self.0, 2, set)?;
        // Read
        let serialized = read_frame(&mut self.0)?;
        Ok(bincode::deserialize(&serialized).unwrap())
    }
}

#[derive(Debug)]
struct EndOfStreamErr;
/// Reads frame from stream.
fn read_frame(stream: &mut TcpStream) -> Result<Vec<u8>, EndOfStreamErr> {
    // A frame consists of:
    // 1. A number defining the length of the input bytes (usize).
    // 2. The bytes (Vec<u8>).
    const S: usize = size_of::<usize>();
    let mut bytes = vec![Default::default(); S];
    let mut i = 0;

    // != or < have same affect here
    while i != S {
        match stream.read(&mut bytes[i..]) {
            // End of stream
            Ok(0) => return Err(EndOfStreamErr),
            Ok(size) => {
                i += size;
            }
            _ => unreachable!(),
        }
    }
    let bytes_length = usize::from_ne_bytes(bytes[0..S].try_into().unwrap());

    // Read data bytes
    let data_length = bytes_length - S;
    let mut data_bytes = vec![Default::default(); data_length];
    let mut j = 0;
    // != or < have same affect here
    while j != data_length {
        match stream.read(&mut data_bytes[j..]) {
            Ok(0) => return Err(EndOfStreamErr),
            Ok(size) => {
                j += size;
            }
            _ => unreachable!(),
        }
    }
    // Returns data bytes
    Ok(data_bytes)
}
/// Writes frame to stream
fn write_frame(
    stream: &mut TcpStream,
    function_index: usize,
    input: impl Serialize,
) -> std::io::Result<()> {
    let serialized = bincode::serialize(&input).unwrap();
    trace!("serialized: {:?}", &serialized);
    let mut write_bytes = Vec::from(serialized.len().to_ne_bytes());
    write_bytes.extend(function_index.to_ne_bytes());
    write_bytes.extend(serialized);
    trace!("write_bytes: {:?}", &write_bytes);
    stream.write_all(&write_bytes)
}
fn stress_test(requests: u64, bar: &ProgressBar) {
    let mut client = Client::new(ADDRESS);
    let mut rng = rand::thread_rng();
    for _ in 0..requests {
        // Picks random query
        let x: usize = rng.gen_range(0..3);
        debug!("x: {}", x);

        let function_result = match x {
            0 => client.get().is_ok(),
            1 => client.set(2).is_ok(),
            2 => client.sum(vec![1,2,3,4,5,6,7,8,9]).is_ok(),
            _ => unreachable!(),
        };
        assert!(function_result);

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
            std::thread::spawn(move || stress_test(requests, &bar))
        })
        .collect::<Vec<_>>();
    // Join all threads
    for handle in handles {
        handle.join().unwrap();
    }
    multibar.clear().unwrap();
    println!("live for: {:?}", now.elapsed());
}
