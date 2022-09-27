#![feature(unchecked_math)]
#![feature(exclusive_range_pattern)]
#![feature(precise_pointer_size_matching)]
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

//! Crate

use std::io::{Read, Write};
use std::mem::size_of;
use std::net::TcpStream;
use std::time::Duration;

use log::trace;
use log_derive::logfn;
use serde::Serialize;

/// We allow the client to hang for up to 100ms awaiting the server transfer process.
///
/// The vast majority of transfers are <1ms, this is just very cautious.
const SERVER_TRANSFER_TIMEOUT: u64 = 100;

/// Result of [`CommunicationErr`].
type CommunicationResult<T> = Result<T, CommunicationErr>;
/// Error type of raw queries covering errors which arise during communication with the server.
#[derive(Debug, thiserror::Error)]
pub enum CommunicationErr {
    /// In attempt to write the input, we discovered the write half of the stream has been
    /// shutdown.
    #[error(
        "In attempt to write the input, we discovered the write half of the stream has been \
         shutdown: {0}"
    )]
    Write(#[from] WriteFrameError),
    /// In attempting to read the result, we discovered the read half of the stream has been
    /// shutdown.
    #[error(
        "In attempting to read the result, we discovered the read half of the stream has been \
         shutdown: {0}"
    )]
    Read(#[from] ReadFrameError),
    /// Failed to deserialize results.
    #[error("Failed to deserialize results: {0}")]
    #[cfg(feature = "bincode")]
    Deserialize(#[from] bincode::Error),
    #[cfg(feature = "json")]
    Deserialize(#[from] serde_json::Error),
    /// Failed to connect to server `TcpStream`.
    #[error("Failed to connect to server `TcpStream`.")]
    Connect(#[from] std::io::Error),
}

/// Result of user defined `get` query.
type GetReturn = Result<u16, ()>;
/// Result of user defined `set` query.
type SetReturn = Result<(), ()>;
/// Result of user defined `sum` query.
type SumReturn = Result<u16, ()>;
/// Client.
#[derive(Debug)]
pub struct Client(TcpStream);
impl Client {
    /// Creates new client.
    ///
    /// # Errors
    ///
    /// When failing to connection to server `TcpStream`.
    #[logfn(Trace)]
    pub fn new(address: impl std::net::ToSocketAddrs) -> Result<Self, std::io::Error> {
        Ok(Self(TcpStream::connect(address)?))
    }

    /// Re-connects to server stream.
    #[logfn(Trace)]
    fn reconnect(&mut self) -> Result<(), std::io::Error> {
        // Wait for server transfer to complete
        std::thread::sleep(Duration::from_millis(SERVER_TRANSFER_TIMEOUT));
        // TODO Can we use `TcpStream::connect_SERVER_TRANSFER_TIMEOUT` here so we don't always
        // need to wait full time?
        // Re-connect to stream
        self.0 = TcpStream::connect(self.0.local_addr()?)?;
        Ok(())
    }

    /// User `get` query.
    // Handling this would add generation complexity, without any performance benefit.
    #[allow(clippy::unit_arg)]
    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn get(&mut self, input: ()) -> CommunicationResult<GetReturn> {
        // Handling this would add generation complexity, without any performance benefit.
        #[allow(clippy::clone_on_copy)]
        self.raw_get(input.clone()).or_else(|_| {
            // Presume error as a result of server transfer
            self.reconnect()?;
            // Re-rerun query
            self.raw_get(input)
        })
    }

    /// User `set` query.
    // Handling this would add generation complexity, without any performance benefit.
    #[allow(clippy::unit_arg)]
    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn set(&mut self, input: u16) -> CommunicationResult<SetReturn> {
        // Handling this would add generation complexity, without any performance benefit.
        #[allow(clippy::clone_on_copy)]
        self.raw_set(input.clone()).or_else(|_| {
            // Presume error as a result of server transfer
            self.reconnect()?;
            // Re-rerun query
            self.raw_set(input)
        })
    }

    /// User `sum` query.
    // Handling this would add generation complexity, without any performance benefit.
    #[allow(clippy::unit_arg)]
    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn sum(&mut self, input: Vec<u16>) -> CommunicationResult<SumReturn> {
        // Handling this would add generation complexity, without any performance benefit.
        #[allow(clippy::clone_on_copy)]
        self.raw_sum(input.clone()).or_else(|_| {
            // Presume error as a result of server transfer
            self.reconnect()?;
            // Re-rerun query
            self.raw_sum(input)
        })
    }

    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn raw_get(&mut self, input: ()) -> CommunicationResult<GetReturn> {
        // Write
        write_frame(&mut self.0, 0, input)?;
        // Read
        let serialized = read_frame(&mut self.0)?;
        // Deserialize
        #[cfg(feature = "bincode")]
        let deserialized = bincode::deserialize(&serialized)?;
        #[cfg(feature = "json")]
        let deserialized = serde_json::from_slice(&serialized)?;
        Ok(deserialized)
    }

    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn raw_set(&mut self, input: u16) -> CommunicationResult<SetReturn> {
        // Write
        write_frame(&mut self.0, 1, input)?;
        // Read
        let serialized = read_frame(&mut self.0)?;
        // Deserialize
        #[cfg(feature = "bincode")]
        let deserialized = bincode::deserialize(&serialized)?;
        #[cfg(feature = "json")]
        let deserialized = serde_json::from_slice(&serialized)?;
        Ok(deserialized)
    }

    #[logfn(ok = "TRACE", err = "ERROR")]
    pub fn raw_sum(&mut self, input: Vec<u16>) -> CommunicationResult<SumReturn> {
        // Write
        write_frame(&mut self.0, 2, input)?;
        // Read
        let serialized = read_frame(&mut self.0)?;
        // Deserialize
        #[cfg(feature = "bincode")]
        let deserialized = bincode::deserialize(&serialized)?;
        #[cfg(feature = "json")]
        let deserialized = serde_json::from_slice(&serialized)?;
        Ok(deserialized)
    }
}

/// Error type for [`read_frame`].
#[derive(Debug, thiserror::Error)]
pub enum ReadFrameError {
    /// The frame contains an invalid length (`(1..4).contains(length)`).
    #[error("The frame contains an invalid length (`(1..4).contains(length)`).")]
    InvalidLength,
    /// Failed to read full frame from client as stream ended.
    #[error("Failed to read full frame from client as stream ended.")]
    EndOfStream,
    /// Failed to read length of the frame from the stream.
    #[error("Failed to read length of the frame from the stream: {0}")]
    ReadLength(std::io::Error),
    /// Failed to read all data of the frame from the stream.
    #[error("Failed to read all data of the frame from the stream: {0}")]
    ReadData(std::io::Error),
}
/// Reads frame from stream.
#[logfn(Trace)]
fn read_frame(stream: &mut TcpStream) -> Result<Vec<u8>, ReadFrameError> {
    // A frame consists of:
    // 1. A number defining the length of the data (including itself).
    // 2. The bytes (Vec<u8>).
    /// Size of length description of frame.
    const S: usize = size_of::<usize>();
    let mut bytes = vec![Default::default(); S];
    let mut i = 0;

    // != or < have same affect here
    while i != S {
        // SAFETY:
        // `i < L` thus `data_bytes[i]` is always valid, thus `data_bytes[i..]` is
        // always valid.
        let slice = unsafe { bytes.get_unchecked_mut(i..) };
        match stream.read(slice) {
            // End of stream
            Ok(0) => return Err(ReadFrameError::EndOfStream),
            Ok(size) => {
                // SAFETY:
                // `size` will never be greater than `L`, thus this is always safe.
                i = unsafe { i.unchecked_add(size) };
            }
            Err(err) => return Err(ReadFrameError::ReadLength(err)),
        }
    }
    // SAFETY:
    // Always safe,
    let length_arr = unsafe { bytes.get_unchecked(0..S).try_into().unwrap_unchecked() };
    let bytes_length = usize::from_ne_bytes(length_arr);
    trace!("bytes_length: {bytes_length}");

    // Read input data bytes
    let data_length = match bytes_length {
        0 => return Err(ReadFrameError::EndOfStream),
        1..S => return Err(ReadFrameError::InvalidLength),
        // SAFETY:
        // We know `input_length >= S` thus `input_length - S` is always valid.
        S.. => unsafe { bytes_length.unchecked_sub(S) },
    };
    trace!("data_length: {data_length}");

    // We size `data_bytes` to exactly `data_length` so we only read this frame, and not into the
    // next one.
    let mut data_bytes = vec![Default::default(); data_length];
    let mut j = 0;
    // Equivalent to `j < data_length`
    while j != data_length {
        // SAFETY:
        // `j` < `data_length` thus `data_bytes[j]` is always valid, thus `data_bytes[j..]` is
        // always valid.
        let slice = unsafe { data_bytes.get_unchecked_mut(j..) };
        match stream.read(slice) {
            Ok(0) => return Err(ReadFrameError::EndOfStream),
            Ok(size) => {
                // SAFETY:
                // `size` will never be greater than `data_length`, thus this is always safe.
                j = unsafe { j.unchecked_add(size) };
            }
            Err(err) => return Err(ReadFrameError::ReadData(err)),
        }
    }
    // Returns data bytes
    Ok(data_bytes)
}

/// Error type for [`write_frame`].
#[derive(Debug, thiserror::Error)]
pub enum WriteFrameError {
    /// Failed to serialize inputs.
    #[error("Failed to serialize input: {0}")]
    #[cfg(feature = "bincode")]
    Serialize(#[from] bincode::Error),
    #[cfg(feature = "json")]
    Serialize(#[from] serde_json::Error),
    /// Failed to write all data to the stream.
    #[error("Failed to write all data to the stream.")]
    Write(std::io::Error),
}

/// Writes frame to stream
#[logfn(Trace)]
fn write_frame(
    stream: &mut TcpStream,
    function_index: usize,
    input: impl Serialize,
) -> Result<(), WriteFrameError> {
    #[cfg(feature = "bincode")]
    let serialized = bincode::serialize(&input)?;
    #[cfg(feature = "json")]
    let serialized = serde_json::to_vec(&input)?;

    trace!("serialized: {:?}", &serialized);

    // SAFETY:
    // For `size_of::<usize>().checked_add(serialized.len()).is_err()` to be
    // true `serialized.len()` would need to be greater than `2^64 - 4`.
    // We can reasonably presume this will only occur in circumstance such as
    // memory corruption where it is impossible to guard against these errors.
    // As such it is reasonable and safe to do an unchecked addition here.
    let len = unsafe { size_of::<usize>().unchecked_add(serialized.len()) };
    let mut write_bytes = Vec::from(len.to_ne_bytes());
    write_bytes.extend(function_index.to_ne_bytes());
    write_bytes.extend(serialized);
    trace!("write_bytes: {:?}", &write_bytes);
    stream
        .write_all(&write_bytes)
        .map_err(WriteFrameError::Write)
}

#[allow(clippy::unwrap_used, clippy::dbg_macro, clippy::unreachable)]
#[cfg(test)]
mod tests {
    use std::time::Instant;

    use indicatif::ProgressBar;
    use rand::Rng;

    use super::*;

    /// Sever address.
    const ADDRESS: &str = "127.0.0.1:8080";

    static INIT_LOGGER: std::sync::Once = std::sync::Once::new();
    fn init_logger() {
        INIT_LOGGER.call_once(|| simple_logger::init_with_level(log::Level::Trace).unwrap());
    }

    #[test]
    fn test_get() {
        init_logger();
        let mut client = Client::new(ADDRESS).unwrap();
        let result = client.get(());
        dbg!(&result);
        assert!(matches!(result, Ok(Ok(12))));
        // std::thread::sleep(Duration::from_secs(3));
    }
    #[test]
    fn main() {
        const REQUESTS_RANGE: std::ops::Range<u64> = 5_000_000..10_000_000;
        const CLIENTS: usize = 6;

        init_logger();

        let now = Instant::now();
        dbg!("started");

        let multi_bar = indicatif::MultiProgress::new();
        let mut rng = rand::thread_rng();

        let handles = (0..CLIENTS)
            .map(|_| {
                let requests = rng.gen_range(REQUESTS_RANGE);
                let bar = multi_bar.add(indicatif::ProgressBar::new(requests));
                std::thread::spawn(move || stress_test(requests, &bar))
            })
            .collect::<Vec<_>>();
        // Join all threads
        for handle in handles {
            handle.join().unwrap();
        }
        multi_bar.clear().unwrap();
        dbg!(now.elapsed());
    }
    fn stress_test(requests: u64, bar: &ProgressBar) {
        let mut client = Client::new(ADDRESS).unwrap();
        let mut rng = rand::thread_rng();
        for _ in 0..requests {
            // Picks random query
            let x: usize = rng.gen_range(0..3);
            dbg!(x);

            let function_result = match x {
                0 => client.get(()).is_ok(),
                1 => client.set(2).is_ok(),
                2 => client.sum(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).is_ok(),
                _ => unreachable!(),
            };
            assert!(function_result);

            #[cfg(debug_assertions)]
            std::thread::sleep(Duration::from_secs(1));

            bar.inc(1);
        }
        bar.finish();
    }
}
