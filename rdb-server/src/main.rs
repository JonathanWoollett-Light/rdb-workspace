#![feature(duration_constants)]
#![warn(clippy::pedantic)]
use core::time;
use std::io::{ErrorKind, Read, Write};
use std::mem::{size_of, MaybeUninit};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::UnixDatagram;
use std::ptr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use log::{info, trace};
use serde::{Deserialize, Serialize};
use shared_memory_allocator::SharedAllocator;
const OLD_PROCESS: &str = "./old";
const NEW_PROCESS: &str = "./new";
const ADDRESS: &str = "127.0.0.1:8080";

type Store = RwLock<Data>;
#[derive(Serialize, Deserialize, Debug)]
#[repr(C)]
struct Data {
    pub x: u16,
}
impl Data {
    fn get(&self) -> u16 {
        self.x
    }

    fn set(&mut self, x: u16) {
        self.x = x;
    }
}

fn function_map(f: usize, input: Vec<u8>) -> Vec<u8> {
    let store = unsafe { &**DATA.assume_init_ref() };
    match f {
        0 => bincode::serialize(&store.read().unwrap().get()),
        1 => bincode::serialize(
            &store
                .write()
                .unwrap()
                .set(bincode::deserialize(&input).unwrap()),
        ),
        _ => unreachable!(),
    }
    .unwrap()
}
enum FrameErr {
    Timeout,
    EndOfStream,
}

/// We read a frame from the stream, returning the function number and serialized input.
///
/// A client could potentially indefinitely block a process transfer if we do not allow short
/// circuiting `read_frame`. Since we do not want to drop client frames, but we do not want to block
/// the transfer process, we have a timeout to `read_frame`.
fn read_frame(stream: &mut TcpStream, timeout: Duration) -> Result<(usize, Vec<u8>), FrameErr> {
    // A frame consists of:
    // 1. A number defining the length of the input bytes (usize).
    // 2. A function number (usize).
    // 3. The input bytes (Vec<u8>).
    const S: usize = size_of::<usize>();
    const L: usize = 2 * S;
    let mut bytes = vec![Default::default(); L];
    let mut i = 0;

    // Presume stream has been previously set as non-blocking
    let start = Instant::now();
    // While we haven't received the number defining the length of the input bytes.
    while i < L {
        match stream.read(&mut bytes[i..]) {
            Ok(size) => {
                i += size;
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {}
            _ => unreachable!(),
        }
        if start.elapsed() > timeout {
            return Err(FrameErr::Timeout);
        }
    }
    let input_length = usize::from_ne_bytes(bytes[0..S].try_into().unwrap());
    if input_length == 0 {
        return Err(FrameErr::EndOfStream);
    }

    // Read input data bytes
    let data_length = input_length - S;
    let mut data_bytes = vec![Default::default(); data_length];
    let mut j = 0;
    while j != data_length {
        match stream.read(&mut data_bytes[j..]) {
            Ok(size) => {
                j += size;
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {}
            _ => unreachable!(),
        }
        if start.elapsed() > timeout {
            return Err(FrameErr::Timeout);
        }
    }
    // Returns function index and input data bytes
    let function_index = usize::from_ne_bytes(bytes[S..L].try_into().unwrap());
    Ok((function_index, data_bytes))
}

// Returns true when server received shutdown signal while processing client stream
fn handle_stream(mut stream: TcpStream, exit: &Arc<Mutex<bool>>) {
    // TODO Check `server_socket` is blocking
    stream.set_nonblocking(true).unwrap();
    loop {
        // We read a u32 from the stream, multiply it by 2, then write it back.
        match read_frame(&mut stream, Duration::SECOND) {
            // Next value
            Ok((function_index, input)) => {
                let return_bytes = {
                    let output_bytes = function_map(function_index, input);
                    trace!("output_bytes: {:?}", output_bytes);
                    let length_bytes = output_bytes.len().to_ne_bytes();
                    [Vec::from(length_bytes), output_bytes].concat()
                };

                let written = stream.write(&return_bytes).unwrap();

                trace!("written: {:?}", written);
            }
            // End of stream
            Err(FrameErr::EndOfStream) => {
                return;
            }
            Err(FrameErr::Timeout) => {
                unimplemented!()
            }
        }
        // Exit signal set true
        if *exit.lock().unwrap() {
            return;
        }
    }
}
fn process(now: Instant) {
    // Exit signal
    let exit = Arc::new(Mutex::new(false));
    // Thread handles
    let mut handles = Vec::new();
    // Once we receive something (indicating a new process has started) we then transfer data.
    let socket = UnixDatagram::bind(OLD_PROCESS).unwrap();
    // Shutdown signal listener
    let listener = TcpListener::bind(ADDRESS).unwrap();

    // Sets streams to non-blocking
    listener.set_nonblocking(true).unwrap();
    socket.set_nonblocking(true).unwrap();

    info!("time to ready: {:?}", now.elapsed());
    info!("ready to receive connections");

    // Work loop
    loop {
        // Accept incoming streams
        match listener.accept() {
            Ok((stream, client_socket)) => {
                info!("client_socket: {:?}", client_socket);

                let exit_clone = exit.clone();
                let handle = std::thread::spawn(move || handle_stream(stream, &exit_clone));
                handles.push(handle);
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {}
            _ => unreachable!(),
        }
        // Check for shutdown signal
        match socket.recv(&mut []) {
            // Received shutdown signal
            Ok(_) => {
                // We set exit signal to true
                *exit.lock().unwrap() = true;
                break;
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {}
            _ => unreachable!(),
        }
    }
    // We await all threads finishing.
    let now = Instant::now();
    info!("awaiting threads");
    for handle in handles {
        handle.join().unwrap();
    }
    info!("awaited threads: {:?}", now.elapsed());
    // Set our unix socket to blocking
    socket.set_nonblocking(false).unwrap();
    // Send the data to our unix socket (and new server process)
    let addr = unsafe { *DATA.assume_init_ref() } as usize;
    let offset = addr - unsafe { ALLOCATOR.assume_init_ref().address() };
    info!("sent offset: {:?}", offset);
    let buf = offset.to_ne_bytes();
    socket.send_to(&buf, NEW_PROCESS).unwrap();
    info!("sent data");

    // Remove old process unix datagram file descriptor
    std::fs::remove_file(OLD_PROCESS).unwrap();
}

const KB: usize = 1024;
const MB: usize = 1024 * KB;
const GB: usize = 1024 * MB;

static mut ALLOCATOR: MaybeUninit<SharedAllocator> = MaybeUninit::uninit();
static mut DATA: MaybeUninit<*mut Store> = MaybeUninit::uninit();

fn main() {
    let now = Instant::now();
    simple_logger::init_with_level(log::Level::Info).unwrap();
    info!("started");

    //  If an old process exists
    if std::path::Path::new(OLD_PROCESS).exists() {
        info!("non-first");

        // We send the shutdown notification then await shared memory address of the data
        let socket = UnixDatagram::bind(NEW_PROCESS).unwrap();

        // Send shutdown signal to old process
        socket.send_to(&[], OLD_PROCESS).unwrap();

        // Await data offset (which also serves as signal old process has terminated)
        let mut buf = [0; size_of::<usize>()];
        socket.recv(&mut buf).unwrap();
        let offset = usize::from_ne_bytes(buf);
        info!("received offset: {:?}", offset as *mut u8);

        // Gets data
        let allocator = unsafe { SharedAllocator::new_process("/tmp/rdb-server") };

        info!("allocator.address(): {}", allocator.address());
        let ptr = (allocator.address() + offset) as *mut u8;
        let data_ptr = ptr.cast::<Store>();
        info!("data_ptr: {:?}", data_ptr);
        info!("data: {:?}", unsafe { &*data_ptr });

        // Store allocator in static (if this allocator is dropped the shared memory is dropped)
        unsafe {
            ALLOCATOR.write(allocator);
            DATA.write(data_ptr);
        }

        // Clear `NEW_PROCESS` unix datagram file descriptor
        std::fs::remove_file(NEW_PROCESS).unwrap();

        process(now);
    } else {
        info!("first");
        let allocator = unsafe { SharedAllocator::new_memory("/tmp/rdb-server", GB) };
        let ptr = allocator.address() as *mut u8;
        let data_ptr = ptr.cast::<Store>();
        info!("data_ptr: {:?}", data_ptr);
        unsafe {
            ptr::write(data_ptr, RwLock::new(Data { x: 12 }));
        }
        // Store allocator in static (if this allocator is dropped the shared memory is dropped)
        unsafe {
            ALLOCATOR.write(allocator);
            DATA.write(data_ptr);
        }
        process(now);
    }
}
