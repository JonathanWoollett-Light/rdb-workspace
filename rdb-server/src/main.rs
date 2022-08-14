#![feature(duration_constants)]
#![warn(clippy::pedantic)]
use std::io::{ErrorKind, Read, Write};
use std::mem::{size_of, MaybeUninit};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::UnixDatagram;
use std::ptr;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

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

// type QueryFn = fn(Arc<RwLock<Database>>, Vec<u8>) -> Vec<u8>;
// static FUNCTIONS: [QueryFn; 1] = [|data: Arc<RwLock<Database>>, buf: Vec<u8>| {
//     let y: u16 = bincode::deserialize(&buf).unwrap();
//     let z = data.read().unwrap().x * y;
//     bincode::serialize(&z).unwrap()
// }];

// Returns true when server received shutdown signal while processing client stream
fn handle_stream(mut stream: TcpStream, exit: &Arc<Mutex<bool>>) {
    // TODO Check `server_socket` is blocking
    stream.set_nonblocking(true).unwrap();
    let mut buf = [0; size_of::<u32>()];
    loop {
        // We read a u32 from the stream, multiply it by 2, then write it back.
        match stream.read(&mut buf) {
            // Next value
            Ok(size) if size == 4 => {
                let x = u32::from_ne_bytes(buf);

                trace!("x: {:?}", x);

                let y = x.overflowing_mul(2).0;

                trace!("y: {:?}", y);

                let data = unsafe { &**DATA.assume_init_ref() };
                let guard = data.read().unwrap();
                let z = y + u32::from(guard.x);
                trace!("z: {}", z);

                let rtn_buf = z.to_ne_bytes();
                let written = stream.write(&rtn_buf).unwrap();

                trace!("written: {:?}", written);
            }
            // End of stream
            Ok(size) if size == 0 => {
                return;
            }
            // Timeout
            Err(err) if err.kind() == ErrorKind::WouldBlock => {}
            // Other values are errors
            _ => unreachable!(),
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
