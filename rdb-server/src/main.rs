#![feature(duration_constants)]
#![feature(anonymous_lifetime_in_impl_trait)]
#![feature(ptr_to_from_bits)]
#![feature(unchecked_math)]
#![feature(exclusive_range_pattern)]
#![feature(precise_pointer_size_matching)]
#![feature(const_inherent_unchecked_arith)]
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

//! A binary.

use std::any::Any;
use std::io::{ErrorKind, Read, Write};
use std::mem::{size_of, MaybeUninit};
use std::net::{TcpListener, TcpStream};
use std::os::unix::net::UnixDatagram;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use clap::Parser;
use log::{info, trace, SetLoggerError};
use log_derive::logfn;
use serde::{Deserialize, Serialize};
use shared_memory_allocator::{
    SharedAllocator, SharedAllocatorNewMemoryError, SharedAllocatorNewProcessError,
};

/// Signal used to communicate across threads for shutdown.
static EXIT_SIGNAL: AtomicBool = AtomicBool::new(false);

/// Default address used for the TCP socket.
const DEFAULT_ADDRESS: &str = "127.0.0.1:8282";
/// Default path used for the old process unix socket.
const DEFAULT_OLD_PROCESS_SOCKET: &str = "/tmp/old_process_socket";
/// Default path used for the new process unix socket.
const DEFAULT_NEW_PROCESS_SOCKET: &str = "/tmp/new_process_socket";
/// Default path used for the file containing the shared memory id.
const DEFAULT_SHARED_MEMORY_KEY: i32 = 0x6F89B;

/// Server command line arguments.
#[derive(Debug, Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Address for client connections.
    #[clap(short, long, value_parser, default_value = DEFAULT_ADDRESS)]
    address: String,
    /// Unix datagram for process transfer.
    #[clap(short, long, value_parser, default_value = DEFAULT_OLD_PROCESS_SOCKET)]
    old_process_socket: String,
    /// Unix datagram for process transfer.
    #[clap(short, long, value_parser, default_value = DEFAULT_NEW_PROCESS_SOCKET)]
    new_process_socket: String,
    /// Log level
    #[clap(short,long,value_parser,default_value_t = log::Level::Trace)]
    log_level: log::Level,
    /// File used to synchronize the shared memory id.
    #[clap(short, long, value_parser, default_value_t = DEFAULT_SHARED_MEMORY_KEY)]
    shared_memory_key: i32,
}

/// The data type storing `OldData`.
type OldStore = RwLock<OldData>;
/// The data in our database of the predecessor process.
#[derive(Debug, Serialize, Deserialize)]
#[repr(C)]
struct OldData {
    /// `x`
    pub x: u16,
}

impl From<OldData> for Data {
    fn from(OldData { x }: OldData) -> Data {
        Data { x }
    }
}
/// The data type storing `Data`.
type Store = RwLock<Data>;
/// The data in our database.
#[derive(Debug, Serialize, Deserialize)]
#[repr(C)]
struct Data {
    /// `x`
    pub x: u16,
}
#[allow(clippy::unnecessary_wraps)]
impl Data {
    /// Gets `x` in our database.
    #[logfn(Trace)]
    fn get(&self, _: ()) -> Result<u16, ()> {
        Ok(self.x)
    }

    /// Sets `x` in our database.
    #[logfn(Trace)]
    fn set(&mut self, x: u16) -> Result<(), ()> {
        self.x = x;
        Ok(())
    }

    /// Sums `set` then multiplies it by `x` in our database.
    #[logfn(Trace)]
    fn sum(&self, set: Vec<u16>) -> Result<u16, ()> {
        set.into_iter().sum::<u16>().checked_mul(self.x).ok_or(())
    }
}

/// Error type for [`function_map`].
#[derive(Debug, thiserror::Error)]
enum FunctionMapError {
    /// Failed to deserialize inputs.
    #[error("Failed to deserialize input: {0}")]
    #[cfg(feature = "bincode")]
    Deserialize(#[from] bincode::Error),
    #[cfg(feature = "json")]
    Deserialize(#[from] serde_json::Error),

    /// Failed to acquire write lock on database data.
    #[error("Failed to acquire write lock on database data.")]
    WriteLock,
    /// Failed to acquire read lock on database data.
    #[error("Failed to acquire read lock on database data.")]
    ReadLock,
    /// Failed to write frame.
    #[error("Failed to write frame: {0}")]
    WriteFrame(#[from] WriteFrameError),
    /// Invalid function index.
    #[error("Invalid function index.")]
    InvalidFunctionIndex,
}

/// Given function index `function_index` and input `input`
#[logfn(Trace)]
fn function_map(
    function_index: usize,
    serialized_input: &[u8],
    stream: &mut TcpStream,
) -> Result<(), FunctionMapError> {
    trace!("serialized_input: {:?}", serialized_input);
    // SAFETY:
    // We know `DATA` will be initialized at this time.
    let store = unsafe { &**DATA.assume_init_ref() };
    // - Where input is `()` we can avoid this step, however this increases code generation
    //   complexity and offers no speedup as we would expect this to be optimized out. TODO Double
    //   check this is optimized out.
    // - We cannot return `PoisonError`, thus we must discard information.
    #[allow(clippy::let_unit_value, clippy::map_err_ignore)]
    match function_index {
        0 => {
            #[cfg(feature = "bincode")]
            let input = bincode::deserialize(serialized_input)?;
            #[cfg(feature = "json")]
            let input = serde_json::from_slice(serialized_input)?;

            let output = store
                .read()
                .map_err(|_| FunctionMapError::ReadLock)?
                .get(input);
            write_frame(stream, output)?;
        }
        1 => {
            #[cfg(feature = "bincode")]
            let input = bincode::deserialize(serialized_input)?;
            #[cfg(feature = "json")]
            let input = serde_json::from_slice(serialized_input)?;

            trace!("input: {:?}", input);

            let output = store
                .write()
                .map_err(|_| FunctionMapError::WriteLock)?
                .set(input);
            write_frame(stream, output)?;
        }
        2 => {
            #[cfg(feature = "bincode")]
            let input = bincode::deserialize(serialized_input)?;
            #[cfg(feature = "json")]
            let input = serde_json::from_slice(serialized_input)?;

            trace!("input: {:?}", input);
            let output = store
                .read()
                .map_err(|_| FunctionMapError::ReadLock)?
                .sum(input);
            write_frame(stream, output)?;
        }
        _ => return Err(FunctionMapError::InvalidFunctionIndex),
    }
    Ok(())
}

/// Error type for [`write_frame`].
#[derive(Debug, thiserror::Error)]
enum WriteFrameError {
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
fn write_frame(stream: &mut TcpStream, input: impl Serialize) -> Result<(), WriteFrameError> {
    #[cfg(feature = "bincode")]
    let serialized = bincode::serialize(&input)?;
    #[cfg(feature = "json")]
    let serialized = serde_json::to_vec(&input)?;

    // SAFETY:
    // For `size_of::<usize>().checked_add(serialized.len()).is_err()` to be
    // true `serialized.len()` would need to be greater than `2^64 - 4`.
    // We can reasonably presume this will only occur in circumstance such as
    // memory corruption where it is impossible to guard against these errors.
    // As such it is reasonable and safe to do an unchecked addition here.
    let len = unsafe { size_of::<usize>().unchecked_add(serialized.len()) };
    let mut write_bytes = Vec::from(len.to_ne_bytes());
    write_bytes.extend(serialized);
    stream
        .write_all(&write_bytes)
        .map_err(WriteFrameError::Write)
}
/// Error type for [`read_frame`].
#[derive(Debug, thiserror::Error)]
enum ReadFrameError {
    /// Failed to read full frame from client due to blocking first read. This
    /// like [`std::io::ErrorKind::WouldBlock`] does not indicate an error in the typical sense.
    #[error("Failed to read full frame from client due to blocking first read.")]
    WouldBlock,
    /// Failed to read full frame from client within `timeout` `Duration`.
    #[error("Failed to read full frame from client within `timeout` duration.")]
    Timeout,
    /// Failed to read full frame from client as stream ended.
    #[error("Failed to read full frame from client as stream ended.")]
    EndOfStream,
    /// Failed to read length of the frame from the stream.
    #[error("Failed to read length of the frame from the stream: {0}")]
    ReadLength(std::io::Error),
    /// Failed to read all data of the frame from the stream.
    #[error("Failed to read all data of the frame from the stream: {0}")]
    ReadData(std::io::Error),
    /// The frame contains an invalid length (`(1..4).contains(length)`).
    #[error("The frame contains an invalid length (`(1..4).contains(length)`).")]
    InvalidLength,
}

/// We read a frame from the stream, returning the function number and serialized input.
///
/// A client could potentially indefinitely block a process transfer if we do not allow short
/// circuiting `read_frame`. Since we do not want to drop client frames, but we do not want to block
/// the transfer process, we have a timeout to `read_frame`.
///
/// Presumes stream has been previously set as non-blocking
// We do not trace `read_frame` since it is non-blocking and may return thousands of
// `ReadFrameError::WouldBlock` in a loop.
#[logfn(Trace)]
fn read_frame(
    stream: &mut TcpStream,
    timeout: Duration,
) -> Result<(usize, Vec<u8>), ReadFrameError> {
    // A frame consists of:
    // 1. A number defining the full length of the data received (size of function number plus
    // length input bytes). 2. A function number (usize).
    // 3. The input bytes (Vec<u8>).

    /// Size of length description of frame.
    const S: usize = size_of::<usize>();
    /// Size of length description of frame and function index.
    #[allow(clippy::undocumented_unsafe_blocks)]
    const L: usize = unsafe { 2usize.unchecked_mul(S) };

    let mut bytes = vec![Default::default(); L];
    let mut i = 0;

    let start = Instant::now();

    // While we haven't received the number defining the length of the input bytes and the function
    // index.
    // Equivalent to `i < L`
    while i != L {
        // SAFETY:
        // `i < L` thus `data_bytes[i]` is always valid, thus `data_bytes[i..]` is
        // always valid.
        let slice = unsafe { bytes.get_unchecked_mut(i..) };
        match stream.read(slice) {
            Ok(0) => return Err(ReadFrameError::EndOfStream),
            Ok(size) => {
                // SAFETY:
                // `size` will never be greater than `L`, thus this is always safe.
                i = unsafe { i.unchecked_add(size) };
            }
            // On the first read if it would block we return the result that this function would
            // block. After the first read we have committed to reading a frame
            // thereafter a timeout error can
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                if i == 0 {
                    return Err(ReadFrameError::WouldBlock);
                }
            }
            Err(err) => return Err(ReadFrameError::ReadLength(err)),
        }
        if start.elapsed() > timeout {
            return Err(ReadFrameError::Timeout);
        }
    }
    // SAFETY:
    // Always safe,
    let length_arr = unsafe { bytes.get_unchecked(0..S).try_into().unwrap_unchecked() };
    let input_length = usize::from_ne_bytes(length_arr);
    trace!("input_length: {input_length}");

    // Read input data bytes
    let data_length = match input_length {
        0..S => return Err(ReadFrameError::InvalidLength),
        // SAFETY:
        // We know `input_length >= S` thus `input_length - S` is always valid.
        S.. => unsafe { input_length.unchecked_sub(S) },
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
            Ok(size) => {
                // SAFETY:
                // `size` will never be greater than `data_length`, thus this is always safe.
                j = unsafe { j.unchecked_add(size) };
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {}
            Err(err) => return Err(ReadFrameError::ReadData(err)),
        }
        if start.elapsed() > timeout {
            return Err(ReadFrameError::Timeout);
        }
    }

    // SAFETY:
    // Always safe,
    let function_index_arr = unsafe { bytes.get_unchecked(S..L).try_into().unwrap_unchecked() };
    // Returns function index and input data bytes
    let function_index = usize::from_ne_bytes(function_index_arr);
    trace!("function_index: {function_index}");

    Ok((function_index, data_bytes))
}

/// Error type for [`handle_stream`].
#[derive(Debug, thiserror::Error)]
enum HandleStreamError {
    /// Failed to set client [`TcpStream`] to non-blocking.
    #[error("Failed to set client [`TcpStream`] to non-blocking: {0}")]
    SetNonBlocking(std::io::Error),
    /// Failed to read frame from client stream.
    #[error("Failed to read frame from client stream: {0}")]
    FrameRead(ReadFrameError),
}

/// Processes a client stream.
///
/// Returns when:
/// - Client stream is shutdown.
/// - Server set shutdown notification.
#[logfn(Trace)]
fn handle_stream(mut stream: TcpStream) -> Result<(), HandleStreamError> {
    // TODO Check `server_socket` is blocking
    stream
        .set_nonblocking(true)
        .map_err(HandleStreamError::SetNonBlocking)?;
    loop {
        // We read a u32 from the stream, multiply it by 2, then write it back.
        match read_frame(&mut stream, Duration::SECOND) {
            // Next value
            Ok((function_index, input)) => {
                // TODO: We presume an error here is a result of client
                // disconnection. This is not entirely logical, we need to fix
                // this.
                if let Err(err) = function_map(function_index, &input, &mut stream) {
                    info!("client disconnected: {err:?}");
                    return Ok(());
                }
            }
            // End of stream
            Err(ReadFrameError::EndOfStream) => {
                info!("client disconnected: {:?}", ReadFrameError::EndOfStream);
                return Ok(());
            }
            Err(ReadFrameError::WouldBlock) => (),
            Err(err) => return Err(HandleStreamError::FrameRead(err)),
        }
        // When exit signal set true
        if EXIT_SIGNAL.load(Ordering::SeqCst) {
            return Ok(());
        }
    }
}

/// Error type for [`process`].
#[derive(Debug, thiserror::Error)]
enum ProcessError {
    /// Failed to bind the transition unix datagram socket.
    #[error("Failed to bind the transition unix datagram socket: {0}")]
    SocketBind(std::io::Error),
    /// Failed to bind the client connection `TcpListener`.
    #[error("Failed to bind the client connection `TcpListener`: {0}")]
    ListenerBind(std::io::Error),
    /// Failed to set client listener to non-blocking.
    #[error("Failed to set client listener to non-blocking: {0}")]
    ListenerBlocking(std::io::Error),
    /// Failed to set transition socket to non-blocking before handling clients.
    #[error("Failed to set transition socket to non-blocking before handling clients: {0}")]
    SocketBlockingBefore(std::io::Error),
    /// Failed to read client connection.
    #[error("Failed to read client connection: {0}")]
    ClientConnection(std::io::Error),
    /// Failed to read shutdown notification.
    #[error("Failed to read shutdown notification: {0}")]
    ShutdownNotification(std::io::Error),
    /// Failed to join client thread.
    #[error("Failed to join client thread: {0:?}")]
    JoinThread(Box<dyn Any + Send + 'static>),
    /// Failed to set the transition socket to blocking after joining all client threads.
    #[error("Failed to set the transition socket to blocking after joining all client threads.")]
    SocketBlockingAfter(std::io::Error),
    /// Offset calculate from subtracting the beginning of the attached shared memory to the
    /// beginning of the allocated data is invalid (less than 0).
    #[error(
        "Offset calculate from subtracting the beginning of the attached shared memory to the \
         beginning of the allocated data is invalid (less than 0)."
    )]
    Offset,
    /// Failed to send data offset to successor process.
    #[error("Failed to send data offset to successor process")]
    Send(std::io::Error),
    /// Failed to remove predecessor process unix datagram
    #[error("Failed to remove old process unix datagram: {0}")]
    RemoveOldProcessDatagram(std::io::Error),
    /// Error while handling client connection.
    #[error("Error while handling client connection: {0}")]
    Client(#[from] HandleStreamError),
}

/// Runs server process
#[logfn(Trace)]
fn process(
    process_start: Instant,
    old_process_socket: &str,
    new_process_socket: &str,
    address: &str,
) -> Result<(), ProcessError> {
    // Thread handles
    let mut handles = Vec::new();
    // Once we receive something (indicating a new process has started) we then transfer data.
    let socket = UnixDatagram::bind(old_process_socket).map_err(ProcessError::SocketBind)?;
    // Client stream listener
    let listener = TcpListener::bind(address).map_err(ProcessError::ListenerBind)?;

    // Sets streams to non-blocking
    listener
        .set_nonblocking(true)
        .map_err(ProcessError::ListenerBlocking)?;
    socket
        .set_nonblocking(true)
        .map_err(ProcessError::SocketBlockingBefore)?;

    info!("time to ready: {:?}", process_start.elapsed());
    info!("ready to receive connections");

    // Work loop
    let transfer = loop {
        // Accept incoming streams
        match listener.accept() {
            Ok((stream, client_socket)) => {
                info!("client_socket: {:?}", client_socket);
                let handle = std::thread::spawn(move || handle_stream(stream));
                handles.push(handle);
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {}
            Err(err) => return Err(ProcessError::ClientConnection(err)),
        }
        // Check for shutdown notification from successor process
        match socket.recv(&mut []) {
            // Received shutdown signal
            Ok(_) => {
                // We set exit signal to true
                EXIT_SIGNAL.store(true, std::sync::atomic::Ordering::SeqCst);
                info!("transfer");
                // We break, noting we do need to transfer data to a successor process.
                break true;
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {}
            Err(err) => return Err(ProcessError::ShutdownNotification(err)),
        }
        // Exit signal can also be set when we catch `SIGINT` (e.g. ctrl-c).
        if EXIT_SIGNAL.load(Ordering::SeqCst) {
            info!("interrupt");
            // We break, noting we do not need to transfer data to a successor process.
            break false;
        }
    };

    // We drop the incoming client stream listener.
    // Importantly this needs to occur before we send the data offset to the successor process and
    // it tries to construct a listener on the same socket. If this hadn't been dropped it would
    // return `Address already in use`.
    drop(listener);

    // We await all threads finishing.
    let now = Instant::now();
    info!("awaiting threads");
    for handle in handles {
        handle.join().map_err(ProcessError::JoinThread)??;
    }
    info!("awaited threads: {:?}", now.elapsed());

    info!("transfer: {}", transfer);

    // If we are transferring to a successor process.
    if transfer {
        // Set our unix socket to blocking
        socket
            .set_nonblocking(false)
            .map_err(ProcessError::SocketBlockingAfter)?;

        // Send the data offset to the unix socket (for the successor server process)
        // Get data offset
        // SAFETY:
        // We know both `DATA` and `ALLOCATOR` have been initialized, are not being currently
        // accessed, and will not be accessed after.
        let data_offset = unsafe {
            let addr = DATA.assume_init_ref().to_bits();
            addr.checked_sub(ALLOCATOR.assume_init_ref().address())
                .ok_or(ProcessError::Offset)?
        };
        info!("data_offset: {data_offset}");
        // Send data offset
        socket
            .send_to(&data_offset.to_ne_bytes(), new_process_socket)
            .map_err(ProcessError::Send)?;
        info!("sent data");
    }

    // Clear `OLD_PROCESS` unix datagram file descriptor.
    // When the transition is completed from this to its successor process, this successor process
    // must be able to create and use this datagram for when it must transition.
    // Or on `SIGINT` we want to clean up and gracefully exit.
    std::fs::remove_file(old_process_socket).map_err(ProcessError::RemoveOldProcessDatagram)?;
    // Drops the shared memory allocator
    // SAFETY:
    // `ALLOCATOR` will always be initialized before this. So this is safe.
    unsafe {
        ALLOCATOR.assume_init_drop();
    }
    Ok(())
}

/// 1kb
const KB: usize = 1024;
/// 1mb
#[allow(clippy::undocumented_unsafe_blocks)]
const MB: usize = unsafe { 1024usize.unchecked_mul(KB) };
/// 1gb
#[allow(clippy::undocumented_unsafe_blocks)]
const GB: usize = unsafe { 1024usize.unchecked_mul(MB) };

/// Data allocator
static mut ALLOCATOR: MaybeUninit<SharedAllocator> = MaybeUninit::uninit();
/// Database data
static mut DATA: MaybeUninit<*mut Store> = MaybeUninit::uninit();

/// Error type for [`main`].
///
/// This handles case where we would otherwise need to call `unwrap` (which we disallow).
#[derive(Debug, thiserror::Error)]
enum MainError {
    /// Failed to set Ctrl-C handler.
    #[error("Failed to set Ctrl-C handler: {0}")]
    CtrlC(#[from] ctrlc::Error),
    /// Failed to initialize logger.
    #[error("Failed to initialize logger: {0}")]
    Logger(#[from] SetLoggerError),
    /// Failed to create socket to send shutdown notification
    #[error("Failed to create socket to send shutdown notification: {0}")]
    Socket(std::io::Error),
    /// Failed to send shutdown notification.
    #[error("Failed to send shutdown notification: {0}")]
    Send(std::io::Error),
    /// Failed to receive data address and previous process shutdown notification
    #[error("Failed to receive data address and previous process shutdown notification: {0}")]
    Receive(std::io::Error),
    /// Failed to remove new process unix datagram
    #[error("Failed to remove new process unix datagram: {0}")]
    RemoveNewProcessDatagram(std::io::Error),
    /// The offset of the data in the shared memory (as given by the old process) is invalid (would
    /// exceed `usize::MAX`).
    #[error(
        "The offset of the data in the shared memory (as given by the old process) is invalid \
         (would exceed `usize::MAX`)."
    )]
    BadAddressOffset,
    /// Error occurred in running process.
    #[error("Error occurred in running process: {0}")]
    Process(#[from] ProcessError),
    /// Failed to deconstruct the old data `RwLock`.
    #[error("Failed to deconstruct the old data `RwLock`.")]
    OldDataIntoInner,
    /// Failed to create shared memory allocator on new memory.
    #[error("Failed to create shared memory allocator on new memory: {0}")]
    NewMemoryAllocator(SharedAllocatorNewMemoryError),
    /// Failed to create shared memory allocator on new process.
    #[error("Failed to create shared memory allocator on new process: {0}")]
    NewProcessAllocator(SharedAllocatorNewProcessError),
}

fn main() -> Result<(), MainError> {
    ctrlc::set_handler(|| EXIT_SIGNAL.store(true, Ordering::SeqCst))?;

    let args = Args::parse();

    let now = Instant::now();
    simple_logger::init_with_level(args.log_level)?;
    info!("started");

    //  If an old process exists
    if std::path::Path::new(&args.old_process_socket).exists() {
        info!("non-first");

        // We construct the shared memory allocator before sending the shutdown signal. We must do
        // it in this order to prevent the shared allocator in the predecessor process being dropped
        // before the shared allocator in this process is constructed, if this occurs the shared
        // memory would be deallocated.
        let allocator = SharedAllocator::new_process(args.shared_memory_key)
            .map_err(MainError::NewProcessAllocator)?;

        // We create the socket on which to send the shutdown notification
        let socket = UnixDatagram::bind(&args.new_process_socket).map_err(MainError::Socket)?;

        // Send shutdown notification to old process
        socket
            .send_to(&[], &args.old_process_socket)
            .map_err(MainError::Send)?;

        // Await data offset (which also serves as notification old process has terminated)
        let mut buf = [0; size_of::<usize>()];
        socket.recv(&mut buf).map_err(MainError::Receive)?;
        let offset = usize::from_ne_bytes(buf);
        info!("received offset: {offset}");

        // Gets data
        info!("allocator.address(): {}", allocator.address());
        let ptr = <*mut u8>::from_bits(
            allocator
                .address()
                .checked_add(offset)
                .ok_or(MainError::BadAddressOffset)?,
        );
        #[allow(clippy::cast_ptr_alignment)]
        let old_data_ptr = ptr.cast::<OldStore>();
        info!("old_data_ptr: {old_data_ptr:?}");
        // SAFETY:
        // We know the pointer to be valid.
        unsafe {
            info!("old_data_ptr: {:?}", &*old_data_ptr);
        }

        // Sets data
        // SAFETY:
        // We know the pointer to be valid.
        let old_data = unsafe { std::ptr::read(old_data_ptr) };
        info!("old_data: {old_data:?}");
        // `Store::from(OldStore)`
        let new_data = {
            // We cannot return `PoisonError`, thus we must discard information.
            #[allow(clippy::map_err_ignore)]
            let unlocked_old_data = old_data
                .into_inner()
                .map_err(|_| MainError::OldDataIntoInner)?;
            let unlocked_new_data = Data::from(unlocked_old_data);
            Store::new(unlocked_new_data)
        };
        info!("new_data: {new_data:?}");
        let new_data_ptr = old_data_ptr.cast::<Store>();
        // SAFETY:
        // We know the pointer to be valid.
        unsafe {
            std::ptr::write(new_data_ptr, new_data);
        }

        // Store allocator in static (if this allocator is dropped the shared memory is dropped)
        // SAFETY:
        // It is always safe, as no other thread will be accessing this stack at this time.
        unsafe {
            ALLOCATOR.write(allocator);
            DATA.write(new_data_ptr);
        }

        // Clear `NEW_PROCESS` unix datagram file descriptor. When successor process to this one is
        // started it must be able to create and use this stream.
        std::fs::remove_file(&args.new_process_socket)
            .map_err(MainError::RemoveNewProcessDatagram)?;

        process(
            now,
            &args.old_process_socket,
            &args.new_process_socket,
            &args.address,
        )?;
    } else {
        info!("first");
        let allocator = SharedAllocator::new_memory(args.shared_memory_key, GB)
            .map_err(MainError::NewMemoryAllocator)?;
        let ptr = <*mut u8>::from_bits(allocator.address());
        #[allow(clippy::cast_ptr_alignment)]
        let data_ptr = ptr.cast::<Store>();
        info!("data_ptr: {:?}", data_ptr);
        // SAFETY:
        // We know the pointer to be valid.
        unsafe {
            ptr::write(data_ptr, RwLock::new(Data { x: 12 }));
        }
        // Store allocator in static (if this allocator is dropped the shared memory is dropped)
        // SAFETY:
        // It is always safe, as no other thread will be accessing this stack at this time.
        unsafe {
            ALLOCATOR.write(allocator);
            DATA.write(data_ptr);
        }
        process(
            now,
            &args.old_process_socket,
            &args.new_process_socket,
            &args.address,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    use super::{DEFAULT_NEW_PROCESS_SOCKET, DEFAULT_OLD_PROCESS_SOCKET};

    fn unique_key() -> i32 {
        use std::sync::atomic::{AtomicI32, Ordering};
        const BASE_KEY: i32 = 453_845i32;
        static BASE: AtomicI32 = AtomicI32::new(BASE_KEY);
        BASE.fetch_add(1, Ordering::SeqCst)
    }

    #[test]
    fn interrupt() {
        let key = unique_key();
        // Start
        #[allow(clippy::unwrap_used)]
        let child = Command::new("cargo")
            .args(["run", "--", "-s", &key.to_string()])
            .spawn()
            .unwrap();

        // Interrupt server (`ctrl+c`)
        // SAFETY:
        // This call should always be safe.
        unsafe {
            #[allow(clippy::unwrap_used)]
            let pid = i32::try_from(child.id()).unwrap();
            libc::kill(pid, libc::SIGINT);
        }

        // Check cleanup
        assert!(!Path::new(DEFAULT_NEW_PROCESS_SOCKET).exists());
        assert!(!Path::new(DEFAULT_OLD_PROCESS_SOCKET).exists());
    }
    #[test]
    fn server_chain() {
        fn overwrite(s: &str) -> std::io::Result<std::fs::File> {
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(s)
        }

        let key = unique_key();
        let max: i32 = 10i32;
        assert!(max < i32::MAX);

        assert!(!std::path::Path::new(DEFAULT_NEW_PROCESS_SOCKET).exists());
        assert!(!std::path::Path::new(DEFAULT_OLD_PROCESS_SOCKET).exists());

        #[allow(clippy::unwrap_used)]
        let mut process = (0i32..max).fold(
            Command::new("cargo")
                .args(["run", "--", "-l", "Trace", "-s", &key.to_string()])
                .stdout(overwrite("/tmp/server_chain_stdout_0.txt").unwrap())
                .stderr(overwrite("/tmp/server_chain_stderr_0.txt").unwrap())
                .spawn()
                .unwrap(),
            |mut predecessor, i| {
                // We wait for the predecessor process to have started.
                std::thread::sleep(std::time::Duration::from_secs(1));
                // SAFETY:
                // Since we previously assert `max < i32::MAX` this will always
                // be safe.
                let j = unsafe { i.unchecked_add(1) };
                let successor = Command::new("cargo")
                    .args(["run", "--", "-l", "Trace", "-s", &key.to_string()])
                    .stdout(overwrite(&format!("/tmp/server_chain_stdout_{j}.txt")).unwrap())
                    .stderr(overwrite(&format!("/tmp/server_chain_stderr_{j}.txt")).unwrap())
                    .spawn()
                    .unwrap();

                let res = predecessor.wait();
                dbg!(&res);
                assert!(matches!(res, Ok(ok) if ok.success()));

                successor
            },
        );

        // Interrupt server (`ctrl+c`)
        // SAFETY:
        // This call should always be safe.
        unsafe {
            #[allow(clippy::unwrap_used)]
            let pid = i32::try_from(process.id()).unwrap();
            libc::kill(pid, libc::SIGINT);
        }
        let res = process.wait();
        dbg!(&res);
        assert!(matches!(res, Ok(ok) if ok.success()));
    }
}
