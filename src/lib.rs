//! This crate provides [`SharedThread`], a wrapper around
//! [`std::thread::JoinHandle`](https://doc.rust-lang.org/std/thread/struct.JoinHandle.html) that
//! lets multiple threads wait on a shared thread and read its output, with an optional timeout.
//!
//! For example code, see [the `SharedThread` example](struct.SharedThread.html#example).

#![deny(unsafe_code)]

use std::fmt;
use std::mem;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::MutexGuard;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread;
use std::time::Duration;
use std::time::Instant;

/// A wrapper around [`std::thread::JoinHandle`] that allows for multiple waiters.
///
/// The high-level differences between `SharedThread` and [`JoinHandle`] are:
///
/// - [`join`][SharedThread::join] takes `&self` rather than `&mut self`.
/// - [`join`][SharedThread::join] returns `&T` rather than `T`. For taking ownership of `T`, see
///   [`into_output`][SharedThread::into_output].
/// - `SharedThread` also provides [`join_timeout`][Self::join_timeout],
///   [`join_deadline`][Self::join_deadline], and [`try_join`][SharedThread::try_join].
/// - Rather than converting panics in into
///   [`std::thread::Result`](https://doc.rust-lang.org/std/thread/type.Result.html), which usually
///   requires the caller to `.unwrap()` every `.join()`, `SharedThread` propagates panics
///   automatically.
///
/// # Example
///
/// ```
/// use shared_thread::SharedThread;
/// use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
///
/// // Use this flag to tell our shared thread when to stop.
/// static EXIT_FLAG: AtomicBool = AtomicBool::new(false);
///
/// // Start a background thread that we'll share with several waiting threads.
/// let shared_thread = SharedThread::spawn(|| {
///     // Pretend this is some expensive, useful background work...
///     while (!EXIT_FLAG.load(Relaxed)) {}
///
///     42
/// });
///
/// // It's up to you how to share the SharedThread object with other threads. In this sense it's
/// // like any other object you might need to share, like say a HashMap or a File. The common
/// // options are to put it in an Arc, or to let "scoped" threads borrow it directly. Let's use
/// // scoped threads.
/// std::thread::scope(|scope| {
///     // Spawn three waiter threads that each wait on the shared thread.
///     let waiter1 = scope.spawn(|| shared_thread.join());
///     let waiter2 = scope.spawn(|| shared_thread.join());
///     let waiter3 = scope.spawn(|| shared_thread.join());
///
///     // In this example, the shared thread is going to keep looping until we set the EXIT_FLAG.
///     // In the meantime, .is_finished() returns false, and .try_join() returns None.
///     assert!(!shared_thread.is_finished());
///     assert_eq!(shared_thread.try_join(), None);
///
///     // Ask the shared thread to stop looping.
///     EXIT_FLAG.store(true, Relaxed);
///
///     // Now all the calls to .join() above will return quickly, and each of the waiter threads
///     // will get a reference to the shared thread's output, &42.
///     assert_eq!(*waiter1.join().unwrap(), 42);
///     assert_eq!(*waiter2.join().unwrap(), 42);
///     assert_eq!(*waiter3.join().unwrap(), 42);
///
///     // Now that the shared thread has finished, .is_finished() returns true, and .try_join()
///     // returns Some(&42).
///     assert!(shared_thread.is_finished());
///     assert_eq!(*shared_thread.try_join().unwrap(), 42);
/// });
///
///  // We can take ownership of the output by consuming the SharedThread object. As with any
///  // non-Copy type in Rust, this requires that the SharedThread is not borrowed.
///  assert_eq!(shared_thread.into_output(), 42);
/// ```
///
/// [`std::thread::JoinHandle`]: https://doc.rust-lang.org/std/thread/struct.JoinHandle.html
/// [`JoinHandle`]: https://doc.rust-lang.org/std/thread/struct.JoinHandle.html
#[derive(Debug)]
pub struct SharedThread<T> {
    state: Mutex<State<T>>,
    exit_signal: Arc<ExitSignal>,
    output: OnceLock<T>,
}

// The shared thread sets this bool to true and signals the condvar when it exits, even if it
// panicks.
#[derive(Debug)]
struct ExitSignal {
    mutex: Mutex<bool>,
    condvar: Condvar,
}

enum State<T> {
    Running(thread::JoinHandle<T>),
    // Note that the return value T goes in the OnceLock. If it lived here in the Exited variant,
    // it would be stuck inside the state Mutex, and we couldn't share it with simple references.
    Exited,
    Panicked,
}

use State::*;

impl<T> fmt::Debug for State<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Running { .. } => write!(f, "Running"),
            Exited => write!(f, "Exited"),
            Panicked => write!(f, "Panicked"),
        }
    }
}

impl<T: Send + 'static> SharedThread<T> {
    /// Spawn a new `SharedThread`.
    ///
    /// # Panics
    ///
    /// This function calls
    /// [`std::thread::spawn`](https://doc.rust-lang.org/std/thread/fn.spawn.html) internally,
    /// which panics if it fails to spawn a thread.
    pub fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() -> T + Send + 'static,
    {
        let exit_signal = Arc::new(ExitSignal {
            mutex: Mutex::new(false),
            condvar: Condvar::new(),
        });
        let exit_signal_clone = Arc::clone(&exit_signal);
        let handle = thread::spawn(move || {
            // Whether or not the closure `f` panics, set the exited flag and notify the condvar.
            // It's not clear to me that the concept of "unwind safety" in the standard library was
            // a good indea, but at least it doesn't require any unsafe code to work around it.
            let unwind_result = catch_unwind(AssertUnwindSafe(f));
            let mut guard = lock_ignoring_poison(&exit_signal_clone.mutex);
            *guard = true;
            exit_signal_clone.condvar.notify_all();
            // Now that we've signaled exit, if there was a panic, propagate it. The first waiter
            // thread will observe it. (Subsequent waiter threads will only see the Panicked state
            // variant.)
            match unwind_result {
                Ok(return_value) => return_value,
                Err(panic) => resume_unwind(panic),
            }
        });
        SharedThread {
            state: Mutex::new(Running(handle)),
            exit_signal,
            output: OnceLock::new(),
        }
    }
}

// A thread that multiple other threads can wait on simultaneously.
impl<T> SharedThread<T> {
    fn join_exited_thread(&self, exit_signal_guard: MutexGuard<bool>) -> &T {
        // It's not really important that we pass down the exit_signal_guard here, but it would
        // probably live across this call anway, so it seems cleaner to take ownership of it.
        debug_assert!(*exit_signal_guard, "the thread exited");

        let mut state_guard = lock_ignoring_poison(&self.state);
        match &*state_guard {
            // Running means we're the thread that needs to .join(). Fall through.
            Running(_) => {}
            // Exited or Panicked means someone already joined.
            Exited => return self.output.get().unwrap(),
            Panicked => panic!("shared thread panicked"),
        };

        // We need to .join(), so take the JoinHandle by value. Use the Panicked state as a
        // placeholder, so that it's the state we leave behind if something does in fact panic.
        // This makes the Panicked state kind of ambiguous between "the other thread panicked" or
        // "we failed an assert somewhere", but at least the initial panic backtrace will make it
        // clear what happened.
        let Running(handle) = mem::replace(&mut *state_guard, Panicked) else {
            unreachable!()
        };

        // The thread has signaled that it's exiting, so .join() will return quickly. (It might
        // block briefly it the thread is still cleaning itself up.) If it panicked, propagate the
        // panic.
        match handle.join() {
            Ok(return_value) => {
                // Because we set `output` while we hold the state mutex, it's guaranteed that
                // subsequent threads that see the Exited state will also see that `output` is set.
                let set_result = self.output.set(return_value);
                assert!(set_result.is_ok(), "output must be previously unset");
                *state_guard = Exited;
                self.output.get().unwrap()
            }
            Err(panic) => resume_unwind(panic),
        }
    }

    /// Wait for the shared thread to finish, then return `&T`. This blocks the current thread.
    ///
    /// # Panics
    ///
    /// This function panics if the shared thread panicked. The original panic is propagated
    /// directly with [`resume_unwind`](https://doc.rust-lang.org/std/panic/fn.resume_unwind.html)
    /// the first time. Subsequent calls panic with a generic message.
    pub fn join(&self) -> &T {
        let mut exit_signal_guard = lock_ignoring_poison(&self.exit_signal.mutex);
        while !*exit_signal_guard {
            exit_signal_guard = wait_ignoring_poison(&self.exit_signal.condvar, exit_signal_guard);
        }
        self.join_exited_thread(exit_signal_guard)
    }

    /// Wait with a timeout for the shared thread to finish. If it finishes in time (or it already
    /// finished), return `Some(&T)`, otherwise return `None`. This blocks the current thread.
    ///
    /// # Panics
    ///
    /// This function panics if the shared thread panicked. The original panic is propagated
    /// directly with [`resume_unwind`](https://doc.rust-lang.org/std/panic/fn.resume_unwind.html)
    /// the first time it's observed. Subsequent calls panic with a generic message.
    pub fn join_timeout(&self, timeout: Duration) -> Option<&T> {
        let deadline = Instant::now() + timeout;
        self.join_deadline(deadline)
    }

    /// Wait with a deadline for the shared thread to finish. If it finishes in time (or it already
    /// finished), return `Some(&T)`, otherwise return `None`. This blocks the current thread.
    ///
    /// # Panics
    ///
    /// This function panics if the shared thread panicked. The original panic is propagated
    /// directly with [`resume_unwind`](https://doc.rust-lang.org/std/panic/fn.resume_unwind.html)
    /// the first time it's observed. Subsequent calls panic with a generic message.
    pub fn join_deadline(&self, deadline: Instant) -> Option<&T> {
        let mut exit_signal_guard = lock_ignoring_poison(&self.exit_signal.mutex);
        while !*exit_signal_guard {
            if Instant::now() > deadline {
                return None;
            }
            exit_signal_guard = wait_deadline_ignoring_poison(
                &self.exit_signal.condvar,
                exit_signal_guard,
                deadline,
            );
        }
        Some(self.join_exited_thread(exit_signal_guard))
    }

    /// Return `Some(&T)` if the shared thread has already finished, otherwise `None`. This always
    /// returns quickly.
    ///
    /// # Panics
    ///
    /// This function panics if the shared thread panicked. The original panic is propagated
    /// directly with [`resume_unwind`](https://doc.rust-lang.org/std/panic/fn.resume_unwind.html)
    /// the first time it's observed. Subsequent calls panic with a generic message.
    pub fn try_join(&self) -> Option<&T> {
        let exit_signal_guard = lock_ignoring_poison(&self.exit_signal.mutex);
        if *exit_signal_guard {
            Some(self.join_exited_thread(exit_signal_guard))
        } else {
            None
        }
    }

    /// Wait for the shared thread to finish, then return `T`. This blocks the current. This
    /// requires ownership of the `SharedThread` and consumes it.
    ///
    /// # Panics
    ///
    /// This function panics if the shared thread panicked. The original panic is propagated
    /// directly with [`resume_unwind`](https://doc.rust-lang.org/std/panic/fn.resume_unwind.html)
    /// the first time it's observed. Subsequent calls panic with a generic message.
    pub fn into_output(self) -> T {
        self.join();
        self.output.into_inner().expect("should be set")
    }

    /// Return `true` if the shared thread has finished, `false` otherwise.
    ///
    /// This function never blocks. If it returns `true`, [`try_join`][Self::try_join],
    /// [`join_timeout`][Self::join_timeout], and [`join_deadline`][Self::join_deadline] are
    /// guaranteed not to return `None`, and all join functions are guaranteed to return quickly.
    pub fn is_finished(&self) -> bool {
        *lock_ignoring_poison(&self.exit_signal.mutex)
    }
}

fn lock_ignoring_poison<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(e) => e.into_inner(),
    }
}

fn wait_ignoring_poison<'guard, T>(
    condvar: &Condvar,
    guard: MutexGuard<'guard, T>,
) -> MutexGuard<'guard, T> {
    match condvar.wait(guard) {
        Ok(guard) => guard,
        Err(e) => e.into_inner(),
    }
}

fn wait_deadline_ignoring_poison<'guard, T>(
    condvar: &Condvar,
    guard: MutexGuard<'guard, T>,
    deadline: Instant,
) -> MutexGuard<'guard, T> {
    let timeout = deadline.saturating_duration_since(Instant::now());
    match condvar.wait_timeout(guard, timeout) {
        Ok((guard, _)) => guard,
        Err(e) => e.into_inner().0,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};

    #[test]
    fn test_join_and_try_join() {
        static STOP_FLAG: AtomicBool = AtomicBool::new(false);
        let bg_thread = SharedThread::spawn(|| {
            while !STOP_FLAG.load(Relaxed) {}
            42
        });
        // Spawn 10 joiner threads that all simultaneously try to join the backgroud thread.
        thread::scope(|scope| {
            let mut joiner_handles = Vec::new();
            for _ in 0..10 {
                joiner_handles.push(scope.spawn(|| {
                    bg_thread.join();
                }));
            }
            // try_join will always return None here.
            for _ in 0..100 {
                assert!(bg_thread.try_join().is_none());
                assert!(!bg_thread.is_finished());
            }
            STOP_FLAG.store(true, Relaxed);
            // One of the joiner threads almost certainly has the underlying thread handle, and
            // eventually it'll set SharedThread::result and one of these try_joins will return Some.
            while !bg_thread.is_finished() {}
            assert_eq!(bg_thread.try_join(), Some(&42));
        });
    }

    #[test]
    fn test_try_join_only() {
        static STOP_FLAG: AtomicBool = AtomicBool::new(false);
        let bg_thread = SharedThread::spawn(|| {
            while !STOP_FLAG.load(Relaxed) {}
            42
        });
        // try_join will always return None here.
        for _ in 0..100 {
            assert!(bg_thread.try_join().is_none());
        }
        STOP_FLAG.store(true, Relaxed);
        // Eventually one of these try_join's will see .is_finished() = true and join the thread.
        while bg_thread.try_join().is_none() {}
        assert_eq!(bg_thread.try_join(), Some(&42));
    }

    #[test]
    fn test_into_inner() {
        let thread = SharedThread::spawn(|| String::from("foo"));
        let result: String = thread.into_output();
        assert_eq!(result, "foo");
    }

    // This is a close port of the test_wait_try_wait_race test in shared_child.rs. The basic
    // principle of the race is the same: If the blocking thread is going to somehow signal that
    // it's blocking (in this case by taking the join handle), it has to check for early
    // termination before it releases its initial lock, or else it creates a window where the
    // non-blocking thread can get confused.
    #[test]
    fn test_join_try_join_race() {
        // Make sure that .join() and .try_join() can't race against each other. The scenario we're
        // worried about is:
        //   1. join() acquires the lock, takes the JoinHandle out, and releases the lock.
        //   2. try_join swoops in, acquires the lock, sees the JoinHandle is missing, checks the
        //      OnceCell, and returns None.
        //   3. join() resumes, actually calls JoinHandle::join, observes the thread has already
        //      finished, retakes the lock immediately, and populates the OnceCell.
        // A race like this could cause .try_join() to report that the thread hasn't finished, even
        // if in fact it finished long ago. A subsequent call to .try_join() would almost certainly
        // report Some, but the first call is still a bug. The way to prevent the bug is by making
        // .join() check JoinHandle::is_finished before releasing the lock.
        //
        // This was a failing test when I first committed it. Most of the time it would fail after
        // a few hundred iterations, but sometimes it took thousands. Default to one second so that
        // the tests don't take too long, but use an env var to configure a really big run in CI.
        use std::time::{Duration, Instant};
        let mut test_duration_secs: u64 = 1;
        if let Ok(test_duration_secs_str) = std::env::var("RACE_TEST_SECONDS") {
            dbg!(&test_duration_secs_str);
            test_duration_secs = test_duration_secs_str.parse().expect("invalid u64");
        }
        let test_duration = Duration::from_secs(test_duration_secs);
        let test_start = Instant::now();
        let mut iterations = 1u64;
        loop {
            // Start a thread that will finish immediately.
            let shared_thread = SharedThread::spawn(|| ());
            // Wait for the thread to finish, without updating the SharedChild state.
            {
                let state_guard = lock_ignoring_poison(&shared_thread.state);
                let Running(handle) = &*state_guard else {
                    unreachable!()
                };
                while !handle.is_finished() {}
            }
            // Spawn two more threads, one to join() and one to try_join(). It should be impossible
            // for the try_join thread to return None at this point. However, we want to make sure
            // there's no race condition between them, where the join() thread has said indicated
            // it's joining and released the lock but hasn't yet actually joined.
            let barrier = std::sync::Barrier::new(2);
            let try_join_ret = std::thread::scope(|scope| {
                scope.spawn(|| {
                    barrier.wait();
                    shared_thread.join();
                });
                scope
                    .spawn(|| {
                        barrier.wait();
                        shared_thread.try_join()
                    })
                    .join()
                    .unwrap()
            });
            let test_time_so_far = Instant::now().saturating_duration_since(test_start);
            assert!(
                try_join_ret.is_some(),
                "encountered the race condition after {:?} ({} iterations)",
                test_time_so_far,
                iterations,
            );
            iterations += 1;

            // If we've met the target test duration (1 sec by default), exit with success.
            // Otherwise keep looping and trying to provoke the race.
            if test_time_so_far >= test_duration {
                return;
            }
        }
    }

    #[test]
    fn test_panic_messages() {
        let thread = SharedThread::spawn(|| panic!("original message"));
        let panic_error = catch_unwind(|| thread.join()).unwrap_err();
        assert_eq!(panic_error.downcast_ref(), Some(&"original message"));

        let second_panic_error = catch_unwind(|| thread.join()).unwrap_err();
        assert_eq!(
            second_panic_error.downcast_ref(),
            Some(&"shared thread panicked"),
        );
    }
}
