//! This crate provides [`SharedThread`], a wrapper around
//! [`std::thread::JoinHandle`](https://doc.rust-lang.org/std/thread/struct.JoinHandle.html) that
//! lets multiple threads wait on a shared thread and read its output.
//!
//! For example code, see [the `SharedThread` example](struct.SharedThread.html#example).

#![deny(unsafe_code)]

use std::fmt;
use std::mem;
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;

enum State<T> {
    Started(thread::JoinHandle<T>),
    Joining,
    Joined,
    Panicked,
}

impl<T> fmt::Debug for State<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Started(_) => write!(f, "Started"),
            Joining => write!(f, "Joining"),
            Joined => write!(f, "Joined"),
            Panicked => write!(f, "Panicked"),
        }
    }
}

use State::*;

/// A wrapper around [`JoinHandle`](https://doc.rust-lang.org/std/thread/struct.JoinHandle.html)
/// that allows for multiple waiters.
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
/// // How to share a SharedThread object with other threads is up to you. In this sense it's like
/// // any other object you might need to share, like say a HashMap or a File. The common options
/// // are to put it in an Arc, or to let "scoped" threads borrow it directly. Let's use scoped
/// // threads.
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
///     // At this point the calls to .join() above will return quickly, and each of the waiter
///     // threads will get a reference to the shared thread's output, &42.
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
#[derive(Debug)]
pub struct SharedThread<T> {
    state: Mutex<State<T>>,
    condvar: Condvar,
    output: OnceLock<T>,
}

impl<T: Send + 'static> SharedThread<T> {
    /// Spawn a new `SharedThread`.
    pub fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() -> T + Send + 'static,
    {
        Self::new(thread::spawn(f))
    }
}

// A thread that multiple other threads can wait on simultaneously.
impl<T> SharedThread<T> {
    /// Wrap an existing
    /// [`JoinHandle`](https://doc.rust-lang.org/std/thread/struct.JoinHandle.html).
    pub fn new(handle: thread::JoinHandle<T>) -> Self {
        SharedThread {
            state: Mutex::new(Started(handle)),
            condvar: Condvar::new(),
            output: OnceLock::new(),
        }
    }

    // .join() and .try_join() both call this when it's clear that they need to join the child (as
    // opposed to sleeping while some other thread does it, or returning None). The incoming state
    // must be Started. The state might transition through Joining, but the end state is guaranteed
    // to be either Joined or Panicked.
    fn do_blocking_join(&self, mut state_guard: MutexGuard<State<T>>) {
        // Use the Panicked state as a placeholder, so that that's the state we leave behind if
        // something does in fact panic. This makes the Panicked state kind of ambiguous between
        // "the other thread panicked" or "we failed an assert somewhere", but at least the initial
        // panic backtrace will make it clear what happened.
        let Started(handle) = mem::replace(&mut *state_guard, Panicked) else {
            panic!("unexpected shared thread state: {:?}", *state_guard);
        };

        // If we released the lock in the Joining state, that would make any calls to .try_join()
        // return None until we reacquired the lock and cleaned up. That's fine if the thread is
        // still running, or if it's exiting "now-ish" (i.e. an "honest" race), but it's not fine
        // if the thread actually exited long ago. To avoid that race condition, we need to check
        // on the thread *before* we release the lock. (See test_join_try_join_race.)
        if handle.is_finished() {
            // The thread has already exited. JoinHandle::join will not block for long, and we do
            // it with the state lock held. In this case we know there are no other waiting
            // threads, and there's no need to notify the condvar.
            let output = handle.join().expect("shared thread panicked");
            self.output.set(output).ok().expect("should be empty");
            *state_guard = Joined;
            return;
        }

        // The thread is still running (or at least, it was running until very recently). We're
        // going to do a potentially long-blocking join, and we need to release the lock while we
        // do this, so that calls to .try_join() in the meantime can observe the Joining state and
        // return None without blocking. After entering the Joining state, we *must* exit that
        // state and signal the condvar before returning, or else other threads might block
        // forever. No short-circuiting in this "critical section".
        *state_guard = Joining;
        drop(state_guard);

        // Do the blocking join. We're not allowed to panic here.
        let result = handle.join();

        // Reacquire the state lock, set the Panicked state again (so that, like in the beginning,
        // that's what we're left with if the other thread panicked or if something else panics
        // below), and signal the condvar. We're still in the critical section, so we suppress
        // Mutex poisoning.
        let mut state_guard = match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned_guard) => poisoned_guard.into_inner(),
        };
        *state_guard = Panicked;
        self.condvar.notify_all();

        // *Now* we're out of the critical section, and panicking is ok again. Clean up.
        let output = result.expect("the shared thread panicked");
        self.output.set(output).ok().expect("should be empty");
        *state_guard = Joined;
    }

    /// Return `Some(&T)` if the shared thread has already finished, otherwise `None`. This never
    /// blocks.
    ///
    /// # Panics
    ///
    /// This function panics if the shared thread panicked.
    pub fn try_join(&self) -> Option<&T> {
        let state_guard = self.state.lock().unwrap();
        match &*state_guard {
            // If the thread has already exited, we can join it ourselves. Otherwise return None,
            // to avoid blocking.
            Started(handle) => {
                if handle.is_finished() {
                    self.do_blocking_join(state_guard);
                } else {
                    return None;
                }
            }
            // Because we know .do_blocking_join() checked .is_finished() before blocking, we can
            // short-circuit here without worrying about a race condition.
            Joining => return None,
            // Just fall through.
            Joined => (),
            // Re-panic.
            Panicked => panic!("something panicked earlier"),
        }
        debug_assert!(self.output.get().is_some());
        self.output.get()
    }

    /// Wait for the shared thread to finish, then return `&T`. This blocks the current thread
    /// until the shared thread is finished.
    ///
    /// # Panics
    ///
    /// This function panics if the shared thread panicked.
    pub fn join(&self) -> &T {
        let mut state_guard = self.state.lock().unwrap();
        match *state_guard {
            // Joing the other thread ourselves.
            Started(_) => self.do_blocking_join(state_guard),
            // Sleep while another thread joins.
            Joining => {
                while matches!(*state_guard, Joining) {
                    state_guard = self.condvar.wait(state_guard).unwrap();
                }
            }
            // Just fall through.
            Joined => (),
            // Re-panic.
            Panicked => panic!("something panicked earlier"),
        }
        self.output.get().expect("must be set")
    }

    /// Wait for the shared thread to finish, then return `T`. This requires ownership of the
    /// `SharedThread` and consumes it. This blocks the current thread until the shared thread is
    /// finished.
    ///
    /// # Panics
    ///
    /// This function panics if the shared thread panicked.
    pub fn into_output(self) -> T {
        self.join();
        self.output.into_inner().expect("should be set")
    }

    /// Return `true` if the shared thread has finished, `false` otherwise.
    ///
    /// This function never blocks. If it returns `true`, [`try_join`][SharedThread::try_join] is
    /// guaranteed to return `Some(T)`, and [`join`][SharedThread::join] is guaranteed to return
    /// quickly.
    ///
    /// # Panics
    ///
    /// This function panics if the shared thread panicked.
    pub fn is_finished(&self) -> bool {
        match &*self.state.lock().unwrap() {
            Started(handle) => handle.is_finished(),
            // Because we know .do_blocking_join() checked .is_finished() before blocking, we don't
            // have to worry about a race condition here.
            Joining => false,
            Joined => true,
            Panicked => panic!("something panicked earlier"),
        }
    }
}

impl<T> From<thread::JoinHandle<T>> for SharedThread<T> {
    fn from(handle: thread::JoinHandle<T>) -> Self {
        Self::new(handle)
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
            }
            STOP_FLAG.store(true, Relaxed);
            // One of the joiner threads almost certainly has the underlying thread handle, and
            // eventually it'll set SharedThread::result and one of these try_joins will return Some.
            while bg_thread.try_join().is_none() {}
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
    fn test_from_and_into_inner() {
        let thread = thread::spawn(|| String::from("foo"));
        let shared: SharedThread<String> = thread.into();
        let result: String = shared.into_output();
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
                let state_guard = shared_thread.state.lock().unwrap();
                let Started(handle) = &*state_guard else {
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
}
