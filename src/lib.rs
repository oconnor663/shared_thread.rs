use std::sync::{Condvar, Mutex, OnceLock};
use std::thread;

/// A wrapper around std::thread::JoinHandle that allows for multiple joiners. Most methods take
/// `&self` and implicitly propagate panics from the shared thread.
#[derive(Debug)]
pub struct SharedThread<T> {
    thread: Mutex<Option<thread::JoinHandle<T>>>,
    condvar: Condvar,
    result: OnceLock<T>,
}

impl<T: Send + 'static> SharedThread<T> {
    pub fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() -> T + Send + 'static,
    {
        Self::new(thread::spawn(f))
    }
}

// A thread that sticks its result in a lazy cell, so that multiple callers can see it.
impl<T> SharedThread<T> {
    pub fn new(handle: thread::JoinHandle<T>) -> Self {
        SharedThread {
            thread: Mutex::new(Some(handle)),
            condvar: Condvar::new(),
            result: OnceLock::new(),
        }
    }

    pub fn try_join(&self) -> Option<&T> {
        let mut guard = self.thread.lock().unwrap();
        if let Some(thread) = &*guard {
            if thread.is_finished() {
                // Joining will not block in this case, and we can do it while holding the Mutex.
                let result = guard
                    .take()
                    .expect("already checked not None")
                    .join()
                    .expect("shared thread panicked");
                self.result.set(result).ok().expect("should be unset")
            }
        }
        self.result.get()
    }

    pub fn join(&self) -> &T {
        let mut guard = self.thread.lock().unwrap();
        if let Some(thread) = guard.take() {
            // To avoid a race condition (see test_join_try_join_race), first do a non-blocking
            // check while holding the lock.
            if thread.is_finished() {
                // The thread has already exited. JoinHandle::join will not block for long. We were
                // the first blocking waiter, so there's no need to notify the condvar.
                let result = thread.join().expect("shared thread panicked");
                self.result.set(result).ok().expect("should be unest");
                return self.result.get().unwrap();
            }
            // The thread hasn't finished, and it's our job to block on join(). Release the mutex
            // while we block, so that we don't interfere with calls to Self::try_join in the
            // meantime.
            drop(guard);
            let maybe_result = thread.join();
            // Retake the mutex so that notify_all() and wait() don't race.
            guard = self.thread.lock().unwrap();
            self.condvar.notify_all();
            let result = maybe_result.expect("shared thread panicked");
            self.result.set(result).ok().expect("should be unest");
            drop(guard); // suppress a value-not-read warning
            return self.result.get().unwrap();
        }
        // Either another thread has already joined, in which case the result will be populated, or
        // another thread is in the process of joining, in which case we'll wait for them to notify
        // the Condvar.
        loop {
            match self.result.get() {
                Some(result) => return result,
                None => {
                    guard = self.condvar.wait(guard).unwrap();
                }
            }
        }
    }

    pub fn into_result(self) -> T {
        self.join();
        self.result.into_inner().unwrap()
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
        let result: String = shared.into_result();
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
            while !shared_thread
                .thread
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .is_finished()
            {}
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
