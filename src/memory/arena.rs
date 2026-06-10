//! Arena Allocator — zero-allocation hot path memory management.
//!
//! Pre-allocates a contiguous memory region at startup. All hot-path
//! allocations come from this arena, avoiding malloc/free overhead
//! and GC pauses. Reset between cycles for O(1) deallocation.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Thread-local bump allocator for hot-path zero-alloc operation.
/// Each MEV/ZK cycle resets the cursor to zero after completion.
pub struct Arena {
    buf: UnsafeCell<Vec<u8>>,
    cursor: AtomicUsize,
    capacity: usize,
}

// Arena is Send+Sync because we use atomic cursor and
// never hand out overlapping mutable references.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(capacity: usize) -> Self {
        let mut buf = Vec::with_capacity(capacity);
        buf.resize(capacity, 0u8);
        Self {
            buf: UnsafeCell::new(buf),
            cursor: AtomicUsize::new(0),
            capacity,
        }
    }

    /// Allocate `size` bytes aligned to `align`.
    /// Returns None if arena is exhausted (never panics on hot path).
    #[inline(always)]
    pub fn alloc(&self, size: usize, align: usize) -> Option<*mut u8> {
        loop {
            let current = self.cursor.load(Ordering::Relaxed);
            // Align up
            let aligned = (current + align - 1) & !(align - 1);
            let new_cursor = aligned + size;
            if new_cursor > self.capacity {
                return None;
            }
            if self.cursor
                .compare_exchange_weak(current, new_cursor, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                let buf = unsafe { &mut *self.buf.get() };
                return Some(buf[aligned..].as_mut_ptr());
            }
        }
    }

    /// Allocate a typed value from the arena.
    #[inline(always)]
    pub fn alloc_typed<T: Sized>(&self) -> Option<&mut T> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc(size, align)?;
        Some(unsafe { &mut *(ptr as *mut T) })
    }

    /// Allocate a slice of N elements from the arena.
    #[inline(always)]
    pub fn alloc_slice<T: Sized>(&self, count: usize) -> Option<&mut [T]> {
        let size = std::mem::size_of::<T>() * count;
        let align = std::mem::align_of::<T>();
        let ptr = self.alloc(size, align)?;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr as *mut T, count) })
    }

    /// O(1) reset — all allocations freed instantly.
    #[inline(always)]
    pub fn reset(&self) {
        self.cursor.store(0, Ordering::Release);
    }

    /// Current usage in bytes.
    #[inline]
    pub fn used(&self) -> usize {
        self.cursor.load(Ordering::Relaxed)
    }

    /// Total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc_and_reset() {
        let arena = Arena::new(4096);
        let p1 = arena.alloc(64, 8).unwrap();
        assert!(!p1.is_null());
        assert_eq!(arena.used(), 64);

        let p2 = arena.alloc(128, 16).unwrap();
        assert!(!p2.is_null());
        assert!(arena.used() >= 192);

        arena.reset();
        assert_eq!(arena.used(), 0);
    }

    #[test]
    fn test_arena_exhaustion() {
        let arena = Arena::new(64);
        assert!(arena.alloc(32, 8).is_some());
        assert!(arena.alloc(32, 8).is_some());
        assert!(arena.alloc(1, 1).is_none()); // Exhausted
    }

    #[test]
    fn test_typed_alloc() {
        let arena = Arena::new(4096);
        let val: &mut u64 = arena.alloc_typed().unwrap();
        *val = 0xDEADBEEF;
        assert_eq!(*val, 0xDEADBEEF);
    }

    #[test]
    fn test_slice_alloc() {
        let arena = Arena::new(4096);
        let slice: &mut [f64] = arena.alloc_slice(8).unwrap();
        for (i, v) in slice.iter_mut().enumerate() {
            *v = i as f64;
        }
        assert_eq!(slice[7], 7.0);
    }
}
