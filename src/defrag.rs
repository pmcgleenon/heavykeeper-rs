//! Shared machinery for relocating a sketch's large heap allocations under
//! active memory defragmentation. Used by every `*TopK` variant.

/// Relocates the sketch's large heap allocations for a host running active
/// memory defragmentation. One generic method covers every element type, so
/// the sketch's private types need not be named by the caller.
///
/// Implementations must not panic, or the field being relocated is left
/// permanently empty.
pub trait Reallocator {
    /// Relocate `boxed`, returning an equal-length, equal-contents `Box<[T]>`.
    fn realloc<T>(&mut self, boxed: Box<[T]>) -> Box<[T]>;
}

/// Relocate the allocation backing `slot` through `reallocator`, in place.
pub(crate) fn realloc_large_heap_allocated_object<E, R: Reallocator>(
    slot: &mut Box<[E]>,
    reallocator: &mut R,
) {
    // `mem::take` leaves an empty box, so nothing dangles if `reallocator` panics.
    *slot = reallocator.realloc(std::mem::take(slot));
}

/// Relocate `vec`'s backing allocation through `reallocator`, in place. Trimmed
/// to a boxed slice first to drop spare capacity; the rebuilt `Vec` has
/// capacity equal to its length.
pub(crate) fn realloc_vec<E, R: Reallocator>(vec: &mut Vec<E>, reallocator: &mut R) {
    let mut boxed = std::mem::take(vec).into_boxed_slice();
    realloc_large_heap_allocated_object(&mut boxed, reallocator);
    *vec = boxed.into_vec();
}
