pub mod binary_stream;
pub mod version_aware;

pub use binary_stream::{
    BinaryStream, MAX_BINARY_ARRAY_ELEMS, SliceReader, checked_array_len, signed_size_to_u64,
};
pub use version_aware::VersionRange;
