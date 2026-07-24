#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Invalid metadata file: {0}")]
    InvalidMetadata(String),

    #[error("Unsupported metadata version: {0}")]
    UnsupportedVersion(i32),

    #[error("Address 0x{0:x} not in any segment")]
    AddressNotMapped(u64),

    #[error("Invalid binary format: {0}")]
    InvalidFormat(String),

    #[error(
        "Read out of bounds: offset 0x{offset:x}, size {size} \
         (truncated file, bad offsets, or encrypted/corrupted data)"
    )]
    OutOfBounds { offset: u64, size: usize },

    #[error("Invalid array size: count={count}, element_size={elem_size} ({context})")]
    InvalidArraySize {
        count: u64,
        elem_size: usize,
        context: String,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Cyclic value-type dependency: {0}")]
    CyclicDependency(String),

    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Enrich low-level read failures with actionable metadata-protection hints.
    pub fn with_metadata_context(self, file_len: u64) -> Self {
        const HINT: &str = "\
Common causes: encrypted/obfuscated global-metadata.dat, truncated extract, or wrong Unity layout. \
Valid files start with magic AF 1B B1 FA. Protected games often need a runtime memory dump of \
decrypted metadata before dumping.";

        match self {
            Error::OutOfBounds { offset, size } => Error::InvalidMetadata(format!(
                "Read past end of metadata while parsing tables \
                 (offset 0x{offset:X}, need {size} bytes, file size {file_len}). {HINT}"
            )),
            Error::Io(ref e)
                if e.kind() == std::io::ErrorKind::UnexpectedEof
                    || e.to_string().contains("fill whole buffer") =>
            {
                Error::InvalidMetadata(format!(
                    "Unexpected end of metadata file while parsing tables (file size {file_len}). {HINT}"
                ))
            }
            Error::InvalidArraySize {
                count,
                elem_size,
                context,
            } => Error::InvalidMetadata(format!(
                "Implausible metadata table size in '{context}' \
                 (count={count}, element_size={elem_size}, file size {file_len}). \
                 Header fields look corrupted — often encryption or a truncated file. {HINT}"
            )),
            other => other,
        }
    }

    /// Short UI-facing string (Android / Tauri toast).
    pub fn user_message(&self) -> String {
        match self {
            Error::InvalidMetadata(msg) => format!("Metadata error: {msg}"),
            Error::UnsupportedVersion(v) => format!(
                "Metadata error: unsupported metadata version {v}. \
                 Update the dumper or check for a corrupted/encrypted file."
            ),
            Error::OutOfBounds { offset, size } => format!(
                "Metadata error: read past end of file (offset 0x{offset:X}, size {size}). \
                 File may be truncated or encrypted."
            ),
            Error::InvalidArraySize { context, .. } => format!(
                "Metadata error: invalid table size ({context}). \
                 File may be encrypted or corrupted."
            ),
            Error::Io(e)
                if e.kind() == std::io::ErrorKind::UnexpectedEof
                    || e.to_string().contains("fill whole buffer") =>
            {
                "Metadata error: unexpected end of file while reading metadata \
                 (often encrypted, truncated, or wrong file)."
                    .into()
            }
            other => other.to_string(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
