#[derive(Debug, Clone, Copy)]
pub struct VersionRange {
    pub min: f64,
    pub max: f64,
}

impl VersionRange {
    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, version: f64) -> bool {
        version >= self.min && version <= self.max
    }
}

impl Default for VersionRange {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: 999.0,
        }
    }
}

#[macro_export]
macro_rules! read_versioned {
    ($stream:expr, $version:expr, $min:expr, $max:expr, $read_fn:ident) => {
        if $version >= $min && $version <= $max {
            $stream.$read_fn()?
        } else {
            Default::default()
        }
    };
    ($stream:expr, $version:expr, $min:expr, $max:expr, $read_fn:ident, $default:expr) => {
        if $version >= $min && $version <= $max {
            $stream.$read_fn()?
        } else {
            $default
        }
    };
}

#[macro_export]
macro_rules! size_versioned {
    ($version:expr, $min:expr, $max:expr, $size:expr) => {
        if $version >= $min && $version <= $max {
            $size
        } else {
            0
        }
    };
}
