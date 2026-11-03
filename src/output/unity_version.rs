use std::cmp::Ordering;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BuildType {
    Unspecified,
    Alpha,
    Beta,
    ReleaseCandidate,
    Final,
    Patch,
}

impl BuildType {
    pub fn from_str_safe(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "" => BuildType::Unspecified,
            "a" => BuildType::Alpha,
            "b" => BuildType::Beta,
            "rc" => BuildType::ReleaseCandidate,
            "f" => BuildType::Final,
            "p" => BuildType::Patch,
            _ => BuildType::Unspecified,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnityVersion {
    pub major: i32,
    pub minor: i32,
    pub update: i32,
    pub build_type: BuildType,
    pub build_number: i32,
}

impl Default for UnityVersion {
    fn default() -> Self {
        Self {
            major: 0,
            minor: 0,
            update: 0,
            build_type: BuildType::Unspecified,
            build_number: 0,
        }
    }
}

impl PartialOrd for UnityVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for UnityVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut cmp = self.major.cmp(&other.major);
        if cmp != Ordering::Equal {
            return cmp;
        }

        cmp = self.minor.cmp(&other.minor);
        if cmp != Ordering::Equal {
            return cmp;
        }

        cmp = self.update.cmp(&other.update);
        if cmp != Ordering::Equal {
            return cmp;
        }

        if self.build_type == BuildType::Unspecified || other.build_type == BuildType::Unspecified {
            return Ordering::Equal;
        }

        cmp = self.build_type.cmp(&other.build_type);
        if cmp != Ordering::Equal {
            return cmp;
        }

        self.build_number.cmp(&other.build_number)
    }
}

impl FromStr for UnityVersion {
    type Err = String;

    fn from_str(version_string: &str) -> Result<Self, Self::Err> {
        let mut parts = version_string.split('.');
        let major = parts
            .next()
            .unwrap_or("")
            .parse::<i32>()
            .map_err(|_| "Invalid major version")?;
        let minor = parts
            .next()
            .unwrap_or("")
            .parse::<i32>()
            .map_err(|_| "Invalid minor version")?;

        let mut update = 0;
        let mut build_type = BuildType::Unspecified;
        let mut build_number = 0;

        if let Some(rest) = parts.next() {
            let mut alpha_idx = None;
            for (i, c) in rest.char_indices() {
                if c.is_ascii_alphabetic() {
                    alpha_idx = Some(i);
                    break;
                }
            }

            if let Some(idx) = alpha_idx {
                update = rest[..idx].parse::<i32>().unwrap_or(0);

                let mut num_idx = None;
                for (i, c) in rest[idx..].char_indices() {
                    if c.is_ascii_digit() {
                        num_idx = Some(idx + i);
                        break;
                    }
                }

                if let Some(n_idx) = num_idx {
                    build_type = BuildType::from_str_safe(&rest[idx..n_idx]);
                    build_number = rest[n_idx..].parse::<i32>().unwrap_or(0);
                } else {
                    build_type = BuildType::from_str_safe(&rest[idx..]);
                }
            } else {
                update = rest.parse::<i32>().unwrap_or(0);
            }
        }

        Ok(UnityVersion {
            major,
            minor,
            update,
            build_type,
            build_number,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnityVersionRange {
    pub min: UnityVersion,
    pub max: Option<UnityVersion>,
}

impl UnityVersionRange {
    pub fn new(min: UnityVersion, max: Option<UnityVersion>) -> Self {
        Self { min, max }
    }

    pub fn contains(&self, version: &UnityVersion) -> bool {
        if version < &self.min {
            return false;
        }
        if let Some(max) = &self.max {
            if version > max {
                return false;
            }
        }
        true
    }

    pub fn intersect(&self, other: &UnityVersionRange) -> Option<UnityVersionRange> {
        let highest_low = if self.min > other.min {
            &self.min
        } else {
            &other.min
        };
        let lowest_high = match (&self.max, &other.max) {
            (Some(max1), Some(max2)) => {
                if max1 < max2 {
                    Some(max1)
                } else {
                    Some(max2)
                }
            }
            (Some(max1), None) => Some(max1),
            (None, Some(max2)) => Some(max2),
            (None, None) => None,
        };

        if let Some(lh) = lowest_high {
            if highest_low > lh {
                return None;
            }
        }

        Some(UnityVersionRange {
            min: highest_low.clone(),
            max: lowest_high.cloned(),
        })
    }

    pub fn from_filename(header_filename: &str) -> Self {
        let base_name = header_filename.trim_end_matches(".h");
        let parts: Vec<&str> = base_name.split('-').collect();

        let actual_parts = if parts.len() > 1
            && parts[0].chars().any(|c| c.is_ascii_digit())
            && parts[0].len() <= 4
        {
            &parts[1..]
        } else {
            &parts[0..]
        };

        let min = UnityVersion::from_str(actual_parts[0]).unwrap_or_default();
        let max = if actual_parts.len() == 1 {
            Some(min.clone())
        } else if actual_parts.len() == 2 && !actual_parts[1].is_empty() {
            Some(UnityVersion::from_str(actual_parts[1]).unwrap_or_default())
        } else {
            None
        };

        UnityVersionRange { min, max }
    }
}
