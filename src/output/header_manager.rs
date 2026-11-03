use super::unity_version::{UnityVersion, UnityVersionRange};
use include_dir::{Dir, include_dir};
use std::collections::HashSet;

static UNITY_HEADERS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/assets/UnityHeaders");
static IL2CPP_API_HEADERS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/assets/Il2CppAPIHeaders");

#[derive(Debug, Clone)]
pub struct UnityResource {
    pub name: String,
    pub version_range: UnityVersionRange,
    pub metadata_version: String,
    pub is_api: bool,
}

impl UnityResource {
    pub fn get_text(&self) -> Option<String> {
        let dir = if self.is_api {
            &IL2CPP_API_HEADERS_DIR
        } else {
            &UNITY_HEADERS_DIR
        };
        if let Some(file) = dir.get_file(&self.name) {
            return file.contents_utf8().map(|s| s.to_string());
        }
        None
    }
}

pub struct UnityHeaders {
    pub metadata_version: String,
    pub version_range: UnityVersionRange,
    pub type_header: UnityResource,
    pub api_header: UnityResource,
}

impl UnityHeaders {
    pub fn new(type_header: UnityResource, api_header: UnityResource) -> Self {
        let version_range = type_header
            .version_range
            .intersect(&api_header.version_range)
            .unwrap_or_else(|| {
                UnityVersionRange::new(
                    type_header.version_range.min.clone(),
                    type_header.version_range.max.clone(),
                )
            });
        let metadata_version = type_header.metadata_version.clone();

        Self {
            metadata_version,
            version_range,
            type_header,
            api_header,
        }
    }

    pub fn get_type_header_text(&self, is_32bit: bool) -> String {
        let mut str = if is_32bit {
            "#define IS_32BIT\n".to_string()
        } else {
            "".to_string()
        };
        str.push_str(&self.type_header.get_text().unwrap_or_default());

        let v5_3_6 = "5.3.6".parse::<UnityVersion>().unwrap();
        let v5_4_6 = "5.4.6".parse::<UnityVersion>().unwrap();

        if self.version_range.min >= v5_3_6
            && self
                .version_range
                .max
                .as_ref()
                .map_or(true, |m| *m <= v5_4_6)
        {
            str.push_str("\nstruct VirtualInvokeData\n{\n    Il2CppMethodPointer methodPtr;\n    const MethodInfo* method;\n};\n");
        }

        str
    }

    pub fn get_api_header_text(&self) -> String {
        self.api_header.get_text().unwrap_or_default()
    }

    pub fn get_all_type_headers() -> Vec<UnityResource> {
        let mut resources = Vec::new();
        for file in UNITY_HEADERS_DIR.files() {
            let name = file.path().to_string_lossy().into_owned();
            if name.ends_with(".h") {
                let version_range = UnityVersionRange::from_filename(&name);
                let metadata_version = name.split('-').next().unwrap_or("").to_string();
                resources.push(UnityResource {
                    name,
                    version_range,
                    metadata_version,
                    is_api: false,
                });
            }
        }
        resources
    }

    pub fn get_all_api_headers() -> Vec<UnityResource> {
        let mut resources = Vec::new();
        for file in IL2CPP_API_HEADERS_DIR.files() {
            let name = file.path().to_string_lossy().into_owned();
            if name.ends_with(".h") {
                let version_range = UnityVersionRange::from_filename(&name);
                let metadata_version = name.split('-').next().unwrap_or("").to_string();
                resources.push(UnityResource {
                    name,
                    version_range,
                    metadata_version,
                    is_api: true,
                });
            }
        }
        resources
    }

    fn get_function_names_from_api_header(text: &str) -> HashSet<String> {
        let mut names = HashSet::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("DO_API") {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    let name = parts[1].trim();
                    names.insert(name.to_string());
                }
            }
        }
        names
    }

    pub fn guess_headers_for_binary(
        metadata_version: f32,
        _is_32bit: bool,
        api_exports: &HashSet<String>,
        binary_field_offsets_are_pointers: bool,
    ) -> Vec<UnityHeaders> {
        let mut type_headers: Vec<UnityResource> = Self::get_all_type_headers()
            .into_iter()
            .filter(|r| {
                let Ok(mv) = r.metadata_version.parse::<f32>() else {
                    return false;
                };
                if (mv.floor() - metadata_version.floor()).abs() > 0.01 {
                    return false;
                }

                if (metadata_version - 21.0).abs() < 0.01 {
                    let v5_3_7 = "5.3.7".parse::<UnityVersion>().unwrap();
                    let v5_4_0 = "5.4.0".parse::<UnityVersion>().unwrap();
                    let header_field_offsets_are_pointers =
                        r.version_range.min >= v5_3_7 && r.version_range.min != v5_4_0;
                    if header_field_offsets_are_pointers != binary_field_offsets_are_pointers {
                        return false;
                    }
                }
                true
            })
            .collect();

        type_headers.sort_by(|a, b| a.version_range.min.cmp(&b.version_range.min));
        if type_headers.is_empty() {
            return vec![];
        }

        let total_range = UnityVersionRange::new(
            type_headers.first().unwrap().version_range.min.clone(),
            type_headers.last().unwrap().version_range.max.clone(),
        );

        let apis: Vec<UnityResource> = Self::get_all_api_headers()
            .into_iter()
            .filter(|a| a.version_range.intersect(&total_range).is_some())
            .collect();

        if apis.is_empty() {
            return vec![];
        }

        if api_exports.is_empty() {
            println!(
                "No IL2CPP API exports found in binary - IL2CPP APIs will be unavailable in C++ project"
            );
            return type_headers
                .into_iter()
                .map(|t| {
                    let matching_api = apis
                        .iter()
                        .rev()
                        .find(|a| a.version_range.intersect(&t.version_range).is_some())
                        .unwrap_or(apis.last().unwrap());
                    UnityHeaders::new(t, matching_api.clone())
                })
                .collect();
        }

        let mut api_matches = Vec::new();
        for api in &apis {
            if let Some(text) = api.get_text() {
                let api_funcs = Self::get_function_names_from_api_header(&text);

                let mut all_match = true;
                for func in &api_funcs {
                    if !api_exports.contains(func) {
                        all_match = false;
                        break;
                    }
                }

                if all_match {
                    api_matches.push(api.clone());
                }
            }
        }

        if !api_matches.is_empty() {
            println!("IL2CPP API discovery was successful");
            let mut results = Vec::new();
            for t in &type_headers {
                for a in &api_matches {
                    if t.version_range.intersect(&a.version_range).is_some() {
                        results.push(UnityHeaders::new(t.clone(), a.clone()));
                    }
                }
            }
            if !results.is_empty() {
                return results;
            }
        }

        println!(
            "No exact match for IL2CPP APIs found in binary - IL2CPP API availability in C++ project will be partial"
        );
        type_headers
            .into_iter()
            .map(|t| {
                let matching_api = apis
                    .iter()
                    .rev()
                    .find(|a| a.version_range.intersect(&t.version_range).is_some())
                    .unwrap_or(apis.last().unwrap());
                UnityHeaders::new(t, matching_api.clone())
            })
            .collect()
    }
}
