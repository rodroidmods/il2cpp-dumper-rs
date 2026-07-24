use std::fmt::Write;
use crate::io::BinaryStream;
use crate::il2cpp::metadata::Metadata;
use crate::il2cpp::enums::Il2CppTypeEnum;
use crate::error::Result;

const MAX_ATTRIBUTE_ARGS: u32 = 1024;

pub struct CustomAttributeDataReader {
    stream: BinaryStream,
    data_len: usize,
    pub count: u32,
    ctor_buffer: u64,
    data_buffer: u64,
}

impl CustomAttributeDataReader {
    pub fn new(data: Vec<u8>) -> Result<Self> {
        let data_len = data.len();
        let mut stream = BinaryStream::new(data);
        let count = stream.read_compressed_u32()?;
        if count > MAX_ATTRIBUTE_ARGS {
            return Err(crate::error::Error::Other(
                format!("Attribute count too large: {count}")
            ));
        }
        let ctor_buffer = stream.position();
        let data_buffer = ctor_buffer + count as u64 * 4;
        if data_buffer > data_len as u64 {
            return Err(crate::error::Error::Other(
                "Attribute data buffer exceeds data length".into()
            ));
        }
        Ok(Self {
            stream,
            data_len,
            count,
            ctor_buffer,
            data_buffer,
        })
    }

    fn remaining(&self) -> usize {
        let pos = self.stream.position() as usize;
        if pos >= self.data_len { 0 } else { self.data_len - pos }
    }

    pub fn get_string_custom_attribute_data(&mut self, metadata: &Metadata) -> Result<String> {
        if self.remaining() == 0 {
            return Err(crate::error::Error::Other("No data remaining".into()));
        }

        self.stream.set_position(self.ctor_buffer);
        let ctor_index = self.stream.read_i32()? as usize;
        let method_def = metadata.method_defs.get(ctor_index).cloned();
        self.ctor_buffer = self.stream.position();

        self.stream.set_position(self.data_buffer);
        let argument_count = self.stream.read_compressed_u32()?.min(MAX_ATTRIBUTE_ARGS);
        let field_count = self.stream.read_compressed_u32()?.min(MAX_ATTRIBUTE_ARGS);
        let property_count = self.stream.read_compressed_u32()?.min(MAX_ATTRIBUTE_ARGS);

        let mut arg_list = Vec::new();

        for _ in 0..argument_count {
            if self.remaining() == 0 { break; }
            match self.read_attribute_data_value() {
                Ok(val) => arg_list.push(val),
                Err(_) => break,
            }
        }

        let type_def = method_def.and_then(|md| metadata.type_defs.get(md.declaring_type as usize).cloned());

        for _ in 0..field_count {
            if self.remaining() == 0 { break; }
            let val = self.read_attribute_data_value().unwrap_or_default();
            if let Some(ref td) = type_def {
                if let Ok((declaring, field_index)) = self.read_named_argument_class_and_index(td, metadata) {
                    if let Some(field_idx) = (declaring.field_start as usize).checked_add(field_index as usize) {
                        if let Some(field_def) = metadata.field_defs.get(field_idx) {
                            let field_def = field_def.clone();
                            if let Ok(name) = metadata.get_string_from_index(field_def.name_index) {
                                arg_list.push(format!("{name} = {val}"));
                                continue;
                            }
                        }
                    }
                }
            }
            arg_list.push(val);
        }

        for _ in 0..property_count {
            if self.remaining() == 0 { break; }
            let val = self.read_attribute_data_value().unwrap_or_default();
            if let Some(ref td) = type_def {
                if let Ok((declaring, prop_index)) = self.read_named_argument_class_and_index(td, metadata) {
                    if let Some(prop_idx) = (declaring.property_start as usize).checked_add(prop_index as usize) {
                        if let Some(prop_def) = metadata.property_defs.get(prop_idx) {
                            let prop_def = prop_def.clone();
                            if let Ok(name) = metadata.get_string_from_index(prop_def.name_index) {
                                arg_list.push(format!("{name} = {val}"));
                                continue;
                            }
                        }
                    }
                }
            }
            arg_list.push(val);
        }

        self.data_buffer = self.stream.position();

        let type_name = type_def
            .and_then(|td| metadata.get_string_from_index(td.name_index).ok())
            .unwrap_or_else(|| "UnknownAttribute".to_string())
            .replace("Attribute", "");

        if arg_list.is_empty() {
            Ok(format!("[{type_name}]"))
        } else {
            Ok(format!("[{type_name}({})]", arg_list.join(", ")))
        }
    }

    pub fn get_ctor_type_name(&mut self, metadata: &Metadata) -> Result<String> {
        if self.remaining() == 0 {
            return Err(crate::error::Error::Other("No data remaining".into()));
        }

        self.stream.set_position(self.ctor_buffer);
        let ctor_index = self.stream.read_i32()? as usize;
        self.ctor_buffer = self.stream.position();

        self.stream.set_position(self.data_buffer);
        let argument_count = self.stream.read_compressed_u32()?.min(MAX_ATTRIBUTE_ARGS);
        let field_count = self.stream.read_compressed_u32()?.min(MAX_ATTRIBUTE_ARGS);
        let property_count = self.stream.read_compressed_u32()?.min(MAX_ATTRIBUTE_ARGS);

        for _ in 0..argument_count {
            if self.remaining() == 0 { break; }
            let _ = self.read_attribute_data_value();
        }
        for _ in 0..field_count {
            if self.remaining() == 0 { break; }
            let _ = self.read_attribute_data_value();
            let method_def = metadata.method_defs.get(ctor_index).cloned();
            let type_def = method_def.and_then(|md| metadata.type_defs.get(md.declaring_type as usize).cloned());
            if let Some(ref td) = type_def {
                let _ = self.read_named_argument_class_and_index(td, metadata);
            }
        }
        for _ in 0..property_count {
            if self.remaining() == 0 { break; }
            let _ = self.read_attribute_data_value();
            let method_def = metadata.method_defs.get(ctor_index).cloned();
            let type_def = method_def.and_then(|md| metadata.type_defs.get(md.declaring_type as usize).cloned());
            if let Some(ref td) = type_def {
                let _ = self.read_named_argument_class_and_index(td, metadata);
            }
        }

        self.data_buffer = self.stream.position();

        let method_def = metadata.method_defs.get(ctor_index).cloned();
        let type_def = method_def.and_then(|md| metadata.type_defs.get(md.declaring_type as usize).cloned());

        let type_name = match type_def {
            Some(td) => {
                let ns = metadata.get_string_from_index(td.namespace_index).unwrap_or_default();
                let name = metadata.get_string_from_index(td.name_index).unwrap_or_default();
                if ns.is_empty() { name } else { format!("{ns}.{name}") }
            }
            None => format!("UnknownAttribute_{ctor_index}"),
        };

        Ok(type_name)
    }

    fn read_attribute_data_value(&mut self) -> Result<String> {
        if self.remaining() == 0 {
            return Err(crate::error::Error::Other("No data remaining".into()));
        }
        let type_byte = self.stream.read_compressed_u32()?;
        let type_enum = Il2CppTypeEnum::from_u8(type_byte as u8);

        match type_enum {
            Some(Il2CppTypeEnum::Boolean) => {
                let v = self.stream.read_u8()?;
                Ok(if v != 0 { "true".into() } else { "false".into() })
            }
            Some(Il2CppTypeEnum::Char) => {
                let v = self.stream.read_u16()?;
                Ok(format!("'\\x{v:x}'"))
            }
            Some(Il2CppTypeEnum::I1) => {
                let v = self.stream.read_i8()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::U1) => {
                let v = self.stream.read_u8()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::I2) => {
                let v = self.stream.read_i16()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::U2) => {
                let v = self.stream.read_u16()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::I4) => {
                let v = self.stream.read_compressed_i32()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::U4) => {
                let v = self.stream.read_compressed_u32()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::I8) => {
                let v = self.stream.read_i64()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::U8) => {
                let v = self.stream.read_u64()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::R4) => {
                let v = self.stream.read_f32()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::R8) => {
                let v = self.stream.read_f64()?;
                Ok(v.to_string())
            }
            Some(Il2CppTypeEnum::String) => {
                let length = self.stream.read_compressed_i32()?;
                if length == -1 {
                    return Ok("null".into());
                }
                if length < 0 || length as usize > self.remaining() {
                    return Err(crate::error::Error::Other("Invalid string length".into()));
                }
                let bytes = self.stream.read_bytes(length as usize)?;
                let s = String::from_utf8_lossy(&bytes).to_string();
                Ok(format!("\"{s}\""))
            }
            Some(Il2CppTypeEnum::SzArray) => {
                let _element_type = self.stream.read_compressed_u32()?;
                let length = self.stream.read_compressed_i32()?;
                if length == -1 {
                    return Ok("null".into());
                }
                if length < 0 || length > MAX_ATTRIBUTE_ARGS as i32 {
                    return Err(crate::error::Error::Other("Invalid array length".into()));
                }
                let mut items = Vec::new();
                for _ in 0..length {
                    if self.remaining() == 0 { break; }
                    items.push(self.read_attribute_data_value()?);
                }
                Ok(format!("new[] {{ {} }}", items.join(", ")))
            }
            Some(Il2CppTypeEnum::Il2CppTypeIndex) => {
                let type_index = self.stream.read_compressed_i32()?;
                if type_index == -1 {
                    Ok("null".into())
                } else {
                    Ok(format!("typeof(/* type index {type_index} */)"))
                }
            }
            _ => {
                Err(crate::error::Error::Other(
                    format!("Unknown attribute type 0x{type_byte:x}")
                ))
            }
        }
    }

    fn read_named_argument_class_and_index(
        &mut self,
        type_def: &crate::il2cpp::structures::Il2CppTypeDefinition,
        metadata: &Metadata,
    ) -> Result<(crate::il2cpp::structures::Il2CppTypeDefinition, i32)> {
        let member_index = self.stream.read_compressed_i32()?;
        if member_index >= 0 {
            return Ok((type_def.clone(), member_index));
        }
        let actual_index = -(member_index + 1);
        let type_index = self.stream.read_compressed_u32()? as usize;
        let declaring = metadata.type_defs.get(type_index)
            .cloned()
            .unwrap_or_else(|| type_def.clone());
        Ok((declaring, actual_index))
    }
}

pub fn format_custom_attribute_data(
    buf: &mut String,
    metadata: &Metadata,
    attr_idx: usize,
    padding: &str,
) -> bool {
    let start_range = match metadata.attribute_data_ranges.get(attr_idx) {
        Some(r) => r.clone(),
        None => return false,
    };
    let end_range = match metadata.attribute_data_ranges.get(attr_idx + 1) {
        Some(r) => r.clone(),
        None => return false,
    };

    if end_range.start_offset <= start_range.start_offset {
        return false;
    }

    let data_offset = metadata.header.attribute_data_offset as u64 + start_range.start_offset as u64;
    let data_size = (end_range.start_offset - start_range.start_offset) as usize;

    if data_size == 0 || data_size > 1024 * 1024 {
        return false;
    }

    let data = match metadata.peek_bytes_at(data_offset, data_size) {
        Ok(d) => d.to_vec(),
        Err(_) => return false,
    };

    let mut reader = match CustomAttributeDataReader::new(data) {
        Ok(r) => r,
        Err(_) => return false,
    };

    if reader.count == 0 {
        return false;
    }

    for _ in 0..reader.count {
        match reader.get_string_custom_attribute_data(metadata) {
            Ok(attr_str) => {
                let _ = writeln!(buf, "{padding}{attr_str}");
            }
            Err(_) => break,
        }
    }

    true
}
