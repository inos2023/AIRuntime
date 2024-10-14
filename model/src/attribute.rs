use derive::{FromCode, GetCode};
use std::any::type_name;
use std::mem::forget;

use super::data::Data;

/// 属性类型
#[derive(Debug, Clone, Copy, PartialEq, GetCode, FromCode)]
pub enum AttType {
    #[code(0)]
    Undefined,
    #[code(1)]
    Float,
    #[code(2)]
    Int,
    #[code(3)]
    String,
    #[code(4)]
    Floats,
    #[code(7)]
    Ints,
    #[code(8)]
    Strings,
}

impl AttType {
    pub fn from_type_name(name: &str) -> Self {
        match name {
            "f32" => Self::Float,
            "&[f32]" => Self::Floats,
            "i64" => Self::Int,
            "&[i64]" => Self::Ints,
            "alloc::string::String" => Self::String,
            "&[alloc::string::String]" => Self::Strings,
            _ => panic!("Not support type name"),
        }
    }
    pub fn type_name(&self) -> &str {
        match self {
            Self::Float => type_name::<f32>(),
            Self::Floats => type_name::<&[f32]>(),
            Self::Int => type_name::<i64>(),
            Self::Ints => type_name::<&[i64]>(),
            Self::String => type_name::<String>(),
            Self::Strings => type_name::<&[String]>(),
            _ => panic!("Not support att_type now"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Attribute {
    /// 类型
    r#type: AttType,
    /// 数据
    data: Data,
}

impl From<i64> for Attribute {
    fn from(value: i64) -> Self {
        Self {
            r#type: AttType::Int,
            data: Data::from_array(&[value]),
        }
    }
}

impl From<&[i64]> for Attribute {
    fn from(value: &[i64]) -> Self {
        Self {
            r#type: AttType::Ints,
            data: Data::from_array(value),
        }
    }
}

impl From<f32> for Attribute {
    fn from(value: f32) -> Self {
        Self {
            r#type: AttType::Float,
            data: Data::from_array(&[value]),
        }
    }
}

impl From<&[f32]> for Attribute {
    fn from(value: &[f32]) -> Self {
        Self {
            r#type: AttType::Floats,
            data: Data::from_array(value),
        }
    }
}

impl From<&str> for Attribute {
    fn from(v: &str) -> Self {
        let v = cast_to_u8_vec_unsafe(String::from(v));

        Self {
            r#type: AttType::String,
            data: Data::from(v),
        }
    }
}

impl From<&[String]> for Attribute {
    fn from(v: &[String]) -> Self {
        Self {
            r#type: AttType::Strings,
            data: Data::from_array(v),
        }
    }
}

impl Into<i64> for Attribute {
    fn into(self) -> i64 {
        debug_assert_eq!(self.r#type, AttType::Int, "type is not Int");
        match self.data.try_into::<i64>() {
            Some(v) => v[0],
            None => unreachable!("guarantee not None, because of from implement"),
        }
    }
}

impl Into<Vec<i64>> for Attribute {
    fn into(self) -> Vec<i64> {
        debug_assert_eq!(self.r#type, AttType::Ints, "type is not Ints");
        match self.data.try_into::<i64>() {
            Some(v) => v,
            None => unreachable!("guarantee not None, because of from implement"),
        }
    }
}

impl Into<f32> for Attribute {
    fn into(self) -> f32 {
        debug_assert_eq!(self.r#type, AttType::Float, "type is not Float");
        match self.data.try_into::<f32>() {
            Some(v) => v[0],
            None => unreachable!("guarantee not None, because of from implement"),
        }
    }
}

impl Into<Vec<f32>> for Attribute {
    fn into(self) -> Vec<f32> {
        debug_assert_eq!(self.r#type, AttType::Floats, "type is not Floats");
        match self.data.try_into::<f32>() {
            Some(v) => v,
            None => unreachable!("guarantee not None, because of from implement"),
        }
    }
}

impl Into<String> for Attribute {
    fn into(self) -> String {
        debug_assert_eq!(self.r#type, AttType::String, "type is not String");
        match self.data.try_into_own_data() {
            Some(v) => cast_to_string_unsafe(v),
            None => unreachable!("guarantee not None, because of from implement"),
        }
    }
}

impl Into<Vec<String>> for Attribute {
    fn into(self) -> Vec<String> {
        debug_assert_eq!(self.r#type, AttType::Strings, "type is not Strings");
        match self.data.try_into::<String>() {
            Some(v) => v,
            None => unreachable!("guarantee not None, because of from implement"),
        }
    }
}

impl Attribute {
    pub fn r#type(&self) -> AttType {
        self.r#type
    }

    pub fn from_vec_u8_as_string(v: Vec<u8>) -> Self {
        Self {
            r#type: AttType::String,
            data: Data::from(v),
        }
    }

    pub fn from_vec_u8_as_strings(v: Vec<Vec<u8>>) -> Self {
        let value = v.into_iter().flatten().collect();
        Self {
            r#type: AttType::Strings,
            data: Data::from(value),
        }
    }
}

// 将String转为u8数组，用于存储
fn cast_to_u8_vec_unsafe(mut v: String) -> Vec<u8> {
    let length = v.len();
    let capacity = v.capacity();
    let ptr = v.as_mut_ptr();

    // 不再执行v的析构
    forget(v);
    // 重新创建Vec<u8>数组
    unsafe { Vec::from_raw_parts(ptr, length, capacity) }
}

fn cast_to_string_unsafe(mut v: Vec<u8>) -> String {
    let length = v.len();
    let capacity = v.capacity();
    let ptr = v.as_mut_ptr();

    // 不再执行v的析构
    forget(v);
    // 重新创建Vec<u8>数组
    unsafe { String::from_raw_parts(ptr, length, capacity) }
}

#[cfg(test)]
mod tests {
    use std::u8;

    use super::*;

    #[test]
    fn it_works() {
        let attr = Attribute::from(-1);
        let v: i64 = attr.into();
        assert_eq!(-1, v);

        let attr = Attribute::from(&[-1_i64, 1_i64] as &[i64]);
        let v: Vec<i64> = attr.into();
        assert_eq!(&[-1, 1], v.as_slice());

        let attr = Attribute::from(-1.0);
        let v: f32 = attr.into();
        assert_eq!(-1.0, v);

        let a: &[f32] = &[1.0, 2.0];
        let attr = Attribute::from(a);
        let v: Vec<f32> = attr.into();
        assert_eq!(a, v.as_slice());

        let attr = Attribute::from("test");
        let v: String = attr.into();
        assert_eq!("test", v);

        let str = String::from("value");
        let attr = Attribute::from_vec_u8_as_string(cast_to_u8_vec_unsafe(str));
        let v: String = attr.into();
        assert_eq!("value", v);

        let attr = Attribute::from(&[String::from("1234"), String::from("123")] as &[String]);
        let v: Vec<String> = attr.into();
        assert_eq!(&[String::from("1234"), String::from("123")], v.as_slice());

        let v1: Vec<u8> = cast_to_u8_vec_unsafe(String::from("value1"));
        let v2: Vec<u8> = cast_to_u8_vec_unsafe(String::from("value2"));
        let mut v: Vec<Vec<u8>> = vec![];
        v.push(v1);
        v.push(v2);
        let attr = Attribute::from_vec_u8_as_strings(v);
        if let Some(u) = attr.data.try_into_own_data() {
            let vs = cast_to_string_unsafe(u);
            assert_eq!(String::from("value1value2"), vs);
        };
    }
}
