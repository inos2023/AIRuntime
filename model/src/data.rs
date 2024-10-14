use std::mem::{forget, size_of};
use std::fmt::Display;
use std::ptr::null_mut;
use super::tensor::Location;

// 将其他类型的数据转为u8数组，转换后的u8数组不能直接使用，需要转回到原来的数据才可以使用
fn cast_to_u8_vec_unsafe<T>(mut v: Vec<T>) -> Vec<u8> {
    let ratio = size_of::<T>() / size_of::<u8>();
    let length = v.len() * ratio;
    let capacity = v.capacity() * ratio;
    let ptr = v.as_mut_ptr() as *mut u8;

    // 不再执行v的析构
    forget(v);
    // 重新创建Vec<u8>数组
    unsafe { Vec::from_raw_parts(ptr, length, capacity) }
}

fn cast_to_t_vec_unsafe<T>(mut v: Vec<u8>) -> Vec<T> {
    let ratio = size_of::<T>() / size_of::<u8>();
    let length = v.len() / ratio;
    let capacity = v.capacity() / ratio;
    let ptr = v.as_mut_ptr() as *mut T;

    // 不再执行v的析构
    forget(v);
    // 重新创建Vec<T>数组
    unsafe { Vec::from_raw_parts(ptr, length, capacity) }
}

#[derive(Debug, Clone)]
pub struct Data {
    location: Location,
    length: usize,
    // 指向数据的堆内存指针
    ptr: *mut u8,
    // 持有所有权，保证raw指针有效
    own_data: Vec<u8>,
    owned: bool,
}

impl Display for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{location: {:?}, ptr: {:p}, length: {}}}", self.location, self.ptr, self.length)
    }
}

impl Data {
    pub fn new() -> Self {
        Self {
            location: Location::Host,
            length: 0,
            ptr: null_mut(),
            own_data: vec![],
            owned: false,
        }
    }

    pub fn from(vs: Vec<u8>) -> Self {
        Self::from_location(vs, Location::Host)
    }

    pub fn from_array<T: Clone>(xs: &[T]) -> Data {
        let v = xs.to_vec();
        Self::from_vec(v)
    }

    pub fn from_vec<T>(v: Vec<T>) -> Data {
        let data = cast_to_u8_vec_unsafe(v);
        Data::from(data)
    }

    pub fn from_ptr(ptr: *mut u8, length: usize, location: Location) -> Self {
        Self { location, length, ptr, own_data: vec![], owned: false }
    }

    pub fn from_location(vs: Vec<u8>, location: Location) -> Self {
        let mut data = Data::new();
        data.set_data(vs);
        data.set_location(location);

        data
    }

    /// 如果原生存储的跟实际转换的类型不一致，将导致不可预知的错误
   pub fn try_into<T>(mut self) -> Option<Vec<T>> {
        if self.owned {
            self.reset();
            Some(cast_to_t_vec_unsafe(self.own_data))
        } else {
            None
        }
    }

    pub fn set_data(&mut self, mut vs: Vec<u8>) {
        self.length = vs.len();
        self.ptr = vs.as_mut_ptr();
        self.own_data = vs;
        self.owned = true;
    }

    pub fn set_location(&mut self, location: Location) {
        self.location = location;
    }

    pub fn location(&self) -> Location {
        self.location
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn try_into_own_data(mut self) -> Option<Vec<u8>> {
        if self.owned {
            self.reset();
            Some(self.own_data)
        } else {
            None
        }
    }

    fn reset(&mut self) {
        self.ptr = null_mut();
        self.length = 0;
        self.owned = false;
    }
}


// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn data_works() {
//         let str = String::from("data");
//         str.as_mut_ptr()
//     }
// }

