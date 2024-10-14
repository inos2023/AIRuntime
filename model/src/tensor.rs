use super::data::Data;
use derive::{FromCode, GetCode};
use std::any::type_name;
use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct Tensor {
    /// 名字
    name: String,
    /// 形状
    shape: Shape,
    /// 布局格式
    format: Format,
    /// 数据类型
    dtype: DType,
    /// 数据
    data: Data,
    /// 类型
    r#type: Type,
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor {{ name: \"{}\", shape: {}, format: {:?}, dtype: {:?}, data: {}, type: {:?} }}",
            self.name, self.shape, self.format, self.dtype, self.data, self.r#type
        )
    }
}

impl Tensor {
    pub fn new(name: &str, format: Format, dtype: DType, r#type: Type) -> Self {
        Self {
            name: String::from(name),
            shape: Shape::new(),
            format,
            dtype,
            data: Data::new(),
            r#type,
        }
    }

    pub fn new_with_shape(
        name: &str,
        shape: &[u32],
        format: Format,
        dtype: DType,
        r#type: Type,
    ) -> Self {
        Self {
            name: String::from(name),
            shape: Shape::from(shape),
            format,
            dtype,
            data: Data::new(),
            r#type,
        }
    }

    pub fn set_array1<T: Clone>(&mut self, vs: &[T]) {
        debug_assert!(
            self.type_match::<T>(),
            "Type not match, need {}, but {}",
            self.dtype(),
            type_name::<T>()
        );
        self.shape = Shape::from(&[vs.len() as u32]);

        self.data = Data::from_array(vs);
    }

    pub fn set_array2<T, A, B>(&mut self, vs: A)
    where
        A: AsRef<[B]>,
        B: AsRef<[T]> + Clone,
    {
        debug_assert!(
            self.type_match::<T>(),
            "Type not match, need {}, but {}",
            self.dtype(),
            type_name::<T>()
        );
        let a = vs.as_ref();
        let b = a[0].as_ref();
        self.shape = Shape::from(&[a.len() as u32, b.len() as u32]);

        self.data = Data::from_array(a);
    }

    pub fn set_array3<T, A, B, C>(&mut self, vs: A)
    where
        A: AsRef<[B]>,
        B: AsRef<[C]> + Clone,
        C: AsRef<[T]>,
    {
        debug_assert!(
            self.type_match::<T>(),
            "Type not match, need {}, but {}",
            self.dtype(),
            type_name::<T>()
        );
        let a = vs.as_ref();
        let b = a[0].as_ref();
        let c = b[0].as_ref();
        self.shape = Shape::from(&[a.len() as u32, b.len() as u32, c.len() as u32]);

        self.data = Data::from_array(a);
    }

    pub fn set_array4<T, A, B, C, D>(&mut self, vs: A)
    where
        A: AsRef<[B]>,
        B: AsRef<[C]> + Clone,
        C: AsRef<[D]>,
        D: AsRef<[T]>,
    {
        debug_assert!(
            self.type_match::<T>(),
            "Type not match, need {}, but {}",
            self.dtype(),
            type_name::<T>()
        );
        let a = vs.as_ref();
        let b = a[0].as_ref();
        let c = b[0].as_ref();
        let d = c[0].as_ref();
        self.shape = Shape::from(&[
            a.len() as u32,
            b.len() as u32,
            c.len() as u32,
            d.len() as u32,
        ]);

        self.data = Data::from_array(a);
    }

    pub fn set_array5<T, A, B, C, D, E>(&mut self, vs: A)
    where
        A: AsRef<[B]>,
        B: AsRef<[C]> + Clone,
        C: AsRef<[D]>,
        D: AsRef<[E]>,
        E: AsRef<[T]>,
    {
        debug_assert!(
            self.type_match::<T>(),
            "Type not match, need {}, but {}",
            self.dtype(),
            type_name::<T>()
        );
        let a = vs.as_ref();
        let b = a[0].as_ref();
        let c = b[0].as_ref();
        let d = c[0].as_ref();
        let e = d[0].as_ref();
        self.shape = Shape::from(&[
            a.len() as u32,
            b.len() as u32,
            c.len() as u32,
            d.len() as u32,
            e.len() as u32,
        ]);

        self.data = Data::from_array(a);
    }

    pub fn set_vec<T>(&mut self, data: Vec<T>) {
        debug_assert!(
            self.type_match::<T>(),
            "Type not match, need {}, but {}",
            self.dtype(),
            type_name::<T>()
        );
        debug_assert_eq!(data.len(), self.shape.len());

        self.data = Data::from_vec(data);
    }

    pub fn set_vec_u8(&mut self, data: Vec<u8>, dtype: DType) {
        debug_assert_eq!(
            dtype,
            self.dtype(),
            "Type not match, need {}, but {}",
            self.dtype(),
            dtype
        );
        self.data = Data::from(data);
    }

    // 不持有ptr的所有权，需要自己确保在使用时ptr指向的内容是有效的
    pub fn set_data(&mut self, ptr: *mut u8, length: usize, location: Location) {
        debug_assert_eq!(
            self.shape.len() * self.dtype.size_of(),
            length,
            "Data length({length}) not match shape len({}).",
            self.shape.len() * self.dtype.size_of()
        );
        self.data = Data::from_ptr(ptr, length, location);
    }

    // 移交data所有权
    pub fn into_vec(self) -> Option<Vec<u8>> {
        self.data.try_into::<u8>()
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn format(&self) -> Format {
        self.format
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn r#type(&self) -> Type {
        self.r#type
    }

    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    pub fn data_ptr(&self) -> *mut u8 {
        self.data.ptr()
    }

    pub fn location(&self) -> Location {
        self.data.location()
    }

    fn type_match<T>(&self) -> bool {
        self.dtype.type_name() == type_name::<T>()
    }
}

#[derive(Debug, Clone)]
pub struct Shape {
    // 维度
    dim: usize,
    // 维度数据
    data: [u32; 8],
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data())
    }
}

impl Shape {
    pub fn new() -> Self {
        let data = [0; 8];
        Self { dim: 0, data: data }
    }

    pub fn from(data: &[u32]) -> Self {
        let mut shape = Self::new();
        shape.set_data(data);
        return shape;
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn data(&self) -> &[u32] {
        &self.data[0..self.dim]
    }

    pub fn set_data(&mut self, data: &[u32]) {
        assert!(data.len() <= 8, "dim need no more than 8");
        self.dim = data.len();
        for i in 0..data.len() {
            self.data[i] = data[i];
        }
    }

    pub fn len(&self) -> usize {
        if self.dim == 0 {
            return 0;
        }
        let mut len = 1;
        for i in 0..self.dim {
            len *= self.data[i];
        }
        len as usize
    }
}

/// Tensor数据布局
#[derive(Debug, Clone, Copy, PartialEq, GetCode, FromCode, Default)]
pub enum Format {
    #[code(1)]
    #[default]
    NCHW,
    #[code(2)]
    NHWC,
    #[code(3)]
    CHWN,
    #[code(4)]
    HWCN,
    #[code(5)]
    NDHWC,
    #[code(6)]
    NCDHW,
}

/// Tensor数据类型
#[derive(Debug, Clone, Copy, PartialEq, GetCode, FromCode)]
pub enum DType {
    #[code(0)]
    Undefined,
    #[code(1)]
    Float32,
    #[code(2)]
    Uint8,
    #[code(3)]
    Int8,
    #[code(4)]
    Uint16,
    #[code(5)]
    Int16,
    #[code(6)]
    Int32,
    #[code(7)]
    Int64,
    #[code(8)]
    String,
    #[code(9)]
    Bool,
    #[code(10)]
    Float16,
    #[code(11)]
    Float64,
    #[code(12)]
    Uint32,
    #[code(13)]
    Uint64,
    #[code(14)]
    Complex64,
    #[code(15)]
    Complex128,
    #[code(16)]
    Bfloat16,
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.type_name())
    }
}

impl DType {
    pub const fn size_of(&self) -> usize {
        match self {
            Self::Float16 => 2,
            Self::Float32 => 4,
            Self::Float64 => 8,
            Self::Uint8 => 1,
            Self::Uint16 => 2,
            Self::Uint32 => 4,
            Self::Uint64 => 8,
            Self::Int8 => 1,
            Self::Int16 => 2,
            Self::Int32 => 4,
            Self::Int64 => 8,
            _ => panic!("Not support dtype now"),
        }
    }

    pub fn type_name(&self) -> &str {
        match self {
            Self::Float16 => "f16",
            Self::Float32 => type_name::<f32>(),
            Self::Float64 => type_name::<f64>(),
            Self::Uint8 => type_name::<u8>(),
            Self::Uint16 => type_name::<u16>(),
            Self::Uint32 => type_name::<u32>(),
            Self::Uint64 => type_name::<u64>(),
            Self::Int8 => type_name::<i8>(),
            Self::Int16 => type_name::<i16>(),
            Self::Int32 => type_name::<i32>(),
            Self::Int64 => type_name::<i64>(),
            _ => panic!("Not support dtype now"),
        }
    }
}

/// Tensor数据存储位置
#[derive(Debug, Clone, Copy, PartialEq, GetCode, FromCode)]
pub enum Location {
    #[code(0)]
    Host,
    #[code(1)]
    Device,
}

/// Tensor类型
#[derive(Debug, Clone, Copy, PartialEq, GetCode, FromCode)]
pub enum Type {
    // 变量类型，默认值
    #[code(0)]
    Variable,
    // 常量类型
    #[code(1)]
    Constant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_works() {
        let shape = Shape::from(&[2, 2, 3]);
        assert_eq!(3, shape.dim());
        assert_eq!(&[2, 2, 3], shape.data());
        assert_eq!(format!("{shape}"), "[2, 2, 3]");
        let shape = Shape::new();
        assert_eq!(0, shape.dim());
        assert_eq!(&[0; 0], shape.data());
        assert_eq!(format!("{shape}"), "[]");
    }

    #[test]
    #[should_panic]
    fn shape_panic() {
        Shape::from(&[2, 2, 3, 2, 2, 3, 2, 2, 3]);
    }

    #[test]
    fn tensor_works() {
        let mut tensor = Tensor::new("name", Format::CHWN, DType::Int32, Type::Constant);
        tensor.set_array1(&[1]);
        assert_eq!(1, tensor.shape().dim());

        let mut tensor = Tensor::new("name", Format::CHWN, DType::Int32, Type::Constant);
        tensor.set_array2(&[[1]]);
        assert_eq!(2, tensor.shape().dim());

        let mut tensor = Tensor::new("name", Format::CHWN, DType::Int32, Type::Constant);
        tensor.set_array3(&[[[1]]]);
        assert_eq!(3, tensor.shape().dim());

        let mut tensor = Tensor::new("name", Format::CHWN, DType::Int32, Type::Constant);
        tensor.set_array4(&[[[[1]]]]);
        assert_eq!(4, tensor.shape().dim());

        let mut tensor = Tensor::new("name", Format::CHWN, DType::Int32, Type::Constant);
        tensor.set_array5(&[[[[[1]]]]]);
        assert_eq!(5, tensor.shape().dim());

        let mut tensor =
            Tensor::new_with_shape("name", &[2, 2], Format::CHWN, DType::Int32, Type::Constant);
        tensor.set_vec(vec![1, 2, 3, 4]);
        assert_eq!(2, tensor.shape().dim());
    }
}
