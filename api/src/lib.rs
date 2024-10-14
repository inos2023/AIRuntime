use std::{collections::HashMap, ffi::*, mem::forget, ptr::null, ptr::null_mut};

use derive::{FromCode, GetCode};
use model::tensor;

// 对外暴露的类型
/// 配置信息
pub type Config = airuntime::Config;
/// 模型推理上下文
pub type Context = airuntime::Context;
/// 张量
pub type Tensor = model::tensor::Tensor;
/// 模型加载回调
pub type LoadCallback = unsafe extern "C" fn(AiruntimeErrCode, *mut c_void);
/// 模型推理回调
pub type RunCallback = unsafe extern "C" fn(*mut TensorVec, c_uint, AiruntimeErrCode, *mut c_void);
/// \0结尾的String
pub type CString = std::ffi::CString;

#[repr(C)]
pub enum AiruntimeErrCode {
    Ok = 0,
    Error = -1,
    InvalidParam = -2,
}

/// 数据布局
#[repr(C)]
#[derive(GetCode, FromCode)]
pub enum TensorFormat {
    #[code(1)]
    NCHW = 1,
    #[code(2)]
    NHWC = 2,
    #[code(3)]
    CHWN = 3,
    #[code(4)]
    HWCN = 4,
    #[code(5)]
    NDHWC = 5,
    #[code(6)]
    NCDHW = 6,
}

/// Tensor数据类型
#[repr(C)]
#[derive(GetCode, FromCode)]
pub enum TensorDType {
    #[code(0)]
    Undefined = 0,
    #[code(1)]
    Float32 = 1,
    #[code(2)]
    Uint8 = 2,
    #[code(3)]
    Int8 = 3,
    #[code(4)]
    Uint16 = 4,
    #[code(5)]
    Int16 = 5,
    #[code(6)]
    Int32 = 6,
    #[code(7)]
    Int64 = 7,
    #[code(8)]
    String = 8,
    #[code(9)]
    Bool = 9,
    #[code(10)]
    Float16 = 10,
    #[code(11)]
    Float64 = 11,
    #[code(12)]
    Uint32 = 12,
    #[code(13)]
    Uint64 = 13,
    #[code(14)]
    Complex64 = 14,
    #[code(15)]
    Complex128 = 15,
    #[code(16)]
    Bfloat16 = 16,
}

/// 候选后端
pub struct Backends {
    pub backends: Vec<String>,
}

/// Tensor数组
pub struct TensorVec {
    pub vs: Vec<*mut Tensor>,
}

/// 获取后续backend
#[no_mangle]
pub extern "C" fn airuntime_get_candidate_backends(
    backends: *mut *mut Backends,
) -> AiruntimeErrCode {
    if backends.is_null() {
        return AiruntimeErrCode::InvalidParam;
    }
    match airuntime::get_candidate_backends() {
        Ok(ids) => {
            unsafe { *backends = Box::into_raw(Box::new(Backends { backends: ids })) };
            AiruntimeErrCode::Ok
        }
        Err(e) => {
            println!("[E][AiRuntime] -> Get candidate backends failed! {}", e);
            AiruntimeErrCode::Error
        }
    }
}

/// 加载模型
#[no_mangle]
pub extern "C" fn airuntime_load(
    ctx: *mut *mut Context,
    path: *const c_char,
    config: *mut Config,
    cb: LoadCallback,
    user_data: *mut c_void,
) -> AiruntimeErrCode {
    if ctx.is_null() || config.is_null() || path.is_null() {
        return AiruntimeErrCode::InvalidParam;
    }

    let mut config = unsafe { Box::from_raw(config) };
    config.model_dir = String::from(c_char_to_str(path));

    let code = match airuntime::load(&config, move |r| {
        let code = match r {
            Ok(_) => AiruntimeErrCode::Ok,
            Err(e) => {
                println!(
                    "[E][AiRuntime] -> Load model callback result has error! {}",
                    e
                );
                AiruntimeErrCode::Error
            }
        };
        unsafe {
            cb(code, user_data);
        }
    }) {
        Ok(c) => {
            let c = Box::new(c);
            unsafe { *ctx = Box::into_raw(c) };
            AiruntimeErrCode::Ok
        }
        Err(e) => {
            println!("[E][AiRuntime] -> Load model failed! {}", e);
            AiruntimeErrCode::Error
        }
    };
    // 确保config不被rust释放
    forget(config);
    code
}

/// 执行推理
#[no_mangle]
pub extern "C" fn airuntime_run(
    ctx: *mut Context,
    inputs: *mut *mut Tensor,
    inputs_len: c_uint,
    cb: RunCallback,
    user_data: *mut c_void,
) -> AiruntimeErrCode {
    if ctx.is_null() || inputs.is_null() {
        return AiruntimeErrCode::InvalidParam;
    }

    let ctx = unsafe { Box::from_raw(ctx) };

    let tensors = unsafe { std::slice::from_raw_parts(inputs, inputs_len as usize) };

    let tensors: Vec<Box<Tensor>> = tensors
        .iter()
        .map(|input| unsafe { Box::from_raw(*input) })
        .collect();
    let inputs: Vec<&Tensor> = tensors.iter().map(|t| t.as_ref()).collect();

    let code = match airuntime::run(ctx.as_ref(), inputs.as_slice(), move |r| {
        match r {
            Ok(outputs) => unsafe {
                let outputs_len = outputs.len();
                let outputs = to_tensor_raw_pointer(outputs);
                println!("outputs: {:?}", outputs);
                cb(
                    outputs,
                    outputs_len as c_uint,
                    AiruntimeErrCode::Ok,
                    user_data,
                );
            },
            Err(e) => {
                println!(
                    "[E][AiRuntime] -> Run model callback result has error! {}",
                    e
                );
                unsafe {
                    cb(std::ptr::null_mut(), 0, AiruntimeErrCode::Error, user_data);
                }
            }
        };
    }) {
        Ok(_) => {
            println!("[I][AiRuntime] -> Run model success!");
            AiruntimeErrCode::Ok
        }
        Err(e) => {
            println!("[E][AiRuntime] -> Run model failed! {}", e);
            AiruntimeErrCode::Error
        }
    };

    // 确保Context和Tensor的内存不被释放
    forget(ctx);
    tensors.iter().for_each(|t| {
        forget(t);
    });

    code
}

/// 销毁上下文
#[no_mangle]
pub extern "C" fn airuntime_context_destory(ctx: *mut Context) -> AiruntimeErrCode {
    let ctx = unsafe { Box::from_raw(ctx) };
    match airuntime::destory_context(*ctx) {
        Ok(_) => AiruntimeErrCode::Ok,
        Err(_) => AiruntimeErrCode::Error,
    }
}

fn to_tensor_raw_pointer(tensors: Vec<Tensor>) -> *mut TensorVec {
    let tensors: Vec<*mut Tensor> = tensors
        .into_iter()
        .map(|t| {
            let r = Box::into_raw(Box::new(t));
            r
        })
        .collect();

    Box::into_raw(Box::new(TensorVec { vs: tensors }))
}

fn config_add_option(config: &mut Config, key: &str, value: &str) {
    config.ops.insert(String::from(key), String::from(value));
}

fn config_get_option<'a>(config: &'a Config, key: &str) -> Option<&'a String> {
    config.ops.get(key)
}

fn c_char_to_str<'a>(c: *const c_char) -> &'a str {
    let c = unsafe { CStr::from_ptr(c) };
    c.to_str().unwrap()
}

/// 创建配置对象
///
/// `backend_id` - 执行的后端推理的ID
/// `de_id` - 实际设备ID
#[no_mangle]
pub extern "C" fn airuntime_config_new(backend_id: *const c_char, dev_id: c_int) -> *mut Config {
    let backend_id = c_char_to_str(backend_id);

    let config = Config {
        model_dir: String::new(),
        backend: String::from(backend_id),
        device_id: dev_id,
        ops: HashMap::new(),
    };
    let config = Box::new(config);
    Box::into_raw(config)
}

/// 销毁配置对象
#[no_mangle]
pub extern "C" fn airuntime_config_destory(config: *mut Config) {
    unsafe {
        drop(Box::from_raw(config));
    }
}

/// 配置对象中添加附加选项
#[no_mangle]
pub extern "C" fn airuntime_config_add_option(
    config: *mut Config,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    let mut config = unsafe { Box::from_raw(config) };
    config_add_option(&mut config, c_char_to_str(key), c_char_to_str(value));
    // 确保config不被rust释放
    forget(config);
    0
}

/// 获取配置对象的Option值
#[no_mangle]
pub extern "C" fn airuntime_config_get_option(
    config: *mut Config,
    key: *const c_char,
) -> *mut CString {
    let config = unsafe { Box::from_raw(config) };

    let ptr = match config_get_option(&config, c_char_to_str(key)) {
        None => null_mut(),
        Some(value) => {
            let value = CString::new(value.as_str()).unwrap();
            Box::into_raw(Box::new(value))
        }
    };
    // 确保config不被rust释放
    forget(config);
    ptr
}

/// 获取backend_id
#[no_mangle]
pub extern "C" fn airuntime_config_get_backend_id(config: *mut Config) -> *mut CString {
    let config = unsafe { Box::from_raw(config) };

    let backend_id = CString::new(config.backend.as_str()).unwrap();

    // 确保不被rust释放
    forget(config);

    Box::into_raw(Box::new(backend_id))
}

/// 获取dev_id
#[no_mangle]
pub extern "C" fn airuntime_config_get_dev_id(config: *mut Config) -> c_int {
    let config = unsafe { Box::from_raw(config) };
    let dev_id = config.device_id;
    // 确保config不被rust释放
    forget(config);
    dev_id
}

/// 打印配置信息
#[no_mangle]
pub extern "C" fn airuntime_config_print(config: *mut Config) {
    let config = unsafe { Box::from_raw(config) };
    println!("[I]{:?}", config);
    // 确保config不被rust释放
    forget(config);
}

/// 创建Tensor
#[no_mangle]
pub extern "C" fn airuntime_tensor_new(
    name: *const c_char,
    dim: c_uint,
    shape: *const c_uint,
    format: TensorFormat,
    dtype: TensorDType,
) -> *mut Tensor {
    let name = c_char_to_str(name);
    let shape = unsafe { std::slice::from_raw_parts(shape, dim as usize) };

    let tensor = Tensor::new_with_shape(
        name,
        shape,
        tensor::Format::from_code(format.get_code()),
        tensor::DType::from_code(dtype.get_code()),
        tensor::Type::Variable,
    );
    let tensor = Box::new(tensor);
    Box::into_raw(tensor)
}

/// 销毁Tensor
#[no_mangle]
pub extern "C" fn airuntime_tensor_destory(tensor: *mut Tensor) {
    unsafe {
        drop(Box::from_raw(tensor));
    }
}

/// 设置数据
/// ptr需要自己释放
#[no_mangle]
pub extern "C" fn airuntime_tensor_set_data(tensor: *mut Tensor, ptr: *mut u8, length: c_uint) {
    let mut tensor = unsafe { Box::from_raw(tensor) };
    tensor.set_data(ptr, length as usize, tensor::Location::Host);
    // 确保不被rust释放
    forget(tensor);
}

/// 获取Tensor name
#[no_mangle]
pub extern "C" fn airuntime_tensor_get_name(tensor: *mut Tensor) -> *mut CString {
    let tensor = unsafe { Box::from_raw(tensor) };

    let name = CString::new(tensor.name().as_str()).unwrap();

    // 确保不被rust释放
    forget(tensor);
    Box::into_raw(Box::new(name))
}

/// 获取Tensor shape
#[no_mangle]
pub extern "C" fn airuntime_tensor_get_shape(
    tensor: *mut Tensor,
    shape: *mut *const c_uint,
) -> c_uint {
    let tensor = unsafe { Box::from_raw(tensor) };
    let len = tensor.shape().dim();
    unsafe { *shape = tensor.shape().data().as_ptr() };
    // 确保不被rust释放
    forget(tensor);
    len as c_uint
}

/// 获取Tensor shape的大小
#[no_mangle]
pub extern "C" fn airuntime_tensor_get_shape_len(tensor: *mut Tensor) -> c_uint {
    let tensor = unsafe { Box::from_raw(tensor) };
    let len = tensor.shape().len();
    // 确保不被rust释放
    forget(tensor);
    len as c_uint
}

#[no_mangle]
pub extern "C" fn airuntime_tensor_get_format(tensor: *mut Tensor) -> TensorFormat {
    let tensor = unsafe { Box::from_raw(tensor) };
    let format = TensorFormat::from_code(tensor.format().get_code());
    // 确保不被rust释放
    forget(tensor);
    format
}

#[no_mangle]
pub extern "C" fn airuntime_tensor_get_dtype(tensor: *mut Tensor) -> TensorDType {
    let tensor = unsafe { Box::from_raw(tensor) };
    let dtype = TensorDType::from_code(tensor.dtype().get_code());
    // 确保不被rust释放
    forget(tensor);
    dtype
}

#[no_mangle]
pub extern "C" fn airuntime_tensor_get_data(tensor: *mut Tensor, data: *mut *mut u8) -> c_uint {
    let tensor = unsafe { Box::from_raw(tensor) };
    unsafe { *data = tensor.data_ptr() };
    let len = tensor.data_len();
    // 确保不被rust释放
    forget(tensor);
    len as c_uint
}

/// 销毁Backends
#[no_mangle]
pub extern "C" fn airuntime_backends_destory(backends: *mut Backends) {
    unsafe {
        drop(Box::from_raw(backends));
    }
}

/// 获取Backends长度
#[no_mangle]
pub extern "C" fn airuntime_backends_get_len(backends: *mut Backends) -> c_uint {
    let backends = unsafe { Box::from_raw(backends) };
    let len = backends.backends.len();
    // 确保不被rust释放
    forget(backends);
    len as c_uint
}

/// 获取元素
#[no_mangle]
pub extern "C" fn airuntime_backends_get(backends: *mut Backends, index: c_uint) -> *const c_char {
    let backends = unsafe { Box::from_raw(backends) };
    let result = match backends.backends.get(index as usize) {
        Some(backend) => backend.as_ptr() as *const c_char,
        None => null(),
    };
    // 确保不被rust释放
    forget(backends);
    result
}

// /// 创建TensorVec
// #[no_mangle]
// pub extern "C" fn airuntime_tensorvec_new() -> *mut TensorVec {
//     let vec = TensorVec {
//         vs: vec![]
//     };
//     Box::into_raw(Box::new(vec))
// }

/// 销毁TensorVec
#[no_mangle]
pub extern "C" fn airuntime_tensorvec_destory(vec: *mut TensorVec) {
    unsafe {
        drop(Box::from_raw(vec));
    }
}

/// 获取TensorVec长度
#[no_mangle]
pub extern "C" fn airuntime_tensorvec_get_len(vec: *mut TensorVec) -> c_uint {
    let vec = unsafe { Box::from_raw(vec) };
    let len = vec.vs.len();
    // 确保不被rust释放
    forget(vec);
    len as c_uint
}

/// 获取Tensor
#[no_mangle]
pub extern "C" fn airuntime_tensorvec_get(vec: *mut TensorVec, index: c_uint) -> *const Tensor {
    let vec = unsafe { Box::from_raw(vec) };
    let result = match vec.vs.get(index as usize) {
        Some(tensor) => *tensor,
        None => null(),
    };
    // 确保不被rust释放
    forget(vec);
    result
}

/// 销毁CString
#[no_mangle]
pub extern "C" fn airuntime_cstring_destory(cstr: *mut CString) {
    unsafe {
        drop(Box::from_raw(cstr));
    }
}

/// 获取
#[no_mangle]
pub extern "C" fn airuntime_cstring_get(cstr: *mut CString) -> *const c_char {
    let cstr = unsafe { Box::from_raw(cstr) };
    let ptr = cstr.as_ptr();
    // 确保不被rust释放
    forget(cstr);
    ptr
}
