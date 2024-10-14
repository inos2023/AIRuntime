use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

use log::trace;
use log::{info, warn};
use model::attribute::AttType;
use model::attribute::Attribute;
use model::operator::Operator;
use model::tensor::*;

use crate::pb::{self, type_proto::Value, *};
use anyhow::*;

pub fn trans_tensor(t: &TensorProto, path: Option<&str>) -> Result<Tensor> {
    let dtype = DType::from_code(t.data_type as u32);
    // 标量按一维处理
    let mut shape: Vec<u32> = t.dims.iter().map(|&x| x as u32).collect();
    if t.dims.len() == 0 {
        shape = vec![1];
    }

    let mut tensor = Tensor::new_with_shape(
        &t.name,
        shape.as_slice(),
        Format::default(),
        dtype,
        Type::Constant,
    );
    let is_external = t.data_location.is_some()
        && t.data_location == Some(tensor_proto::DataLocation::External.into());
    if t.raw_data.len() > 0 {
        tensor.set_vec_u8(t.raw_data.to_vec(), dtype);
    } else if is_external {
        if let Some(model_path) = path {
            // external files will be loaded and fed to the tensor if necessary
            info!(
                "number of external file needed for this tensor: {}",
                t.external_data.len()
            );
            let mut tensor_data: Vec<u8> = Vec::new();
            for external_data in t.external_data.iter()
            // according to the onnx format, it is possible to have multiple files for one tensor
            {
                let p = PathBuf::from(format!("{}/{}", model_path, external_data.value));
                info!("external file detected: {:?}", p);
                extend_bytes_from_path(&mut tensor_data, p)?;
                info!("external file loaded");
            }
            tensor.set_vec(tensor_data);
        } else {
            warn!("no model path was specified in the parsing context, yet external data was detected. aborting");
        }
    } else {
        match dtype {
            DType::Float16 => tensor.set_vec(t.float_data.to_vec()),
            DType::Int64 => tensor.set_vec(t.int64_data.to_vec()),
            DType::Int32 => tensor.set_vec(t.int32_data.to_vec()),
            DType::Float32 => tensor.set_vec(t.float_data.to_vec()),
            // TODO 待验证
            // DType::Int16 => tensor.set_vec(t.int32_data),
            // DType::Int8 => tensor.set_vec(t.int32_data),
            // DType::Uint16 => tensor.set_vec(t.int32_data),
            // DType::Uint32 => ptr = t.int32_data.as_ptr() as *mut u8,
            // DType::Uint8 => ptr = t.int32_data.as_ptr() as *mut u8,
            // DType::Uint64 => ptr = t.uint64_data.as_ptr() as *mut u8,
            // DType::Float16 => ptr = t.float_data.as_ptr() as *mut u8,
            // DType::String => ptr = t.string_data.as_ptr() as *mut u8,
            // DType::Double => ptr = t.double_data.as_ptr() as *mut u8,
            // DType::Bool => ptr = t.int32_data.as_ptr() as *mut u8,
            _ => panic!("dtype {} is not exist", dtype),
        };
    }

    Ok(tensor)
}

fn extend_bytes_from_path(buf: &mut Vec<u8>, p: impl AsRef<Path>) -> Result<()> {
    use std::fs;

    let file = fs::File::open(p)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    buf.extend_from_slice(&mmap);
    Ok(())
}

pub fn trans_valueinfo(v: &ValueInfoProto) -> Tensor {
    let mut dims = vec![];
    let mut dtype: DType = DType::Undefined;
    if let Some(t) = &v.r#type {
        if let Some(value) = &t.value {
            let Value::TensorType(tt) = value;
            dtype = DType::from_code(tt.elem_type as u32);
            if let Some(d) = tt.shape.clone() {
                for i in d.dim.iter() {
                    if let Some(y) = &i.value {
                        if let pb::tensor_shape_proto::dimension::Value::DimValue(dv) = y {
                            let udv: u32 = *dv as u32;
                            dims.push(udv);
                        }
                    }
                }
            }
        }
    }

    Tensor::new_with_shape(&v.name, &dims, Format::default(), dtype, Type::Variable)
}

pub fn build_op(
    name: String,
    pbnode: &pb::NodeProto,
    initializers: &mut HashMap<String, Tensor>,
    value_infos: &mut HashMap<String, Tensor>,
) -> Result<Operator> {
    let mut op = Operator::new(&name, &pbnode.op_type);
    //input
    for i in 0..pbnode.input.len() {
        let iname = pbnode.input.get(i).unwrap();
        let tag = i.to_string();

        // init constant tensor
        if let Some(tensor) = initializers.remove(iname) {
            op = op.add_input(&tag, tensor).unwrap();
            trace!("op {} add input tensor {}", name, iname);
        } else {
            // input variable edge
            if value_infos.contains_key(iname) {
                // 因为value_infos存的Tensor只有描述信息，没有数据，
                // 这里通过clone复制一个Tensor，不会增加很大的内存开销
                // 如果move的方式，将导致只有input或者output能拿到正确的信息
                if let Some(iv) = value_infos.get(iname) {
                    op = op.add_input(&tag, iv.clone()).unwrap();
                }
            } else {
                // 对于既不是initializer中，又不存在value_infos中的输入，可能是一个非法的ONNX模型
                panic!("Invalid ONNX model, Undefine input Tensor {} for op {}, create a default Tensor with none shape.", iname, name);
            }
        }
    }
    // output
    for i in 0..pbnode.output.len() {
        let oname = pbnode.output.get(i).unwrap();
        let tag = i.to_string();

        if value_infos.contains_key(oname) {
            // 因为value_infos存的Tensor只有描述信息，没有数据，
            // 这里通过clone复制一个Tensor，不会增加很大的内存开销
            // 如果move的方式，将导致只有input或者output能拿到正确的信息
            if let Some(ov) = value_infos.get(oname) {
                op = op.add_output(&tag, ov.clone()).unwrap();
            }
        } else {
            // 对于存在value_infos中的输出，可能是一个非法的ONNX模型
            panic!("Invalid ONNX model, Undefine output Tensor {} for op {}, create a default Tensor with none shape.", oname, name);
        }
    }

    //attributes
    for a in &pbnode.attribute {
        let attr = trans_attr(a)?;
        op = op.add_attribute(&a.name, attr).unwrap();
    }

    Ok(op)
}

fn trans_attr(a: &AttributeProto) -> Result<Attribute> {
    let tp = AttType::from_code(a.r#type as u32);
    let attr = match tp {
        AttType::Floats => Attribute::from(a.floats.as_slice()),
        AttType::Ints => Attribute::from(a.ints.as_slice()),
        // TODO 待验证
        AttType::Strings => Attribute::from_vec_u8_as_strings(a.strings.to_vec()),
        AttType::String => Attribute::from_vec_u8_as_string(a.s.to_vec()),
        AttType::Float => Attribute::from(a.f),
        AttType::Int => Attribute::from(a.i),

        _ => unimplemented!("FIXME, struct tensor loading"),
    };

    Ok(attr)
}
