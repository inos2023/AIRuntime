use super::ffi::{ffi, CompileCallback, ExecuteCallback, GraphWrapper, TensorWrapper};
use super::{parser_error, parser_error_from_code, Error};
use cxx::UniquePtr;
use model::graph::Graph;
use model::tensor;
use model::tensor::Tensor;
use std::collections::HashMap;

pub type Context = UniquePtr<ffi::CxxContext>;

pub fn create_context(
    backend_id: &str,
    dev_id: i32,
    opts: HashMap<String, String>,
) -> Result<Context, Error> {
    let mut rust_opts = ffi::RustOptions {
        keys: Vec::new(),
        values: Vec::new(),
    };
    for ele in opts {
        rust_opts.keys.push(ele.0);
        rust_opts.values.push(ele.1);
    }

    match ffi::CreateContext(backend_id, dev_id, &rust_opts) {
        Err(e) => match parser_error(e) {
            Some(e) => Err(e),
            None => panic!("Not reachable!"),
        },
        Ok(ctx) => Ok(ctx),
    }
}

pub fn destory_context(ctx: Context) -> Result<(), Error> {
    match ffi::DestoryContext(ctx) {
        Ok(_) => Ok(()),
        Err(e) => match parser_error(e) {
            Some(e) => Err(e),
            None => panic!("Not reachable!"),
        },
    }
}

pub fn compile_graph<C>(ctx: &Context, graph: &Graph, cb: C) -> Result<(), Error>
where
    C: FnOnce(Result<(), Error>) + 'static,
{
    let wrapper = GraphWrapper::new(graph);

    let rust_cb = CompileCallback::new(cb);

    match ffi::CompileGraph(
        ctx,
        &wrapper,
        |rust_cb, rc| {
            let cb = rust_cb.cb;
            cb(match parser_error_from_code(rc) {
                None => Ok(()),
                Some(e) => Err(e),
            });
        },
        Box::new(rust_cb),
    ) {
        Ok(_) => Ok(()),
        Err(e) => match parser_error(e) {
            Some(e) => Err(e),
            None => panic!("Not reachable!"),
        },
    }
}
pub fn excute<C>(ctx: &Context, inputs: &[&Tensor], cb: C) -> Result<(), Error>
where
    C: FnOnce(Result<Vec<Tensor>, Error>) + 'static,
{
    let tag_unuse = String::new();
    let inputs: Vec<TensorWrapper> = inputs
        .iter()
        .map(|input| TensorWrapper::new(&tag_unuse, input))
        .collect();
    let rust_cb = ExecuteCallback::new(cb);

    match ffi::Execute(
        ctx,
        &inputs,
        |rust_cb, outputs, rc| {
            let cb = rust_cb.cb;
            match parser_error_from_code(rc) {
                None => {
                    let outputs = outputs
                        .into_iter()
                        .map(|rust_tensor| {
                            let mut tensor = Tensor::new_with_shape(
                                &rust_tensor.name,
                                rust_tensor.dims.as_slice(),
                                tensor::Format::from_code(rust_tensor.layout),
                                tensor::DType::from_code(rust_tensor.dtype),
                                tensor::Type::Variable,
                            );
                            tensor.set_data(
                                rust_tensor.data,
                                rust_tensor.len,
                                tensor::Location::Host,
                            );
                            tensor
                        })
                        .collect();
                    cb(Ok(outputs));
                }
                Some(e) => cb(Err(e)),
            };
        },
        Box::new(rust_cb),
    ) {
        Ok(_) => Ok(()),
        Err(e) => match parser_error(e) {
            Some(e) => Err(e),
            None => panic!("Not reachable!"),
        },
    }
}
