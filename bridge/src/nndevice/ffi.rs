use super::Error;
use model::attribute::Attribute;
use model::graph::Graph;
use model::operator::Operator;
use model::tensor::Tensor;

#[cxx::bridge(namespace = "inos::aichip::nndevice")]
pub mod ffi {
    struct RustOptions {
        keys: Vec<String>,
        values: Vec<String>,
    }

    struct RustTensor {
        name: String,
        dtype: u32,
        dims: Vec<u32>,
        layout: u32,
        data: *mut u8,
        len: usize,
    }

    // 暴露Rust接口到C++
    #[namespace = "inos::aichip::nndevice::bridge"]
    extern "Rust" {
        type GraphWrapper<'a>;
        type OperatorWrapper<'a>;
        type TensorWrapper<'a>;
        type AttributeWrapper<'a>;

        #[rust_name = "graph_name"]
        unsafe fn GraphName<'a>(self: &'a GraphWrapper) -> &'a String;
        #[rust_name = "graph_all_operators"]
        unsafe fn GraphAllOperators<'a>(self: &'a GraphWrapper) -> Vec<OperatorWrapper<'a>>;
        #[rust_name = "name"]
        unsafe fn Name<'a>(self: &'a OperatorWrapper) -> &'a String;
        #[rust_name = "type1"]
        unsafe fn Type<'a>(self: &'a OperatorWrapper) -> &'a String;
        #[rust_name = "inputs"]
        unsafe fn Inputs<'a>(self: &'a OperatorWrapper) -> Vec<TensorWrapper<'a>>;
        #[rust_name = "outputs"]
        unsafe fn Outputs<'a>(self: &'a OperatorWrapper) -> Vec<TensorWrapper<'a>>;
        #[rust_name = "attributes"]
        unsafe fn Attributes<'a>(self: &'a OperatorWrapper) -> Vec<AttributeWrapper<'a>>;
        #[rust_name = "name"]
        unsafe fn Name<'a>(self: &'a TensorWrapper) -> &'a String;
        #[rust_name = "tag"]
        unsafe fn Tag<'a>(self: &'a TensorWrapper) -> &'a String;
        #[rust_name = "dtype"]
        unsafe fn Dtype<'a>(self: &'a TensorWrapper) -> u32;
        #[rust_name = "dims"]
        unsafe fn Dims<'a>(self: &'a TensorWrapper) -> Vec<u32>;
        #[rust_name = "layout"]
        unsafe fn Layout<'a>(self: &'a TensorWrapper) -> u32;
        #[rust_name = "data"]
        unsafe fn Data<'a>(self: &'a TensorWrapper) -> *const u8;
        #[rust_name = "data_len"]
        unsafe fn DataLen<'a>(self: &'a TensorWrapper) -> usize;
        #[rust_name = "name"]
        unsafe fn Name<'a>(self: &'a AttributeWrapper) -> &'a String;
        #[rust_name = "type1"]
        unsafe fn Type<'a>(self: &'a AttributeWrapper) -> u32;
        #[rust_name = "as_int"]
        unsafe fn AsInt<'a>(self: &'a AttributeWrapper) -> i64;
        #[rust_name = "as_ints"]
        unsafe fn AsInts<'a>(self: &'a AttributeWrapper) -> Vec<i64>;
        #[rust_name = "as_float"]
        unsafe fn AsFloat<'a>(self: &'a AttributeWrapper) -> f32;
        #[rust_name = "as_floats"]
        unsafe fn AsFloats<'a>(self: &'a AttributeWrapper) -> Vec<f32>;
        #[rust_name = "as_string"]
        unsafe fn AsString<'a>(self: &'a AttributeWrapper) -> String;
        #[rust_name = "as_strings"]
        unsafe fn AsStrings<'a>(self: &'a AttributeWrapper) -> Vec<String>;

        #[cxx_name = "RustCompileCallback"]
        type CompileCallback;
        #[cxx_name = "RustExecuteCallback"]
        type ExecuteCallback;
    }

    // 暴露C++接口到Rust
    unsafe extern "C++" {
        include!("nndevice.h");
        include!("inos/ai_chip/nn_device/context.h");

        pub fn GetCandidateBackends() -> Vec<String>;

        #[cxx_name = "Context"]
        type CxxContext;
        pub fn CreateContext(
            backend_id: &str,
            dev_id: i32,
            opts: &RustOptions,
        ) -> Result<UniquePtr<CxxContext>>;
        pub fn DestoryContext(ctx: UniquePtr<CxxContext>) -> Result<()>;
        pub fn CompileGraph(
            ctx: &UniquePtr<CxxContext>,
            graph: &GraphWrapper,
            cb: fn(Box<CompileCallback>, i32),
            rust_cb: Box<CompileCallback>,
        ) -> Result<()>;
        pub fn Execute(
            ctx: &UniquePtr<CxxContext>,
            inputs: &Vec<TensorWrapper>,
            cb: fn(Box<ExecuteCallback>, &Vec<RustTensor>, i32),
            rust_cb: Box<ExecuteCallback>,
        ) -> Result<()>;
    }
}

pub struct CompileCallback {
    pub cb: Box<dyn FnOnce(Result<(), Error>)>,
}
impl CompileCallback {
    pub fn new(cb: impl FnOnce(Result<(), Error>) + 'static) -> Self {
        Self { cb: Box::new(cb) }
    }
}

pub struct ExecuteCallback {
    pub cb: Box<dyn FnOnce(Result<Vec<Tensor>, Error>)>,
}
impl ExecuteCallback {
    pub fn new(cb: impl FnOnce(Result<Vec<Tensor>, Error>) + 'static) -> Self {
        Self { cb: Box::new(cb) }
    }
}

pub struct GraphWrapper<'a> {
    pub graph: &'a Graph,
}
impl<'a> GraphWrapper<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        Self { graph }
    }

    fn graph_name(&'a self) -> &'a String {
        &self.graph.name()
    }

    fn graph_all_operators(&'a self) -> Vec<OperatorWrapper<'a>> {
        self.graph
            .operators()
            .into_iter()
            .map(|op| OperatorWrapper::new(op))
            .collect()
    }
}

pub struct TensorWrapper<'a> {
    pub tag: &'a String,
    pub tensor: &'a Tensor,
}
impl<'a> TensorWrapper<'a> {
    pub fn new(tag: &'a String, tensor: &'a Tensor) -> Self {
        Self { tag, tensor }
    }

    fn name(&'a self) -> &'a String {
        self.tensor.name()
    }

    fn tag(&'a self) -> &'a String {
        self.tag
    }

    fn dtype(&'a self) -> u32 {
        self.tensor.dtype().get_code()
    }

    fn dims(&'a self) -> Vec<u32> {
        Vec::from(self.tensor.shape().data())
    }

    fn layout(&'a self) -> u32 {
        self.tensor.format().get_code()
    }

    fn data(&'a self) -> *const u8 {
        self.tensor.data_ptr()
    }

    fn data_len(&'a self) -> usize {
        self.tensor.data_len()
    }
}

pub struct OperatorWrapper<'a> {
    pub op: &'a Operator,
}
impl<'a> OperatorWrapper<'a> {
    pub fn new(op: &'a Operator) -> Self {
        Self { op }
    }

    fn name(&'a self) -> &'a String {
        self.op.name()
    }

    fn type1(&'a self) -> &'a String {
        self.op.r#type()
    }

    fn inputs(&'a self) -> Vec<TensorWrapper<'a>> {
        self.op
            .inputs()
            .into_iter()
            .map(|v| TensorWrapper::new(v.0, v.1))
            .collect()
    }

    fn outputs(&'a self) -> Vec<TensorWrapper<'a>> {
        self.op
            .outputs()
            .into_iter()
            .map(|v| TensorWrapper::new(v.0, v.1))
            .collect()
    }

    fn attributes(&'a self) -> Vec<AttributeWrapper<'a>> {
        self.op
            .attributes()
            .into_iter()
            .map(|v| AttributeWrapper::new(v.0, v.1))
            .collect()
    }
}

pub struct AttributeWrapper<'a> {
    pub name: &'a String,
    pub attr: &'a Attribute,
    pub iv: i32,
}
impl<'a> AttributeWrapper<'a> {
    pub fn new(name: &'a String, attr: &'a Attribute) -> Self {
        Self { name, attr, iv: 0 }
    }

    fn name(&'a self) -> &'a String {
        &self.name
    }

    fn type1(&'a self) -> u32 {
        self.attr.r#type().get_code()
    }

    fn as_int(&'a self) -> i64 {
        let attr = self.attr.clone();
        attr.into()
    }

    fn as_ints(&'a self) -> Vec<i64> {
        let attr = self.attr.clone();
        attr.into()
    }

    fn as_float(&'a self) -> f32 {
        let attr = self.attr.clone();
        attr.into()
    }

    fn as_floats(&'a self) -> Vec<f32> {
        let attr = self.attr.clone();
        attr.into()
    }

    fn as_string(&'a self) -> String {
        let attr = self.attr.clone();
        attr.into()
    }

    fn as_strings(&'a self) -> Vec<String> {
        let attr = self.attr.clone();
        attr.into()
    }
}
