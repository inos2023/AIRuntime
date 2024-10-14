use bridge::nndevice::engine::Context;
pub struct CompileContext {
    pub ctx: Context,
}

impl CompileContext {
    pub fn new(ctx: Context) -> Self {
        Self { ctx }
    }
}

pub mod Pipline {
    use std::collections::HashMap;

    use crate::{Config, loader};
    use anyhow::*;
    use bridge::nndevice;
    use bridge::nndevice::engine::Context;
    use model::graph::Graph;
    use model::tensor::Tensor;

    pub fn load_model(config: Config) -> Result<Graph> {
        let graph = loader::load(config.model_dir.as_str())?;
        Ok(graph)
    }

    /// 创建context
    pub fn create_context(
        backend_id: &str,
        dev_id: i32,
        opts: &HashMap<String, String>,
    ) -> Result<Context> {
        let ctx = nndevice::engine::create_context(backend_id, dev_id, opts.clone()).unwrap();

        Ok(ctx)
    }
    /// 图编译
    pub fn compile_graph(
        graph: Graph,
        backend: &str,
        dev_id: i32,
        opts: &HashMap<String, String>,
    ) -> Result<Context> {
        // 创建context
        let ctx = create_context(backend, dev_id, opts)?;
        // 图编译
        nndevice::engine::compile_graph(&ctx, graph, |_result| {});
        Ok(ctx)
        // 销毁context
        // destory_context(ctx)
    }
    /// 销毁context
    pub fn destory_context(ctx: Context) -> Result<(), Error> {
        let res = nndevice::engine::destory_context(ctx).unwrap();
        Ok(res)
    }
    /// 执行推理
    pub fn excute(ctx: &Context, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        let mut res = vec![];
        nndevice::engine::excute(&ctx, inputs, |result| unimplemented!("TODO"));
        Ok(res)
    }
}
