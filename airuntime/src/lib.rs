pub use anyhow::*;

mod loader;

use bridge::nndevice::{self, engine};
use log::error;
use model::graph::Graph;
use model::tensor::Tensor;
use std::collections::HashMap;

#[allow(clippy::all)]
pub mod pb {
    include!("prost/onnx.rs");
}

#[derive(Clone, Debug, Default)]
pub struct Config {
    // pub op_register: OnnxOpRegister,
    // pub ignore_output_shapes: bool,
    // pub ignore_output_types: bool,
    /// 模型路径
    pub model_dir: String,
    // // 模型类型
    // pub model_type: ModelType,
    /// 目标硬件类型
    pub backend: String,
    /// 设备id
    pub device_id: i32,
    // pub device_type: DeviceType,
    // /// 精度类型
    // pub precision: PrecisionType,
    pub ops: HashMap<String, String>,
}

/// 模型推理上下文
pub struct Context {
    pub graph: Graph,
    pub bridge_ctx: engine::Context,
}

pub fn get_candidate_backends() -> Result<Vec<String>> {
    Ok(nndevice::get_candidate_backends())
}

pub fn load<C>(config: &Config, cb: C) -> Result<Context>
where
    C: FnOnce(Result<()>) + 'static,
{
    let graph = loader::load(config.model_dir.as_str())?;

    let ctx: Context = Context {
        graph,
        bridge_ctx: engine::create_context(&config.backend, config.device_id, config.ops.clone())?,
    };
    engine::compile_graph(&ctx.bridge_ctx, &ctx.graph, |r| {
        match r {
            std::result::Result::Ok(_) => cb(Ok(())),
            std::result::Result::Err(e) => {
                error!("模型编译失败, {}", e);
                cb(Err(anyhow!("模型编译失败")));
            }
        };
    })?;

    Ok(ctx)
}

pub fn run<C>(ctx: &Context, inputs: &[&Tensor], cb: C) -> Result<()>
where
    C: FnOnce(Result<Vec<Tensor>>) + 'static,
{
    engine::excute(&ctx.bridge_ctx, inputs, |r| match r {
        std::result::Result::Ok(outputs) => cb(Ok(outputs)),
        std::result::Result::Err(e) => {
            error!("模型编译失败, {}", e);
            cb(Err(anyhow!("模型编译失败")));
        }
    })?;

    Ok(())
}

pub fn destory_context(ctx: Context) -> Result<()> {
    engine::destory_context(ctx.bridge_ctx)?;

    Ok(())
}
