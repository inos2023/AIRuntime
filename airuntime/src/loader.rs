mod transform;

use anyhow::*;
use log::*;

use prost::Message;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::pb;
use model::graph::*;

pub fn load(model_file: &str) -> Result<Graph> {
    let model_file = Path::new(model_file);
    let graph = parser_model(model_file)?;

    Ok(graph)
}

fn parser_model(model_file: &Path) -> Result<Graph> {
    if !model_file.exists() {
        warn!("{:?}模型文件不存在", model_file.to_str());
        return Err(anyhow!("模型文件不存在"));
    }
    let mut model_path = PathBuf::new();
    model_path.push(&model_file);

    if let Some(dir) = model_path.parent() {
        let proto = parser_proto(model_path.as_path())?;
        let graph = parser(&proto, dir)?;

        return Ok(graph);
    };

    Err(anyhow!("解析失败"))
}

fn parser_proto(model_file: &Path) -> Result<pb::ModelProto> {
    let map = unsafe { memmap2::Mmap::map(&fs::File::open(model_file)?)? };
    let pb = crate::pb::ModelProto::decode(&*map)?;
    Ok(pb)
}

fn parser(proto: &pb::ModelProto, dir: &Path) -> Result<Graph> {
    let onnx_operator_set_version = proto
        .opset_import
        .iter()
        .find(|import| import.domain.is_empty() || import.domain == "ai.onnx")
        .map(|op| op.version)
        .unwrap_or(0);
    let pbgraph = proto
        .graph
        .as_ref()
        .ok_or_else(|| anyhow!("model proto does not contain a graph"))?;
    debug!("ONNX operator set version: {:?}", onnx_operator_set_version);
    // if onnx_operator_set_version != 0 && !(9..19).contains(&onnx_operator_set_version) {
    //     warn!("ONNX operator for your model is {}, tract is only tested against \
    //           operator set 9 to 18 (included). Your model may still work so this is not a hard fail.",
    //           onnx_operator_set_version);
    // }
    let ctx = ParsingContext {
        // framework: self,
        model: proto,
        parent_graphs: vec![],

        subgraph: vec![],
        onnx_operator_set_version,
        model_path: dir.to_str(),
        // device_type: self.device_type,
        // device_options: vec![],
        // symbol_table: symbol_table.clone(),
    };
    trace!("created ParsingContext");
    ctx.parse_graph(pbgraph)
}

#[derive(Clone)]
struct ParsingContext<'a> {
    pub onnx_operator_set_version: i64,
    pub subgraph: Vec<&'a Graph>,
    pub model: &'a pb::ModelProto,
    pub parent_graphs: Vec<&'a Graph>,
    pub model_path: Option<&'a str>,
    // pub device_type: DeviceType,
    // 可选设备列表
    // pub device_options: Vec<i32>,
    // pub symbol_table: SymbolTable,
}

impl<'a> ParsingContext<'a> {
    pub fn parse_graph(&self, pbgraph: &pb::GraphProto) -> Result<Graph> {
        // let ctx = self.clone();

        let mut graph = Graph::new(&pbgraph.name);
        // graph.name = pbgraph.name.clone();

        //遍历构建所有初始化张量的Map
        let mut initializers = HashMap::new();
        for t in pbgraph.initializer.iter() {
            let tensor = transform::trans_tensor(t, self.model_path)?;
            initializers.insert(t.name.clone(), tensor);
        }

        // 获取张量形状信息
        let mut value_infos = HashMap::new();
        // 获取中间张量的形状信息
        pbgraph.value_info.iter().for_each(|v| {
            let vi = transform::trans_valueinfo(v);
            value_infos.insert(v.name.clone(), vi);
        });
        // 获取输入的张量形状信息
        pbgraph.input.iter().for_each(|v| {
            let vi = transform::trans_valueinfo(v);
            value_infos.insert(v.name.clone(), vi);
        });
        // 获取输出的张量形状信息
        pbgraph.output.iter().for_each(|v| {
            let vi = transform::trans_valueinfo(v);
            value_infos.insert(v.name.clone(), vi);
        });

        //构建node
        for (i, pbnode) in pbgraph.node.iter().enumerate() {
            let name = if !pbnode.name.is_empty() {
                // pbnode.name.to_string().replace("/", "_")
                pbnode.name.to_string()
            } else if pbnode.output.len() > 0 && !pbnode.output[0].is_empty() {
                pbnode.output[0].to_owned()
            } else {
                // ??
                format!("{}-{}", i, pbnode.op_type)
            };
            trace!("Creating op {}", name);

            // graph.add_op(name, pbnode, &initializers, &value_infos)?;
            let op = transform::build_op(name, pbnode, &mut initializers, &mut value_infos)?;
            graph = graph.add_operator(op).unwrap();
        }

        // // 构建graph input
        // let gi = Tensor::trans_valueinfo(&pbgraph.input.get(0).unwrap())?;
        // // 构建graph output
        // let go = Tensor::trans_valueinfo(&pbgraph.output.get(0).unwrap())?;
        // graph.inputs.push(gi);
        // graph.outputs.push(go);
        Ok(graph)
    }
}
