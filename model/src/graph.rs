use super::operator::Operator;
use std::collections::HashMap;

use anyhow::{Result, Ok};

#[derive(Debug)]
pub struct Graph {
    /// 图的名字
    name: String,
    /// 模型所有的节点
    operators: HashMap<String, Operator>,
}

impl Graph {
    pub fn new(name: &str) -> Self {
        Self {
            name: String::from(name),
            operators: HashMap::new(),
        }
    }

    pub fn add_operator(mut self, op: Operator) -> Result<Self> {
        self.operators.insert(String::from(op.name()), op);

        Ok(self)
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn operators(&self) -> Vec<&Operator> {
        self.operators.iter().map(|v| v.1).collect()
    }

    pub fn get_operator(&self, name: &str) -> Option<&Operator> {
        self.operators.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::*;
    use crate::tensor::*;
    use crate::attribute::*;

    #[test]
    fn data_works() {
        let graph = Graph::new("graph")
            .add_operator(
                Operator::new("name", "Add")
                    .add_input(
                        "X1",
                        Tensor::new_with_shape(
                            "add1",
                            &[2, 3],
                            Format::CHWN,
                            DType::Int32,
                            Type::Variable,
                        ),
                    ).unwrap()
                    .add_input(
                        "X2",
                        Tensor::new_with_shape(
                            "add2",
                            &[2, 3],
                            Format::CHWN,
                            DType::Int32,
                            Type::Variable,
                        ),
                    ).unwrap()
                    .add_output(
                        "Y",
                        Tensor::new_with_shape(
                            "sum",
                            &[2, 3],
                            Format::CHWN,
                            DType::Int32,
                            Type::Variable,
                        ),
                    ).unwrap()
                    .add_attribute("B", Attribute::from("test")).unwrap(),
            ).unwrap();

        println!("{:?}", graph);
    }
}
