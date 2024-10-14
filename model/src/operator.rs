use std::collections::HashMap;

use super::tensor::Tensor;
use super::attribute::Attribute;
use anyhow::Result;

#[derive(Debug)]
pub struct Operator {
    /// 名字
    name: String,
    /// 类型
    r#type: String,
    /// 输入Tensor
    inputs: HashMap<String, Tensor>,
    /// 输出Tensor
    outputs: HashMap<String, Tensor>,
    /// 属性
    attributes: HashMap<String, Attribute>,
}

impl Operator {
    pub fn new(name: &str, r#type: &str) -> Self {
        Self {
            name: String::from(name),
            r#type: String::from(r#type),
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            attributes: HashMap::new(),
        }
    }

    pub fn add_input(mut self, tag: &str, input: Tensor) -> Result<Self> {
        self.inputs.insert(String::from(tag), input);

        Ok(self)
    }

    pub fn add_output(mut self, tag: &str, output: Tensor) -> Result<Self> {
        self.outputs.insert(String::from(tag), output);

        Ok(self)
    }

    pub fn add_attribute(mut self, tag: &str, attr: Attribute) -> Result<Self> {
        self.attributes.insert(String::from(tag), attr);

        Ok(self)
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn r#type(&self) -> &String {
        &self.r#type
    }

    pub fn inputs(&self) -> &HashMap<String, Tensor> {
        &self.inputs
    }

    pub fn get_input(&self, tag: &str) -> Option<&Tensor> {
        self.inputs.get(tag)
    }

    pub fn outputs(&self) -> &HashMap<String, Tensor> {
        &self.outputs
    }

    pub fn get_output(&self, tag: &str) -> Option<&Tensor> {
        self.outputs.get(tag)
    }

    pub fn attributes(&self) -> &HashMap<String, Attribute> {
        &self.attributes
    }

    pub fn get_attribute(&self, tag: &str) -> Option<&Attribute> {
        self.attributes.get(tag)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::*;

    #[test]
    fn data_works() {
        let op = Operator::new("name", "Add")
            .add_input(
                "X1",
                Tensor::new_with_shape("add1", &[2, 3], Format::CHWN, DType::Int32, Type::Variable),
            ).unwrap()
            .add_input(
                "X2",
                Tensor::new_with_shape("add2", &[2, 3], Format::CHWN, DType::Int32, Type::Variable),
            ).unwrap()
            .add_output(
                "Y",
                Tensor::new_with_shape("sum", &[2, 3], Format::CHWN, DType::Int32, Type::Variable),
            ).unwrap()
            .add_attribute("B", Attribute::from("test")).unwrap();

        println!("{:?}", op);
    }
}
