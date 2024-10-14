use bridge::nndevice;
// use glog::Flags;
use log::*;
use model::attribute::*;
use model::graph::*;
use model::operator::*;
use model::tensor::*;

pub fn main() {
    // glog::new().init(Flags::default()).unwrap();

    // glog::new()
    //     .with_year(true)
    //     .reduced_log_levels(true)
    //     .set_application_fingerprint("bride-sample")
    //     .init(Flags {
    //         colorlogtostderr: true,
    //         minloglevel: Level::Trace,
    //         alsologtostderr: true,
    //         ..Default::default()
    //     })
    //     .unwrap();

    info!("start main");

    let backends = nndevice::get_candidate_backends();
    info!("{:?}", backends);
    if backends.len() == 0 {
        warn!("Not backends exist!");
        return;
    }

    let ctx = nndevice::engine::create_context(
        &backends[0],
        1,
        vec![
            (String::from("key1"), String::from("value1")),
            (String::from("key2"), String::from("value2")),
        ]
        .into_iter()
        .collect(),
    );

    let ctx = match ctx {
        Ok(ctx) => ctx,
        Err(e) => {
            panic!("{}", e);
        }
    };

    let graph = Graph::new("graph")
        .add_operator(
            Operator::new("add1", "Add")
                .add_input(
                    "X1",
                    Tensor::new_with_shape(
                        "data1",
                        &[2, 2],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_input(
                    "X2",
                    Tensor::new_with_shape(
                        "data2",
                        &[2, 2],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_output(
                    "Y",
                    Tensor::new_with_shape(
                        "sum1",
                        &[2, 2],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_attribute("A", Attribute::from("test"))
                .unwrap(),
        )
        .unwrap()
        .add_operator(
            Operator::new("add2", "Add")
                .add_input(
                    "X1",
                    Tensor::new_with_shape(
                        "data3",
                        &[2, 3],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_input(
                    "X2",
                    Tensor::new_with_shape(
                        "data4",
                        &[2, 3],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_output(
                    "Y",
                    Tensor::new_with_shape(
                        "sum2",
                        &[2, 3],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_attribute("B", Attribute::from(&[-1_i64, 1_i64] as &[i64]))
                .unwrap()
                .add_attribute("B1", Attribute::from(12))
                .unwrap(),
        )
        .unwrap()
        .add_operator(
            Operator::new("mult1", "Mult")
                .add_input(
                    "X1",
                    Tensor::new_with_shape(
                        "sum1",
                        &[2, 2],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_input(
                    "X2",
                    Tensor::new_with_shape(
                        "sum2",
                        &[2, 3],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_output(
                    "Y",
                    Tensor::new_with_shape(
                        "mult1",
                        &[2, 3],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_attribute(
                    "C",
                    Attribute::from(&[String::from("1234"), String::from("321")] as &[String]),
                )
                .unwrap(),
        )
        .unwrap()
        .add_operator(
            Operator::new("mult2", "Mult")
                .add_input(
                    "X1",
                    Tensor::new_with_shape(
                        "sum1",
                        &[2, 2],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_input(
                    "X2",
                    Tensor::new_with_shape(
                        "mult1",
                        &[2, 3],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_output(
                    "Y",
                    Tensor::new_with_shape(
                        "mult2",
                        &[2, 3],
                        Format::NCHW,
                        DType::Int32,
                        Type::Variable,
                    ),
                )
                .unwrap()
                .add_attribute("D", Attribute::from(&[3.0_f32, 4.0_f32] as &[f32]))
                .unwrap()
                .add_attribute("E", Attribute::from(5.0))
                .unwrap(),
        )
        .unwrap();
    match nndevice::engine::compile_graph(&ctx, &graph, |result| {
        match result {
            Ok(_) => info!("compile return ok"),
            Err(e) => {
                panic!("compile return failed, {}", e)
            }
        };
    }) {
        Ok(_) => info!("compile graph ok"),
        Err(e) => {
            panic!("compile graph failed, {}", e)
        }
    };

    let mut data1 = Tensor::new("data1", Format::NCHW, DType::Int32, Type::Variable);
    data1.set_array2(&[[-1, 2], [3, 4]]);

    let mut data2 = Tensor::new("data2", Format::NCHW, DType::Int32, Type::Variable);
    data2.set_array2(&[[1, -2], [3, 4]]);

    let mut data3 =
        Tensor::new_with_shape("data3", &[2, 3], Format::NCHW, DType::Int32, Type::Variable);
    data3.set_array2(&[[1, 2, -3], [4, 5, 6]]);

    let mut data4 = Tensor::new("data4", Format::NCHW, DType::Int32, Type::Variable);
    data4.set_array2(&[[1, 2, 3], [-4, 5, 6]]);
    let inputs = [&data1, &data2, &data3, &data4];

    match nndevice::engine::excute(&ctx, &inputs, |result| {
        match result {
            Ok(outputs) => {
                info!("excute return ok");
                for output in outputs {
                    info!("{}", output);
                    unsafe {
                        let prt = output.data_ptr() as *mut i32;
                        let mut p_str = String::new();
                        for i in 0..output.shape().len() {
                            let value: i32 = *prt.add(i);
                            p_str.push_str(value.to_string().as_str());
                            p_str.push_str(",")
                        }
                        info!("value: [{}]", p_str)
                    }
                }
            }
            Err(e) => {
                panic!("excute return failed, {}", e)
            }
        };
    }) {
        Ok(_) => info!("excute ok"),
        Err(e) => {
            panic!("excute failed, {}", e)
        }
    }

    match nndevice::engine::destory_context(ctx) {
        Ok(_) => info!("destory_context ok"),
        Err(e) => {
            panic!("destory_context failed. {}", e);
        }
    }
}
