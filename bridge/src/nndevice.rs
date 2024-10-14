pub mod engine;
mod ffi;

use cxx::Exception;
use ffi as nndevice;
use thiserror;
use log::*;


#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("未知错误")]
    UnknowErr,
    #[error("参数错误")]
    InvalidParam,
    #[error("加载动态库错误")]
    LoadSoErr,
    #[error("动态库不存在")]
    SoNotExist,
    #[error("NnDevice关闭失败")]
    NnDeviceCloseErr,
    #[error("后端调用错误")]
    NnDeviceDriverErr,
}

/**
 * 要与inos/ai_chip/result.h中的定义匹配
 */
pub fn parser_error(e: Exception) -> Option<Error> {
    let rc = match  e.what().parse::<i32>() {
        Ok(rc) => rc,
        Err(_) => {
            warn!("throw unknow exception for cxx! {e}");
            return Some(Error::UnknowErr);
        }
    };
    parser_error_from_code(rc)
}

pub fn parser_error_from_code(rc: i32) -> Option<Error> {
    match rc {
        0 => None,
        -1 => Some(Error::InvalidParam),
        -2 => Some(Error::LoadSoErr),
        -3 => Some(Error::SoNotExist),
        -201 => Some(Error::NnDeviceDriverErr),
        -202 => Some(Error::NnDeviceCloseErr),
        _ => Some(Error::UnknowErr)
    }
}

pub fn get_candidate_backends() -> Vec<String> {
    nndevice::ffi::GetCandidateBackends()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_candidate_backends() {
        let ctx = engine::create_context(
            "backend_id1",
            0,
            vec![
                (String::from("key1"), String::from("value1")),
                (String::from("key2"), String::from("value2")),
            ]
            .into_iter()
            .collect(),
        );
        let ctx = match ctx {
            Ok(c) => c,
            Err(e) => {
                panic!("{}", e);
            }
        };

        // let s = vec!["test"];

        // engine::compile_graph(|e: i32| {
        //     println!("{}, {:?}", e, s);
        //     // s.push("value");
        //     // println!("new value {:?}", s);
        // });
        // // s.push("value1");
        // println!("new value {:?}", s);

        // engine::compile_graph(test);

        // let backends = get_candidate_backends();
        // assert_eq!(0, backends.len());
    }
}
