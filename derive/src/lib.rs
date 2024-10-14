//! 参考[enum-code](https://github.com/Kunduin/enum-code)仓库，并增加from_code实现
//!

use crate::code::{parse_get_code_stream, parse_from_code_stream};

mod code;

#[proc_macro_derive(GetCode, attributes(code))]
pub fn get_code(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    parse_get_code_stream(input)
}

#[proc_macro_derive(FromCode, attributes(code))]
pub fn from_code(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    parse_from_code_stream(input)
}
