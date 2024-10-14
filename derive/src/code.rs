use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Attribute};

fn get_code_value(attrs: &Vec<Attribute>) -> u32 {
    attrs.iter().find_map(|attr| {
        if attr.path().is_ident("code") {
            let code_in_attr = attr
                .parse_args::<syn::LitInt>()
                .expect("#[code()] value must be integer")
                .base10_parse::<u32>()
                .expect("#[code()] value is not a integer");
            Some(code_in_attr)
        } else {
            None
        }
    }).unwrap_or(0)
}

pub fn parse_get_code_stream(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let getters: Vec<_> = match &input.data {
        Data::Enum(e) => e
            .variants
            .iter()
            .map(|variant| {
                let variant_ident = variant.ident.clone();
                let variant_fields = variant.fields.clone();

                let code_value = get_code_value(&variant.attrs);

                match variant_fields {
                    Fields::Named(..) => quote! {
                        #name::#variant_ident { .. } => #code_value
                    },
                    Fields::Unnamed(..) => quote! {
                        #name::#variant_ident ( .. ) => #code_value
                    },
                    Fields::Unit => quote! {
                        #name::#variant_ident => #code_value
                    },
                }
            })
            .collect(),
        _ => panic!("Code attribute is only applicable to enums!"),
    };

    let output = quote! {
        impl #name {
            pub const fn get_code(&self) -> u32 {
                match self {
                    #(#getters),*
                }
            }
        }
    };
    proc_macro::TokenStream::from(output)
}

pub fn parse_from_code_stream(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    let setters: Vec<_> = match &input.data {
        Data::Enum(e) => e
            .variants
            .iter()
            .map(|variant| {
                let variant_ident = variant.ident.clone();
                let variant_fields = variant.fields.clone();

                let code_value = get_code_value(&variant.attrs);

                match variant_fields {
                    Fields::Named(..) => quote! {
                        #code_value => #name::#variant_ident { .. }
                    },
                    Fields::Unnamed(..) => quote! {
                        #code_value => #name::#variant_ident ( .. )
                    },
                    Fields::Unit => quote! {
                        #code_value => #name::#variant_ident
                    },
                }
            })
            .collect(),
        _ => panic!("Code attribute is only applicable to enums!"),
    };

    let output = quote! {
        impl #name {
            pub fn from_code(code: u32) -> Self {
                match code {
                    #(#setters),*
                    ,_ => panic!("Invalid code value, which is {}", code),
                }
            }
        }
    };
    proc_macro::TokenStream::from(output)
}
