//! Implementation of the `#[tool]` attribute.
//!
//! The work splits in two: [`Tool::parse`] reads an `ItemFn` and either
//! rejects it with a [`syn::Error`] pointing at the offending token or yields
//! everything the expansion needs, and [`Tool::expand`] renders that into
//! tokens. Nothing in this module panics on user input — a panic inside a
//! proc macro surfaces to the user as an internal-compiler-error-shaped
//! message with no indication of what in *their* code caused it.
//!
//! # Hygiene
//!
//! Generated code is compiled in the *caller's* crate, where the only names
//! guaranteed to resolve are the ones we fully qualify. So:
//!
//! - every path into this framework is rooted at `::acton_ai::`, never
//!   `crate::` and never a bare `acton_ai::` (which a `use` could shadow);
//! - every path into the standard library is rooted at `::std::`;
//! - `serde_json` is reached as `::acton_ai::serde_json`, so the user's crate
//!   needs no `serde_json` dependency of its own;
//! - no derive is placed on generated code. Deriving `Deserialize` or
//!   `JsonSchema` would require the user to depend on serde and schemars by
//!   those exact names, which is precisely the coupling this avoids;
//! - no user-written identifier is reused as a generated binding. Locals are
//!   prefixed `__acton_ai_`, so a parameter named `args` or `result` cannot
//!   shadow anything the expansion depends on.
//!
//! Consequently this macro must never be used inside `acton-ai` itself, where
//! `::acton_ai` does not resolve. It is exercised from `tests/`, `examples/`,
//! and doc tests.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{
    Expr, ExprLit, FnArg, GenericArgument, Ident, ItemFn, Lit, Meta, Pat, PathArguments, ReturnType,
    Type, Visibility,
};

/// Prefix for every binding the expansion introduces.
///
/// Spans produced by `Span::call_site()` are not hygienic, so a generated
/// `let args = …` really would shadow a parameter named `args`. Prefixing
/// sidesteps the whole class of collision.
const LOCAL_PREFIX: &str = "__acton_ai";

/// Entry point: parse, then expand, converting any error into the
/// `compile_error!` invocation the compiler reports at the right span.
pub(crate) fn expand(args: TokenStream, input: TokenStream) -> TokenStream {
    // The original item is emitted even when validation fails, so the user
    // sees one targeted error about the attribute rather than a cascade of
    // "cannot find function" errors from the rest of their module.
    let original = input.clone();

    match run(args, input) {
        Ok(tokens) => tokens,
        Err(error) => {
            let compile_error = error.to_compile_error();
            quote! { #original #compile_error }
        }
    }
}

/// The fallible half of [`expand`].
fn run(args: TokenStream, input: TokenStream) -> syn::Result<TokenStream> {
    if !args.is_empty() {
        return Err(syn::Error::new_spanned(
            args,
            "`#[tool]` takes no arguments; the tool's name comes from the \
             function name and its description from the doc comment",
        ));
    }

    let function: ItemFn = syn::parse2(input)?;
    Ok(Tool::parse(&function)?.expand(&function))
}

/// One parameter of a generated tool.
#[derive(Debug)]
struct Parameter {
    /// The parameter name, which is also the JSON property name.
    name: String,
    /// The declared type, used verbatim for both schema and deserialization.
    ty: Type,
    /// Whether the type is spelled `Option<…>`. Syntactic by necessity.
    optional: bool,
    /// Span of the parameter, for error reporting.
    span: Span,
}

/// Everything the expansion needs, extracted from a validated `ItemFn`.
///
/// `Debug` is why this crate takes syn's `extra-traits` feature: it is what
/// lets a failing parse test print what it actually got.
#[derive(Debug)]
struct Tool {
    /// The tool name the model sees: the function name, verbatim.
    name: String,
    /// The function to call, by identifier.
    function: Ident,
    /// The generated unit struct's name, in `PascalCase`.
    struct_name: Ident,
    /// The generated struct's visibility, inherited from the function.
    visibility: Visibility,
    /// The description, from the doc comment.
    description: String,
    /// Parameters in declaration order.
    parameters: Vec<Parameter>,
}

impl Tool {
    /// Validates `function` and extracts what the expansion needs.
    fn parse(function: &ItemFn) -> syn::Result<Self> {
        let signature = &function.sig;

        if signature.asyncness.is_none() {
            return Err(syn::Error::new(
                signature.fn_token.span(),
                "`#[tool]` requires an `async fn`; tools are awaited by the \
                 prompt loop, so add `async` before `fn`",
            ));
        }

        if let Some(receiver) = signature.receiver() {
            return Err(syn::Error::new_spanned(
                receiver,
                "`#[tool]` applies to free functions, not methods; a tool is \
                 a unit struct with no state to borrow, so remove the `self` \
                 parameter",
            ));
        }

        if let Some(variadic) = &signature.variadic {
            return Err(syn::Error::new_spanned(
                variadic,
                "`#[tool]` does not support variadic functions; every \
                 parameter has to appear in the schema by name",
            ));
        }

        if let Some(parameter) = signature.generics.params.first() {
            return Err(syn::Error::new_spanned(
                parameter,
                "`#[tool]` does not support generic functions; the generated \
                 tool is a unit struct with no way to choose the type \
                 arguments, so use concrete types in the signature",
            ));
        }

        if let Some(clause) = &signature.generics.where_clause {
            return Err(syn::Error::new_spanned(
                clause,
                "`#[tool]` does not support `where` clauses; the generated \
                 tool is a unit struct with no generic parameters to bound",
            ));
        }

        ensure_returns_tool_result(&signature.output)?;

        let name = signature.ident.to_string();
        let description = description_of(function)?;
        let parameters = signature
            .inputs
            .iter()
            .map(parameter_of)
            .collect::<syn::Result<Vec<_>>>()?;
        ensure_names_are_unique(&parameters)?;

        Ok(Self {
            struct_name: Ident::new(&pascal_case(&name), signature.ident.span()),
            function: signature.ident.clone(),
            visibility: function.vis.clone(),
            name,
            description,
            parameters,
        })
    }

    /// Renders the original function plus the generated struct and impl.
    fn expand(&self, function: &ItemFn) -> TokenStream {
        let Self {
            name,
            function: call_target,
            struct_name,
            visibility,
            description,
            parameters,
        } = self;

        let args_local = local("args");

        // Schema: one entry per parameter, in signature order.
        let schema_entries = parameters.iter().map(|parameter| {
            let Parameter {
                name, ty, optional, ..
            } = parameter;
            let required = !optional;
            quote! {
                (
                    #name,
                    ::acton_ai::__private::inline_schema_for::<#ty>(),
                    #required,
                )
            }
        });

        // Extraction: one `let` per parameter, bound to a prefixed local so
        // no user parameter name can shadow `args` mid-sequence.
        let bindings = parameters.iter().enumerate().map(|(index, parameter)| {
            let Parameter {
                name, ty, optional, ..
            } = parameter;
            let binding = local(&format!("arg_{index}"));
            let required = !optional;
            quote! {
                let #binding = ::acton_ai::__private::argument::<#ty>(
                    #name,
                    #name,
                    #required,
                    &#args_local,
                )?;
            }
        });

        // The call itself, in the same order.
        let call_arguments = (0..parameters.len()).map(|index| local(&format!("arg_{index}")));

        let result_local = local("result");
        let doc = format!(
            "The [`{name}`] tool.\n\n\
             Generated by [`#[tool]`](acton_ai::tool) from the `{name}` \
             function; register it with `.add_tool({struct_name})`.\n\n\
             # Description given to the model\n\n{description}"
        );

        quote! {
            #function

            #[doc = #doc]
            #[derive(::std::fmt::Debug, ::std::clone::Clone, ::std::marker::Copy)]
            #visibility struct #struct_name;

            impl ::acton_ai::tools::Tool for #struct_name {
                fn name(&self) -> &'static str {
                    #name
                }

                fn description(&self) -> &'static str {
                    #description
                }

                fn input_schema(&self) -> ::acton_ai::serde_json::Value {
                    ::acton_ai::__private::object_schema(
                        ::std::vec![ #( #schema_entries ),* ]
                    )
                }

                fn call(
                    &self,
                    #args_local: ::acton_ai::serde_json::Value,
                ) -> ::acton_ai::tools::ToolFuture {
                    ::std::boxed::Box::pin(async move {
                        #( #bindings )*
                        let #result_local = #call_target( #( #call_arguments ),* ).await?;
                        ::acton_ai::__private::output(#name, #result_local)
                    })
                }
            }
        }
    }
}

/// Builds one of the expansion's reserved local identifiers.
fn local(suffix: &str) -> Ident {
    format_ident!("{}_{}", LOCAL_PREFIX, suffix)
}

/// Extracts and validates one parameter.
fn parameter_of(argument: &FnArg) -> syn::Result<Parameter> {
    let FnArg::Typed(typed) = argument else {
        // `parse` rejects receivers before this runs, so reaching here means
        // a receiver in a position other than first — still not a tool.
        return Err(syn::Error::new_spanned(
            argument,
            "`#[tool]` applies to free functions, not methods",
        ));
    };

    let Pat::Ident(pattern) = typed.pat.as_ref() else {
        return Err(syn::Error::new_spanned(
            &typed.pat,
            "`#[tool]` needs a plain name for each parameter, because the name \
             becomes the JSON property the model fills in; bind the whole \
             value to one identifier and destructure it in the body",
        ));
    };

    if let Some((_, subpattern)) = &pattern.subpat {
        return Err(syn::Error::new_spanned(
            subpattern,
            "`#[tool]` needs a plain name for each parameter; `name @ pattern` \
             bindings have no single JSON property to map to",
        ));
    }

    Ok(Parameter {
        name: pattern.ident.to_string(),
        optional: is_option(&typed.ty),
        ty: typed.ty.as_ref().clone(),
        span: typed.span(),
    })
}

/// Rejects duplicate parameter names.
///
/// Rust already forbids this, but the check costs nothing and the message is
/// better than the one about JSON properties silently colliding would be.
fn ensure_names_are_unique(parameters: &[Parameter]) -> syn::Result<()> {
    for (index, parameter) in parameters.iter().enumerate() {
        if parameters[..index].iter().any(|prior| prior.name == parameter.name) {
            return Err(syn::Error::new(
                parameter.span,
                format!(
                    "duplicate parameter `{}`; each parameter becomes a \
                     distinct JSON property",
                    parameter.name
                ),
            ));
        }
    }
    Ok(())
}

/// Reports whether a type is *spelled* `Option<…>`.
///
/// Deliberately syntactic: a macro sees tokens, and resolving a type alias
/// would require type information that does not exist at expansion time. The
/// three spellings a reader would expect to work all do; an alias does not,
/// which the macro's documentation states.
fn is_option(ty: &Type) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    if path.qself.is_some() {
        return false;
    }

    let segments = &path.path.segments;
    let Some(last) = segments.last() else {
        return false;
    };
    if last.ident != "Option" {
        return false;
    }

    // Accept `Option<T>`, `std::option::Option<T>`, `core::option::Option<T>`,
    // and their leading-`::` forms; reject anything else called `Option`.
    let qualified = match segments.len() {
        1 => true,
        3 => {
            let root = &segments[0].ident;
            (root == "std" || root == "core") && segments[1].ident == "option"
        }
        _ => false,
    };
    if !qualified {
        return false;
    }

    // `Option` without a type argument is not a type at all, but checking
    // keeps the predicate honest about what it claims.
    matches!(&last.arguments, PathArguments::AngleBracketed(arguments)
        if arguments.args.iter().any(|argument| matches!(argument, GenericArgument::Type(_))))
}

/// Requires the return type to be spelled `Result<_, ToolError>`.
///
/// A tool that returns anything else would fail later with a trait-bound
/// error pointing at generated code, which is a poor thing to hand a user.
/// The check is syntactic, so it accepts any path ending in `ToolError`
/// (`ToolError`, `acton_ai::tools::ToolError`, an aliased import) and rejects
/// an aliased *`Result`*, which the message explains.
fn ensure_returns_tool_result(output: &ReturnType) -> syn::Result<()> {
    const EXPECTED: &str = "`#[tool]` requires the return type to be spelled \
                            `Result<T, ToolError>`, where `T: serde::Serialize`; \
                            the `Ok` value is serialized and sent back to the model";

    let ReturnType::Type(arrow, ty) = output else {
        return Err(syn::Error::new(
            output.span(),
            format!("{EXPECTED}\n\nthis function returns `()`"),
        ));
    };

    let error = || syn::Error::new_spanned(quote! { #arrow #ty }, EXPECTED);

    let Type::Path(path) = ty.as_ref() else {
        return Err(error());
    };
    let last = path.path.segments.last().ok_or_else(error)?;
    if last.ident != "Result" {
        return Err(error());
    }

    let PathArguments::AngleBracketed(arguments) = &last.arguments else {
        return Err(error());
    };
    let types: Vec<&Type> = arguments
        .args
        .iter()
        .filter_map(|argument| match argument {
            GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect();

    // A one-argument `Result<T>` is an alias with a baked-in error type; the
    // macro cannot tell whether that error is `ToolError`.
    let [_, error_type] = types.as_slice() else {
        return Err(syn::Error::new_spanned(
            ty,
            format!(
                "{EXPECTED}\n\nwrite both type arguments explicitly, even if \
                 you normally use a `Result` alias"
            ),
        ));
    };

    let names_tool_error = matches!(error_type, Type::Path(path)
        if path.path.segments.last().is_some_and(|segment| segment.ident == "ToolError"));

    if !names_tool_error {
        return Err(syn::Error::new_spanned(error_type, EXPECTED));
    }

    Ok(())
}

/// Joins a function's `///` lines into the description the model reads.
///
/// A missing doc comment is an error rather than an empty description. The
/// description is the entire basis on which a model decides whether to call a
/// tool: an undescribed tool is not a tool with a minor documentation gap, it
/// is a tool that never gets called, and failing at compile time is far
/// cheaper than discovering that from a model's behavior.
fn description_of(function: &ItemFn) -> syn::Result<String> {
    let lines: Vec<String> = function
        .attrs
        .iter()
        .filter(|attribute| attribute.path().is_ident("doc"))
        .filter_map(|attribute| match &attribute.meta {
            Meta::NameValue(pair) => match &pair.value {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(text),
                    ..
                }) => Some(text.value()),
                _ => None,
            },
            _ => None,
        })
        .collect();

    let description = join_doc_lines(&lines);

    if description.is_empty() {
        return Err(syn::Error::new(
            function.sig.ident.span(),
            format!(
                "`#[tool]` requires a doc comment on `{}`: add a `///` comment \
                 describing what the tool does and when to use it. It becomes \
                 the tool description the model sees, and is the only thing it \
                 has to go on when deciding whether to call this tool",
                function.sig.ident
            ),
        ));
    }

    Ok(description)
}

/// Normalizes doc-comment lines into a description string.
///
/// Each `///` line arrives with the leading space the writer put after the
/// slashes, which is punctuation of the comment syntax rather than of the
/// text. Trailing blank lines are dropped and interior ones preserved, so a
/// doc comment with paragraphs reaches the model as paragraphs.
fn join_doc_lines(lines: &[String]) -> String {
    lines
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Converts a `snake_case` function name to a `PascalCase` type name.
///
/// Runs of underscores collapse, and a leading underscore is dropped: `_read`
/// and `read_file` become `Read` and `ReadFile`. Characters that are already
/// uppercase are left alone rather than lowercased, so `read_URL` yields
/// `ReadURL` instead of the surprising `ReadUrl`.
fn pascal_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut capitalize = true;

    for character in name.chars() {
        if character == '_' {
            capitalize = true;
            continue;
        }
        if capitalize {
            out.extend(character.to_uppercase());
            capitalize = false;
        } else {
            out.push(character);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_type(text: &str) -> Type {
        syn::parse_str(text).expect("test fixture must parse")
    }

    fn parse_return(text: &str) -> ReturnType {
        syn::parse_str(text).expect("test fixture must parse")
    }

    // -- pascal_case ------------------------------------------------------

    #[test]
    fn pascal_case_capitalizes_a_single_word() {
        assert_eq!(pascal_case("calculator"), "Calculator");
    }

    #[test]
    fn pascal_case_joins_snake_case_words() {
        assert_eq!(pascal_case("read_file"), "ReadFile");
        assert_eq!(pascal_case("get_weather_for_city"), "GetWeatherForCity");
    }

    #[test]
    fn pascal_case_collapses_repeated_underscores() {
        assert_eq!(pascal_case("read__file"), "ReadFile");
    }

    #[test]
    fn pascal_case_drops_a_leading_underscore() {
        assert_eq!(pascal_case("_read"), "Read");
    }

    #[test]
    fn pascal_case_preserves_existing_uppercase() {
        assert_eq!(pascal_case("read_URL"), "ReadURL");
    }

    #[test]
    fn pascal_case_handles_digits() {
        assert_eq!(pascal_case("base64_decode"), "Base64Decode");
    }

    // -- is_option --------------------------------------------------------

    #[test]
    fn is_option_accepts_the_bare_spelling() {
        assert!(is_option(&parse_type("Option<u8>")));
    }

    #[test]
    fn is_option_accepts_fully_qualified_spellings() {
        assert!(is_option(&parse_type("std::option::Option<u8>")));
        assert!(is_option(&parse_type("::std::option::Option<u8>")));
        assert!(is_option(&parse_type("core::option::Option<String>")));
    }

    #[test]
    fn is_option_rejects_a_plain_type() {
        assert!(!is_option(&parse_type("String")));
        assert!(!is_option(&parse_type("Vec<u8>")));
    }

    #[test]
    fn is_option_rejects_an_unrelated_option_path() {
        assert!(
            !is_option(&parse_type("mycrate::Option<u8>")),
            "only the standard Option spellings count as optional"
        );
    }

    #[test]
    fn is_option_rejects_an_alias() {
        assert!(
            !is_option(&parse_type("MaybeU8")),
            "aliases are a documented limitation: the macro sees tokens only"
        );
    }

    #[test]
    fn is_option_rejects_option_without_a_type_argument() {
        assert!(!is_option(&parse_type("Option")));
    }

    // -- ensure_returns_tool_result ---------------------------------------

    #[test]
    fn return_type_accepts_the_expected_shape() {
        assert!(ensure_returns_tool_result(&parse_return("-> Result<String, ToolError>")).is_ok());
    }

    #[test]
    fn return_type_accepts_a_qualified_tool_error() {
        assert!(ensure_returns_tool_result(&parse_return(
            "-> Result<serde_json::Value, acton_ai::tools::ToolError>"
        ))
        .is_ok());
    }

    #[test]
    fn return_type_rejects_a_missing_return_type() {
        let error = ensure_returns_tool_result(&parse_return(""))
            .expect_err("a unit return must be rejected");

        assert!(error.to_string().contains("Result<T, ToolError>"));
    }

    #[test]
    fn return_type_rejects_a_non_result() {
        let error = ensure_returns_tool_result(&parse_return("-> String"))
            .expect_err("a bare String return must be rejected");

        assert!(error.to_string().contains("Result<T, ToolError>"));
    }

    #[test]
    fn return_type_rejects_a_foreign_error_type() {
        let error = ensure_returns_tool_result(&parse_return("-> Result<String, std::io::Error>"))
            .expect_err("a non-ToolError error type must be rejected");

        assert!(error.to_string().contains("ToolError"));
    }

    #[test]
    fn return_type_rejects_a_single_argument_result_alias() {
        let error = ensure_returns_tool_result(&parse_return("-> Result<String>"))
            .expect_err("an aliased Result must be rejected");

        assert!(
            error.to_string().contains("both type arguments"),
            "the message must say how to fix it: {error}"
        );
    }

    // -- join_doc_lines ---------------------------------------------------

    #[test]
    fn doc_lines_are_trimmed_and_joined() {
        let joined = join_doc_lines(&[
            " Computes a result.".to_string(),
            " Use it for arithmetic.".to_string(),
        ]);

        assert_eq!(joined, "Computes a result.\nUse it for arithmetic.");
    }

    #[test]
    fn doc_lines_preserve_interior_blank_lines() {
        let joined = join_doc_lines(&[
            " Summary.".to_string(),
            String::new(),
            " Detail.".to_string(),
        ]);

        assert_eq!(joined, "Summary.\n\nDetail.");
    }

    #[test]
    fn doc_lines_drop_surrounding_blank_lines() {
        let joined = join_doc_lines(&[String::new(), " Body.".to_string(), String::new()]);

        assert_eq!(joined, "Body.");
    }

    #[test]
    fn no_doc_lines_yields_an_empty_description() {
        assert_eq!(join_doc_lines(&[]), "");
        assert_eq!(join_doc_lines(&[String::new()]), "");
    }

    // -- local ------------------------------------------------------------

    #[test]
    fn locals_are_prefixed_so_they_cannot_shadow_user_names() {
        assert_eq!(local("args").to_string(), "__acton_ai_args");
        assert_eq!(local("arg_0").to_string(), "__acton_ai_arg_0");
    }

    // -- end-to-end parsing ----------------------------------------------

    fn parse_fn(text: &str) -> syn::Result<Tool> {
        let function: ItemFn = syn::parse_str(text).expect("test fixture must parse");
        Tool::parse(&function)
    }

    #[test]
    fn parse_extracts_name_description_and_parameters() {
        let tool = parse_fn(
            "/// Computes things.\n\
             async fn calculator(expr: String, precision: Option<u8>) \
             -> Result<serde_json::Value, ToolError> { todo!() }",
        )
        .expect("a well-formed tool must parse");

        assert_eq!(tool.name, "calculator");
        assert_eq!(tool.struct_name.to_string(), "Calculator");
        assert_eq!(tool.description, "Computes things.");
        assert_eq!(tool.parameters.len(), 2);
        assert_eq!(tool.parameters[0].name, "expr");
        assert!(!tool.parameters[0].optional);
        assert_eq!(tool.parameters[1].name, "precision");
        assert!(tool.parameters[1].optional);
    }

    #[test]
    fn parse_accepts_a_zero_parameter_tool() {
        let tool = parse_fn(
            "/// Reports the time.\nasync fn now() -> Result<String, ToolError> { todo!() }",
        )
        .expect("a zero-parameter tool must parse");

        assert!(tool.parameters.is_empty());
    }

    #[test]
    fn parse_rejects_a_missing_doc_comment() {
        let error = parse_fn("async fn f() -> Result<String, ToolError> { todo!() }")
            .expect_err("a tool without a doc comment must be rejected");

        assert!(error.to_string().contains("doc comment"), "{error}");
    }

    #[test]
    fn parse_rejects_a_non_async_function() {
        let error = parse_fn("/// Doc.\nfn f() -> Result<String, ToolError> { todo!() }")
            .expect_err("a non-async tool must be rejected");

        assert!(error.to_string().contains("async"), "{error}");
    }

    #[test]
    fn parse_rejects_generics() {
        let error =
            parse_fn("/// Doc.\nasync fn f<T>(x: T) -> Result<String, ToolError> { todo!() }")
                .expect_err("a generic tool must be rejected");

        assert!(error.to_string().contains("generic"), "{error}");
    }

    #[test]
    fn parse_rejects_a_where_clause() {
        let error = parse_fn(
            "/// Doc.\nasync fn f(x: String) -> Result<String, ToolError> \
             where String: Clone { todo!() }",
        )
        .expect_err("a where clause must be rejected");

        assert!(error.to_string().contains("where"), "{error}");
    }

    #[test]
    fn parse_rejects_a_destructuring_parameter() {
        let error = parse_fn(
            "/// Doc.\nasync fn f((a, b): (u8, u8)) -> Result<String, ToolError> { todo!() }",
        )
        .expect_err("a tuple pattern must be rejected");

        assert!(error.to_string().contains("plain name"), "{error}");
    }

    #[test]
    fn parse_accepts_a_mut_parameter_binding() {
        let tool = parse_fn(
            "/// Doc.\nasync fn f(mut x: String) -> Result<String, ToolError> { todo!() }",
        )
        .expect("`mut` on a parameter is a binding mode, not a pattern");

        assert_eq!(tool.parameters[0].name, "x");
    }

    #[test]
    fn parse_joins_a_multi_line_doc_comment() {
        let tool = parse_fn(
            "/// Summary line.\n\
             ///\n\
             /// Detail line.\n\
             async fn f() -> Result<String, ToolError> { todo!() }",
        )
        .expect("a multi-line doc comment must parse");

        assert_eq!(tool.description, "Summary line.\n\nDetail line.");
    }

    #[test]
    fn parse_inherits_the_function_visibility() {
        let tool =
            parse_fn("/// Doc.\npub async fn f() -> Result<String, ToolError> { todo!() }")
                .expect("a pub tool must parse");

        assert!(matches!(tool.visibility, Visibility::Public(_)));
    }

    // -- expansion smoke checks ------------------------------------------

    #[test]
    fn expansion_emits_the_original_function_and_the_struct() {
        let function: ItemFn = syn::parse_str(
            "/// Computes things.\n\
             async fn calculator(expr: String) -> Result<serde_json::Value, ToolError> { todo!() }",
        )
        .expect("fixture must parse");
        let tokens = Tool::parse(&function)
            .expect("fixture must validate")
            .expand(&function)
            .to_string();

        assert!(tokens.contains("async fn calculator"), "{tokens}");
        assert!(tokens.contains("struct Calculator"), "{tokens}");
        assert!(
            tokens.contains(":: acton_ai :: tools :: Tool for Calculator"),
            "{tokens}"
        );
    }

    #[test]
    fn expansion_roots_every_framework_path_at_the_crate() {
        let function: ItemFn = syn::parse_str(
            "/// Doc.\nasync fn f(x: String) -> Result<String, ToolError> { todo!() }",
        )
        .expect("fixture must parse");
        let tokens = Tool::parse(&function)
            .expect("fixture must validate")
            .expand(&function)
            .to_string();

        // Every framework path must carry the leading `::` that makes it
        // absolute — a bare `acton_ai::…` resolves against whatever the
        // caller's module happens to have in scope, and a `use` of their own
        // could shadow it. Counting occurrences with and without the prefix
        // proves there is no unqualified one hiding among the qualified ones.
        for path in [
            "acton_ai :: tools :: Tool",
            "acton_ai :: tools :: ToolFuture",
            "acton_ai :: __private :: argument",
            "acton_ai :: __private :: object_schema",
            "acton_ai :: __private :: inline_schema_for",
            "acton_ai :: __private :: output",
            "acton_ai :: serde_json :: Value",
        ] {
            let occurrences = tokens.matches(path).count();
            let absolute = tokens.matches(&format!(":: {path}")).count();

            assert!(occurrences > 0, "expansion should use `{path}`: {tokens}");
            assert_eq!(
                occurrences, absolute,
                "every `{path}` must be spelled absolutely: {tokens}"
            );
        }

        // Likewise for the standard library.
        assert_eq!(
            tokens.matches("std :: boxed :: Box").count(),
            tokens.matches(":: std :: boxed :: Box").count(),
            "{tokens}"
        );
    }

    #[test]
    fn expansion_never_binds_a_user_parameter_name() {
        let function: ItemFn = syn::parse_str(
            "/// Doc.\n\
             async fn f(args: String, result: String) -> Result<String, ToolError> { todo!() }",
        )
        .expect("fixture must parse");
        let tokens = Tool::parse(&function)
            .expect("fixture must validate")
            .expand(&function)
            .to_string();

        // The generated `call` binds `__acton_ai_args`, never `args`, so a
        // parameter called `args` cannot shadow the arguments value between
        // one extraction and the next.
        assert!(tokens.contains("__acton_ai_args"), "{tokens}");
        assert!(tokens.contains("__acton_ai_arg_0"), "{tokens}");
        assert!(
            !tokens.contains("let args"),
            "a user parameter name must never become a generated binding: {tokens}"
        );
    }

    #[test]
    fn expansion_marks_optional_parameters_as_not_required() {
        let function: ItemFn = syn::parse_str(
            "/// Doc.\n\
             async fn f(a: String, b: Option<u8>) -> Result<String, ToolError> { todo!() }",
        )
        .expect("fixture must parse");
        let tokens = Tool::parse(&function)
            .expect("fixture must validate")
            .expand(&function)
            .to_string();

        assert!(tokens.contains("\"a\" , :: acton_ai"), "{tokens}");
        // One `true` and one `false` in the schema entry list.
        assert!(tokens.contains("true"), "{tokens}");
        assert!(tokens.contains("false"), "{tokens}");
    }

    #[test]
    fn a_failed_expansion_still_emits_the_original_item() {
        let expanded = expand(
            TokenStream::new(),
            "async fn f() -> Result<String, ToolError> { todo!() }"
                .parse()
                .expect("fixture must tokenize"),
        )
        .to_string();

        assert!(
            expanded.contains("async fn f"),
            "the original item must survive so the user gets one error, not a \
             cascade of 'cannot find function': {expanded}"
        );
        assert!(expanded.contains("compile_error"), "{expanded}");
    }

    #[test]
    fn attribute_arguments_are_rejected() {
        let expanded = expand(
            quote! { name = "other" },
            "/// Doc.\nasync fn f() -> Result<String, ToolError> { todo!() }"
                .parse()
                .expect("fixture must tokenize"),
        )
        .to_string();

        assert!(expanded.contains("compile_error"), "{expanded}");
        assert!(expanded.contains("takes no arguments"), "{expanded}");
    }
}
