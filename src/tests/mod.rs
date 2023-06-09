#[cfg(test)]
mod tests {
    use std::env;
    use std::fs::read_to_string;
    use std::path::PathBuf;

    use inkwell::context::Context;
    use lalrpop_util::lalrpop_mod;

    use crate::llvm::Compiler;

    lalrpop_mod!(sysy);

    macro_rules! test_case {
        ($testFuncName: ident, $filename: expr) => {
            #[test]
            fn $testFuncName() {
                let path = get_current_dir();
                let in_path: PathBuf = [&path, "src", "tests", "cases", $filename].iter().collect();
                let input = read_to_string(&in_path).unwrap();

                let program_ast = sysy::CompUnitParser::new().parse(&input).unwrap();
    
                let context = Context::create();
                let mut compiler = Compiler::new(&context);
                let res = compiler.generate(&program_ast);

                let out_path = format!("{}/src/tests/output/{}", path, $filename);
                compiler.write_ir_to_file(&out_path);
                assert!(res.is_ok());
            }
        };
    }

    pub fn get_current_dir() -> String {
        if let Ok(path) = env::var("CARGO_MANIFEST_DIR") {
            path
        }
        else {
            String::from(env::current_dir().unwrap().to_str().unwrap())
        }
    }

    test_case!(test_just_main, "just_main");
    test_case!(test_unary, "unary");
    test_case!(test_add_exp, "add_exp");
    test_case!(test_rel_exp, "rel_exp");
    test_case!(test_const, "const");
    test_case!(test_var, "var");
    test_case!(test_block, "block");
    test_case!(test_if, "if");
    test_case!(test_shortcut, "shortcut");
    test_case!(test_while, "while");
    test_case!(test_break, "break");
    test_case!(test_continue, "continue");
    test_case!(test_function, "function");
    test_case!(test_global, "global");
    test_case!(test_array, "array");
    test_case!(test_array_param, "array_param");
    test_case!(test_array_init, "array_init");
}
