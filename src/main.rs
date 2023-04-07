mod parser;
mod tests;
mod llvm_ir;

use std::fs::read_to_string;

use inkwell::context::Context;
use lalrpop_util::lalrpop_mod;

use crate::llvm_ir::Compiler;

lalrpop_mod!(sysy);

fn main() {
    // parse args
    let cmd = clap::Command::new("toy-lang")
        .version(clap::crate_version!())
        .arg(
            clap::Arg::new("input")
                .short('i')
                .default_value("input/hello.c")
                .help("Input file needs to be parsed.")
                .action(clap::ArgAction::Set),
            ).arg(
                clap::Arg::new("output")
                    .short('o')
                    .default_value("output/hello.ir")
                    .help("Output llvm ir file")
                    .action(clap::ArgAction::Set),
            );

    let matches = cmd.get_matches();
    let input = matches.get_one::<String>("input").unwrap();
    let output = matches.get_one::<String>("output").unwrap();

    // parse input file
    let input = read_to_string(input).unwrap();
    let program_ast = sysy::CompUnitParser::new().parse(&input).unwrap();
    // println!("AST:\n{}", program_ast);

    // generate LLVM IR
    let context = Context::create();
    let mut compiler = Compiler::new(&context);
    compiler.generate(&program_ast).unwrap();
    compiler.optimize();
    compiler.write_to_file(output);
    println!("LLVM IR:\n{:#?}", compiler.get_llvm_ir());
}
