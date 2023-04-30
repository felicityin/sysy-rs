mod llvm;
mod parser;
mod tests;

use std::fs::read_to_string;
use std::path::Path;

use inkwell::context::Context;
use lalrpop_util::lalrpop_mod;

use crate::llvm::Compiler;

lalrpop_mod!(sysy);

fn main() {
    // parse args
    let cmd = clap::Command::new("toy-lang")
        .version(clap::crate_version!())
        .arg(
            clap::Arg::new("input")
                .short('i')
                .default_value("input/hello.c")
                .required(true)
                .help("Input file needs to be parsed.")
                .action(clap::ArgAction::Set),
            ).arg(
                clap::Arg::new("output-ir")
                    .short('o')
                    .default_value("output/hello.ll")
                    .required(true)
                    .help("Output llvm ir file")
                    .action(clap::ArgAction::Set),
            ).arg(
                clap::Arg::new("output-riskv")
                    .short('s')
                    .default_value("output/hello.riscv")
                    .required(true)
                    .help("Output riscv file")
                    .action(clap::ArgAction::Set),
            );

    let matches = cmd.get_matches();
    let input = matches.get_one::<String>("input").unwrap();
    let output_ir = matches.get_one::<String>("output-ir").unwrap();
    let output_riscv = matches.get_one::<String>("output-riskv").unwrap();

    // parse input file
    let input = read_to_string(input).unwrap();
    let program_ast = sysy::CompUnitParser::new().parse(&input).unwrap();

    // generate LLVM IR
    let context = Context::create();
    let mut compiler = Compiler::new(&context);
    compiler.generate(&program_ast).unwrap();
    compiler.optimize();
    compiler.write_ir_to_file(output_ir);
    compiler.gen_riskv(Path::new(&output_riscv));
    println!("LLVM IR:\n{:#?}", compiler.get_llvm_ir());
}
