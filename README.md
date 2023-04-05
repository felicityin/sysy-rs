# Description

Simple PoC Language built in Rust using [LALRPOP](https://crates.io/crates/lalrpop) and LLVM ([inkwell](https://crates.io/crates/inkwell)).

The grammar is copied from [pku-minic](https://pku-minic.github.io/online-doc/#/misc-app-ref/sysy-spec).

# Test
```
cargo test
```

# Run
```
cargo run -- -i input/hello.c -o output/hello.ir
```

