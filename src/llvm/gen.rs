use std::collections::HashMap;
use std::path::Path;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassManager;
use inkwell::types::{ArrayType, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, IntType, VoidType};
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::values::{
    ArrayValue, BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, IntValue, GlobalValue, PointerValue
};
use inkwell::AddressSpace;
use inkwell::IntPredicate;

use crate::llvm::error::{CompileErr, Result};
use crate::llvm::values::Initializer;
use crate::parser::ast::*;

use super::eval::Evaluate;

pub struct Compiler<'ast, 'ctx> {
    pub context: &'ctx Context,
    pub builder: Builder<'ctx>,
    pub module: Module<'ctx>,

    pub int_type: IntType<'ctx>,
    pub void_type: VoidType<'ctx>,

    pub current_fn: Option<Function<'ast, 'ctx>>,
    pub loops: Vec<Loop<'ctx>>,
    pub scopes: Vec<HashMap<&'ast str, Variable<'ctx>>>,
}

pub struct Loop<'ctx> {
    /// Saves the loop_start basic block (for `continue`)
    loop_head: BasicBlock<'ctx>,
    /// Saves the after_loop basic block (for `break`)
    after_loop: BasicBlock<'ctx>,
}

pub struct Function<'ast, 'ctx> {
    /// Specifies the name of the function.
    name: &'ast str,
    /// Holds the LLVM function value.
    llvm_value: FunctionValue<'ctx>,
    return_type: &'ast FuncType,
}

#[derive(Clone)]
pub struct Variable<'ctx>(pub VariableValue<'ctx>);

impl<'ctx> Variable<'ctx> {
    pub fn new_mut(ptr: PointerValue<'ctx>, type_: BasicTypeEnum<'ctx>, origin_type: BasicTypeEnum<'ctx>) -> Self {
        Self(VariableValue::Mut((ptr, type_, origin_type)))
    }

    pub fn new_const(value: BasicValueEnum<'ctx>) -> Self {
        Self(VariableValue::Const(value))
    }
}

#[derive(Clone, Copy, Debug)]
pub enum VariableValue<'ctx> {
    /// A mutable variable which can be assigned to later.
    Mut((PointerValue<'ctx>, BasicTypeEnum<'ctx>, BasicTypeEnum<'ctx>)),
    /// A static variable which is only declared and used.
    Const(BasicValueEnum<'ctx>),
}

impl<'ast, 'ctx> Compiler<'ast, 'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        Compiler {
            context,
            builder: context.create_builder(),
            module: context.create_module("module"),
            int_type: context.i32_type(),
            void_type: context.void_type(),
            loops: Vec::new(),
            current_fn: None,
            scopes: vec![HashMap::new()],
        }
    }

    pub fn generate(&mut self, program: &'ast CompUnit) -> Result<()> {
        program.generate(self)
    }

    pub fn get_llvm_ir(&self) -> String {
        self.module.print_to_string().to_string()
    }

    pub fn write_ir_to_file(&self, path: &str) {
        self.module.print_to_file(path).unwrap()
    }

    pub fn optimize(&self) {
        let pass_manager = PassManager::create(());

        pass_manager.add_promote_memory_to_register_pass();
        pass_manager.add_function_inlining_pass();
        pass_manager.add_global_dce_pass();
        pass_manager.add_constant_merge_pass();

        pass_manager.run_on(&self.module);
    }

    pub fn gen_riskv(&self, path: &Path) {
        Target::initialize_riscv(&InitializationConfig::default());
        let triple = TargetTriple::create("riscv64-unknown-elf");
        let target  = Target::from_triple(&triple).unwrap();
        let target_machine = target
            .create_target_machine(
                &triple,
                TargetMachine::get_host_cpu_name().to_str().unwrap_or_default(),
                TargetMachine::get_host_cpu_features().to_str().unwrap_or_default(),
                inkwell::OptimizationLevel::Default,
                RelocMode::Default,
                CodeModel::Default,
            )
            .unwrap();

        target_machine.write_to_file(&self.module, FileType::Assembly, path).unwrap();
    }

    pub fn no_terminator(&self) -> bool {
        let block = self.builder.get_insert_block();
        let terminator = block.unwrap().get_terminator();
        terminator.is_none()
    }

    pub fn current_fn(&self) -> &Function<'ast, 'ctx> {
        self.current_fn
            .as_ref()
            .expect("this is only called from functions")
    }

    pub fn new_value(&mut self, id: &'ast str, value: Variable<'ctx>) -> Result<()> {
        let cur = self.scopes.last_mut().unwrap();
        if cur.contains_key(id) {
            Err(CompileErr::DuplicatedSymbol(id.to_owned()))
        } else {
            cur.insert(id, value);
            Ok(())
        }
    }

    pub fn value(&self, name: &str) -> Result<Variable<'ctx>> {
        for scope in self.scopes.iter().rev() {
            if let Some(variable) = scope.get(name) {
                return Ok(variable.to_owned());
            }
        }
        Err(CompileErr::SymbolNotFound(name.to_owned()))
    }

    pub fn in_globals(&self, name: &str) -> bool {
        self.scopes[0].contains_key(name)
    }
}

pub trait GenerateProgram<'ast, 'ctx> {
    type Out;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out>;
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for CompUnit {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        for item in &self.items {
            item.generate(compiler)?;
        }
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for GlobalItem {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Decl(decl) => decl.generate(compiler)?,
            Self::FuncDef(def) => def.generate(compiler)?,
        }
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for Decl {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Const(c) => c.generate(compiler),
            Self::Var(v) => v.generate(compiler),
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for ConstDecl {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        for def in &self.defs {
            def.generate(compiler)?;
        }
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for ConstDef {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        let init = self.init.generate(compiler)?;

        match init {
            Initializer::Const(v) => compiler.new_value(&self.id, Variable::new_const(compiler
                .int_type
                .const_int(v, false)
                .as_basic_value_enum()
            ))?,
            Initializer::Value(_) => unreachable!(),
            Initializer::List(_) => {
                let dims = self.dims
                    .iter()
                    .rev()
                    .map(|x| {
                        x.eval(compiler).unwrap() as u32
                    }).collect::<Vec<u32>>();

                let init = init.reshape(&dims)?;
                let len = init.len();

                let (const_array, ty) = init_global_array(compiler, &self.id, init, &dims)?;

                if compiler.current_fn.is_none() {  // global
                    compiler.new_value(&self.id, Variable::new_const(const_array.as_basic_value_enum()))?;
                } else {  // local
                    let ptr = compiler.builder.build_alloca(ty, &self.id);
                    init_local_const_array(compiler, &const_array, ptr, len);
                    compiler.new_value(&self.id, Variable::new_const(ptr.as_basic_value_enum()))?;
                }
            }
        }

        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for ConstInitVal {
    type Out = Initializer<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        Ok(match self {
            Self::Exp(exp) => Initializer::Const(
                exp.eval(compiler).ok_or(CompileErr::DerefInt)?.try_into().unwrap()
            ),
            Self::List(list) => Initializer::List(
                list
                    .iter()
                    .map(|v| v.generate(compiler))
                    .collect::<Result<_>>()?,
            ),
        })
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for VarDecl {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        for def in &self.defs {
            def.generate(compiler)?;
        }
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for VarDef {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        let dims = self.dims
            .iter()
            .rev()
            .map(|x| {
                x.eval(compiler).unwrap() as u32
            })
            .collect::<Vec<u32>>();

        let llvm_type = if dims.is_empty() {
            compiler.int_type.as_basic_type_enum()
        } else {
            dims.iter()
                .fold(
                    compiler.int_type.as_basic_type_enum(),
                    |acc, len| acc.array_type(len.to_owned()).as_basic_type_enum(),
                )
        };

        if compiler.current_fn.is_none() {  // global
            if let Some(ref init) = self.init {
                let init = init.generate(compiler)?;
                match init {
                    Initializer::Const(_) | Initializer::Value(_) => {
                        let global = compiler.module.add_global(compiler.int_type, None, &self.id);
                        let v = match init {
                            Initializer::Const(c) => compiler.int_type.const_int(c, false).as_basic_value_enum(),
                            Initializer::Value(v) => v,
                            _ => unimplemented!(),
                        };
                        global.set_initializer(&v);
                        compiler.new_value(&self.id, Variable::new_mut(
                            global.as_pointer_value(),
                            compiler.int_type.as_basic_type_enum(),
                            compiler.int_type.as_basic_type_enum()
                        ))?;
                    }
                    Initializer::List(_) => {
                        let init = init.reshape(&dims)?;
                        let (global, ty) = init_global_array(compiler, &self.id, init, &dims)?;
                        compiler.new_value(&self.id, Variable::new_mut(
                            global.as_pointer_value(),
                            ty.as_basic_type_enum(),
                            ty.as_basic_type_enum()
                        ))?;
                    }
                }
            } else {
                let global = compiler.module.add_global(llvm_type, None, &self.id);
                compiler.new_value(&self.id, Variable::new_mut(global.as_pointer_value(), llvm_type, llvm_type))?;
            }
        } else {  // local
            let ptr = compiler.builder.build_alloca(llvm_type, &self.id);
            if let Some(ref init) = self.init {
                let init = init.generate(compiler)?;
                match init {
                    Initializer::Const(v) => {
                        let v = compiler
                            .int_type
                            .const_int(v, false);
                        compiler.builder.build_store(ptr, v);
                    }
                    Initializer::Value(v) => {
                        compiler.builder.build_store(ptr, v);
                    }
                    Initializer::List(_) => {
                        let init = init.reshape(&dims)?;
                        let len = init.len();

                        let mut const_count = 0;
                        for v in init.iter() {
                            if let Initializer::Const(_) = v {
                                const_count += 1;
                            }
                        }

                        if const_count == len {  // all elements in the initialization list are constants
                            let (const_array, _) = init_global_array(compiler, &self.id, init, &dims)?;
                            init_local_const_array(compiler, &const_array, ptr, len);
                        } else {
                            for (i, v) in init.iter().enumerate() {
                                let mut index = vec![compiler.int_type.const_zero()];

                                let mut e = i as u32;

                                for d in dims.iter() {
                                    index.insert(1, compiler.int_type.const_int((e % d).into(), false));
                                    e /= d;
                                }

                                let elemptr = unsafe {
                                    compiler.builder
                                            .build_in_bounds_gep(llvm_type, ptr, &index, &format!("elemptr{i}"))
                                };

                                let elem = match v {
                                    Initializer::Const(c) => compiler
                                        .int_type
                                        .const_int(c.to_owned(), false)
                                        .as_basic_value_enum(),
                                    Initializer::Value(v) => v.to_owned(),
                                    _ => unreachable!(),
                                };

                                compiler.builder.build_store(elemptr, elem);
                            }
                        }
                    }
                }
            }
            compiler.new_value(&self.id, Variable::new_mut(ptr, llvm_type, llvm_type))?;
        }

        Ok(())
    }
}

fn init_global_array<'ast, 'ctx>(
    compiler: &mut Compiler<'ast, 'ctx>,
    id: &str,
    init: Vec<Initializer>,
    dims: &Vec<u32>,
) -> Result<(GlobalValue<'ctx>, ArrayType<'ctx>)> {
    let mut values = Vec::new();
    for v in init.into_iter() {
        if let Initializer::Const(c) = v {
            values.push(compiler.int_type.const_int(c, false))
        } else {
            return Err(CompileErr::InvalidInit);
        }
    }

    let mut dims = dims.iter();
    let top_size = *dims.next().unwrap();

    // Create a vector of ArrayValues
    let mut arrays = values
        .chunks(top_size as usize)
        .map(|a| compiler.int_type.const_array(a))
        .collect::<Vec<ArrayValue>>();

    let mut ty = compiler.int_type.array_type(top_size);

    // for each dimension, split the array into futher arrays
    for d in dims {
        arrays = arrays
            .chunks(*d as usize)
            .map(|a| ty.const_array(a))
            .collect::<Vec<ArrayValue>>();

        ty = ty.array_type(*d);
    }

    let const_array = compiler.module.add_global(
        ty,
        Some(AddressSpace::default()),
        &id
    );

    // We actually end up with an array with a single entry
    const_array.set_initializer(&arrays[0]);
    const_array.set_constant(true);

    Ok((const_array, ty))
}

fn init_local_const_array<'ast, 'ctx>(
    compiler: &mut Compiler<'ast, 'ctx>,
    const_array: &GlobalValue<'ctx>,
    ptr: PointerValue<'ctx>,
    len: usize,
) {
    const_array.set_visibility(inkwell::GlobalVisibility::Hidden);
    const_array.set_linkage(inkwell::module::Linkage::Private);
    const_array.set_unnamed_addr(true);

    let bytes_to_copy = len * std::mem::size_of::<i32>();
    compiler.builder.build_memcpy(
        ptr,
        4,
        const_array.as_pointer_value(),
        4,
        compiler.int_type.const_int(bytes_to_copy as u64, false)
    ).unwrap();
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for InitVal {
    type Out = Initializer<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        Ok(match self {
            Self::Exp(exp) => if compiler.current_fn.is_none() {
                Initializer::Const(
                    exp.eval(compiler).ok_or(CompileErr::DerefInt)?.try_into().unwrap()
                )
            } else {
                if let Some(v) = exp.eval(compiler) {
                    Initializer::Const(v.try_into().unwrap())
                } else {
                    Initializer::Value(exp.generate(compiler)?)
                }
            },
            Self::List(list) => Initializer::List(
                list
                    .iter()
                    .map(|v| v.generate(compiler))
                    .collect::<Result<_>>()?,
            ),
        })
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for FuncDef {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        if compiler.in_globals(&self.id) {
            return Err(CompileErr::DuplicatedSymbol(self.id.clone()));
        }

        let mut params_type = Vec::new();
        for param in self.params.iter() {
            let param_type = if param.dims.is_some() {
                BasicMetadataTypeEnum::from(
                    compiler.int_type.ptr_type(AddressSpace::default())
                )
            } else {
                BasicMetadataTypeEnum::from(compiler.int_type)
            };
            params_type.push(param_type);
        }

        let fn_type = match self.ty {
            FuncType::Int => compiler.int_type.fn_type(params_type.as_ref(), false),
            FuncType::Void => compiler.void_type.fn_type(params_type.as_ref(), false),
        };

        let function = compiler.module.add_function(&self.id, fn_type, None);

        // create basic blocks for the function
        let entry_block = compiler.context.append_basic_block(function, "entry");

        // set the current function environment
        compiler.current_fn = Some(Function {
            name: &self.id,
            llvm_value: function,
            return_type: &self.ty,
        });

        // create new scope for the function
        compiler.scopes.push(HashMap::new());

        // bind each parameter to the original value (for later reference)
        compiler.builder.position_at_end(entry_block);

        for (idx, param) in self.params.iter().enumerate() {
            let (llvm_type, origin_type) = if let Some(ref dims) = param.dims {
                (
                    compiler.int_type.ptr_type(AddressSpace::default()).as_basic_type_enum(),
                    dims
                        .iter()
                        .rev()
                        .map(|x| {
                            x.eval(compiler).unwrap() as u32
                        })
                        .fold(
                            compiler.int_type.as_basic_type_enum(),
                            |acc, len| acc.array_type(len).as_basic_type_enum(),
                        )
                        .as_basic_type_enum()
                )
            } else {
                (compiler.int_type.as_basic_type_enum(), compiler.int_type.as_basic_type_enum())
            };

            // allocate a pointer for each parameter (allows mutability)
            let ptr = compiler.builder.build_alloca(
                llvm_type,
                &param.id,
            );
            // get the param's value from the function
            let value = function
                .get_nth_param(idx as u32)
                .expect("the function signatures have been added before");
            value.set_name(&param.id);

            // store the param value in the pointer
            compiler.builder.build_store(ptr, value);

            // insert the pointer / parameter into the current scope
            compiler.new_value(&param.id, Variable::new_mut(ptr, llvm_type, origin_type))?;
        }

        // compile the body
        self.block.generate(compiler)?;

        if !function.verify(true) {
            function.print_to_stderr();
        }

        compiler.scopes.pop();
        compiler.current_fn = None;
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for Block {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        compiler.scopes.push(HashMap::new());

        for item in &self.items {
            item.generate(compiler)?;
        }

        compiler.scopes.pop();
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for BlockItem {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Decl(decl) => decl.generate(compiler),
            Self::Stmt(stmt) => stmt.generate(compiler),
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for Stmt {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Assign(s) => s.generate(compiler),
            Self::ExpStmt(s) => s.generate(compiler),
            Self::Block(s) => s.generate(compiler),
            Self::If(s) => s.generate(compiler),
            Self::While(s) => s.generate(compiler),
            Self::Break(s) => s.generate(compiler),
            Self::Continue(s) => s.generate(compiler),
            Self::Return(s) => s.generate(compiler),
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for Assign {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        let lval = self.lval.generate(compiler)?.into_pointer_value();
        let rhs = self.exp.generate(compiler)?.into_int_value();
        compiler.builder.build_store(lval, rhs);
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for ExpStmt {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        if let Some(exp) = &self.exp {
            exp.generate(compiler)?;
        }
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for If {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        let func_val = compiler.current_fn().llvm_value;

        let if_block = compiler.context.append_basic_block(func_val, "if_block");
        let after_block;

        if let Some(ref else_stmt) = self.else_then {
            let else_block = compiler.context.append_basic_block(func_val, "else_block");
            after_block = compiler.context.append_basic_block(func_val, "after_block");

            let cond_value = self.cond.generate(compiler)?;
            compiler.builder
                .build_conditional_branch(cond_value.into_int_value(), if_block, else_block);

            compiler.builder.position_at_end(if_block);
            self.then.generate(compiler)?;
            if compiler.no_terminator() {
                compiler.builder.build_unconditional_branch(after_block);
            }

            compiler.builder.position_at_end(else_block);
            else_stmt.generate(compiler)?;
        } else {
            after_block = compiler.context.append_basic_block(func_val, "after_block");

            let cond_int_value = self.cond.generate(compiler)?;
            compiler.builder
                .build_conditional_branch(cond_int_value.into_int_value(), if_block, after_block);

            compiler.builder.position_at_end(if_block);
            self.then.generate(compiler)?;
        }

        if compiler.no_terminator() {
            compiler.builder.build_unconditional_branch(after_block);
        }

        compiler.builder.position_at_end(after_block);
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for While {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        let fv = compiler.current_fn().llvm_value;

        let while_head = compiler.context.append_basic_block(fv, "while_head");
        let while_body = compiler.context.append_basic_block(fv, "while_body");
        let after_while = compiler.context.append_basic_block(fv, "after_while");

        // enter the loop from outside
        compiler.builder.build_unconditional_branch(while_head);

        // compile the condition check
        compiler.builder.position_at_end(while_head);
        let cond = self.cond.generate(compiler)?;

        // if the condition is true, jump into the while body, otherwise, quit the loop
        if compiler.no_terminator() {
            compiler.builder
                .build_conditional_branch(cond.into_int_value(), while_body, after_while);
        }

        compiler.loops.push(Loop {
            loop_head: while_head,
            after_loop: after_while,
        });

        // compile the loop body
        compiler.builder.position_at_end(while_body);
        self.body.generate(compiler)?;

        // jump back to the loop head
        if compiler.no_terminator() {
            compiler.builder.build_unconditional_branch(while_head);
        }

        // remove the loop from `loops`
        compiler.loops.pop();

        // place the builder cursor at the end of the `after_loop`
        compiler.builder.position_at_end(after_while);
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for Break {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        let after_loop_block = compiler
            .loops
            .last()
            .as_ref()
            .expect("break is only called in loop bodies")
            .after_loop;
        compiler.builder.build_unconditional_branch(after_loop_block);
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for Continue {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        let loop_start_block = compiler
            .loops
            .last()
            .as_ref()
            .expect("continue is only called in loop bodies")
            .loop_head;
        compiler.builder.build_unconditional_branch(loop_start_block);
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for Return {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        if let FuncType::Int = compiler.current_fn().return_type {
            if let Some(val) = &self.exp {
                let value = val.generate(compiler)?;
                compiler.builder.build_return(Some(&value));
            }
        } else if self.exp.is_some() {
            return Err(CompileErr::RetValInVoidFunc(compiler.current_fn().name.to_owned()));
        } else {
            compiler.builder.build_return(None);
        }
        Ok(())
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for LVal {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match compiler.value(&self.id)?.0 {
            VariableValue::Const(c) => Ok(c),
            VariableValue::Mut((mut value, type_, origin_type)) => {
                if self.indices.is_empty() {
                    Ok(value.as_basic_value_enum())
                } else if type_.is_array_type() {
                    let mut idx_vals = vec![compiler.int_type.const_zero()];
                    idx_vals.extend(self.indices
                        .iter()
                        .map(|expr| compiler
                            .int_type
                            .const_int(expr.eval(compiler).unwrap() as u64, false)
                        ),
                    );
                    Ok(unsafe {
                        compiler.builder.build_in_bounds_gep(
                            type_,
                            value,
                            idx_vals.as_ref(),
                            "index_access"
                        ).as_basic_value_enum()
                    })
                } else {  // array param as left var
                    value = compiler.builder.build_load(
                        compiler.int_type.ptr_type(AddressSpace::default()),
                        value,
                        "array_ptr"
                    ).into_pointer_value();

                    let idx_vals = self.indices
                        .iter()
                        .map(|expr| compiler
                            .int_type
                            .const_int(expr.eval(compiler).unwrap() as u64, false)
                        ).collect::<Vec<IntValue>>();

                    Ok(unsafe {
                        compiler.builder.build_in_bounds_gep(
                            origin_type,
                            value,
                            idx_vals.as_ref(),
                            "index_access"
                        ).as_basic_value_enum()
                    })
                }
            }
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for Exp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        self.lor.generate(compiler)
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for LOrExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::LAnd(exp) => exp.generate(compiler),
            Self::LOrLAnd(lhs, rhs) => {
                let lhs = lhs.generate(compiler)?.into_int_value();
                let rhs = rhs.generate(compiler)?.into_int_value();
                Ok(compiler.builder.build_or(lhs, rhs, "logical_or").as_basic_value_enum())
            }
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for LAndExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Eq(exp) => exp.generate(compiler),
            Self::LAndEq(lhs, rhs) => {
                let lhs = lhs.generate(compiler)?.into_int_value();
                let rhs = rhs.generate(compiler)?.into_int_value();
                Ok(compiler.builder.build_and(lhs, rhs, "logical_and").as_basic_value_enum())
            }
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for EqExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Rel(exp) => exp.generate(compiler),
            Self::EqRel(lhs, op, rhs) => {
                let lhs = lhs.generate(compiler)?.into_int_value();
                let rhs = rhs.generate(compiler)?.into_int_value();
                let v = match op {
                    EqOp::Eq => compiler.builder.build_int_compare(IntPredicate::EQ, lhs, rhs, "int_eq"),
                    EqOp::Neq => compiler.builder.build_int_compare(IntPredicate::NE, lhs, rhs, "int_ne"),
                };
                Ok(v.as_basic_value_enum())
            }
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for RelExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Add(exp) => exp.generate(compiler),
            Self::RelAdd(lhs, op, rhs) => {
                let lhs = lhs.generate(compiler)?.into_int_value();
                let rhs = rhs.generate(compiler)?.into_int_value();
                let v = match op {
                    RelOp::Lt => compiler.builder.build_int_compare(IntPredicate::SLT, lhs, rhs, "int_lt"),
                    RelOp::Gt => compiler.builder.build_int_compare(IntPredicate::SGT, lhs, rhs, "int_gt"),
                    RelOp::Le => compiler.builder.build_int_compare(IntPredicate::SLE, lhs, rhs, "int_le"),
                    RelOp::Ge => compiler.builder.build_int_compare(IntPredicate::SGE, lhs, rhs, "int_ge"),
                };
                Ok(v.as_basic_value_enum())
            }
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for AddExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Mul(exp) => exp.generate(compiler),
            Self::AddMul(lhs, op, rhs) => {
                let lhs = lhs.generate(compiler)?.into_int_value();
                let rhs = rhs.generate(compiler)?.into_int_value();
                let v = match op {
                    AddOp::Add => compiler.builder.build_int_add(lhs, rhs, "int_add"),
                    AddOp::Sub => compiler.builder.build_int_sub(lhs, rhs, "int_sub"),
                };
                Ok(v.as_basic_value_enum())
            }
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for MulExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Unary(exp) => exp.generate(compiler),
            Self::MulUnary(lhs, op, rhs) => {
                let lhs = lhs.generate(compiler)?.into_int_value();
                let rhs = rhs.generate(compiler)?.into_int_value();
                let v = match op {
                    MulOp::Mul => compiler.builder.build_int_mul(lhs, rhs, "int_mul"),
                    MulOp::Div => compiler.builder.build_int_signed_div(lhs, rhs, "int_div"),
                    MulOp::Mod => compiler.builder.build_int_signed_rem(lhs, rhs, "int_mod"),
                };
                Ok(v.as_basic_value_enum())
            }
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for UnaryExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Primary(exp) => exp.generate(compiler),
            Self::Call(call) => call.generate(compiler),
            Self::Unary(op, exp) => {
                let exp = exp.generate(compiler)?.into_int_value();
                let v = match op {
                    UnaryOp::Neg => compiler.builder.build_int_neg(exp, "int_neg"),
                    UnaryOp::LNot => compiler.builder.build_int_compare(
                        IntPredicate::EQ,
                        compiler.int_type.const_int(0_u64, true),
                        exp,
                        "logical_not_result_int",
                    ),
                };
                Ok(v.as_basic_value_enum())
            }
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for PrimaryExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        match self {
            Self::Exp(exp) => exp.generate(compiler),
            Self::Number(num) => Ok(compiler
                .int_type
                .const_int(num.to_owned() as u64, false)
                .as_basic_value_enum()
            ),
            Self::LVal(lval) => {  // as a right var
                Ok(match compiler.value(&lval.id)?.0 {
                    VariableValue::Const(c) => c,
                    VariableValue::Mut((ptr, type_, origin_type)) => {
                        if type_.is_int_type() {
                            compiler.builder.build_load(
                                compiler.int_type,
                                ptr,
                                &lval.id
                            )
                        } else {
                            if lval.indices.is_empty() {  // pass an array as a param to a function
                                unsafe {
                                    compiler.builder.build_in_bounds_gep(
                                        origin_type,
                                        ptr,
                                        vec![compiler.int_type.const_zero(), compiler.int_type.const_zero()].as_ref(),
                                        "array"
                                    ).as_basic_value_enum()
                                }
                            } else if type_.is_array_type() {  // access array
                                let mut idx_vals = vec![compiler.int_type.const_zero()];
                                idx_vals.extend(lval.indices
                                    .iter()
                                    .map(|expr| compiler
                                        .int_type
                                        .const_int(expr.eval(compiler).unwrap() as u64, false)
                                    ),
                                );
                                compiler.builder.build_load(
                                    compiler.int_type,
                                    unsafe {
                                        compiler.builder.build_in_bounds_gep(
                                            type_,
                                            ptr,
                                            idx_vals.as_ref(),
                                            "index_access"
                                        )
                                    },
                                    "array_item"
                                )
                            } else {  // acces array param
                                let idx_vals = lval.indices
                                    .iter()
                                    .map(|expr| compiler
                                        .int_type
                                        .const_int(expr.eval(compiler).unwrap() as u64, false)
                                    ).collect::<Vec<IntValue>>();

                                compiler.builder.build_load(
                                    compiler.int_type,
                                    unsafe {
                                        compiler.builder.build_in_bounds_gep(
                                            origin_type,
                                            ptr,
                                            idx_vals.as_ref(),
                                            "index_access"
                                        )
                                    },
                                    "array_item"
                                )
                            }
                        }
                    }
                })
            }
        }
    }
}

impl<'ast, 'ctx> GenerateProgram<'ast, 'ctx> for FuncCall {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ast, 'ctx>) -> Result<Self::Out> {
        let fv = compiler.module.get_function(&self.id).unwrap();
        if self.args.len() != fv.get_type().count_param_types() as usize {
            return Err(CompileErr::ArgCountMismatch {
                id: self.id.clone(),
                expect: fv.get_type().count_param_types() as usize,
                found: self.args.len(),
            });
        }

        let llvm_params_value = self.args
            .iter()
            .by_ref()
            .map(|arg| arg.generate(compiler).unwrap().into())
            .collect::<Vec<BasicMetadataValueEnum>>();

        Ok(
            compiler.builder.build_call(
                fv,
                llvm_params_value.as_slice(),
                &self.id,
            )
            .try_as_basic_value()
            .left()
            .unwrap_or(compiler.int_type.const_zero().as_basic_value_enum())
        )
    }
}
