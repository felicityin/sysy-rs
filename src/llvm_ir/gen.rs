use std::collections::HashMap;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassManager;
use inkwell::types::{BasicMetadataTypeEnum, IntType, VoidType, BasicType};
use inkwell::values::BasicValue;
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue, PointerValue};
use inkwell::IntPredicate;

use crate::llvm_ir::error::CompileErr;
use crate::parser::ast::*;

use super::eval::Evaluate;

pub type Result<T> = std::result::Result<T, CompileErr>;

pub struct Compiler<'ctx, 'ast> {
    pub context: &'ctx Context,
    pub builder: Builder<'ctx>,
    pub module: Module<'ctx>,

    pub int_type: IntType<'ctx>,
    pub void_type: VoidType<'ctx>,

    pub current_fn: Option<Function<'ctx, 'ast>>,
    pub loops: Vec<Loop<'ctx>>,
    pub scopes: Vec<HashMap<&'ast str, Variable<'ctx>>>,
}

pub struct Loop<'ctx> {
    /// Saves the loop_start basic block (for `continue`)
    loop_head: BasicBlock<'ctx>,
    /// Saves the after_loop basic block (for `break`)
    after_loop: BasicBlock<'ctx>,
}

pub struct Function<'ctx, 'ast> {
    /// Specifies the name of the function.
    name: &'ast str,
    /// Holds the LLVM function value.
    llvm_value: FunctionValue<'ctx>,
    return_type: &'ast FuncType,
}

#[derive(Clone)]
pub struct Variable<'ctx>(pub VariableValue<'ctx>);

impl<'ctx> Variable<'ctx> {
    pub fn new_mut(ptr: PointerValue<'ctx>) -> Self {
        Self(VariableValue::Mut(ptr))
    }

    pub fn new_const(value: BasicValueEnum<'ctx>) -> Self {
        Self(VariableValue::Const(value))
    }

    pub fn new_const_val(v: i32) -> Self {
        Self(VariableValue::ConstVal(v))
    }
}

#[derive(Clone, Copy, Debug)]
pub enum VariableValue<'ctx> {
    /// A mutable variable which can be assigned to later.
    Mut(PointerValue<'ctx>),
    /// A static variable which is only declared and used.
    Const(BasicValueEnum<'ctx>),
    /// Const value.
    ConstVal(i32),
}

impl<'ctx, 'ast> Compiler<'ctx, 'ast> {
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

    pub fn no_terminator(&self) -> bool {
        let block = self.builder.get_insert_block();
        let terminator = block.unwrap().get_terminator();
        terminator.is_none()
    }

    pub fn get_llvm_ir(&self) -> String {
        self.module.print_to_string().to_string()
    }

    pub fn write_to_file(&self, path: &str) {
        self.module.print_to_file(path).unwrap()
    }

    pub fn _optimize(&self) {
        let pass_manager = PassManager::create(());

        pass_manager.add_promote_memory_to_register_pass();
        pass_manager.add_function_inlining_pass();
        pass_manager.add_global_dce_pass();
        pass_manager.add_constant_merge_pass();

        pass_manager.run_on(&self.module);
    }

    pub fn current_fn(&self) -> &Function<'ctx, 'ast> {
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

pub trait GenerateProgram<'ctx, 'ast> {
    type Out;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out>;
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for CompUnit {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        for item in &self.items {
            item.generate(compiler)?;
        }
        Ok(())
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for GlobalItem {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        match self {
            Self::Decl(decl) => decl.generate(compiler)?,
            Self::FuncDef(def) => def.generate(compiler)?,
        }
        Ok(())
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for Decl {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        match self {
            Self::Const(c) => c.generate(compiler),
            Self::Var(v) => v.generate(compiler),
        }
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for ConstDecl {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        for def in &self.defs {
            def.generate(compiler)?;
        }
        Ok(())
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for ConstDef {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        let init = self.init.generate(compiler)?;

        if self.dims.is_empty() {
            compiler.new_value(&self.id, Variable::new_const_val(init.1))?;
        } else {
            let ptr = if compiler.current_fn.is_none() {  // global
                let global = compiler.module.add_global(compiler.int_type, None, &self.id);
                global.set_initializer(&init.0);
                global.set_constant(true);
                global.as_basic_value_enum()
            } else {  // local
                let ptr = compiler.builder.build_alloca(compiler.int_type, &self.id);
                compiler.builder.build_store(ptr, init.0);
                ptr.as_basic_value_enum()
            };
            compiler.new_value(&self.id, Variable::new_const(ptr))?;
        }
        Ok(())
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for ConstInitVal {
    type Out = (BasicValueEnum<'ctx>, i32);

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        match self {
            Self::Exp(exp) => {
                let v = exp.eval(compiler).unwrap();
                Ok((compiler
                    .int_type
                    .const_int(v as u64, false)
                    .as_basic_value_enum(), v))
            }
            Self::List(_list) => Err(CompileErr::NotImplement),
        }
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for VarDecl {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        for def in &self.defs {
            def.generate(compiler)?;
        }
        Ok(())
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for VarDef {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        let ptr = if compiler.current_fn.is_none() {  // global
            let global = compiler.module.add_global(compiler.int_type, None, &self.id);
            if let Some(ref init) = self.init {
                let init = init.generate(compiler)?;
                global.set_initializer(&init);
            }
            global.as_pointer_value()
        } else {  // local
            let ptr = compiler.builder.build_alloca(compiler.int_type, &self.id);
            if let Some(ref init) = self.init {
                let init = init.generate(compiler)?;
                compiler.builder.build_store(ptr, init);
            }
            ptr
        };
        compiler.new_value(&self.id, Variable::new_mut(ptr))?;
        Ok(())
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for InitVal {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        match self {
            Self::Exp(exp) => Ok(
                if compiler.current_fn.is_none() {
                    compiler
                        .int_type
                        .const_int(exp.eval(compiler).unwrap() as u64, false)
                        .as_basic_value_enum()
                } else {
                    exp.generate(compiler)?
                }
            ),
            Self::List(_list) => Err(CompileErr::NotImplement),
        }
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for FuncDef {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        if compiler.in_globals(&self.id) {
            return Err(CompileErr::DuplicatedSymbol(self.id.clone()));
        }

        let mut params_type = Vec::new();
        for param in self.params.iter() {
            if let Some(ref _dims) = param.dims {
                return Err(CompileErr::NotImplement);
            } else {
                params_type.push(BasicMetadataTypeEnum::IntType(compiler.int_type));
            }
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
            if let Some(ref _dims) = param.dims {
                return Err(CompileErr::NotImplement);
            } else {
                // allocate a pointer for each parameter (allows mutability)
                let ptr = compiler.builder.build_alloca(
                    compiler.context.i32_type().as_basic_type_enum(),
                    &param.id,
                );
                // get the param's value from the function
                let value = function
                    .get_nth_param(idx as u32)
                    .expect("the function signatures have been added before");

                // store the param value in the pointer
                compiler.builder.build_store(ptr, value);
                // insert the pointer / parameter into the current scope
                compiler.new_value(&param.id, Variable::new_mut(ptr))?;
            }
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for Block {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        compiler.scopes.push(HashMap::new());

        for item in &self.items {
            item.generate(compiler)?;
        }

        compiler.scopes.pop();
        Ok(())
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for BlockItem {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        match self {
            Self::Decl(decl) => decl.generate(compiler),
            Self::Stmt(stmt) => stmt.generate(compiler),
        }
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for Stmt {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for Assign {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        let lval = self.lval.generate(compiler)?.into_pointer_value();
        let rhs = self.exp.generate(compiler)?.into_int_value();
        compiler.builder.build_store(lval, rhs);
        Ok(())
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for ExpStmt {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        if let Some(exp) = &self.exp {
            exp.generate(compiler)?;
        }
        Ok(())
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for If {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        let func_val = compiler.current_fn().llvm_value;

        let if_block = compiler.context.append_basic_block(func_val, "if_block");
        let after_block;

        if let Some(ref else_stmt) = self.else_then {
            let else_block = compiler.context.append_basic_block(func_val, "else_block");
            after_block = compiler.context.append_basic_block(func_val, "after_block");

            let cond_int_value = self.cond.generate(compiler)?;
            compiler.builder
                .build_conditional_branch(cond_int_value.into_int_value(), if_block, else_block);

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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for While {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        let fv = compiler.current_fn().llvm_value;

        // create the `while_head` block
        let while_head = compiler.context.append_basic_block(fv, "while_head");

        // create the `while_body` block
        let while_body = compiler.context.append_basic_block(fv, "while_body");

        // create the `after_while` block
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for Break {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for Continue {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for Return {
    type Out = ();

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for LVal {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        let value = match compiler.value(&self.id)?.0 {
            VariableValue::Mut(value) => value.as_basic_value_enum(),
            VariableValue::Const(c) => c,
            VariableValue::ConstVal(num) => {
                if self.indices.is_empty() {
                    compiler
                        .int_type
                        .const_int(num as u64, false)
                        .as_basic_value_enum()
                } else {
                    return Err(CompileErr::DerefInt);
                }
            }
        };

        if !self.indices.is_empty() {
            return Err(CompileErr::NotImplement);
        }
        Ok(value)
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for Exp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        self.lor.generate(compiler)
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for LOrExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for LAndExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for EqExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for RelExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for AddExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for MulExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for UnaryExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
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

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for PrimaryExp {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        match self {
            Self::Exp(exp) => exp.generate(compiler),
            Self::Number(num) => Ok(compiler
                .int_type
                .const_int(num.to_owned() as u64, false)
                .as_basic_value_enum()
            ),
            Self::LVal(lval) => {
                if !lval.indices.is_empty() {
                    return Err(CompileErr::NotImplement);
                }
                Ok(match compiler.value(&lval.id)?.0 {
                    VariableValue::Mut(ptr) => compiler.builder.build_load(ptr, &lval.id),
                    VariableValue::Const(c) => c,
                    VariableValue::ConstVal(num) => {
                        if lval.indices.is_empty() {
                            compiler
                                .int_type
                                .const_int(num as u64, false)
                                .as_basic_value_enum()
                        } else {
                            return Err(CompileErr::DerefInt);
                        }
                    }
                })
            }
        }
    }
}

impl<'ctx, 'ast> GenerateProgram<'ctx, 'ast> for FuncCall {
    type Out = BasicValueEnum<'ctx>;

    fn generate(&'ast self, compiler: &mut Compiler<'ctx, 'ast>) -> Result<Self::Out> {
        let fv = compiler.module.get_function(&self.id).unwrap();
        if self.args.len() != fv.get_type().count_param_types() as usize {
            return Err(CompileErr::ArgCountMismatch {
                id: self.id.clone(),
                expect: fv.get_type().count_param_types() as usize,
                found: self.args.len(),
            });
        }

        let llvm_params_value: Vec<BasicMetadataValueEnum> = self.args
            .iter()
            .by_ref()
            .map(|arg| arg.generate(compiler).unwrap().into())
            .collect();

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
