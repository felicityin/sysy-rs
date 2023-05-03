use crate::llvm::gen::VariableValue;
use crate::llvm::Compiler;
use crate::parser::ast::*;

/// Trait for evaluating constant.
pub trait Evaluate {
    fn eval(&self, compiler: &Compiler) -> Option<i32>;
}

impl Evaluate for ConstExp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        self.exp.eval(compiler)
    }
}

impl Evaluate for Exp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        self.lor.eval(compiler)
    }
}

impl Evaluate for LOrExp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        match self {
            Self::LAnd(exp) => exp.eval(compiler),
            Self::LOrLAnd(lhs, rhs) => match (lhs.eval(compiler), rhs.eval(compiler)) {
                (Some(lhs), Some(rhs)) => Some((lhs != 0 || rhs != 0) as i32),
                _ => None,
            },
        }
    }
}

impl Evaluate for LAndExp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        match self {
            Self::Eq(exp) => exp.eval(compiler),
            Self::LAndEq(lhs, rhs) => match (lhs.eval(compiler), rhs.eval(compiler)) {
                (Some(lhs), Some(rhs)) => Some((lhs != 0 && rhs != 0) as i32),
                _ => None,
            },
        }
    }
}

impl Evaluate for EqExp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        match self {
            Self::Rel(exp) => exp.eval(compiler),
            Self::EqRel(lhs, op, rhs) => match (lhs.eval(compiler), rhs.eval(compiler)) {
                (Some(lhs), Some(rhs)) => Some(match op {
                    EqOp::Eq => (lhs == rhs) as i32,
                    EqOp::Neq => (lhs != rhs) as i32,
                }),
                _ => None,
            },
        }
    }
}

impl Evaluate for RelExp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        match self {
            Self::Add(exp) => exp.eval(compiler),
            Self::RelAdd(lhs, op, rhs) => match (lhs.eval(compiler), rhs.eval(compiler)) {
                (Some(lhs), Some(rhs)) => Some(match op {
                    RelOp::Lt => (lhs < rhs) as i32,
                    RelOp::Gt => (lhs > rhs) as i32,
                    RelOp::Le => (lhs <= rhs) as i32,
                    RelOp::Ge => (lhs >= rhs) as i32,
                }),
                _ => None,
            },
        }
    }
}

impl Evaluate for AddExp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        match self {
            Self::Mul(exp) => exp.eval(compiler),
            Self::AddMul(lhs, op, rhs) => match (lhs.eval(compiler), rhs.eval(compiler)) {
                (Some(lhs), Some(rhs)) => Some(match op {
                        AddOp::Add => lhs + rhs,
                        AddOp::Sub => lhs - rhs,
                    }),
                _ => None,
            },
        }
    }
}

impl Evaluate for MulExp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        match self {
            Self::Unary(exp) => exp.eval(compiler),
            Self::MulUnary(lhs, op, rhs) => match (lhs.eval(compiler), rhs.eval(compiler)) {
                (Some(lhs), Some(rhs)) => match op {
                    MulOp::Mul => Some(lhs * rhs),
                    MulOp::Div => (rhs != 0).then_some(lhs / rhs),
                    MulOp::Mod => (rhs != 0).then_some(lhs % rhs),
                },
                _ => None,
            },
        }
    }
}

impl Evaluate for UnaryExp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        match self {
            Self::Primary(primary) => primary.eval(compiler),
            Self::Call(_) => None,
            Self::Unary(op, exp) => exp.eval(compiler).map(|exp| match op {
                UnaryOp::Neg => -exp,
                UnaryOp::LNot => (exp == 0) as i32,
            }),
        }
    }
}

impl Evaluate for PrimaryExp {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        match self {
            Self::Exp(exp) => exp.eval(compiler),
            Self::LVal(lval) => lval.eval(compiler),
            Self::Number(num) => Some(*num),
        }
    }
}

impl Evaluate for LVal {
    fn eval(&self, compiler: &Compiler) -> Option<i32> {
        let val = compiler.value(&self.id).ok()?;
        if self.indices.is_empty() {
            match val.0 {
                VariableValue::Const(c) => if c.is_int_value() {
                    Some(c.into_int_value().get_zero_extended_constant().unwrap() as i32)
                } else {
                    None
                }
                _ => None,
            }
        } else {
            None
        }
    }
}
