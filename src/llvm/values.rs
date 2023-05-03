use std::iter::repeat_with;

use inkwell::values::BasicValueEnum;

use crate::llvm::error::{CompileErr, Result};

/// An initializer.
#[derive(Clone)]
pub enum Initializer<'ctx> {
    Const(u64),
    Value(BasicValueEnum<'ctx>),
    List(Vec<Initializer<'ctx>>),
}

impl<'ctx> Initializer<'ctx> {
    /// Reshapes the current initializer by using the given type.
    /// Returns the reshaped initializer.
    pub fn reshape(self, dims: &Vec<u32>) -> Result<Vec<Initializer<'ctx>>> {
        // array `int a[2][3][4]` yields [(4, 4), (3, 12), (2, 24)]
        let mut last_len = 1;
        let lens = dims
            .iter()
            .map(|l| {
                last_len *= l;
                (l.to_owned(), last_len)
            })
            .collect::<Vec<(u32, u32)>>();

        let mut values = Vec::new();

        match self {
            Self::List(l) if !lens.is_empty() => Self::reshape_impl(l, &lens, &mut values)?,
            _ => return Err(CompileErr::InvalidInit),
        };

        Ok(values)
    }

    fn reshape_impl(inits: Vec<Self>, lens: &[(u32, u32)], values: &mut Vec<Initializer<'ctx>>) -> Result<Self> {
        let mut reshaped: Vec<Vec<Self>> = repeat_with(Vec::new).take(lens.len() + 1).collect();
        let mut len = 0;

        // handle initializer elements
        for init in inits {
            // too many elements
            if len >= lens.last().unwrap().1 {
                return Err(CompileErr::InvalidInit);
            }
            match init {
                Self::List(list) => {
                    // get the next-level length list
                    let next_lens = match reshaped.iter().position(|v| !v.is_empty()) {
                        // not aligned
                        Some(0) => return Err(CompileErr::InvalidInit),
                        Some(i) => &lens[..i],
                        None => &lens[..lens.len() - 1],
                    };
                    // reshape, and add to reshaped initializer list
                    reshaped[next_lens.len()].push(Self::reshape_impl(list, next_lens, values)?);
                    Self::carry(&mut reshaped, lens);
                    len += next_lens.last().unwrap().1;
                }
                _ => {
                    // just push
                    reshaped[0].push(init.clone());
                    values.push(init);
                    Self::carry(&mut reshaped, lens);
                    len += 1;
                }
            }
        }

        // fill zeros
        while len < lens.last().unwrap().1 {
            reshaped[0].push(Self::Const(0));
            values.push(Self::Const(0));
            Self::carry(&mut reshaped, lens);
            len += 1;
        }
        Ok(reshaped.pop().unwrap().pop().unwrap())
    }

    fn carry(reshaped: &mut [Vec<Self>], lens: &[(u32, u32)]) {
        for (i, &(len, _)) in lens.iter().enumerate() {
            if reshaped[i].len() as u32 == len {
                let init = Self::List(reshaped[i].drain(..).collect());
                reshaped[i + 1].push(init);
            }
        }
    }
}
