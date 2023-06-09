use thiserror::Error;

pub type Result<T> = std::result::Result<T, CompileErr>;

#[derive(Error, Debug)]
pub enum CompileErr {
    #[error("Duplicated symbol {0}")]
    DuplicatedSymbol(String),

    #[error("Symbol not found {0}")]
    SymbolNotFound(String),

    #[error("Parameter count of function {id:?} is incorrect, expect {expect:?}, found {found:?}")]
    ArgCountMismatch { id: String, expect: usize, found: usize },

    #[error("Return val in void function {0}")]
    RetValInVoidFunc(String),

    #[error("Deref int error ")]
    DerefInt,

    #[error("Invalid int error ")]
    InvalidInit,
}
