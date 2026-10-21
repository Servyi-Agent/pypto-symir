//! Parser and interpreter message constants (servyi `inline_literal`:
//! literals live in named consts; clippy `useless_format`: no argless
//! `format!` at the use sites).

// parser: structure
pub const MSG_NOT_A_DAG: &str = "the extracted graph contains a cycle";
pub const MSG_NO_FUNCTIONS: &str = "the program has no functions";
pub const MSG_INTERNAL_NO_FRAME: &str = "internal: no active frame";
pub const MSG_INTERNAL_NODE_RANGE: &str = "internal: node index out of range";
