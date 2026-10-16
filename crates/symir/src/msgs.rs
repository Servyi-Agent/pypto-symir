//! Parser and interpreter message constants (servyi `inline_literal`:
//! literals live in named consts; clippy `useless_format`: no argless
//! `format!` at the use sites).

// parser: structure
pub const MSG_EMPTY_CONTENT_LINE: &str = "empty content line";
pub const MSG_LINE_KIND: &str = "line kind";
pub const MSG_STATEMENT_KIND: &str = "statement kind";
pub const MSG_NO_PROGRAM_CLASS: &str = "no @pl.program class found";
pub const MSG_STMT_OUTSIDE_CLASS: &str = "statement outside the program class";
pub const MSG_INCONSISTENT_INDENT: &str = "inconsistent indentation";
pub const MSG_HEADER_WITHOUT_BODY: &str = "block header without a body";
pub const MSG_ELIF_WITHOUT_IF: &str = "elif/else without a preceding if";
pub const MSG_NESTED_DEFINITION: &str = "nested definition inside a function body";
pub const MSG_FUNCTION_WITHOUT_BODY_TMPL: &str = "function";

// parser: headers and statements
pub const MSG_FOR_WITHOUT_ITERATOR: &str = "for header without iterator";
pub const MSG_WITH_WITHOUT_SCOPE: &str = "with header without a scope";
pub const MSG_EXPECTED_EXPRESSION: &str = "expected an expression";
pub const MSG_EXPECTED_IDENTIFIER: &str = "expected an identifier";
pub const MSG_EXPECTED_DEF_NAME: &str = "expected a name in the definition header";
pub const MSG_ASSIGN_WITHOUT_VALUE: &str = "assignment without value";
pub const MSG_EMPTY_STATEMENT: &str = "empty statement";
pub const MSG_PARAM_WITHOUT_NAME: &str = "parameter without a name";

// parser: expressions
pub const MSG_EXPRESSION_WITHOUT_BODY: &str = "expression without body";
pub const MSG_EMPTY_EXPRESSION: &str = "empty expression";
pub const MSG_EMPTY_NAME: &str = "empty name";
pub const MSG_EMPTY_DOTTED_NAME: &str = "empty dotted name";
pub const MSG_EMPTY_STRING: &str = "empty string";
pub const MSG_EMPTY_GROUP: &str = "empty group";
pub const MSG_EMPTY_UNARY: &str = "empty unary";
pub const MSG_EMPTY_NOT_EXPR: &str = "empty not_expr";
pub const MSG_EMPTY_CHAIN: &str = "empty chain";
pub const MSG_UNARY_WITHOUT_OPERAND: &str = "unary without operand";
pub const MSG_NOT_WITHOUT_OPERAND: &str = "not without operand";
pub const MSG_POSTFIX_WITHOUT_PRIMARY: &str = "postfix without primary";
pub const MSG_COMPARISON_WITHOUT_LHS: &str = "comparison without lhs";
pub const MSG_COMPARISON_WITHOUT_OP: &str = "comparison without op";
pub const MSG_COMPARISON_WITHOUT_RHS: &str = "comparison without rhs";
pub const MSG_BOOL_CHAIN_WITHOUT_OPERANDS: &str = "boolean chain without operands";
pub const MSG_KWARG_WITHOUT_KEY: &str = "kwarg without key";
pub const MSG_KWARG_WITHOUT_VALUE: &str = "kwarg without value";
pub const MSG_DICT_ENTRY_WITHOUT_KEY: &str = "dict entry without key";
pub const MSG_DICT_ENTRY_WITHOUT_VALUE: &str = "dict entry without value";
pub const MSG_DICT_KEY_WITHOUT_CONTENT: &str = "dict key without content";
pub const MSG_DECORATOR_WITHOUT_NAME: &str = "decorator without a name";
pub const MSG_CALL_ARGS_WITHOUT_PAYLOAD: &str = "call arguments without payload";
pub const MSG_UNKNOWN_OPERATOR: &str = "unknown operator";

// interpreter
pub const MSG_NOT_A_DAG: &str = "the extracted graph contains a cycle";
pub const MSG_NO_FUNCTIONS: &str = "the program has no functions";
pub const MSG_INTERNAL_NO_FRAME: &str = "internal: no active frame";
pub const MSG_INTERNAL_NODE_RANGE: &str = "internal: node index out of range";
