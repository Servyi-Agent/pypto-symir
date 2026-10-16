//! Parser: printed PyPTO IR → [`crate::ast`].
//!
//! Two layers:
//!
//! 1. The PEG grammar in `pypto_ir.pest` (an off-the-shelf parser generator,
//!    per the servyi `custom_parser` policy — no hand-rolled parsing
//!    primitives anywhere in this crate) tokenizes each line and parses its
//!    expression content.
//! 2. This module assembles block structure by comparing INDENT widths
//!    (plain arithmetic on `char` counts) and converts pest pairs into the
//!    [`ast`] types.

use std::collections::VecDeque;

use pest::iterators::Pair;
use pest::Parser;
use pest_derive::Parser as DeriveParser;

use crate::ast::{BinOp, Expr, Function, IndexItem, Param, Program, Stmt, UnaryOp};
use crate::consts;
use crate::error::SymirError;

/// How far one step through the flat line list moves.
use consts::LINE_ADVANCE;

#[derive(DeriveParser)]
#[grammar = "pypto_ir.pest"]
struct PyptoParser;

/// One parsed source line.
struct RawLine {
    indent: usize,
    kind: LineKind,
    line: usize,
}

/// The statement-ish shapes a line can have.
enum LineKind {
    Comment,
    Decorator(Expr),
    ClassHeader(String),
    DefHeader {
        name: String,
        params: Vec<Param>,
    },
    For {
        index_vars: Vec<String>,
        iter_arg_vars: Vec<String>,
        iterator: Expr,
    },
    If(Expr),
    Elif(Expr),
    While(Expr),
    ElseHeader,
    With {
        scope: Expr,
        binder: Option<String>,
    },
    Return(Option<Expr>),
    Pass,
    Simple(Stmt),
}

/// Parses a whole printed program.
pub fn parse_program(src: &str) -> Result<Program, SymirError> {
    let parsed =
        PyptoParser::parse(Rule::source_file, src).map_err(|e| SymirError::Parse(e.to_string()))?;
    let mut lines = Vec::new();
    for top in parsed {
        collect_lines(top, &mut lines)?;
    }
    build_program(&lines)
}

/// Flattens the pest tree into [`RawLine`]s.
fn collect_lines(pair: Pair<'_, Rule>, out: &mut Vec<RawLine>) -> Result<(), SymirError> {
    if pair.as_rule() == Rule::content_line {
        let line_no = pair.as_span().start_pos().line_col().0;
        let mut kids = pair.into_inner();
        let mut indent = usize::default();
        if let Some(first) = kids.peek() {
            if first.as_rule() == Rule::INDENT {
                indent = first.as_str().chars().count();
                let _ = kids.next();
            }
        }
        let body = kids.next().ok_or_else(|| SymirError::Unsupported {
            line: line_no,
            message: crate::msgs::MSG_EMPTY_CONTENT_LINE.to_owned(),
        })?;
        let kind = classify(body, line_no)?;
        out.push(RawLine {
            indent,
            kind,
            line: line_no,
        });
        return Ok(());
    }
    for child in pair.into_inner() {
        collect_lines(child, out)?;
    }
    Ok(())
}

/// Turns one line-content pair into a [`LineKind`].
fn classify(body: Pair<'_, Rule>, line_no: usize) -> Result<LineKind, SymirError> {
    match body.as_rule() {
        Rule::comment_line | Rule::import_line => Ok(LineKind::Comment),
        Rule::decorator_line => Ok(LineKind::Decorator(decorator_of(body)?)),
        Rule::class_header => Ok(LineKind::ClassHeader(single_ident_child(body, line_no)?)),
        Rule::def_header => {
            let mut kids: VecDeque<Pair<'_, Rule>> = body.into_inner().collect();
            let name = next_rule(&mut kids, Rule::ident, line_no)?;
            let mut params = Vec::new();
            while let Some(child) = kids.pop_front() {
                if child.as_rule() == Rule::param_list {
                    for p in child.into_inner() {
                        if p.as_rule() == Rule::param {
                            // `param` wraps either `KW_SELF` (stripped,
                            // matching the C++ side) or `annotated_param`.
                            for inner in p.into_inner() {
                                if inner.as_rule() == Rule::annotated_param {
                                    params.push(param_of(inner, line_no)?);
                                }
                            }
                        }
                    }
                }
            }
            Ok(LineKind::DefHeader { name, params })
        }
        Rule::for_header => {
            let mut index_vars = Vec::new();
            let mut iter_arg_vars = Vec::new();
            let mut iterator = None;
            for child in body.into_inner() {
                match child.as_rule() {
                    Rule::for_targets => {
                        for target in child.into_inner() {
                            collect_for_target(target, &mut index_vars, &mut iter_arg_vars);
                        }
                    }
                    Rule::expr => iterator = Some(build_expr(child)?),
                    _ => {}
                }
            }
            let iterator = iterator.ok_or_else(|| SymirError::Unsupported {
                line: line_no,
                message: crate::msgs::MSG_FOR_WITHOUT_ITERATOR.to_owned(),
            })?;
            Ok(LineKind::For {
                index_vars,
                iter_arg_vars,
                iterator,
            })
        }
        Rule::if_header => Ok(LineKind::If(only_expr(body, line_no)?)),
        Rule::while_header => Ok(LineKind::While(only_expr(body, line_no)?)),
        Rule::elif_header => Ok(LineKind::Elif(only_expr(body, line_no)?)),
        Rule::else_header => Ok(LineKind::ElseHeader),
        Rule::with_header => {
            let mut scope = None;
            let mut binder = None;
            for child in body.into_inner() {
                if child.as_rule() == Rule::expr {
                    scope = Some(build_expr(child)?);
                } else if child.as_rule() == Rule::ident {
                    binder = Some(child.as_str().to_owned());
                }
            }
            let scope = scope.ok_or_else(|| SymirError::Unsupported {
                line: line_no,
                message: crate::msgs::MSG_WITH_WITHOUT_SCOPE.to_owned(),
            })?;
            Ok(LineKind::With { scope, binder })
        }
        Rule::return_stmt => {
            // ret_items holds one expr or several (unparenthesized tuple).
            let mut items = Vec::new();
            for child in body.into_inner() {
                if child.as_rule() == Rule::expr {
                    items.push(build_expr(child)?);
                }
            }
            Ok(LineKind::Return(match items.len() {
                0 => None,
                1 => items.pop(),
                _ => Some(Expr::Tuple(items)),
            }))
        }
        Rule::pass_stmt => Ok(LineKind::Pass),
        Rule::statement => Ok(LineKind::Simple(stmt_of(body, line_no)?)),
        other => Err(SymirError::Unsupported {
            line: line_no,
            message: format!("line kind {other:?}"),
        }),
    }
}

/// Splits one `for_target` into index vars (bare) or iter-arg vars (grouped).
fn collect_for_target(
    target: Pair<'_, Rule>,
    index_vars: &mut Vec<String>,
    iter_arg_vars: &mut Vec<String>,
) {
    match target.as_rule() {
        Rule::bare_target => index_vars.push(target.as_str().to_owned()),
        Rule::target_group => {
            for ident_pair in target.into_inner() {
                iter_arg_vars.push(ident_pair.as_str().to_owned());
            }
        }
        _ => {}
    }
}

/// Builds the decorator expression: `@dotted.name(args...)`.
fn decorator_of(body: Pair<'_, Rule>) -> Result<Expr, SymirError> {
    let mut kids: VecDeque<Pair<'_, Rule>> = body.into_inner().collect();
    let name = kids
        .pop_front()
        .ok_or_else(|| SymirError::Parse(crate::msgs::MSG_DECORATOR_WITHOUT_NAME.to_owned()))?;
    let mut expr = build_expr(name)?;
    if let Some(call) = kids.pop_front() {
        expr = apply_trailer(expr, call)?;
    }
    Ok(expr)
}

/// Unwraps a `call_args` pair into its `call_args_inner` payload.
fn call_args_of(pair: Pair<'_, Rule>) -> Result<Pair<'_, Rule>, SymirError> {
    for child in pair.into_inner() {
        if child.as_rule() == Rule::call_args_inner {
            return Ok(child);
        }
    }
    Err(SymirError::Parse(
        crate::msgs::MSG_CALL_ARGS_WITHOUT_PAYLOAD.to_owned(),
    ))
}

/// Converts a `statement` pair into an ast [`Stmt`].
fn stmt_of(body: Pair<'_, Rule>, line_no: usize) -> Result<Stmt, SymirError> {
    let child = body
        .into_inner()
        .next()
        .ok_or_else(|| SymirError::Unsupported {
            line: line_no,
            message: crate::msgs::MSG_EMPTY_STATEMENT.to_owned(),
        })?;
    match child.as_rule() {
        Rule::expr_stmt => Ok(Stmt::Expr(only_expr(child, line_no)?)),
        Rule::assign_stmt => {
            let mut targets = Vec::new();
            let mut exprs: VecDeque<Pair<'_, Rule>> = VecDeque::new();
            // `assign_targets` and `assign_target` are silent: their
            // children (bare_target / target_group / expr) attach directly.
            for inner in child.into_inner() {
                match inner.as_rule() {
                    Rule::expr => exprs.push_back(inner),
                    Rule::bare_target => targets.push(inner.as_str().to_owned()),
                    Rule::target_group => {
                        for ident_pair in inner.into_inner() {
                            targets.push(ident_pair.as_str().to_owned());
                        }
                    }
                    _ => {}
                }
            }
            let annotation = if exprs.len() > 1 {
                Some(pop_expr(&mut exprs, line_no)?)
            } else {
                None
            };
            let value = pop_expr(&mut exprs, line_no)?;
            Ok(Stmt::Assign {
                targets,
                annotation,
                value,
            })
        }
        other => Err(SymirError::Unsupported {
            line: line_no,
            message: format!("statement kind {other:?}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Program assembly from RawLines
// ---------------------------------------------------------------------------

/// Builds the [`Program`] out of the flat line list.
fn build_program(lines: &[RawLine]) -> Result<Program, SymirError> {
    let mut pos = usize::default();
    let mut decorators = Vec::new();
    let mut program_name = None;
    let mut functions = Vec::new();
    let mut globals = Vec::new();
    while let Some(current) = lines.get(pos) {
        match &current.kind {
            LineKind::Comment | LineKind::Pass => {}
            LineKind::Decorator(expr) => decorators.push(expr.clone()),
            LineKind::ClassHeader(name) => {
                program_name = Some(name.clone());
                decorators.clear();
            }
            LineKind::DefHeader { name, params, .. } => {
                let header_pos = pos;
                let body_indent = body_indent_after(lines, header_pos).ok_or_else(|| {
                    SymirError::Unsupported {
                        line: current.line,
                        message: format!("function {name} without a body"),
                    }
                })?;
                pos += LINE_ADVANCE;
                let body = parse_block(lines, &mut pos, body_indent)?;
                functions.push(Function {
                    name: name.clone(),
                    params: params.clone(),
                    decorators: decorators.clone(),
                    body,
                });
                decorators.clear();
                continue;
            }
            LineKind::Simple(Stmt::Assign { targets, value, .. }) if program_name.is_none() => {
                for target in targets {
                    globals.push((target.clone(), value.clone()));
                }
            }
            _ => {
                if program_name.is_none() {
                    return Err(SymirError::Unsupported {
                        line: current.line,
                        message: crate::msgs::MSG_STMT_OUTSIDE_CLASS.to_owned(),
                    });
                }
            }
        }
        pos += LINE_ADVANCE;
    }
    let name = program_name.ok_or_else(|| SymirError::Unsupported {
        line: 0,
        message: crate::msgs::MSG_NO_PROGRAM_CLASS.to_owned(),
    })?;
    Ok(Program {
        name,
        functions,
        globals,
    })
}

/// The indent of the first meaningful line strictly below `header_pos`'s.
fn body_indent_after(lines: &[RawLine], header_pos: usize) -> Option<usize> {
    let header_indent = lines.get(header_pos)?.indent;
    let mut probe = header_pos + LINE_ADVANCE;
    while let Some(candidate) = lines.get(probe) {
        if matches!(candidate.kind, LineKind::Comment) {
            probe += LINE_ADVANCE;
            continue;
        }
        if candidate.indent > header_indent {
            return Some(candidate.indent);
        }
        return None;
    }
    None
}

/// Parses statements at `indent` until dedent; `pos` is advanced.
fn parse_block(lines: &[RawLine], pos: &mut usize, indent: usize) -> Result<Vec<Stmt>, SymirError> {
    let mut stmts = Vec::new();
    while let Some(current) = lines.get(*pos) {
        if matches!(current.kind, LineKind::Comment) {
            *pos += LINE_ADVANCE;
            continue;
        }
        if current.indent < indent {
            break;
        }
        if current.indent > indent {
            return Err(SymirError::Unsupported {
                line: current.line,
                message: crate::msgs::MSG_INCONSISTENT_INDENT.to_owned(),
            });
        }
        match &current.kind {
            LineKind::Comment | LineKind::Pass => {
                *pos += LINE_ADVANCE;
            }
            LineKind::If(cond) => {
                let header_pos = *pos;
                let cond = cond.clone();
                let then_arm = arm_after(lines, pos, header_pos)?;
                let mut arms = vec![(Some(cond), then_arm)];
                while let Some(follow) = lines.get(*pos) {
                    if follow.indent != lines.get(header_pos).map(|l| l.indent).unwrap_or_default()
                    {
                        break;
                    }
                    match &follow.kind {
                        LineKind::Elif(next_cond) => {
                            let next_cond = next_cond.clone();
                            let elif_pos = *pos;
                            let arm = arm_after(lines, pos, elif_pos)?;
                            arms.push((Some(next_cond), arm));
                        }
                        LineKind::ElseHeader => {
                            let else_pos = *pos;
                            let arm = arm_after(lines, pos, else_pos)?;
                            arms.push((None, arm));
                            break;
                        }
                        _ => break,
                    }
                }
                stmts.push(Stmt::If { arms });
            }
            LineKind::While(cond) => {
                let header_pos = *pos;
                let cond = cond.clone();
                let body = block_after(lines, pos, header_pos, current.line)?;
                stmts.push(Stmt::While { cond, body });
            }
            LineKind::Elif(_) | LineKind::ElseHeader => {
                return Err(SymirError::Unsupported {
                    line: current.line,
                    message: crate::msgs::MSG_ELIF_WITHOUT_IF.to_owned(),
                });
            }
            LineKind::For {
                index_vars,
                iter_arg_vars,
                iterator,
            } => {
                let header_pos = *pos;
                let index_vars = index_vars.clone();
                let iter_arg_vars = iter_arg_vars.clone();
                let iterator = iterator.clone();
                let body = block_after(lines, pos, header_pos, current.line)?;
                stmts.push(Stmt::For {
                    index_vars,
                    iter_arg_vars,
                    iterator,
                    body,
                });
            }
            LineKind::With { scope, binder } => {
                let header_pos = *pos;
                let scope = scope.clone();
                let binder = binder.clone();
                let body = block_after(lines, pos, header_pos, current.line)?;
                stmts.push(Stmt::With {
                    scope,
                    binder,
                    body,
                });
            }
            LineKind::Return(value) => {
                let value = value.clone();
                stmts.push(Stmt::Return(value));
                *pos += LINE_ADVANCE;
            }
            LineKind::Simple(stmt) => {
                let stmt = stmt.clone();
                stmts.push(stmt);
                *pos += LINE_ADVANCE;
            }
            LineKind::Decorator(_) | LineKind::ClassHeader { .. } | LineKind::DefHeader { .. } => {
                return Err(SymirError::Unsupported {
                    line: current.line,
                    message: crate::msgs::MSG_NESTED_DEFINITION.to_owned(),
                });
            }
        }
    }
    Ok(stmts)
}

/// Parses the body of the block header at `header_pos`, advancing `pos`.
fn block_after(
    lines: &[RawLine],
    pos: &mut usize,
    header_pos: usize,
    header_line: usize,
) -> Result<Vec<Stmt>, SymirError> {
    let body_indent =
        body_indent_after(lines, header_pos).ok_or_else(|| SymirError::Unsupported {
            line: header_line,
            message: crate::msgs::MSG_HEADER_WITHOUT_BODY.to_owned(),
        })?;
    *pos = header_pos + LINE_ADVANCE;
    parse_block(lines, pos, body_indent)
}

/// Parses one if-arm body (cond or else) after the header at `header_pos`.
fn arm_after(
    lines: &[RawLine],
    pos: &mut usize,
    header_pos: usize,
) -> Result<Vec<Stmt>, SymirError> {
    let header_line = lines.get(header_pos).map(|l| l.line).unwrap_or_default();
    block_after(lines, pos, header_pos, header_line)
}

// ---------------------------------------------------------------------------
// Expression building
// ---------------------------------------------------------------------------

/// Extracts the single `expr` child (required).
fn only_expr(pair: Pair<'_, Rule>, line_no: usize) -> Result<Expr, SymirError> {
    for child in pair.into_inner() {
        if child.as_rule() == Rule::expr {
            return build_expr(child);
        }
    }
    Err(SymirError::Unsupported {
        line: line_no,
        message: crate::msgs::MSG_EXPECTED_EXPRESSION.to_owned(),
    })
}

/// Reads the sole identifier child of a header pair.
fn single_ident_child(pair: Pair<'_, Rule>, line_no: usize) -> Result<String, SymirError> {
    for child in pair.into_inner() {
        if child.as_rule() == Rule::ident {
            return Ok(child.as_str().to_owned());
        }
    }
    Err(SymirError::Unsupported {
        line: line_no,
        message: crate::msgs::MSG_EXPECTED_IDENTIFIER.to_owned(),
    })
}

/// Pops the next pair with the expected rule.
fn next_rule(
    kids: &mut VecDeque<Pair<'_, Rule>>,
    rule: Rule,
    line_no: usize,
) -> Result<String, SymirError> {
    while let Some(front) = kids.pop_front() {
        if front.as_rule() == rule {
            return Ok(front.as_str().to_owned());
        }
    }
    Err(SymirError::Unsupported {
        line: line_no,
        message: crate::msgs::MSG_EXPECTED_DEF_NAME.to_owned(),
    })
}

/// Pops one expression pair off the queue (required).
fn pop_expr(queue: &mut VecDeque<Pair<'_, Rule>>, line_no: usize) -> Result<Expr, SymirError> {
    let front = queue.pop_front().ok_or_else(|| SymirError::Unsupported {
        line: line_no,
        message: crate::msgs::MSG_ASSIGN_WITHOUT_VALUE.to_owned(),
    })?;
    build_expr(front)
}

/// Converts one `annotated_param` pair (ident + annotation).
fn param_of(pair: Pair<'_, Rule>, line_no: usize) -> Result<Param, SymirError> {
    let mut name = None;
    let mut annotation = None;
    for child in pair.into_inner() {
        if child.as_rule() == Rule::ident {
            name = Some(child.as_str().to_owned());
        } else if child.as_rule() == Rule::expr {
            annotation = Some(build_expr(child)?);
        }
    }
    let name = name.ok_or_else(|| SymirError::Unsupported {
        line: line_no,
        message: crate::msgs::MSG_PARAM_WITHOUT_NAME.to_owned(),
    })?;
    Ok(Param {
        name,
        annotation: annotation.unwrap_or(Expr::Name(String::new())),
    })
}

/// Recursive pest-pair → ast-Expr conversion.
fn build_expr(pair: Pair<'_, Rule>) -> Result<Expr, SymirError> {
    match pair.as_rule() {
        Rule::expr => {
            let inner = pair
                .into_inner()
                .next()
                .ok_or_else(|| SymirError::Parse(crate::msgs::MSG_EMPTY_EXPRESSION.to_owned()))?;
            build_expr(inner)
        }
        Rule::or_expr => chain_expr(pair, Rule::and_expr, BinOp::Or),
        Rule::and_expr => chain_expr(pair, Rule::not_expr, BinOp::And),
        Rule::not_expr => {
            let mut kids: VecDeque<Pair<'_, Rule>> = pair.into_inner().collect();
            if let Some(first) = kids.pop_front() {
                if first.as_rule() == Rule::KW_NOT {
                    let inner = kids.pop_front().ok_or_else(|| {
                        SymirError::Parse(crate::msgs::MSG_NOT_WITHOUT_OPERAND.to_owned())
                    })?;
                    return Ok(Expr::UnaryOp(UnaryOp::Not, Box::new(build_expr(inner)?)));
                }
                return build_expr(first);
            }
            Err(SymirError::Parse(
                crate::msgs::MSG_EMPTY_NOT_EXPR.to_owned(),
            ))
        }
        Rule::comparison => {
            let mut kids: VecDeque<Pair<'_, Rule>> = pair.into_inner().collect();
            let lhs = build_expr(kids.pop_front().ok_or_else(|| {
                SymirError::Parse(crate::msgs::MSG_COMPARISON_WITHOUT_LHS.to_owned())
            })?)?;
            if kids.is_empty() {
                return Ok(lhs);
            }
            let op_pair = kids.pop_front().ok_or_else(|| {
                SymirError::Parse(crate::msgs::MSG_COMPARISON_WITHOUT_OP.to_owned())
            })?;
            let rhs_pair = kids.pop_front().ok_or_else(|| {
                SymirError::Parse(crate::msgs::MSG_COMPARISON_WITHOUT_RHS.to_owned())
            })?;
            let op = binop_of_token(&op_pair)?;
            Ok(Expr::BinOp(
                op,
                Box::new(lhs),
                Box::new(build_expr(rhs_pair)?),
            ))
        }
        Rule::additive | Rule::multiplicative => {
            let mut operands = Vec::new();
            let mut ops = Vec::new();
            for child in pair.into_inner() {
                if child.as_rule() == Rule::add_op || child.as_rule() == Rule::mul_op {
                    ops.push(binop_of_token(&child)?);
                } else {
                    operands.push(build_expr(child)?);
                }
            }
            if operands.len() == ops.len().saturating_add(1) && !ops.is_empty() {
                let mut folded = operands.remove(0);
                for (op, operand) in ops.into_iter().zip(operands) {
                    folded = Expr::BinOp(op, Box::new(folded), Box::new(operand));
                }
                return Ok(folded);
            }
            operands
                .pop()
                .ok_or_else(|| SymirError::Parse(crate::msgs::MSG_EMPTY_CHAIN.to_owned()))
        }
        Rule::unary => {
            let mut kids: VecDeque<Pair<'_, Rule>> = pair.into_inner().collect();
            if let Some(first) = kids.pop_front() {
                if first.as_rule() == Rule::unary_op {
                    let inner = kids.pop_front().ok_or_else(|| {
                        SymirError::Parse(crate::msgs::MSG_UNARY_WITHOUT_OPERAND.to_owned())
                    })?;
                    return Ok(Expr::UnaryOp(UnaryOp::Neg, Box::new(build_expr(inner)?)));
                }
                return build_expr(first);
            }
            Err(SymirError::Parse(crate::msgs::MSG_EMPTY_UNARY.to_owned()))
        }
        Rule::postfix => {
            let mut kids: VecDeque<Pair<'_, Rule>> = pair.into_inner().collect();
            let primary = kids.pop_front().ok_or_else(|| {
                SymirError::Parse(crate::msgs::MSG_POSTFIX_WITHOUT_PRIMARY.to_owned())
            })?;
            let mut expr = build_expr(primary)?;
            while let Some(trailer) = kids.pop_front() {
                expr = apply_trailer(expr, trailer)?;
            }
            Ok(expr)
        }
        Rule::int_lit => {
            let value = pair
                .as_str()
                .parse::<i64>()
                .map_err(|e| SymirError::Parse(format!("{e}")))?;
            Ok(Expr::Int(value))
        }
        Rule::float_lit => {
            let value = pair
                .as_str()
                .parse::<f64>()
                .map_err(|e| SymirError::Parse(format!("{e}")))?;
            Ok(Expr::Float(value))
        }
        Rule::string_lit => {
            let mut inner = None;
            for wrapper in pair.into_inner() {
                // dq_string / sq_string wrap the captured inner text.
                for child in wrapper.into_inner() {
                    match child.as_rule() {
                        Rule::dq_inner | Rule::sq_inner => {
                            inner = Some(child.as_str().to_owned());
                        }
                        _ => {}
                    }
                }
            }
            Ok(Expr::Str(inner.ok_or_else(|| {
                SymirError::Parse(crate::msgs::MSG_EMPTY_STRING.to_owned())
            })?))
        }
        Rule::bool_lit => Ok(Expr::Bool(pair.as_str() == consts::BUILTIN_TRUE)),
        Rule::dict_lit => {
            let mut entries = Vec::new();
            for child in pair.into_inner() {
                if child.as_rule() == Rule::dict_entry {
                    let mut pair_kids = child.into_inner();
                    let key = pair_kids.next().ok_or_else(|| {
                        SymirError::Parse(crate::msgs::MSG_DICT_ENTRY_WITHOUT_KEY.to_owned())
                    })?;
                    let value = pair_kids.next().ok_or_else(|| {
                        SymirError::Parse(crate::msgs::MSG_DICT_ENTRY_WITHOUT_VALUE.to_owned())
                    })?;
                    // `dict_key` wraps a string/int/bool literal.
                    let key_expr = if key.as_rule() == Rule::dict_key {
                        key.into_inner().next()
                    } else {
                        Some(key)
                    };
                    let key_expr = key_expr.ok_or_else(|| {
                        SymirError::Parse(crate::msgs::MSG_DICT_KEY_WITHOUT_CONTENT.to_owned())
                    })?;
                    entries.push((build_expr(key_expr)?, build_expr(value)?));
                }
            }
            Ok(Expr::Dict(entries))
        }
        Rule::list_lit => {
            let mut items = Vec::new();
            for child in pair.into_inner() {
                if child.as_rule() == Rule::expr {
                    items.push(build_expr(child)?);
                }
            }
            Ok(Expr::List(items))
        }
        Rule::tuple_lit => {
            let mut items = Vec::new();
            for child in pair.into_inner() {
                if child.as_rule() == Rule::expr {
                    items.push(build_expr(child)?);
                }
            }
            Ok(Expr::Tuple(items))
        }
        Rule::group_lit => {
            let inner = pair
                .into_inner()
                .next()
                .ok_or_else(|| SymirError::Parse(crate::msgs::MSG_EMPTY_GROUP.to_owned()))?;
            build_expr(inner)
        }
        Rule::unit_lit => Ok(Expr::Tuple(Vec::new())),
        Rule::name => {
            let inner = pair
                .into_inner()
                .next()
                .ok_or_else(|| SymirError::Parse(crate::msgs::MSG_EMPTY_NAME.to_owned()))?;
            build_expr(inner)
        }
        Rule::dotted_name => dotted_name_of(pair),
        Rule::ident => Ok(Expr::Name(pair.as_str().to_owned())),
        other => Err(SymirError::Parse(format!(
            "unexpected expression rule {other:?}"
        ))),
    }
}

/// Builds a left-associated boolean chain (`a and b and c`).
fn chain_expr(pair: Pair<'_, Rule>, operand_rule: Rule, op: BinOp) -> Result<Expr, SymirError> {
    let mut operands = Vec::new();
    for child in pair.into_inner() {
        if child.as_rule() == operand_rule {
            operands.push(build_expr(child)?);
        }
    }
    if operands.is_empty() {
        return Err(SymirError::Parse(
            crate::msgs::MSG_BOOL_CHAIN_WITHOUT_OPERANDS.to_owned(),
        ));
    }
    let mut folded = operands.remove(0);
    for operand in operands {
        folded = Expr::BinOp(op, Box::new(folded), Box::new(operand));
    }
    Ok(folded)
}

/// Builds the attribute chain out of a `dotted_name` pair.
fn dotted_name_of(pair: Pair<'_, Rule>) -> Result<Expr, SymirError> {
    let mut parts: VecDeque<String> = pair.into_inner().map(|p| p.as_str().to_owned()).collect();
    let head = parts
        .pop_front()
        .ok_or_else(|| SymirError::Parse(crate::msgs::MSG_EMPTY_DOTTED_NAME.to_owned()))?;
    let mut expr = Expr::Name(head);
    while let Some(part) = parts.pop_front() {
        expr = Expr::Attr(Box::new(expr), part);
    }
    Ok(expr)
}

/// Applies one postfix trailer (call, subscript, attribute) to `expr`.
fn apply_trailer(base: Expr, trailer: Pair<'_, Rule>) -> Result<Expr, SymirError> {
    match trailer.as_rule() {
        Rule::call_args => {
            let inner = call_args_of(trailer)?;
            call_args_inner_of(base, inner)
        }
        Rule::subscript => {
            let mut indices = Vec::new();
            // `index_item` is silent and wraps a non-silent `index_expr`,
            // whose child is the `expr` — descend both levels.
            for child in trailer.into_inner() {
                match child.as_rule() {
                    Rule::slice_all => indices.push(IndexItem::SliceAll),
                    Rule::index_expr => {
                        let Some(inner) = child.into_inner().next() else {
                            return Err(SymirError::Parse(
                                crate::msgs::MSG_EMPTY_EXPRESSION.to_owned(),
                            ));
                        };
                        indices.push(IndexItem::Expr(Box::new(build_expr(inner)?)));
                    }
                    _ => {}
                }
            }
            Ok(Expr::Subscript {
                base: Box::new(base),
                indices,
            })
        }
        Rule::ident => Ok(Expr::Attr(Box::new(base), trailer.as_str().to_owned())),
        other => Err(SymirError::Parse(format!("unexpected trailer {other:?}"))),
    }
}

/// Builds `base(args..., kwargs...)` out of a `call_args_inner` pair.
fn call_args_inner_of(base: Expr, inner: Pair<'_, Rule>) -> Result<Expr, SymirError> {
    let mut args = Vec::new();
    let mut kwargs = Vec::new();
    // `call_arg` is silent: `kwarg` and `expr` pairs attach directly.
    for arg in inner.into_inner() {
        if arg.as_rule() == Rule::kwarg {
            let mut kv = arg.into_inner();
            let key = kv
                .next()
                .ok_or_else(|| SymirError::Parse(crate::msgs::MSG_KWARG_WITHOUT_KEY.to_owned()))?;
            let value = kv.next().ok_or_else(|| {
                SymirError::Parse(crate::msgs::MSG_KWARG_WITHOUT_VALUE.to_owned())
            })?;
            kwargs.push((key.as_str().to_owned(), build_expr(value)?));
        } else {
            args.push(build_expr(arg)?);
        }
    }
    Ok(Expr::Call {
        func: Box::new(base),
        args,
        kwargs,
    })
}

/// Maps an operator token pair to its [`BinOp`].
fn binop_of_token(pair: &Pair<'_, Rule>) -> Result<BinOp, SymirError> {
    let text = pair.as_str();
    if text == consts::TOK_PLUS {
        return Ok(BinOp::Add);
    }
    if text == consts::TOK_MINUS {
        return Ok(BinOp::Sub);
    }
    if text == consts::TOK_STAR {
        return Ok(BinOp::Mul);
    }
    if text == consts::TOK_SLASH {
        return Ok(BinOp::FloatDiv);
    }
    if text == consts::TOK_DOUBLE_SLASH {
        return Ok(BinOp::FloorDiv);
    }
    if text == consts::TOK_PERCENT {
        return Ok(BinOp::Mod);
    }
    if text == consts::TOK_EQ {
        return Ok(BinOp::Eq);
    }
    if text == consts::TOK_NE {
        return Ok(BinOp::Ne);
    }
    if text == consts::TOK_LT {
        return Ok(BinOp::Lt);
    }
    if text == consts::TOK_LE {
        return Ok(BinOp::Le);
    }
    if text == consts::TOK_GT {
        return Ok(BinOp::Gt);
    }
    if text == consts::TOK_GE {
        return Ok(BinOp::Ge);
    }
    Err(SymirError::Parse(
        crate::msgs::MSG_UNKNOWN_OPERATOR.to_owned(),
    ))
}
