//! Exact integer/linear programming bridge exposed to Python.
//!
//! All Python-side solving (constructor permutation MILP, the milp engine,
//! and the physical-layout model) goes through this single entry point, so
//! the whole project shares the one Z3 build embedded in this extension.

use std::collections::BTreeMap;

use num_bigint::BigInt;
use num_traits::{One, ToPrimitive, Zero};
use pyo3::prelude::*;
use serde_json::{json, Value};
use z3::ast::{Ast, Bool, Int, Real};
use z3::{Config, Context, Optimize, SatResult};

use crate::Rat;

fn parse_rat(raw: &str) -> Result<Rat, String> {
    let raw = raw.trim();
    let (numerator, denominator) = raw.split_once('/').unwrap_or((raw, "1"));
    let numerator = numerator
        .trim()
        .parse::<BigInt>()
        .map_err(|error| format!("invalid numerator {numerator:?}: {error}"))?;
    let denominator = denominator
        .trim()
        .parse::<BigInt>()
        .map_err(|error| format!("invalid denominator {denominator:?}: {error}"))?;
    if denominator.is_zero() {
        return Err("zero denominator".into());
    }
    Ok(Rat::new(numerator, denominator))
}

/// Parses a Z3 numeral rendered in SMT-LIB style, e.g. `(/ 3 4)`, `(- 2)`,
/// or a plain integer.
fn parse_z3_numeral(raw: &str) -> Result<Rat, String> {
    fn integer(raw: &str) -> Result<BigInt, String> {
        let raw = raw.trim().strip_suffix(".0").unwrap_or(raw.trim());
        raw.parse::<BigInt>()
            .map_err(|error| format!("invalid Z3 integer {raw:?}: {error}"))
    }

    let raw = raw.trim();
    if let Some(inner) = raw.strip_prefix("(- ").and_then(|s| s.strip_suffix(')')) {
        return Ok(-parse_z3_numeral(inner)?);
    }
    if let Some(inner) = raw.strip_prefix("(/ ").and_then(|s| s.strip_suffix(')')) {
        let mut pieces = inner.split_whitespace();
        let numerator = pieces
            .next()
            .ok_or_else(|| format!("invalid Z3 division numeral {raw:?}"))?;
        let denominator = pieces
            .next()
            .ok_or_else(|| format!("invalid Z3 division numeral {raw:?}"))?;
        if pieces.next().is_some() {
            return Err(format!("invalid Z3 division numeral {raw:?}"));
        }
        return Ok(Rat::new(integer(numerator)?, integer(denominator)?));
    }
    let (numerator, denominator) = raw.split_once('/').unwrap_or((raw, "1"));
    Ok(Rat::new(integer(numerator)?, integer(denominator)?))
}

fn rat_to_real<'ctx>(ctx: &'ctx Context, value: &Rat) -> Real<'ctx> {
    Real::from_real_str(
        ctx,
        &value.numer().to_str_radix(10),
        &value.denom().to_str_radix(10),
    )
    .expect("rational numeral")
}

fn get_string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("missing string field {key:?}"))
}

fn string_array(value: &Value) -> Result<Vec<String>, String> {
    let entries = value
        .as_array()
        .ok_or_else(|| "expected an array of strings".to_owned())?;
    entries
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "expected an array of strings".to_owned())
        })
        .collect()
}

enum VarAst<'ctx> {
    Bool(Bool<'ctx>),
    Int(Int<'ctx>),
    Real(Real<'ctx>),
}

struct BridgeSolver<'ctx> {
    optimize: Optimize<'ctx>,
    variables: BTreeMap<String, VarAst<'ctx>>,
    /// Real form of every variable, for linear expressions.
    linear: BTreeMap<String, Real<'ctx>>,
}

impl<'ctx> BridgeSolver<'ctx> {
    fn linear_of(&self, name: &str) -> Result<Real<'ctx>, String> {
        self.linear
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown variable {name:?}"))
    }

    fn bool_of(&self, name: &str) -> Result<Bool<'ctx>, String> {
        match self.variables.get(name) {
            Some(VarAst::Bool(ast)) => Ok(ast.clone()),
            _ => Err(format!("variable {name:?} is not boolean")),
        }
    }

    fn expr(&self, ctx: &'ctx Context, terms: &[Value]) -> Result<Real<'ctx>, String> {
        let mut expression = Real::from_real(ctx, 0, 1);
        for term in terms {
            let pair = term
                .as_array()
                .ok_or_else(|| "constraint term must be [coefficient, variable]".to_owned())?;
            if pair.len() != 2 {
                return Err("constraint term must be [coefficient, variable]".into());
            }
            let coefficient = parse_rat(
                pair[0]
                    .as_str()
                    .ok_or_else(|| "term coefficient must be a string".to_owned())?,
            )?;
            let name = pair[1]
                .as_str()
                .ok_or_else(|| "term variable must be a string".to_owned())?;
            expression = expression + rat_to_real(ctx, &coefficient) * self.linear_of(name)?;
        }
        Ok(expression)
    }

    /// `only_if` is `null`, a variable name, `"!name"`, or a list of both.
    fn conditions(&self, only_if: Option<&Value>) -> Result<Vec<Bool<'ctx>>, String> {
        let entries: Vec<String> = match only_if {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::String(name)) => vec![name.clone()],
            Some(value) => string_array(value)?,
        };
        let mut conditions = Vec::with_capacity(entries.len());
        for entry in entries {
            let (negated, name) = match entry.strip_prefix('!') {
                Some(rest) => (true, rest.to_owned()),
                None => (false, entry),
            };
            let VarAst::Bool(ast) = self
                .variables
                .get(&name)
                .ok_or_else(|| format!("unknown only_if variable {name:?}"))?
            else {
                return Err(format!("only_if variable {name:?} is not boolean"));
            };
            conditions.push(if negated { ast.not() } else { ast.clone() });
        }
        Ok(conditions)
    }

    /// Pseudo-boolean encoding for constraints whose terms are all boolean
    /// variables with integer weights; far faster than linear arithmetic on
    /// ite-expanded booleans.
    fn assert_pseudo_boolean(
        &self,
        ctx: &'ctx Context,
        terms: &[Value],
        op: &str,
        rhs: i64,
        conditions: &[Bool<'ctx>],
    ) -> Result<(), String> {
        let mut literals: Vec<(Bool<'ctx>, i64)> = Vec::with_capacity(terms.len());
        let mut offset: i64 = 0;
        for term in terms {
            let pair = term.as_array().ok_or("term must be [weight, variable]")?;
            let coefficient = parse_rat(pair[0].as_str().ok_or("weight must be a string")?)?;
            if !coefficient.is_integer() {
                return Err("non-integer weight".into());
            }
            let weight = coefficient.numer().to_i64().ok_or("weight out of range")?;
            let name = pair[1].as_str().ok_or("variable must be a string")?;
            let ast = self.bool_of(name)?;
            if weight >= 0 {
                literals.push((ast, weight));
            } else {
                // c*x with c < 0  <=>  c - (-c)*(not x)
                offset += weight;
                literals.push((ast.not(), -weight));
            }
        }
        let k = rhs - offset;
        let references: Vec<(&Bool<'ctx>, i32)> = literals
            .iter()
            .map(|(ast, weight)| (ast, *weight as i32))
            .collect();
        let assertion = match op {
            "le" => Bool::pb_le(ctx, &references, k as i32),
            "ge" => Bool::pb_ge(ctx, &references, k as i32),
            "eq" => Bool::pb_eq(ctx, &references, k as i32),
            other => return Err(format!("unknown constraint op {other:?}")),
        };
        let assertion = conditions
            .iter()
            .rfold(assertion, |implication, condition| {
                condition.implies(&implication)
            });
        self.optimize.assert(&assertion);
        Ok(())
    }

    fn assert_linear(
        &self,
        ctx: &'ctx Context,
        expression: Real<'ctx>,
        op: &str,
        rhs: &Rat,
        conditions: &[Bool<'ctx>],
    ) -> Result<(), String> {
        let rhs_ast = rat_to_real(ctx, rhs);
        let constraint = match op {
            "le" => expression.le(&rhs_ast),
            "ge" => expression.ge(&rhs_ast),
            "eq" => expression._eq(&rhs_ast),
            other => return Err(format!("unknown constraint op {other:?}")),
        };
        let assertion = conditions
            .iter()
            .rfold(constraint, |implication, condition| {
                condition.implies(&implication)
            });
        self.optimize.assert(&assertion);
        Ok(())
    }
}

fn var_ast<'ctx>(ctx: &'ctx Context, kind: &str, name: &str) -> Result<VarAst<'ctx>, String> {
    match kind {
        "bool" => Ok(VarAst::Bool(Bool::new_const(ctx, name.to_owned()))),
        "int" => Ok(VarAst::Int(Int::new_const(ctx, name.to_owned()))),
        "real" => Ok(VarAst::Real(Real::new_const(ctx, name.to_owned()))),
        other => Err(format!("unknown variable kind {other:?}")),
    }
}

fn to_real_bool<'ctx>(ctx: &'ctx Context, ast: &Bool<'ctx>) -> Real<'ctx> {
    let one = Int::from_i64(ctx, 1);
    let zero = Int::from_i64(ctx, 0);
    Real::from_int(&ast.ite(&one, &zero))
}

fn read_value(model: &z3::Model<'_>, variable: &VarAst<'_>) -> Value {
    match variable {
        VarAst::Bool(ast) => match model.eval(ast, true).and_then(|value| value.as_bool()) {
            Some(true) => json!("1"),
            Some(false) => json!("0"),
            None => json!(null),
        },
        VarAst::Int(ast) => match model.eval(ast, true).map(|value| value.to_string()) {
            Some(raw) => match parse_z3_numeral(&raw) {
                Ok(value) => json!(value.numer().to_str_radix(10)),
                Err(_) => json!(null),
            },
            None => json!(null),
        },
        VarAst::Real(ast) => match model.eval(ast, true).map(|value| value.to_string()) {
            Some(raw) => match parse_z3_numeral(&raw) {
                Ok(value) => json!(format!(
                    "{}/{}",
                    value.numer().to_str_radix(10),
                    value.denom().to_str_radix(10)
                )),
                Err(_) => json!(null),
            },
            None => json!(null),
        },
    }
}

fn build<'ctx>(
    ctx: &'ctx Context,
    spec: &Value,
    with_hints: bool,
) -> Result<(BridgeSolver<'ctx>, Option<Real<'ctx>>), String> {
    let mut solver = BridgeSolver {
        optimize: Optimize::new(ctx),
        variables: BTreeMap::new(),
        linear: BTreeMap::new(),
    };

    let vars = spec
        .get("vars")
        .and_then(Value::as_array)
        .ok_or_else(|| "spec must contain a vars array".to_owned())?;
    for item in vars {
        let name = get_string(item, "name")?;
        let kind = get_string(item, "kind")?;
        let ast = var_ast(ctx, &kind, &name)?;
        let linear = match &ast {
            VarAst::Bool(bool_ast) => to_real_bool(ctx, bool_ast),
            VarAst::Int(int_ast) => Real::from_int(int_ast),
            VarAst::Real(real_ast) => real_ast.clone(),
        };
        if let Some(lo) = item.get("lo").and_then(Value::as_str) {
            solver
                .optimize
                .assert(&linear.ge(&rat_to_real(ctx, &parse_rat(lo)?)));
        }
        if let Some(hi) = item.get("hi").and_then(Value::as_str) {
            solver
                .optimize
                .assert(&linear.le(&rat_to_real(ctx, &parse_rat(hi)?)));
        }
        solver.variables.insert(name.clone(), ast);
        solver.linear.insert(name, linear);
    }

    if let Some(constraints) = spec.get("constraints").and_then(Value::as_array) {
        for constraint in constraints {
            let terms = constraint
                .get("terms")
                .and_then(Value::as_array)
                .ok_or_else(|| "constraint must contain a terms array".to_owned())?;
            let op = get_string(constraint, "op")?;
            let rhs = parse_rat(
                constraint
                    .get("rhs")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "constraint must contain an rhs string".to_owned())?,
            )?;
            let conditions = solver.conditions(constraint.get("only_if"))?;
            // Prefer the native pseudo-boolean encoding for pure boolean
            // constraints; it is dramatically faster than linear arithmetic
            // on ite-expanded booleans.
            let all_boolean = terms.iter().all(|term| {
                term.as_array()
                    .and_then(|pair| pair[1].as_str())
                    .and_then(|name| solver.variables.get(name))
                    .is_some_and(|variable| matches!(variable, VarAst::Bool(_)))
            });
            let pb_done = all_boolean
                && rhs.is_integer()
                && solver
                    .assert_pseudo_boolean(
                        ctx,
                        terms,
                        &op,
                        rhs.numer().to_i64().unwrap_or(0),
                        &conditions,
                    )
                    .is_ok();
            if !pb_done {
                let expression = solver.expr(ctx, terms)?;
                solver.assert_linear(ctx, expression, &op, &rhs, &conditions)?;
            }
        }
    }

    if let Some(groups) = spec.get("all_different").and_then(Value::as_array) {
        for group in groups {
            let names = string_array(group)?;
            let mut distinct = Vec::with_capacity(names.len());
            for name in &names {
                distinct.push(solver.linear_of(name)?);
            }
            let references: Vec<&Real<'ctx>> = distinct.iter().collect();
            solver.optimize.assert(&Ast::distinct(ctx, &references));
        }
    }

    if let Some(groups) = spec.get("exactly_one").and_then(Value::as_array) {
        for group in groups {
            let names = string_array(group)?;
            let bools: Vec<Bool<'ctx>> = names
                .iter()
                .map(|name| solver.bool_of(name))
                .collect::<Result<_, _>>()?;
            let references: Vec<(&Bool<'ctx>, i32)> =
                bools.iter().map(|ast| (ast, 1)).collect();
            solver.optimize.assert(&Bool::pb_eq(ctx, &references, 1));
        }
    }

    if let Some(tables) = spec.get("table").and_then(Value::as_array) {
        for table in tables {
            let vars_field = table
                .get("vars")
                .ok_or_else(|| "table must contain a vars array".to_owned())?;
            let names = string_array(vars_field)?;
            let tuples = table
                .get("tuples")
                .and_then(Value::as_array)
                .ok_or_else(|| "table must contain a tuples array".to_owned())?;
            let mut alternatives = Vec::with_capacity(tuples.len());
            for tuple in tuples {
                let values = tuple
                    .as_array()
                    .ok_or_else(|| "table tuples must be arrays".to_owned())?;
                if values.len() != names.len() {
                    return Err("table tuple length mismatches vars length".into());
                }
                let mut equalities: Vec<Bool<'ctx>> = Vec::with_capacity(names.len());
                for (name, value) in names.iter().zip(values) {
                    let value = value
                        .as_i64()
                        .ok_or_else(|| "table values must be integers".to_owned())?;
                    equalities.push(
                        solver
                            .linear_of(name)?
                            ._eq(&Real::from_real(ctx, value as i32, 1)),
                    );
                }
                let references: Vec<&Bool<'ctx>> = equalities.iter().collect();
                alternatives.push(Bool::and(ctx, &references));
            }
            if !alternatives.is_empty() {
                let references: Vec<&Bool<'ctx>> = alternatives.iter().collect();
                solver.optimize.assert(&Bool::or(ctx, &references));
            }
        }
    }

    if with_hints {
        if let Some(hints) = spec.get("hints").and_then(Value::as_array) {
            for hint in hints {
                let pair = hint
                    .as_array()
                    .ok_or_else(|| "hints entries must be [name, value] pairs".to_owned())?;
                if pair.len() != 2 {
                    return Err("hints entries must be [name, value] pairs".into());
                }
                let name = pair[0]
                    .as_str()
                    .ok_or_else(|| "hint variable must be a string".to_owned())?;
                let value = pair[1]
                    .as_i64()
                    .ok_or_else(|| "hint value must be an integer".to_owned())?;
                solver.optimize.assert(
                    &solver
                        .linear_of(name)?
                        ._eq(&Real::from_real(ctx, value as i32, 1)),
                );
            }
        }
    }

    let objective = match spec.get("objective") {
        None | Some(Value::Null) => None,
        Some(objective) => {
            let terms = objective
                .get("terms")
                .and_then(Value::as_array)
                .ok_or_else(|| "objective must contain a terms array".to_owned())?;
            let expression = solver.expr(ctx, terms)?;
            match objective.get("dir").and_then(Value::as_str) {
                Some("min") => solver.optimize.minimize(&expression),
                Some("max") => solver.optimize.maximize(&expression),
                other => return Err(format!("unknown objective direction {other:?}")),
            }
            Some(expression)
        }
    };

    Ok((solver, objective))
}

#[pyfunction]
fn solve_ilp_exact(spec: &str) -> PyResult<String> {
    let parsed: Value = serde_json::from_str(spec).map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!("invalid spec JSON: {error}"))
    })?;
    let timeout_ms = parsed.get("timeout_ms").and_then(Value::as_u64).unwrap_or(0);
    let has_hints = parsed.get("hints").is_some();
    let has_objective = parsed
        .get("objective")
        .is_some_and(|value| !value.is_null());

    for attempt in 0..2 {
        let with_hints = attempt == 0 && has_hints;
        let mut config = Config::new();
        if timeout_ms > 0 {
            config.set_param_value("timeout", &timeout_ms.to_string());
        }
        let ctx = Context::new(&config);
        let (solver, objective) = match build(&ctx, &parsed, with_hints) {
            Ok(solver) => solver,
            Err(message) => {
                let response = json!({ "status": "error", "error": message, "values": {}, "objective": null });
                return Ok(response.to_string());
            }
        };
        match solver.optimize.check(&[]) {
            SatResult::Sat => {
                let Some(model) = solver.optimize.get_model() else {
                    let response = json!({ "status": "error", "error": "no model", "values": {}, "objective": null });
                    return Ok(response.to_string());
                };
                let mut values = serde_json::Map::new();
                for (name, variable) in &solver.variables {
                    values.insert(name.clone(), read_value(&model, variable));
                }
                let objective_value = objective.and_then(|ast| {
                    let evaluated = model.eval(&ast, true)?;
                    let raw = evaluated.to_string();
                    parse_z3_numeral(&raw).ok().map(|value| {
                        json!(format!(
                            "{}/{}",
                            value.numer().to_str_radix(10),
                            value.denom().to_str_radix(10)
                        ))
                    })
                });
                let status = if has_objective { "optimal" } else { "feasible" };
                let response = json!({
                    "status": status,
                    "values": Value::Object(values),
                    "objective": objective_value,
                });
                return Ok(response.to_string());
            }
            SatResult::Unsat if with_hints => continue,
            SatResult::Unsat => {
                let response = json!({ "status": "infeasible", "values": {}, "objective": null });
                return Ok(response.to_string());
            }
            SatResult::Unknown if with_hints => continue,
            SatResult::Unknown => {
                let response = json!({ "status": "timeout", "values": {}, "objective": null });
                return Ok(response.to_string());
            }
        }
    }
    unreachable!("the hint fallback attempt always terminates")
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(solve_ilp_exact, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_ilp() {
        let spec = json!({
            "vars": [
                {"name": "x", "kind": "int", "lo": "0", "hi": "5"},
                {"name": "y", "kind": "int", "lo": "0", "hi": "5"},
            ],
            "constraints": [
                {"terms": [["1", "x"], ["1", "y"]], "op": "le", "rhs": "4"}
            ],
            "objective": {"dir": "max", "terms": [["1", "x"], ["2", "y"]]},
        });
        let config = Config::new();
        let ctx = Context::new(&config);
        let (solver, _) = build(&ctx, &spec, false).expect("build");
        assert_eq!(solver.optimize.check(&[]), SatResult::Sat);
        let model = solver.optimize.get_model().expect("model");
        let VarAst::Int(ast) = &solver.variables["y"] else {
            panic!("y must be an int variable");
        };
        assert_eq!(model.eval(ast, true).map(|value| value.to_string()), Some("4".to_owned()));
    }
}
