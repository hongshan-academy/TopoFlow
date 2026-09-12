//! Z3-backed exact LP for the Karzanov solver's aggregate step.
//!
//! The aggregate LP is solved single-threaded (`smt.threads = 1`) so repeated
//! runs select the same optimal vertex.

use std::sync::Once;

use num_traits::Zero;
use z3::ast::{Ast, Real};
use z3::{Context, Optimize, SatResult};

use crate::ilp_bridge::parse_z3_numeral;
use crate::Rat;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpStatus {
    Optimal,
    Infeasible,
}

fn real_of<'ctx>(ctx: &'ctx Context, value: &Rat) -> Real<'ctx> {
    Real::from_real_str(
        ctx,
        &value.numer().to_str_radix(10),
        &value.denom().to_str_radix(10),
    )
    .expect("valid rational numeral")
}

/// Pin Z3's randomness knobs process-wide so repeated runs select the same
/// optimal vertex. Z3 is already deterministic for a fixed input at the
/// default seed; this makes the intent explicit and guards against a future
/// build changing the defaults.
fn pin_deterministic_globals() {
    static PIN: Once = Once::new();
    PIN.call_once(|| {
        for (key, value) in [
            ("smt.random_seed", "0"),
            ("sat.random_seed", "0"),
            ("sat.random_freq", "0"),
            ("smt.phase_selection", "3"),
            ("smt.arith.random_initial_value", "false"),
        ] {
            z3::set_global_param(key, value);
        }
    });
}

pub fn solve_lp(
    c: &[Rat],
    a_ub: &[Vec<Rat>],
    b_ub: &[Rat],
    a_eq: &[Vec<Rat>],
    b_eq: &[Rat],
) -> (LpStatus, Vec<Rat>) {
    let n = c.len();
    pin_deterministic_globals();
    let ctx = crate::context_with_threads(1);
    let optimize = Optimize::new(&ctx);

    let zero = Real::from_real(&ctx, 0, 1);
    let vars: Vec<Real> = (0..n).map(|j| Real::new_const(&ctx, format!("x_{j}"))).collect();
    for var in &vars {
        optimize.assert(&var.ge(&zero));
    }

    let linear = |row: &[Rat]| -> Real {
        let mut lhs = zero.clone();
        for (j, coefficient) in row.iter().enumerate() {
            if !coefficient.is_zero() {
                lhs = lhs + real_of(&ctx, coefficient) * vars[j].clone();
            }
        }
        lhs
    };
    for (row, rhs) in a_ub.iter().zip(b_ub) {
        optimize.assert(&linear(row).le(&real_of(&ctx, rhs)));
    }
    for (row, rhs) in a_eq.iter().zip(b_eq) {
        optimize.assert(&linear(row)._eq(&real_of(&ctx, rhs)));
    }

    let mut objective = zero.clone();
    for (j, coefficient) in c.iter().enumerate() {
        if !coefficient.is_zero() {
            objective = objective + real_of(&ctx, coefficient) * vars[j].clone();
        }
    }
    optimize.minimize(&objective);

    if optimize.check(&[]) != SatResult::Sat {
        return (LpStatus::Infeasible, Vec::new());
    }
    let Some(model) = optimize.get_model() else {
        return (LpStatus::Infeasible, Vec::new());
    };
    let mut solution = vec![Rat::zero(); n];
    for (j, var) in vars.iter().enumerate() {
        if let Some(value) = model.eval(var, true) {
            if let Ok(parsed) = parse_z3_numeral(&value.to_string()) {
                solution[j] = parsed;
            }
        }
    }
    (LpStatus::Optimal, solution)
}
