mod graph;
mod ilp_bridge;
mod karzanov;
mod rank_smt;
mod z3_lp;

use std::sync::Mutex;

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use pyo3::prelude::*;
use z3::{Config, Context};

pub(crate) type Rat = BigRational;

static Z3_CONTEXT_LOCK: Mutex<()> = Mutex::new(());

/// Create a Z3 context with an explicit `smt.threads` value.
///
/// `Optimize` does not expose `set_params`, so the thread count is applied as a
/// global parameter immediately before the context is created; the lock keeps
/// concurrent contexts from observing each other's value.
pub(crate) fn context_with_threads(threads: u32) -> Context {
    context_with_options(threads, 0)
}

/// Like [`context_with_threads`], with an optional solver timeout in ms.
pub(crate) fn context_with_options(threads: u32, timeout_ms: u64) -> Context {
    let _guard = Z3_CONTEXT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    z3::set_global_param("smt.threads", &threads.max(1).to_string());
    let mut config = Config::new();
    if timeout_ms > 0 {
        config.set_param_value("timeout", &timeout_ms.to_string());
    }
    Context::new(&config)
}

pub(crate) fn rat(value: i64) -> Rat {
    BigRational::from_integer(BigInt::from(value))
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct NativeFlowSolution {
    #[pyo3(get)]
    pub backend: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub feasible: bool,
    #[pyo3(get)]
    pub flows: Vec<f64>,
    #[pyo3(get)]
    pub total_flow: f64,
    #[pyo3(get)]
    pub iterations: usize,
    #[pyo3(get)]
    pub flow_numerators: Vec<BigInt>,
    #[pyo3(get)]
    pub flow_denominators: Vec<BigInt>,
    #[pyo3(get)]
    pub total_numerator: BigInt,
    #[pyo3(get)]
    pub total_denominator: BigInt,
    #[pyo3(get)]
    pub full_rank: bool,
    /// Per-edge state flags; empty when the backend does not classify edges.
    #[pyo3(get)]
    pub can_in: Vec<bool>,
    #[pyo3(get)]
    pub can_out: Vec<bool>,
}

impl NativeFlowSolution {
    pub fn from_fractions(
        backend: String,
        status: String,
        feasible: bool,
        flows: Vec<BigRational>,
        total: BigRational,
        iterations: usize,
        full_rank: bool,
    ) -> Self {
        let floats: Vec<f64> = flows
            .iter()
            .map(|value| value.to_f64().unwrap_or(f64::NAN))
            .collect();
        let total_flow = total.to_f64().unwrap_or(f64::NAN);
        let flow_numerators: Vec<BigInt> =
            flows.iter().map(|value| value.numer().clone()).collect();
        let flow_denominators: Vec<BigInt> =
            flows.iter().map(|value| value.denom().clone()).collect();
        Self {
            backend,
            status,
            feasible,
            flows: floats,
            total_flow,
            iterations,
            flow_numerators,
            flow_denominators,
            total_numerator: total.numer().clone(),
            total_denominator: total.denom().clone(),
            full_rank,
            can_in: Vec::new(),
            can_out: Vec::new(),
        }
    }

    /// Attach per-edge state flags (used by the rank-SMT backend).
    pub fn with_states(mut self, can_in: Vec<bool>, can_out: Vec<bool>) -> Self {
        self.can_in = can_in;
        self.can_out = can_out;
        self
    }
}

#[pyfunction]
#[pyo3(signature = (edges, fixed_edges=None, workers=16))]
fn solve_rank_smt(
    edges: Vec<(String, String)>,
    fixed_edges: Option<Vec<usize>>,
    workers: u32,
) -> PyResult<NativeFlowSolution> {
    rank_smt::solve(edges, fixed_edges, workers).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (edges, max_iterations=100000, tolerance=1e-10))]
fn solve_karzanov(
    edges: Vec<(String, String)>,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<NativeFlowSolution> {
    karzanov::solve(edges, max_iterations, tolerance).map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn topoflow_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeFlowSolution>()?;
    module.add_function(wrap_pyfunction!(solve_rank_smt, module)?)?;
    module.add_function(wrap_pyfunction!(solve_karzanov, module)?)?;
    module.add_function(wrap_pyfunction!(native_version, module)?)?;
    ilp_bridge::register(module)?;
    Ok(())
}
