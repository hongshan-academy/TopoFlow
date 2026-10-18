use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use z3::ast::{Ast, Int, Real};
use z3::{Optimize, SatResult};

use crate::graph::{GraphData, Kind};
use crate::{NativeFlowSolution, Rat};

fn solve_exact(rows: Vec<(Vec<Rat>, Rat)>, variables: usize) -> Result<(Vec<Rat>, bool), String> {
    let mut matrix: Vec<Vec<Rat>> = rows
        .into_iter()
        .map(|(mut coefficients, rhs)| {
            coefficients.push(rhs);
            coefficients
        })
        .collect();
    let mut pivot_columns: Vec<usize> = Vec::new();
    let mut row = 0usize;
    for column in 0..variables {
        let pivot = (row..matrix.len()).find(|&r| !matrix[r][column].is_zero());
        let Some(pivot) = pivot else { continue };
        matrix.swap(row, pivot);
        let pivot_value = matrix[row][column].clone();
        for c in column..=variables {
            matrix[row][c] = &matrix[row][c] / &pivot_value;
        }
        for r in 0..matrix.len() {
            if r == row || matrix[r][column].is_zero() {
                continue;
            }
            let factor = matrix[r][column].clone();
            for c in column..=variables {
                let subtract = &factor * &matrix[row][c];
                matrix[r][c] -= subtract;
            }
        }
        pivot_columns.push(column);
        row += 1;
        if row == matrix.len() {
            break;
        }
    }
    for r in row..matrix.len() {
        if matrix[r][..variables].iter().all(|value| value.is_zero())
            && !matrix[r][variables].is_zero()
        {
            return Err("exact flow system is inconsistent".into());
        }
    }
    let full_rank = pivot_columns.len() == variables;
    let mut solution = vec![Rat::zero(); variables];
    for (r, &column) in pivot_columns.iter().enumerate() {
        solution[column] = matrix[r][variables].clone();
    }
    Ok((solution, full_rank))
}

pub fn solve(
    edges: Vec<(String, String)>,
    fixed_edges: Option<Vec<usize>>,
    workers: u32,
) -> Result<NativeFlowSolution, String> {
    let graph = GraphData::new(edges)?;
    let m = graph.edges.len();
    let explicit_fixed = fixed_edges.is_some();
    let mut explicitly_selected = vec![false; m];
    if let Some(edge_ids) = fixed_edges {
        if edge_ids.is_empty() {
            return Err("fixed_edges must be nonempty when provided".into());
        }
        for edge in edge_ids {
            if edge >= m {
                return Err(format!("fixed edge {edge} is outside 0..{m}"));
            }
            explicitly_selected[edge] = true;
        }
    }

    let ctx = crate::context_with_threads(workers);
    let optimize = Optimize::new(&ctx);

    let zero = Real::from_real(&ctx, 0, 1);
    let one = Real::from_real(&ctx, 1, 1);
    let zero_int = Int::from_i64(&ctx, 0);
    let one_int = Int::from_i64(&ctx, 1);
    let big_m = Real::from_real(&ctx, 1, 1);

    let flow: Vec<Real> = (0..m).map(|i| Real::new_const(&ctx, format!("flow_{i}"))).collect();
    let margin = Real::new_const(&ctx, "strict_margin");
    let empty: Vec<Int> = (0..m).map(|i| Int::new_const(&ctx, format!("empty_{i}"))).collect();
    let full_state: Vec<Int> = (0..m).map(|i| Int::new_const(&ctx, format!("full_{i}"))).collect();
    let partial: Vec<Int> = (0..m).map(|i| Int::new_const(&ctx, format!("partial_{i}"))).collect();
    let fixed: Vec<Int> = (0..m).map(|i| Int::new_const(&ctx, format!("fixed_{i}"))).collect();

    optimize.assert(&margin.ge(&zero));
    optimize.assert(&margin.le(&one));
    let mut fixed_sums: Vec<Int> = (0..graph.component_count)
        .map(|_| Int::from_i64(&ctx, 0))
        .collect();

    for edge_id in 0..m {
        let edge = &graph.edges[edge_id];
        optimize.assert(&flow[edge_id].ge(&margin));
        optimize.assert(&flow[edge_id].ge(&zero));
        optimize.assert(&flow[edge_id].le(&one));
        // A control unit throttles its incoming edge to
        // ``[0, gear/denominator]`` (item 5, pipe 20).
        let cap = match graph.kinds[edge.v] {
            Kind::ControlUnit { gear, denominator } => {
                Real::from_real(&ctx, gear as i32, denominator as i32)
            }
            _ => one.clone(),
        };
        optimize.assert(&flow[edge_id].le(&cap));
        for variable in [&empty[edge_id], &full_state[edge_id], &partial[edge_id], &fixed[edge_id]] {
            optimize.assert(&variable.ge(&zero_int));
            optimize.assert(&variable.le(&one_int));
        }
        optimize.assert(
            &(empty[edge_id].clone() + full_state[edge_id].clone() + partial[edge_id].clone())
                ._eq(&one_int),
        );

        let candidate = matches!(graph.kinds[edge.u], Kind::Source | Kind::Merger)
            && matches!(
                graph.kinds[edge.v],
                Kind::Sink | Kind::Splitter | Kind::ControlUnit { .. }
            );
        let component = graph.component_of[edge.u];
        fixed_sums[component] = fixed_sums[component].clone() + fixed[edge_id].clone();
        if explicit_fixed {
            let selected = if explicitly_selected[edge_id] { one_int.clone() } else { zero_int.clone() };
            optimize.assert(&fixed[edge_id]._eq(&selected));
        } else if !candidate {
            optimize.assert(&fixed[edge_id]._eq(&zero_int));
        }
        if candidate {
            let fixed_real = Real::from_int(&fixed[edge_id]);
            let scaled = &cap * &fixed_real;
            optimize.assert(&flow[edge_id].ge(&scaled));
            optimize.assert(&flow[edge_id].le(&(&cap - &margin + &scaled)));
        } else if explicitly_selected[edge_id] {
            optimize.assert(&flow[edge_id].ge(&Real::from_int(&fixed[edge_id])));
        }

        let buffer = graph.kinds[edge.u] == Kind::Splitter && graph.kinds[edge.v] == Kind::Merger;
        if !buffer {
            optimize.assert(&partial[edge_id]._eq(&zero_int));
        }
        if graph.kinds[edge.u] == Kind::Source {
            optimize.assert(&full_state[edge_id]._eq(&one_int));
        }
        if graph.kinds[edge.v] == Kind::Sink {
            optimize.assert(&empty[edge_id].ge(&(one_int.clone() - fixed[edge_id].clone())));
        }
    }
    for fixed_sum in &fixed_sums {
        optimize.assert(&fixed_sum.ge(&one_int));
    }

    for node in 0..graph.names.len() {
        if matches!(graph.kinds[node], Kind::Source | Kind::Sink) {
            continue;
        }
        let mut incoming = zero.clone();
        for &edge in &graph.incoming[node] {
            incoming = incoming + flow[edge].clone();
        }
        let mut outgoing = zero.clone();
        for &edge in &graph.outgoing[node] {
            outgoing = outgoing + flow[edge].clone();
        }
        optimize.assert(&incoming._eq(&outgoing));
    }

    for (first, other) in graph.parallel_pairs() {
        optimize.assert(&flow[first]._eq(&flow[other]));
    }

    for node in 0..graph.names.len() {
        match graph.kinds[node] {
            Kind::Splitter => {
                let input = graph.incoming[node][0];
                for &output in &graph.outgoing[node] {
                    optimize.assert(
                        &(empty[output].clone() + partial[output].clone()).le(
                            &(empty[input].clone()
                                + partial[input].clone()
                                + fixed[input].clone()
                                + fixed[output].clone()),
                        ),
                    );
                }
                for &a in &graph.outgoing[node] {
                    for &b in &graph.outgoing[node] {
                        if a != b {
                            let penalty = Real::from_int(
                                &(fixed[a].clone() + fixed[b].clone() + full_state[a].clone()),
                            );
                            let lhs = flow[a].clone() + &big_m * penalty;
                            optimize.assert(&lhs.ge(&flow[b]));
                        }
                    }
                }
            }
            Kind::Merger => {
                let output = graph.outgoing[node][0];
                for &input in &graph.incoming[node] {
                    optimize.assert(
                        &(full_state[input].clone() + partial[input].clone()).le(
                            &(full_state[output].clone()
                                + partial[output].clone()
                                + fixed[input].clone()
                                + fixed[output].clone()),
                        ),
                    );
                }
                for &a in &graph.incoming[node] {
                    for &b in &graph.incoming[node] {
                        if a != b {
                            let penalty = Real::from_int(
                                &(fixed[a].clone() + fixed[b].clone() + empty[a].clone()),
                            );
                            let lhs = flow[a].clone() + &big_m * penalty;
                            optimize.assert(&lhs.ge(&flow[b]));
                        }
                    }
                }
            }
            Kind::ControlUnit { gear, denominator } => {
                let input = graph.incoming[node][0];
                let output = graph.outgoing[node][0];
                // Downstream blocked => upstream blocked (equivalently, upstream
                // flowing => downstream flowing).
                optimize.assert(&full_state[output].le(&full_state[input]));
                // Upstream blocked => the incoming edge saturates at gear/denominator.
                let cap = Real::from_real(&ctx, gear as i32, denominator as i32);
                let blocked = Real::from_int(&full_state[input]);
                optimize.assert(&flow[input].ge(&(&cap - &big_m * &(&one - &blocked))));
            }
            _ => {}
        }
    }

    optimize.maximize(&margin);
    if optimize.check(&[]) != SatResult::Sat {
        return Err("MIP has no solution satisfying the strict flow bounds".into());
    }
    let model = optimize.get_model().ok_or("optimizer returned no model")?;
    let margin_value = model
        .eval(&margin, true)
        .and_then(|value| value.as_real())
        .ok_or("failed to read the strict margin")?;
    if margin_value.0 <= 0 {
        return Err("MIP has no solution satisfying the strict flow bounds".into());
    }
    let floats: Vec<f64> = flow
        .iter()
        .map(|variable| {
            model
                .eval(variable, true)
                .and_then(|value| value.as_real())
                .map(|(num, den)| num as f64 / den as f64)
                .unwrap_or(f64::NAN)
        })
        .collect();

    // --- exact reconstruction from the chosen state / fixed assignment ------
    let mut is_fixed = vec![false; m];
    let mut can_in = vec![false; m];
    let mut can_out = vec![false; m];
    for edge in 0..m {
        let read = |variable: &Int| -> bool {
            model
                .eval(variable, true)
                .and_then(|value| value.as_i64())
                .map(|value| value == 1)
                .unwrap_or(false)
        };
        is_fixed[edge] = read(&fixed[edge]);
        let is_partial = read(&partial[edge]);
        let is_full = read(&full_state[edge]);
        let is_empty = read(&empty[edge]);
        can_in[edge] = is_empty || is_partial;
        can_out[edge] = is_full || is_partial;
    }

    let mut rows: Vec<(Vec<Rat>, Rat)> = Vec::new();
    for node in 0..graph.names.len() {
        if matches!(graph.kinds[node], Kind::Source | Kind::Sink) {
            continue;
        }
        let mut row = vec![Rat::zero(); m];
        for &edge in &graph.incoming[node] {
            row[edge] += Rat::one();
        }
        for &edge in &graph.outgoing[node] {
            row[edge] -= Rat::one();
        }
        rows.push((row, Rat::zero()));
    }
    for (first, other) in graph.parallel_pairs() {
        let mut row = vec![Rat::zero(); m];
        row[first] = Rat::one();
        row[other] = -Rat::one();
        rows.push((row, Rat::zero()));
    }
    for edge in 0..m {
        if is_fixed[edge] {
            let mut row = vec![Rat::zero(); m];
            row[edge] = Rat::one();
            let rhs = match graph.kinds[graph.edges[edge].v] {
                Kind::ControlUnit { gear, denominator } => {
                    crate::rat_ratio(gear as i64, denominator as i64)
                }
                _ => Rat::one(),
            };
            rows.push((row, rhs));
        }
    }
    for node in 0..graph.names.len() {
        let selected: Vec<usize> = match graph.kinds[node] {
            Kind::Splitter => graph.outgoing[node]
                .iter()
                .copied()
                .filter(|&edge| !is_fixed[edge] && can_in[edge])
                .collect(),
            Kind::Merger => graph.incoming[node]
                .iter()
                .copied()
                .filter(|&edge| !is_fixed[edge] && can_out[edge])
                .collect(),
            _ => continue,
        };
        for &index in selected.iter().skip(1) {
            let mut row = vec![Rat::zero(); m];
            row[selected[0]] = Rat::one();
            row[index] = -Rat::one();
            rows.push((row, Rat::zero()));
        }
    }
    for node in 0..graph.names.len() {
        if let Kind::ControlUnit { gear, denominator } = graph.kinds[node] {
            let input = graph.incoming[node][0];
            // A blocked (delivering-only) incoming edge sits exactly at gear/denominator.
            if can_out[input] {
                let mut row = vec![Rat::zero(); m];
                row[input] = Rat::one();
                rows.push((row, crate::rat_ratio(gear as i64, denominator as i64)));
            }
        }
    }

    let (exact_flows, rank_full) = match solve_exact(rows, m) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("rank-smt: exact reconstruction failed ({err}); using float approximation");
            (
                floats
                    .iter()
                    .map(|&value| Rat::from_f64(value).unwrap_or_else(Rat::zero))
                    .collect(),
                false,
            )
        }
    };
    let nonnegative = exact_flows.iter().all(|value| value >= &Rat::zero());
    let consistent = rank_full
        && exact_flows
            .iter()
            .zip(&floats)
            .all(|(value, &float)| (value.to_f64().unwrap_or(f64::NAN) - float).abs() <= 1e-6);
    let (flows, full_rank) = if nonnegative {
        (exact_flows, consistent)
    } else {
        (
            floats
                .iter()
                .map(|&value| Rat::from_f64(value).unwrap_or_else(Rat::zero))
                .collect(),
            false,
        )
    };

    let total = graph.sources.iter().fold(Rat::zero(), |sum, &source| {
        sum + &flows[graph.outgoing[source][0]]
    });
    let sink_total = graph.sinks.iter().fold(Rat::zero(), |sum, &sink| {
        sum + &flows[graph.incoming[sink][0]]
    });
    if total != sink_total {
        return Err(format!(
            "stable assignment has unequal aggregate source and sink totals: {total} vs {sink_total}"
        ));
    }
    Ok(NativeFlowSolution::from_fractions(
        "rust-rank-smt".into(),
        "Optimal".into(),
        true,
        flows,
        total,
        1,
        full_rank,
    )
    .with_states(can_in, can_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn edges(raw: &[(&str, &str)]) -> Vec<(String, String)> {
        raw.iter()
            .map(|&(u, v)| (u.to_owned(), v.to_owned()))
            .collect()
    }

    #[test]
    fn solves_multiple_boundary_components() {
        let result = solve(
            edges(&[("InputA", "OutputA"), ("InputB", "OutputB")]),
            None,
            1,
        )
        .expect("multiple boundary components are supported");
        assert_eq!(result.flow_numerators, vec![BigInt::from(1), BigInt::from(1)]);
        assert_eq!(result.flow_denominators, vec![BigInt::from(1), BigInt::from(1)]);
        assert_eq!(result.total_numerator, BigInt::from(2));
        assert_eq!(result.total_denominator, BigInt::from(1));
    }
}
