use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use z3::ast::{Ast, Int, Real};
use z3::{Config, Context, Optimize, SatResult};

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

    let cfg = Config::new();
    let ctx = Context::new(&cfg);
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
    let mut fixed_sum = Int::from_i64(&ctx, 0);

    for edge_id in 0..m {
        let edge = &graph.edges[edge_id];
        optimize.assert(&flow[edge_id].ge(&margin));
        optimize.assert(&flow[edge_id].ge(&zero));
        optimize.assert(&flow[edge_id].le(&one));
        for variable in [&empty[edge_id], &full_state[edge_id], &partial[edge_id], &fixed[edge_id]] {
            optimize.assert(&variable.ge(&zero_int));
            optimize.assert(&variable.le(&one_int));
        }
        optimize.assert(
            &(empty[edge_id].clone() + full_state[edge_id].clone() + partial[edge_id].clone())
                ._eq(&one_int),
        );

        let candidate = matches!(graph.kinds[edge.u], Kind::Source | Kind::Merger)
            && matches!(graph.kinds[edge.v], Kind::Sink | Kind::Splitter);
        fixed_sum = fixed_sum + fixed[edge_id].clone();
        if explicit_fixed {
            let selected = if explicitly_selected[edge_id] { one_int.clone() } else { zero_int.clone() };
            optimize.assert(&fixed[edge_id]._eq(&selected));
        } else if !candidate {
            optimize.assert(&fixed[edge_id]._eq(&zero_int));
        }
        if candidate {
            let fixed_real = Real::from_int(&fixed[edge_id]);
            optimize.assert(&flow[edge_id].ge(&fixed_real));
            optimize.assert(&flow[edge_id].le(&(&one - &margin + &fixed_real)));
        } else if explicitly_selected[edge_id] {
            optimize.assert(&flow[edge_id].ge(&Real::from_int(&fixed[edge_id])));
        }

        let buffer = graph.kinds[edge.u] == Kind::Splitter && graph.kinds[edge.v] == Kind::Merger;
        if !buffer {
            optimize.assert(&partial[edge_id]._eq(&zero_int));
        }
        if edge.u == graph.source {
            optimize.assert(&full_state[edge_id]._eq(&one_int));
        }
        if edge.v == graph.sink {
            optimize.assert(&empty[edge_id].ge(&(one_int.clone() - fixed[edge_id].clone())));
        }
    }
    optimize.assert(&fixed_sum.ge(&one_int));

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

    for first in 0..m {
        for other in (first + 1)..m {
            if graph.edges[first].u == graph.edges[other].u
                && graph.edges[first].v == graph.edges[other].v
            {
                optimize.assert(&flow[first]._eq(&flow[other]));
            }
        }
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
    for first in 0..m {
        for other in (first + 1)..m {
            if graph.edges[first].u == graph.edges[other].u
                && graph.edges[first].v == graph.edges[other].v
            {
                let mut row = vec![Rat::zero(); m];
                row[first] = Rat::one();
                row[other] = -Rat::one();
                rows.push((row, Rat::zero()));
            }
        }
    }
    for edge in 0..m {
        if is_fixed[edge] {
            let mut row = vec![Rat::zero(); m];
            row[edge] = Rat::one();
            rows.push((row, Rat::one()));
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

    let (exact_flows, rank_full) = match solve_exact(rows, m) {
        Ok(result) => result,
        Err(_) => (
            floats
                .iter()
                .map(|&value| Rat::from_f64(value).unwrap_or_else(Rat::zero))
                .collect(),
            false,
        ),
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

    let source_edge = graph.outgoing[graph.source][0];
    let total = flows[source_edge].clone();
    Ok(NativeFlowSolution::from_fractions(
        "rust-rank-smt-z3".into(),
        "Optimal".into(),
        true,
        flows,
        total,
        1,
        full_rank,
    ))
}
