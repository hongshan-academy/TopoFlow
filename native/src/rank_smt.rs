use num_traits::{One, Zero};
use z3::ast::{Ast, Int, Real};
use z3::{Model, Optimize, SatResult};

use crate::graph::{GraphData, Kind};
use crate::{NativeFlowSolution, Rat};

fn read_real(model: &Model<'_>, variable: &Real<'_>) -> Result<Rat, String> {
    let value = model
        .eval(variable, true)
        .ok_or_else(|| format!("missing model value for {variable}"))?;
    crate::ilp_bridge::parse_z3_numeral(&value.to_string())
        .map_err(|err| format!("invalid model value for {variable}: {err}"))
}

fn solve_exact(rows: Vec<(Vec<Rat>, Rat)>, model_flows: &[Rat]) -> Result<(Vec<Rat>, bool), String> {
    let variables = model_flows.len();
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
    // Keep the model's free coordinates. Reduced rows have zero coefficients
    // for every other pivot, so the pivot values can be computed independently.
    let mut solution = model_flows.to_vec();
    for (r, &column) in pivot_columns.iter().enumerate() {
        solution[column] = matrix[r][column + 1..variables]
            .iter()
            .zip(&model_flows[column + 1..])
            .filter(|(coefficient, _)| !coefficient.is_zero())
            .fold(matrix[r][variables].clone(), |rhs, (coefficient, flow)| {
                rhs - coefficient * flow
            });
    }
    Ok((solution, full_rank))
}

pub fn solve(
    edges: Vec<(String, String)>,
    fixed_edges: Option<Vec<usize>>,
    _workers: u32,
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

    // Keep rank-SMT single-threaded. Z3's parallel SMT portfolio (smt.threads)
    // can return different optimal models for the same input, which makes the
    // reconstructed flows non-deterministic.
    let ctx = crate::context_with_threads(1);
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
    // Keep strict positivity as a hard constraint: the lexicographic
    // objectives below may otherwise settle on a margin-zero model.
    optimize.assert(&margin.gt(&zero));
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

        // A splitter output into a CU may be a fixed anchor, but only when the
        // splitter is not itself fed by a CU.  A CU-fed splitter is distributing
        // an already throttled flow; fixing one of several outputs would let the
        // solver break the equal split and change the boundary.  A single-output
        // splitter has no split to preserve, so it may still anchor the CU.
        let fed_by_control = graph.incoming[edge.u]
            .first()
            .map(|&input_edge| {
                matches!(graph.kinds[graph.edges[input_edge].u], Kind::ControlUnit { .. })
            })
            .unwrap_or(false);
        let single_output = graph.outgoing[edge.u].len() == 1;
        let splitter_cu_candidate = matches!(graph.kinds[edge.u], Kind::Splitter)
            && matches!(graph.kinds[edge.v], Kind::ControlUnit { .. })
            && (!fed_by_control || single_output);
        let candidate = (matches!(graph.kinds[edge.u], Kind::Source | Kind::Merger)
            || splitter_cu_candidate)
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
            // Explicit fixed edges outside the candidate set may still point
            // into a CU; their full speed is the target capacity, not one.
            let fixed_real = Real::from_int(&fixed[edge_id]);
            let scaled = &cap * &fixed_real;
            optimize.assert(&flow[edge_id].ge(&scaled));
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
                let cap = Real::from_real(&ctx, gear as i32, denominator as i32);
                let two = Real::from_real(&ctx, 2, 1);

                // can_in/can_out already encode the cut side; a CU is a
                // pass-through, so can_in propagates downstream and can_out
                // propagates upstream.
                let can_in_x = empty[input].clone() + partial[input].clone();
                let can_out_x = full_state[input].clone() + partial[input].clone();
                let can_in_y = empty[output].clone() + partial[output].clone();
                let can_out_y = full_state[output].clone() + partial[output].clone();
                optimize.assert(&can_in_x.le(&can_in_y));
                optimize.assert(&can_out_y.le(&can_out_x));

                // x.can_out and y.can_in => the cut is at the CU, so x = cap.
                let both = can_out_x.clone() + can_in_y.clone();
                optimize.assert(
                    &flow[input].ge(&(&cap - &big_m * &(&two - &Real::from_int(&both)))),
                );

                // x.not(can_out) => x is upstream, hence strictly below cap.
                optimize.assert(
                    &flow[input].le(
                        &(&cap - &margin + &big_m * &Real::from_int(&can_out_x)),
                    ),
                );
            }
            _ => {}
        }
    }

    // Lexicographic objective: first maximize total flow, then keep the
    // number of fixed Splitter->CU edges minimal (so a splitter only
    // saturates an output when the CU caps force it), then maximize margin.
    let total = graph.sources.iter().fold(Real::from_real(&ctx, 0, 1), |sum, &source| {
        sum + flow[graph.outgoing[source][0]].clone()
    });
    optimize.maximize(&total);

    let new_fixed = graph
        .edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| {
            matches!(graph.kinds[edge.u], Kind::Splitter)
                && matches!(graph.kinds[edge.v], Kind::ControlUnit { .. })
        })
        .fold(Int::from_i64(&ctx, 0), |sum, (index, _)| {
            sum + fixed[index].clone()
        });
    optimize.minimize(&Real::from_int(&new_fixed));

    optimize.maximize(&margin);
    if optimize.check(&[]) != SatResult::Sat {
        return Err("MIP has no solution satisfying the strict flow bounds".into());
    }
    let model = optimize.get_model().ok_or("optimizer returned no model")?;
    if read_real(&model, &margin)? <= Rat::zero() {
        return Err("MIP has no solution satisfying the strict flow bounds".into());
    }
    let model_flows = flow
        .iter()
        .map(|variable| read_real(&model, variable))
        .collect::<Result<Vec<_>, _>>()?;

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
            let output = graph.outgoing[node][0];
            // The cut is at the CU exactly when the input is downstream and
            // the output can receive; then the input sits at capacity.
            if can_out[input] && can_in[output] {
                let mut row = vec![Rat::zero(); m];
                row[input] = Rat::one();
                rows.push((row, crate::rat_ratio(gear as i64, denominator as i64)));
            }
        }
    }

    let (flows, full_rank) = solve_exact(rows, &model_flows)?;
    if let Some(edge) = flows.iter().zip(&model_flows).position(|(flow, model)| flow != model) {
        return Err(format!("exact flow reconstruction differs from the SMT model at edge {edge}"));
    }

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

    #[test]
    fn reads_model_rationals_beyond_i64() {
        let ctx = crate::context_with_threads(1);
        let solver = z3::Solver::new(&ctx);
        let variable = Real::new_const(&ctx, "large_denominator");
        let denominator = "9223372036854775809";
        solver.assert(&variable._eq(&Real::from_real_str(&ctx, "1", denominator).unwrap()));
        assert_eq!(solver.check(), SatResult::Sat);
        let model = solver.get_model().unwrap();
        assert_eq!(
            read_real(&model, &variable).unwrap(),
            Rat::new(BigInt::from(1), denominator.parse().unwrap())
        );
    }

    #[test]
    fn anchors_single_output_splitter_into_control_unit() {
        let result = solve(
            edges(&[
                ("In", "ICU4_0"),
                ("ICU4_0", "S2_0"),
                ("S2_0", "ICU1_1"),
                ("ICU1_1", "Out"),
            ]),
            None,
            1,
        )
        .expect("single-output splitter may anchor a CU");
        assert_eq!(result.flow_numerators, vec![BigInt::from(1); 4]);
        assert_eq!(result.flow_denominators, vec![BigInt::from(5); 4]);
        assert_eq!(result.total_numerator, BigInt::from(1));
        assert_eq!(result.total_denominator, BigInt::from(5));
    }

    #[test]
    fn explicit_fixed_control_unit_edge_uses_capacity() {
        let result = solve(
            edges(&[
                ("In", "ICU4_0"),
                ("ICU4_0", "ICU1_1"),
                ("ICU1_1", "Out"),
            ]),
            Some(vec![1]),
            1,
        )
        .expect("explicit fixed CU edge should use its capacity");
        assert_eq!(result.flow_numerators, vec![BigInt::from(1); 3]);
        assert_eq!(result.flow_denominators, vec![BigInt::from(5); 3]);
    }

    #[test]
    fn reconstructs_positive_flows_for_deficient_rank() {
        let graph = edges(&[
            ("In", "S0"), ("S0", "S2"), ("S0", "C1"), ("S2", "C0"),
            ("S2", "C1"), ("C0", "Out"), ("C1", "C0"),
        ]);
        for fixed in [None, Some(vec![0, 5])] {
            let result = solve(graph.clone(), fixed, 1).unwrap();
            assert!(result.feasible);
            assert!(!result.full_rank);
            let flows: Vec<Rat> = result.flow_numerators.into_iter()
                .zip(result.flow_denominators)
                .map(|(num, den)| Rat::new(num, den))
                .collect();
            assert!(flows.iter().all(|flow| flow > &Rat::zero()));
            assert_eq!(flows[0], Rat::one());
            assert_eq!(flows[5], Rat::one());
            assert_eq!(flows[0], &flows[1] + &flows[2]);
            assert_eq!(flows[1], &flows[3] + &flows[4]);
            assert_eq!(flows[6], &flows[2] + &flows[4]);
            assert_eq!(flows[5], &flows[3] + &flows[6]);
        }
    }
}
