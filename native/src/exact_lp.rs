use num_traits::{One, Zero};

use crate::Rat;

/// Exact two-phase simplex with Bland's rule.
///
/// The pivot choices are fully deterministic (smallest-index entering
/// variable, smallest-index leaving variable on ratio ties), so the same
/// problem always yields the same basic optimum regardless of solver state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpStatus {
    Optimal,
    Infeasible,
    Unbounded,
}

fn pivot(table: &mut [Vec<Rat>], col: usize, row: usize) {
    let ncols = table[row].len();
    let pivot_value = table[row][col].clone();
    for entry in table[row].iter_mut() {
        *entry = &*entry / &pivot_value;
    }
    for i in 0..table.len() {
        if i == row {
            continue;
        }
        let factor = table[i][col].clone();
        if factor.is_zero() {
            continue;
        }
        for j in 0..ncols {
            table[i][j] = &table[i][j] - &factor * &table[row][j];
        }
    }
}

fn objective_value(table: &[Vec<Rat>], costs: &[Rat], basis: &[usize]) -> Rat {
    let mut total = Rat::zero();
    for (i, row) in table.iter().enumerate() {
        let value = row.last().expect("tableau row has an RHS column");
        total += &costs[basis[i]] * value;
    }
    total
}

/// Runs the simplex on an already-normalized tableau; returns `None` when the
/// objective is unbounded.
fn bland_simplex(table: &mut [Vec<Rat>], costs: &[Rat], basis: &mut [usize]) -> Option<Rat> {
    let ncols = table[0].len() - 1;
    let mut is_basic = vec![false; ncols];
    for &basic in basis.iter() {
        if basic < ncols {
            is_basic[basic] = true;
        }
    }
    loop {
        // Bland's rule: first column with a negative reduced cost enters.
        let mut entering: Option<usize> = None;
        'columns: for j in 0..ncols {
            if is_basic[j] {
                continue;
            }
            let mut reduced = costs[j].clone();
            for (i, row) in table.iter().enumerate() {
                reduced -= &costs[basis[i]] * &row[j];
            }
            if reduced < Rat::zero() {
                entering = Some(j);
                break 'columns;
            }
        }
        let Some(entering) = entering else { break };
        // Bland's rule: among minimum-ratio ties the smallest basic index leaves.
        let mut leaving: Option<usize> = None;
        let mut best_ratio: Option<Rat> = None;
        for i in 0..table.len() {
            let coefficient = &table[i][entering];
            if coefficient <= &Rat::zero() {
                continue;
            }
            let ratio = &table[i][ncols] / coefficient;
            let better = match &best_ratio {
                None => true,
                Some(current) => {
                    ratio < *current || (ratio == *current && basis[i] < basis[leaving.expect("set")])
                }
            };
            if better {
                best_ratio = Some(ratio);
                leaving = Some(i);
            }
        }
        let Some(leaving) = leaving else { return None };
        let old = basis[leaving];
        pivot(table, entering, leaving);
        if old < ncols {
            is_basic[old] = false;
        }
        basis[leaving] = entering;
        is_basic[entering] = true;
    }
    Some(objective_value(table, costs, basis))
}

/// Solves `minimize c.x` subject to `a_ub x <= b_ub`, `a_eq x == b_eq`,
/// `x >= 0`, using exact rational arithmetic.
///
/// On `Optimal`, `solution` holds the basic optimum. The returned objective is
/// `c.x` at that solution.
pub fn solve_lp(
    c: &[Rat],
    a_ub: &[Vec<Rat>],
    b_ub: &[Rat],
    a_eq: &[Vec<Rat>],
    b_eq: &[Rat],
) -> (LpStatus, Vec<Rat>) {
    let n = c.len();
    let m_ub = a_ub.len();
    let m_eq = a_eq.len();
    let total_cols = n + m_ub + m_eq;

    if m_ub == 0 && m_eq == 0 {
        if c.iter().any(|cost| *cost < Rat::zero()) {
            return (LpStatus::Unbounded, Vec::new());
        }
        return (LpStatus::Optimal, vec![Rat::zero(); n]);
    }

    let mut rows: Vec<Vec<Rat>> = Vec::with_capacity(m_ub + m_eq);
    let mut rhs: Vec<Rat> = Vec::with_capacity(m_ub + m_eq);
    let mut basis: Vec<usize> = Vec::with_capacity(m_ub + m_eq);

    // Inequality rows get a slack as their basic variable.
    for (i, row) in a_ub.iter().enumerate() {
        let mut normalized = vec![Rat::zero(); total_cols];
        normalized[..n].clone_from_slice(row);
        normalized[n + i] = Rat::one();
        if b_ub[i] < Rat::zero() {
            for entry in normalized.iter_mut() {
                *entry = -entry.clone();
            }
            rhs.push(-b_ub[i].clone());
        } else {
            rhs.push(b_ub[i].clone());
        }
        rows.push(normalized);
        basis.push(n + i);
    }

    // Equality rows get an artificial as their basic variable.
    for (i, row) in a_eq.iter().enumerate() {
        let mut normalized = vec![Rat::zero(); total_cols];
        normalized[..n].clone_from_slice(row);
        normalized[n + m_ub + i] = Rat::one();
        if b_eq[i] < Rat::zero() {
            for entry in normalized.iter_mut() {
                *entry = -entry.clone();
            }
            rhs.push(-b_eq[i].clone());
        } else {
            rhs.push(b_eq[i].clone());
        }
        rows.push(normalized);
        basis.push(n + m_ub + i);
    }

    let mut table: Vec<Vec<Rat>> = rows
        .into_iter()
        .zip(rhs)
        .map(|(mut row, value)| {
            row.push(value);
            row
        })
        .collect();

    // Phase 1: minimize the sum of artificial variables.
    let phase1_costs: Vec<Rat> = (0..n + m_ub)
        .map(|_| Rat::zero())
        .chain((0..m_eq).map(|_| Rat::one()))
        .collect();
    let mut basis1 = basis.clone();
    match bland_simplex(&mut table, &phase1_costs, &mut basis1) {
        None => return (LpStatus::Unbounded, Vec::new()),
        Some(value) if value > Rat::zero() => return (LpStatus::Infeasible, Vec::new()),
        Some(_) => {}
    }

    // Drop the artificial columns.
    let cols_now = n + m_ub;
    for row in table.iter_mut() {
        let mut trimmed: Vec<Rat> = Vec::with_capacity(cols_now + 1);
        for (j, value) in row.iter().enumerate() {
            if j < cols_now || j == total_cols {
                trimmed.push(value.clone());
            }
        }
        *row = trimmed;
    }

    // Pivot any zero-valued artificial out of the basis; drop redundant rows.
    let mut i = 0;
    while i < table.len() {
        if basis1[i] < n + m_ub {
            i += 1;
            continue;
        }
        let mut pivoted = false;
        for j in 0..cols_now {
            if !table[i][j].is_zero() {
                pivot(&mut table, j, i);
                basis1[i] = j;
                pivoted = true;
                break;
            }
        }
        if pivoted {
            i += 1;
        } else {
            table.remove(i);
            basis1.remove(i);
        }
    }

    // Phase 2: the real objective.
    let mut phase2_costs = Vec::with_capacity(n + m_ub);
    phase2_costs.extend_from_slice(c);
    phase2_costs.extend(std::iter::repeat_n(Rat::zero(), m_ub));
    if bland_simplex(&mut table, &phase2_costs, &mut basis1).is_none() {
        return (LpStatus::Unbounded, Vec::new());
    }

    let mut solution = vec![Rat::zero(); n];
    for (i, row) in table.iter().enumerate() {
        if basis1[i] < n {
            solution[basis1[i]] = row.last().expect("tableau row has an RHS column").clone();
        }
    }
    (LpStatus::Optimal, solution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rat;

    fn row(values: &[i64]) -> Vec<Rat> {
        values.iter().map(|&value| rat(value)).collect()
    }

    #[test]
    fn solves_a_small_lp_exactly() {
        // minimize -x1 - x2  s.t.  x1 + x2 <= 3, x1 <= 2, x2 <= 2
        let (status, solution) = solve_lp(
            &row(&[-1, -1]),
            &[row(&[1, 1]), row(&[1, 0]), row(&[0, 1])],
            &row(&[3, 2, 2]),
            &[],
            &[],
        );
        assert_eq!(status, LpStatus::Optimal);
        assert_eq!(solution[0], rat(2));
        assert_eq!(solution[1], rat(1));
    }

    #[test]
    fn detects_infeasibility() {
        let (status, _) = solve_lp(
            &row(&[1]),
            &[row(&[1])],
            &row(&[1]),
            &[row(&[1])],
            &row(&[3]),
        );
        assert_eq!(status, LpStatus::Infeasible);
    }

    #[test]
    fn is_deterministic() {
        let (_, first) = solve_lp(
            &row(&[-1, -1, -1]),
            &[row(&[1, 1, 1]), row(&[1, 0, 0]), row(&[0, 1, 0]), row(&[0, 0, 1])],
            &row(&[5, 2, 2, 2]),
            &[],
            &[],
        );
        for _ in 0..4 {
            let (_, again) = solve_lp(
                &row(&[-1, -1, -1]),
                &[row(&[1, 1, 1]), row(&[1, 0, 0]), row(&[0, 1, 0]), row(&[0, 0, 1])],
                &row(&[5, 2, 2, 2]),
                &[],
                &[],
            );
            assert_eq!(first, again);
        }
    }
}
