use std::cmp::{max, min};
use std::collections::{BTreeMap, HashMap};

use num_traits::{One, Zero};

use crate::graph::{GraphData, Kind};
use crate::{NativeFlowSolution, Rat, rat};

#[derive(Clone)]
struct Contract {
    left: usize,
    right: usize,
    capacity: Rat,
    left_tie: usize,
    right_tie: usize,
    original: Option<(usize, usize)>,
}

struct Instance {
    edges: Vec<Contract>,
    left_incident: Vec<Vec<usize>>,
    right_incident: Vec<Vec<usize>>,
    quotas_left: Vec<Rat>,
    quotas_right: Vec<Rat>,
    original_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ChoiceInfo {
    filled: bool,
    critical_tie: Option<usize>,
    head: Vec<usize>,
    critical_edges: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Signature {
    saturated_or_lowered: Vec<bool>,
    left: Vec<ChoiceInfo>,
    right: Vec<ChoiceInfo>,
}

fn build_instance(graph: &GraphData) -> Result<Instance, String> {
    let mut left_of = vec![None; graph.names.len()];
    let mut right_of = vec![None; graph.names.len()];
    let mut left_count = 0;
    let mut right_count = 0;
    for node in 0..graph.names.len() {
        if graph.kinds[node] != Kind::Sink {
            left_of[node] = Some(left_count);
            left_count += 1;
        }
        if graph.kinds[node] != Kind::Source {
            right_of[node] = Some(right_count);
            right_count += 1;
        }
    }

    let mut base = Vec::new();
    for (edge_id, edge) in graph.edges.iter().enumerate() {
        base.push(Contract {
            left: left_of[edge.u].ok_or("edge leaves sink")?,
            right: right_of[edge.v].ok_or("edge enters source")?,
            capacity: rat(1),
            left_tie: if graph.kinds[edge.u] == Kind::Source {
                0
            } else {
                1
            },
            right_tie: if graph.kinds[edge.v] == Kind::Sink {
                0
            } else {
                1
            },
            original: Some((edge_id, 0)),
        });
    }
    for node in 0..graph.names.len() {
        if matches!(graph.kinds[node], Kind::Source | Kind::Sink) {
            continue;
        }
        let left = left_of[node].unwrap();
        let right = right_of[node].unwrap();
        base.push(Contract {
            left,
            right,
            capacity: rat(4),
            left_tie: 0,
            right_tie: 2,
            original: None,
        });
        base.push(Contract {
            left,
            right,
            capacity: rat(4),
            left_tie: 2,
            right_tie: 0,
            original: None,
        });
    }

    let mut labels: HashMap<(usize, usize), usize> = HashMap::new();
    let mut covered = Vec::with_capacity(3 * base.len());
    for edge in base {
        let label = labels.entry((edge.left, edge.right)).or_insert(0);
        if *label >= 3 {
            return Err("auxiliary endpoint multiplicity exceeds three".into());
        }
        let shift = *label;
        *label += 1;
        for copy in 0..3 {
            covered.push(Contract {
                left: 3 * edge.left + copy,
                right: 3 * edge.right + (copy + shift) % 3,
                capacity: edge.capacity.clone(),
                left_tie: edge.left_tie,
                right_tie: edge.right_tie,
                original: edge.original.map(|(id, _)| (id, copy)),
            });
        }
    }
    let mut left_incident = vec![Vec::new(); 3 * left_count];
    let mut right_incident = vec![Vec::new(); 3 * right_count];
    for (edge_id, edge) in covered.iter().enumerate() {
        left_incident[edge.left].push(edge_id);
        right_incident[edge.right].push(edge_id);
    }
    Ok(Instance {
        edges: covered,
        left_incident,
        right_incident,
        quotas_left: vec![rat(4); 3 * left_count],
        quotas_right: vec![rat(4); 3 * right_count],
        original_count: graph.edges.len(),
    })
}

fn choose_vertex(
    request: &[Rat],
    incident: &[usize],
    tie_of: impl Fn(usize) -> usize,
    quota: &Rat,
) -> (Vec<(usize, Rat)>, ChoiceInfo) {
    let total = incident
        .iter()
        .fold(Rat::zero(), |sum, &edge| sum + &request[edge]);
    if &total < quota {
        return (
            incident
                .iter()
                .map(|&edge| (edge, request[edge].clone()))
                .collect(),
            ChoiceInfo {
                filled: false,
                critical_tie: None,
                head: Vec::new(),
                critical_edges: Vec::new(),
            },
        );
    }
    let max_tie = incident.iter().map(|&edge| tie_of(edge)).max().unwrap_or(0);
    let mut accepted = Vec::with_capacity(incident.len());
    let mut remaining = quota.clone();
    for tie in 0..=max_tie {
        let group: Vec<usize> = incident
            .iter()
            .copied()
            .filter(|&edge| tie_of(edge) == tie)
            .collect();
        let group_sum = group
            .iter()
            .fold(Rat::zero(), |sum, &edge| sum + &request[edge]);
        if group_sum < remaining {
            accepted.extend(group.iter().map(|&edge| (edge, request[edge].clone())));
            remaining -= group_sum;
            continue;
        }
        let mut values: Vec<Rat> = group.iter().map(|&edge| request[edge].clone()).collect();
        values.sort();
        let mut level = Rat::zero();
        let mut rest = max(remaining.clone(), Rat::zero());
        for (index, value) in values.iter().enumerate() {
            let width = rat((values.len() - index) as i64);
            let cost = max(value.clone() - &level, Rat::zero()) * &width;
            if rest <= cost {
                level += &rest / &width;
                rest = Rat::zero();
                break;
            }
            rest -= cost;
            level = value.clone();
        }
        if rest > Rat::zero() && !values.is_empty() {
            level += rest / rat(values.len() as i64);
        }
        accepted.extend(
            group
                .iter()
                .map(|&edge| (edge, min(request[edge].clone(), level.clone()))),
        );
        for later in (tie + 1)..=max_tie {
            accepted.extend(
                incident
                    .iter()
                    .copied()
                    .filter(|&edge| tie_of(edge) == later)
                    .map(|edge| (edge, Rat::zero())),
            );
        }
        let mut head: Vec<usize> = group
            .iter()
            .copied()
            .filter(|&edge| request[edge] >= level)
            .collect();
        head.sort_unstable();
        return (
            accepted,
            ChoiceInfo {
                filled: true,
                critical_tie: Some(tie),
                head,
                critical_edges: group,
            },
        );
    }
    unreachable!("quota-exceeding request must have a critical tie")
}

fn sweep(
    instance: &Instance,
    boundary: &[Rat],
) -> (Vec<Rat>, Vec<Rat>, Vec<ChoiceInfo>, Vec<ChoiceInfo>) {
    let mut x = vec![Rat::zero(); instance.edges.len()];
    let mut left_info = Vec::with_capacity(instance.left_incident.len());
    for (left, incident) in instance.left_incident.iter().enumerate() {
        let (chosen, info) = choose_vertex(
            boundary,
            incident,
            |edge| instance.edges[edge].left_tie,
            &instance.quotas_left[left],
        );
        for (edge, value) in chosen {
            x[edge] = value;
        }
        left_info.push(info);
    }
    let mut y = vec![Rat::zero(); instance.edges.len()];
    let mut right_info = Vec::with_capacity(instance.right_incident.len());
    for (right, incident) in instance.right_incident.iter().enumerate() {
        let (chosen, info) = choose_vertex(
            &x,
            incident,
            |edge| instance.edges[edge].right_tie,
            &instance.quotas_right[right],
        );
        for (edge, value) in chosen {
            y[edge] = value;
        }
        right_info.push(info);
    }
    (x, y, left_info, right_info)
}

fn validate_stable_assignment(instance: &Instance, assignment: &[Rat]) -> Result<(), String> {
    if assignment.len() != instance.edges.len() {
        return Err("stable assignment has the wrong edge count".into());
    }
    for (edge, (value, contract)) in assignment.iter().zip(&instance.edges).enumerate() {
        if value < &Rat::zero() || value > &contract.capacity {
            return Err(format!("edge {edge} violates its capacity: {value}"));
        }
    }

    let mut left_info = Vec::with_capacity(instance.left_incident.len());
    for (left, incident) in instance.left_incident.iter().enumerate() {
        let (chosen, info) = choose_vertex(
            assignment,
            incident,
            |edge| instance.edges[edge].left_tie,
            &instance.quotas_left[left],
        );
        if chosen
            .iter()
            .any(|(edge, value)| value != &assignment[*edge])
        {
            return Err(format!("left vertex {left} is not stationary"));
        }
        left_info.push(info);
    }

    let mut right_info = Vec::with_capacity(instance.right_incident.len());
    for (right, incident) in instance.right_incident.iter().enumerate() {
        let (chosen, info) = choose_vertex(
            assignment,
            incident,
            |edge| instance.edges[edge].right_tie,
            &instance.quotas_right[right],
        );
        if chosen
            .iter()
            .any(|(edge, value)| value != &assignment[*edge])
        {
            return Err(format!("right vertex {right} is not stationary"));
        }
        right_info.push(info);
    }

    let willing = |tie: usize, edge: usize, info: &ChoiceInfo| -> bool {
        if !info.filled {
            return true;
        }
        let critical = info
            .critical_tie
            .expect("filled choices have a critical tie");
        tie < critical || (tie == critical && !info.head.contains(&edge))
    };
    for (edge, contract) in instance.edges.iter().enumerate() {
        if assignment[edge] >= contract.capacity {
            continue;
        }
        if willing(contract.left_tie, edge, &left_info[contract.left])
            && willing(contract.right_tie, edge, &right_info[contract.right])
        {
            return Err(format!("edge {edge} blocks the assignment"));
        }
    }
    Ok(())
}

struct AggregateOutcome {
    assignment: Vec<Rat>,
    changes_parameter: bool,
    must_terminate: bool,
    left_info: Vec<ChoiceInfo>,
}

#[allow(clippy::too_many_arguments)]
fn aggregate(
    instance: &Instance,
    capacities: &[Rat],
    boundary: &[Rat],
    x: &[Rat],
    y: &[Rat],
    left_info: &[ChoiceInfo],
    right_info: &[ChoiceInfo],
) -> Result<AggregateOutcome, String> {
    let debug = std::env::var_os("TOPOFLOW_DEBUG_STABLE").is_some();
    // Appendix A defines a(y,f) as the common value on H_f. A positive phi_f
    // is possible only when those edges were not cut by W, have one common
    // value, and are not themselves W-head edges. Freeze an ineligible local
    // vertex at phi_f=0 instead of rejecting independent augmentations.
    let mut left_can_advance = vec![true; left_info.len()];
    for (left, info) in left_info.iter().enumerate().filter(|(_, info)| info.filled) {
        let first = *info
            .head
            .first()
            .ok_or_else(|| format!("filled left vertex {left} has an empty head"))?;
        let head_value = &y[first];
        for &edge in &info.head {
            if x[edge] != y[edge] {
                left_can_advance[left] = false;
            }
            if &y[edge] != head_value {
                left_can_advance[left] = false;
            }
            let right = instance.edges[edge].right;
            if right_info[right].head.contains(&edge) {
                left_can_advance[left] = false;
            }
        }
    }
    let filled_left: Vec<usize> = left_info
        .iter()
        .enumerate()
        .filter_map(|(index, info)| info.filled.then_some(index))
        .collect();
    let filled_right: Vec<usize> = right_info
        .iter()
        .enumerate()
        .filter_map(|(index, info)| info.filled.then_some(index))
        .collect();
    if filled_left.is_empty() {
        return Err("aggregate snapshot has no filled left vertex".into());
    }
    let left_slot: BTreeMap<usize, usize> = filled_left
        .iter()
        .enumerate()
        .map(|(slot, &node)| (node, slot))
        .collect();
    let right_slot: BTreeMap<usize, usize> = filled_right
        .iter()
        .enumerate()
        .map(|(slot, &node)| (node, slot))
        .collect();

    let variable_count = filled_left.len() + filled_right.len();
    let zero_row = || vec![Rat::zero(); variable_count];
    let mut costs = vec![Rat::zero(); variable_count];
    for (slot, &left) in filled_left.iter().enumerate() {
        // A.1 asks to maximize eta(phi); the simplex minimizes -eta.
        costs[slot] = -rat(left_info[left].head.len() as i64);
    }
    let mut a_ub: Vec<Vec<Rat>> = Vec::new();
    let mut b_ub: Vec<Rat> = Vec::new();
    let mut a_eq: Vec<Vec<Rat>> = Vec::new();
    let mut b_eq: Vec<Rat> = Vec::new();

    for (slot, &left) in filled_left.iter().enumerate() {
        if !left_can_advance[left] {
            let mut row = zero_row();
            row[slot] = Rat::one();
            a_eq.push(row);
            b_eq.push(Rat::zero());
        }
    }
    // The constructed graph is a cyclic three-cover and every ordinary state
    // starts symmetric. Averaging an optimum over the cover automorphism gives
    // a symmetric optimum, so these equalities select such a model without
    // changing A.1's optimum. This makes copy 0 a genuine base-graph stable
    // assignment instead of an arbitrary cross-copy projection.
    for copies in (0..instance.left_incident.len())
        .collect::<Vec<_>>()
        .chunks(3)
    {
        let slots: Vec<Option<usize>> = copies
            .iter()
            .map(|left| left_slot.get(left).copied())
            .collect();
        match slots.as_slice() {
            [Some(a), Some(b), Some(c)] => {
                for other in [b, c] {
                    let mut row = zero_row();
                    row[*a] = Rat::one();
                    row[*other] = -Rat::one();
                    a_eq.push(row);
                    b_eq.push(Rat::zero());
                }
            }
            [None, None, None] => {}
            _ => return Err("left cover copies disagree on filled status".into()),
        }
    }
    for copies in (0..instance.right_incident.len())
        .collect::<Vec<_>>()
        .chunks(3)
    {
        let slots: Vec<Option<usize>> = copies
            .iter()
            .map(|right| right_slot.get(right).copied())
            .collect();
        match slots.as_slice() {
            [Some(a), Some(b), Some(c)] => {
                for other in [b, c] {
                    let mut row = zero_row();
                    row[filled_left.len() + a] = Rat::one();
                    row[filled_left.len() + other] = -Rat::one();
                    a_eq.push(row);
                    b_eq.push(Rat::zero());
                }
            }
            [None, None, None] => {}
            _ => return Err("right cover copies disagree on filled status".into()),
        }
    }

    for (&right, &rslot) in &right_slot {
        let head_value = right_info[right]
            .head
            .first()
            .map(|&edge| y[edge].clone())
            .ok_or_else(|| format!("filled right vertex {right} has an empty head"))?;
        // A.2: h_w * psi_w - sum_{fw in H_f} phi_f = 0
        let mut row = zero_row();
        row[filled_left.len() + rslot] = rat(right_info[right].head.len() as i64);
        for &edge in &instance.right_incident[right] {
            let left = instance.edges[edge].left;
            if left_info[left].head.contains(&edge) {
                if let Some(&lslot) = left_slot.get(&left) {
                    row[lslot] -= Rat::one();
                }
            }
        }
        a_eq.push(row);
        b_eq.push(Rat::zero());
        // A.3: psi_w <= a(y,w)
        let mut bound = zero_row();
        bound[filled_left.len() + rslot] = Rat::one();
        a_ub.push(bound);
        b_ub.push(head_value);
    }

    for (&left, &lslot) in &left_slot {
        let quota_gap = &instance.quotas_left[left]
            - instance.left_incident[left]
                .iter()
                .fold(Rat::zero(), |sum, &edge| sum + &y[edge]);
        if quota_gap < Rat::zero() {
            return Err(format!(
                "aggregate input violates left quota at vertex {left}: gap={quota_gap}"
            ));
        }
        // A.4: h_f * phi_f - sum_{fw in L_f} psi_w <= q(f) - |y_f|
        let mut row = zero_row();
        row[lslot] = rat(left_info[left].head.len() as i64);
        for &edge in &instance.left_incident[left] {
            if &boundary[edge] < &capacities[edge] {
                let right = instance.edges[edge].right;
                if right_info[right].head.contains(&edge)
                    && let Some(&rslot) = right_slot.get(&right)
                {
                    row[filled_left.len() + rslot] -= Rat::one();
                }
            }
        }
        a_ub.push(row);
        b_ub.push(quota_gap);
        // A.5: phi_f <= b(fw) - y(fw) for each fw in H_f
        for &edge in &left_info[left].head {
            let slack = &capacities[edge] - &y[edge];
            if slack < Rat::zero() {
                return Err(format!("aggregate input exceeds capacity on edge {edge}"));
            }
            let mut bound = zero_row();
            bound[lslot] = Rat::one();
            a_ub.push(bound);
            b_ub.push(slack);
        }
    }

    for (&right, &rslot) in &right_slot {
        let head_value = right_info[right]
            .head
            .first()
            .map(|&edge| y[edge].clone())
            .ok_or_else(|| format!("filled right vertex {right} has an empty head"))?;
        // A.6: phi_f + psi_w <= a(y,w) - y(fw) for fw in pi^c_w(y) - H_w
        for &edge in &right_info[right].critical_edges {
            if right_info[right].head.contains(&edge) {
                continue;
            }
            let left = instance.edges[edge].left;
            if let Some(&lslot) = left_slot.get(&left) {
                let slack = &head_value - &y[edge];
                if slack < Rat::zero() {
                    return Err(format!(
                        "aggregate input has a negative A.6 slack on edge {edge}"
                    ));
                }
                let mut row = zero_row();
                row[lslot] = Rat::one();
                row[filled_left.len() + rslot] = Rat::one();
                a_ub.push(row);
                b_ub.push(slack);
            }
        }
    }
    // A.7: sum_{fw in E_w} phi_f <= q(w) - |y_w| for w in W - W^=
    for right in 0..instance.right_incident.len() {
        if right_slot.contains_key(&right) {
            continue;
        }
        let gap = &instance.quotas_right[right]
            - instance.right_incident[right]
                .iter()
                .fold(Rat::zero(), |sum, &edge| sum + &y[edge]);
        if gap < Rat::zero() {
            return Err(format!(
                "aggregate input violates right quota at vertex {right}: gap={gap}"
            ));
        }
        let mut row = zero_row();
        for &edge in &instance.right_incident[right] {
            let left = instance.edges[edge].left;
            if let Some(&lslot) = left_slot.get(&left) {
                row[lslot] += Rat::one();
            }
        }
        a_ub.push(row);
        b_ub.push(gap);
    }

    let (status, solution) = crate::z3_lp::solve_lp(&costs, &a_ub, &b_ub, &a_eq, &b_eq);
    match status {
        crate::z3_lp::LpStatus::Optimal => {}
        crate::z3_lp::LpStatus::Infeasible => {
            return Err("Appendix A linear program is infeasible".into())
        }
    }
    let phi_values: Vec<Rat> = solution[..filled_left.len()].to_vec();
    let psi_values: Vec<Rat> = solution[filled_left.len()..].to_vec();
    let objective: Rat = filled_left.iter().enumerate().fold(Rat::zero(), |sum, (slot, &left)| {
        sum + rat(left_info[left].head.len() as i64) * &phi_values[slot]
    });
    if debug {
        eprintln!(
            "aggregate: filled_left={} filled_right={} objective={objective}",
            filled_left.len(),
            filled_right.len()
        );
    }
    if objective <= Rat::zero() {
        return Err(format!(
            "Appendix A linear program made no progress: objective={objective}"
        ));
    }
    let mut advanced = y.to_vec();
    for (&left, &lslot) in &left_slot {
        let delta = &phi_values[lslot];
        for &edge in &left_info[left].head {
            advanced[edge] += delta;
        }
    }
    for (&right, &rslot) in &right_slot {
        let delta = &psi_values[rslot];
        for &edge in &right_info[right].head {
            advanced[edge] -= delta;
        }
    }

    for (edge, value) in advanced.iter().enumerate() {
        if value < &Rat::zero() || value > &capacities[edge] {
            return Err(format!(
                "Appendix A produced an out-of-bounds y' on edge {edge}: {value}"
            ));
        }
    }
    for (left, incident) in instance.left_incident.iter().enumerate() {
        let total = incident
            .iter()
            .fold(Rat::zero(), |sum, &edge| sum + &advanced[edge]);
        if total > instance.quotas_left[left] {
            return Err(format!(
                "Appendix A produced a left quota violation at vertex {left}: {total}"
            ));
        }
    }
    for (right, incident) in instance.right_incident.iter().enumerate() {
        let total = incident
            .iter()
            .fold(Rat::zero(), |sum, &edge| sum + &advanced[edge]);
        if total > instance.quotas_right[right] {
            return Err(format!(
                "Appendix A produced a right quota violation at vertex {right}: {total}"
            ));
        }
        if right_info[right].filled && total != instance.quotas_right[right] {
            return Err(format!(
                "Appendix A failed to preserve a filled right vertex {right}: {total}"
            ));
        }
    }

    // The proof requires either a tight constraint in A.3/A.5-A.7 (a
    // monotone parameter changes) or every A.4 constraint to be tight (the
    // process terminates). Verify the exact model rather than accepting an
    // arbitrary point from the optimal face.
    let mut changes_parameter = false;
    for (&right, &rslot) in &right_slot {
        changes_parameter |= psi_values[rslot] == y[right_info[right].head[0]];
    }
    let mut all_a4_tight = true;
    for (&left, &lslot) in &left_slot {
        let gap = &instance.quotas_left[left]
            - instance.left_incident[left]
                .iter()
                .fold(Rat::zero(), |sum, &edge| sum + &y[edge]);
        let mut lhs = rat(left_info[left].head.len() as i64) * &phi_values[lslot];
        for &edge in &instance.left_incident[left] {
            if boundary[edge] < capacities[edge] {
                let right = instance.edges[edge].right;
                if right_info[right].head.contains(&edge)
                    && let Some(&rslot) = right_slot.get(&right)
                {
                    lhs -= &psi_values[rslot];
                }
            }
        }
        all_a4_tight &= lhs == gap;
        for &edge in &left_info[left].head {
            changes_parameter |= phi_values[lslot] == &capacities[edge] - &y[edge];
        }
    }
    for (&right, &rslot) in &right_slot {
        let head_value = &y[right_info[right].head[0]];
        for &edge in &right_info[right].critical_edges {
            if right_info[right].head.contains(&edge) {
                continue;
            }
            if let Some(&lslot) = left_slot.get(&instance.edges[edge].left) {
                changes_parameter |=
                    &phi_values[lslot] + &psi_values[rslot] == head_value - &y[edge];
            }
        }
    }
    for (right, incident) in instance.right_incident.iter().enumerate() {
        if right_slot.contains_key(&right) {
            continue;
        }
        let gap = &instance.quotas_right[right]
            - incident
                .iter()
                .fold(Rat::zero(), |sum, &edge| sum + &y[edge]);
        let lhs = incident.iter().fold(Rat::zero(), |sum, &edge| {
            match left_slot.get(&instance.edges[edge].left) {
                Some(&lslot) => sum + &phi_values[lslot],
                None => sum,
            }
        });
        changes_parameter |= lhs == gap;
    }
    let mut post_left_info = Vec::with_capacity(instance.left_incident.len());
    for (left, incident) in instance.left_incident.iter().enumerate() {
        let (chosen, info) = choose_vertex(
            &advanced,
            incident,
            |edge| instance.edges[edge].left_tie,
            &instance.quotas_left[left],
        );
        if chosen.iter().any(|(edge, value)| value != &advanced[*edge]) {
            return Err(format!(
                "Appendix A y' is not stationary at left vertex {left}"
            ));
        }
        post_left_info.push(info);
    }
    // Karzanov's appendix claims H_f cannot change inside Q and consequently
    // omits it from the positive parameters. With mixed ties a critical F-tie
    // can move when A.4 becomes tight. Treat that structural event as progress
    // instead of falsely applying the A.4-only termination case.
    changes_parameter |= post_left_info != left_info;
    if !changes_parameter && !all_a4_tight {
        return Err(
            "Appendix A optimum has neither a parameter-changing tight constraint nor all A.4 constraints tight"
                .into(),
        );
    }
    if debug {
        eprintln!(
            "  y' checked: changes_parameter={changes_parameter} all_a4_tight={all_a4_tight}"
        );
    }
    Ok(AggregateOutcome {
        assignment: advanced,
        changes_parameter,
        must_terminate: !changes_parameter && all_a4_tight,
        left_info: post_left_info,
    })
}

fn is_positive(previous: &Signature, current: &Signature) -> Result<bool, String> {
    // Appendix: an iteration is positive if at least one of the following
    // parameters strictly advances.
    // (a) the set of saturated-or-lowered edges grows (for some f in F);
    let mut positive = false;
    for (edge, (now, before)) in current
        .saturated_or_lowered
        .iter()
        .zip(&previous.saturated_or_lowered)
        .enumerate()
    {
        if *before && !*now {
            return Err(format!(
                "Appendix A monotonicity violation: saturated-or-lowered edge {edge} disappeared"
            ));
        }
        positive |= *now && !*before;
    }
    // (b) the set W^= of fully filled right vertices grows;
    // (c) for a filled right vertex, the head grows or the critical tie improves.
    for (right, (now, before)) in current.right.iter().zip(&previous.right).enumerate() {
        if before.filled && !now.filled {
            return Err(format!(
                "Appendix A monotonicity violation: filled right vertex {right} became deficient"
            ));
        }
        if now.filled && !before.filled {
            positive = true;
        }
        if !now.filled {
            continue;
        }
        if let (Some(now_tie), Some(before_tie)) = (now.critical_tie, before.critical_tie) {
            if now_tie < before_tie {
                positive = true;
            } else if now_tie > before_tie {
                return Err(format!(
                    "Appendix A monotonicity violation: critical tie worsened at right vertex {right}"
                ));
            } else {
                if before.head.iter().any(|edge| !now.head.contains(edge)) {
                    return Err(format!(
                        "Appendix A monotonicity violation: right head shrank at vertex {right}"
                    ));
                }
                positive |= now.head.iter().any(|edge| !before.head.contains(edge));
            }
        }
    }
    // The Appendix states that filled F-heads are monotone non-increasing and
    // therefore does not list their changes separately. Mixed-tie instances
    // can move the critical F-tie when an already-lowered better edge keeps
    // decreasing. Such a transition cannot belong to a Q whose LP freezes
    // H_f, so conservatively end Q whenever the F choice structure changes.
    positive |= current.left != previous.left;
    Ok(positive)
}

fn update_boundary(boundary: &[Rat], x: &[Rat], y: &[Rat]) -> Result<Vec<Rat>, String> {
    let mut next = boundary.to_vec();
    for edge in 0..next.len() {
        if y[edge] > x[edge] {
            return Err(format!(
                "ordinary iteration increased edge {edge} at W: {} > {}",
                y[edge], x[edge]
            ));
        }
        if y[edge] < x[edge] {
            next[edge] = y[edge].clone();
        }
        if next[edge] > boundary[edge] || next[edge] < Rat::zero() {
            return Err(format!(
                "ordinary boundary invariant failed on edge {edge}: {} -> {}",
                boundary[edge], next[edge]
            ));
        }
    }
    Ok(next)
}

fn parameter_signature(
    capacities: &[Rat],
    boundary_after_iteration: &[Rat],
    saturated_assignment: &[Rat],
    left: Vec<ChoiceInfo>,
    right: Vec<ChoiceInfo>,
) -> Signature {
    Signature {
        // L_i changes at rule (3.1), which is the end of the iteration.  Using
        // the incoming boundary here is one iteration late and can classify a
        // positive iteration as the first member of Q.
        saturated_or_lowered: saturated_assignment
            .iter()
            .zip(boundary_after_iteration)
            .zip(capacities)
            .map(|((value, bound), capacity)| value == capacity || bound < capacity)
            .collect(),
        left,
        right,
    }
}

fn right_infos_for_assignment(
    instance: &Instance,
    assignment: &[Rat],
) -> Result<Vec<ChoiceInfo>, String> {
    let mut infos = Vec::with_capacity(instance.right_incident.len());
    for (right, incident) in instance.right_incident.iter().enumerate() {
        let (chosen, info) = choose_vertex(
            assignment,
            incident,
            |edge| instance.edges[edge].right_tie,
            &instance.quotas_right[right],
        );
        if chosen
            .iter()
            .any(|(edge, value)| value != &assignment[*edge])
        {
            return Err(format!(
                "Appendix A y' is not stationary at right vertex {right}"
            ));
        }
        infos.push(info);
    }
    Ok(infos)
}

fn verify_q_successor(
    instance: &Instance,
    stored: &PendingState,
    x: &[Rat],
    y: &[Rat],
) -> Result<Rat, String> {
    let increase = x
        .iter()
        .zip(&stored.y)
        .fold(Rat::zero(), |sum, (now, before)| sum + now - before);
    if increase <= Rat::zero() {
        return Err(format!(
            "a non-positive Q successor has no positive first-stage increase: {increase}"
        ));
    }
    for edge in 0..x.len() {
        let left = instance.edges[edge].left;
        let expected =
            if stored.left_info[left].filled && stored.left_info[left].head.contains(&edge) {
                let first = stored.left_info[left].head[0];
                &x[first] - &stored.y[first]
            } else {
                Rat::zero()
            };
        if expected < Rat::zero() || x[edge] != &stored.y[edge] + &expected {
            return Err(format!(
                "Q successor is not representable by one phi value at edge {edge}"
            ));
        }
    }
    for edge in 0..x.len() {
        let right = instance.edges[edge].right;
        let expected =
            if stored.right_info[right].filled && stored.right_info[right].head.contains(&edge) {
                let first = stored.right_info[right].head[0];
                &x[first] - &y[first]
            } else {
                Rat::zero()
            };
        if expected < Rat::zero() || y[edge] != &x[edge] - &expected {
            return Err(format!(
                "Q successor is not representable by one psi value at edge {edge}"
            ));
        }
    }
    Ok(increase)
}

struct PendingState {
    boundary: Vec<Rat>,
    x: Vec<Rat>,
    y: Vec<Rat>,
    left_info: Vec<ChoiceInfo>,
    right_info: Vec<ChoiceInfo>,
    signature: Signature,
}

fn stable_assignment(
    instance: &Instance,
    max_iterations: usize,
) -> Result<(Vec<Rat>, usize, usize), String> {
    if max_iterations == 0 {
        return Err("max_iterations must be positive".into());
    }
    let capacities: Vec<Rat> = instance.edges.iter().map(|e| e.capacity.clone()).collect();
    let mut boundary = capacities.clone();
    let mut previous_signature: Option<Signature> = None;
    let mut pending: Option<PendingState> = None;
    let mut big_iterations = 0usize;
    let debug = std::env::var_os("TOPOFLOW_DEBUG_STABLE").is_some();
    for iteration in 1..=max_iterations {
        let (x, y, left_info, right_info) = sweep(instance, &boundary);
        if x == y {
            validate_stable_assignment(instance, &x).map_err(|error| {
                format!("ordinary iteration reached an invalid fixed point: {error}")
            })?;
            if debug {
                eprintln!("iter {iteration}: converged");
            }
            return Ok((x, iteration + big_iterations, big_iterations));
        }
        let next_boundary = update_boundary(&boundary, &x, &y)?;
        let signature = parameter_signature(
            &capacities,
            &next_boundary,
            &x,
            left_info.clone(),
            right_info.clone(),
        );
        let positive = match &previous_signature {
            Some(previous) => is_positive(previous, &signature)?,
            None => true,
        };
        if debug {
            eprintln!("iter {iteration}: positive={positive}");
        }
        if positive {
            pending = None;
        } else if let Some(stored) = pending.take() {
            // Second (or later) non-positive iteration of the sequence Q: the
            // whole remainder of Q is replaced by one big iteration using the
            // state of the FIRST non-positive iteration of Q.
            let successor_increase = verify_q_successor(instance, &stored, &x, &y)?;
            if debug {
                eprintln!("  successor first-stage increase={successor_increase}");
            }
            let outcome = aggregate(
                instance,
                &capacities,
                &stored.boundary,
                &stored.x,
                &stored.y,
                &stored.left_info,
                &stored.right_info,
            )?;

            let stability_error = match validate_stable_assignment(instance, &outcome.assignment) {
                Ok(()) => {
                    if debug {
                        eprintln!("iter {iteration}: big iteration terminated the process");
                    }
                    return Ok((
                        outcome.assignment,
                        iteration + big_iterations + 1,
                        big_iterations + 1,
                    ));
                }
                Err(error) => error,
            };
            if outcome.must_terminate {
                return Err(format!(
                    "Appendix A says the big iteration terminates, but y' is not stable: {stability_error}"
                ));
            }
            if !outcome.changes_parameter {
                return Err("Appendix A big iteration made no monotone-parameter change".into());
            }

            // The first member of Q is retained as an ordinary iteration. The
            // big iteration only lowers bounds on edges that were already in
            // L at the start of Q; H_f edges keep their original capacities.
            let mut boundary_after_big = update_boundary(&stored.boundary, &stored.x, &stored.y)?;
            for edge in 0..boundary_after_big.len() {
                if stored.boundary[edge] < capacities[edge]
                    && outcome.assignment[edge] < stored.y[edge]
                {
                    boundary_after_big[edge] = min(
                        boundary_after_big[edge].clone(),
                        outcome.assignment[edge].clone(),
                    );
                }
                if boundary_after_big[edge] > stored.boundary[edge]
                    || boundary_after_big[edge] < Rat::zero()
                {
                    return Err(format!(
                        "big iteration produced an invalid boundary on edge {edge}: {} -> {}",
                        stored.boundary[edge], boundary_after_big[edge]
                    ));
                }
            }
            let post_right = right_infos_for_assignment(instance, &outcome.assignment)?;
            let post_signature = parameter_signature(
                &capacities,
                &boundary_after_big,
                &outcome.assignment,
                outcome.left_info,
                post_right,
            );
            if !is_positive(&stored.signature, &post_signature)? {
                return Err(
                    "Appendix A claimed a parameter-changing big iteration, but no tracked parameter advanced"
                        .into(),
                );
            }
            boundary = boundary_after_big;
            big_iterations += 1;
            previous_signature = Some(post_signature);
            continue;
        } else {
            // First non-positive iteration of a potential Q: remember its state.
            pending = Some(PendingState {
                boundary: boundary.clone(),
                x: x.clone(),
                y: y.clone(),
                left_info: left_info.clone(),
                right_info: right_info.clone(),
                signature: signature.clone(),
            });
        }
        previous_signature = Some(signature);
        boundary = next_boundary;
    }
    Err(format!(
        "stable assignment did not converge within {max_iterations} iterations"
    ))
}

fn validate_cover_symmetry(instance: &Instance, assignment: &[Rat]) -> Result<(), String> {
    if assignment.len() != instance.edges.len() || !assignment.len().is_multiple_of(3) {
        return Err("covered assignment has an invalid edge count".into());
    }
    for (base_edge, copies) in assignment.chunks_exact(3).enumerate() {
        if copies[0] != copies[1] || copies[0] != copies[2] {
            return Err(format!(
                "stable assignment broke cyclic-cover symmetry at base edge {base_edge}"
            ));
        }
    }
    Ok(())
}

fn validate_original_flow(graph: &GraphData, flows: &[Rat]) -> Result<(), String> {
    if flows.len() != graph.edges.len() {
        return Err("projected flow has the wrong edge count".into());
    }
    let mut parallel: HashMap<(usize, usize), &Rat> = HashMap::new();
    for (edge_id, (edge, value)) in graph.edges.iter().zip(flows).enumerate() {
        if value <= &Rat::zero() || value > &rat(1) {
            return Err(format!(
                "projected original edge {edge_id} is outside (0,1]: {value}"
            ));
        }
        if let Some(previous) = parallel.insert((edge.u, edge.v), value)
            && previous != value
        {
            return Err(format!(
                "projected parallel edges disagree between {} and {}",
                graph.names[edge.u], graph.names[edge.v]
            ));
        }
    }
    for node in 0..graph.names.len() {
        if matches!(graph.kinds[node], Kind::Source | Kind::Sink) {
            continue;
        }
        let incoming = graph.incoming[node]
            .iter()
            .fold(Rat::zero(), |sum, &edge| sum + &flows[edge]);
        let outgoing = graph.outgoing[node]
            .iter()
            .fold(Rat::zero(), |sum, &edge| sum + &flows[edge]);
        if incoming != outgoing {
            return Err(format!(
                "projected flow violates conservation at {}: {incoming} != {outgoing}",
                graph.names[node]
            ));
        }
    }
    Ok(())
}

pub fn solve(
    edges: Vec<(String, String)>,
    max_iterations: usize,
    _tolerance: f64,
) -> Result<NativeFlowSolution, String> {
    let graph = GraphData::new(edges)?;
    let instance = build_instance(&graph)?;
    let source_edge = graph.outgoing[graph.source][0];
    let sink_edge = graph.incoming[graph.sink][0];
    let (assignment, iterations, _) = stable_assignment(&instance, max_iterations)?;
    validate_stable_assignment(&instance, &assignment)
        .map_err(|error| format!("stable validation failed: {error}"))?;
    validate_cover_symmetry(&instance, &assignment)?;

    let mut flows = vec![Rat::zero(); instance.original_count];
    for (edge_id, contract) in instance.edges.iter().enumerate() {
        if let Some((original, copy)) = contract.original {
            if copy == 0 {
                flows[original] = assignment[edge_id].clone();
            }
        }
    }
    validate_original_flow(&graph, &flows)?;
    let total = flows[source_edge].clone();
    let sink = flows[sink_edge].clone();
    if total != sink {
        return Err(format!(
            "stable assignment has unequal source and sink totals: {total} vs {sink}"
        ));
    }
    Ok(NativeFlowSolution::from_fractions(
        "rust-karzanov-exact".into(),
        "Optimal".into(),
        true,
        flows,
        total,
        iterations,
        false,
    ))
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

    fn assert_boundary(raw: &[(&str, &str)], numerator: i64, denominator: i64) {
        let graph = GraphData::new(edges(raw)).expect("valid regression graph");
        let instance = build_instance(&graph).expect("valid stable instance");
        let (assignment, _, _) =
            stable_assignment(&instance, 10_000).expect("stable solver converges");
        validate_stable_assignment(&instance, &assignment).expect("stable assignment");
        validate_cover_symmetry(&instance, &assignment).expect("symmetric cover assignment");
        let source_edge = graph.outgoing[graph.source][0];
        let covered_edge = instance
            .edges
            .iter()
            .enumerate()
            .find_map(|(edge, contract)| {
                (contract.original == Some((source_edge, 0))).then_some(edge)
            })
            .expect("source copy zero");
        assert_eq!(
            assignment[covered_edge],
            Rat::new(BigInt::from(numerator), BigInt::from(denominator))
        );
    }

    #[test]
    fn appendix_a_fraction_regressions() {
        assert_boundary(
            &[
                ("In", "C2_0"),
                ("S2_0", "C2_0"),
                ("C2_0", "S2_0"),
                ("S2_0", "Out"),
            ],
            1,
            2,
        );
        assert_boundary(
            &[
                ("In", "C3_0"),
                ("C3_0", "S3_0"),
                ("S3_0", "Out"),
                ("S3_0", "C3_0"),
                ("S3_0", "C3_0"),
            ],
            1,
            3,
        );
        assert_boundary(
            &[
                ("In", "C3_0"),
                ("C3_0", "S3_0"),
                ("S3_0", "Out"),
                ("S3_0", "S2_0"),
                ("S2_0", "C3_0"),
                ("S3_0", "C2_0"),
                ("S2_0", "C2_0"),
                ("C2_0", "C3_0"),
            ],
            2,
            5,
        );
        assert_boundary(
            &[
                ("In", "C3_0"),
                ("C3_0", "S3_0"),
                ("S3_0", "Out"),
                ("S3_0", "S3_1"),
                ("S3_1", "C2_0"),
                ("S3_1", "C2_0"),
                ("C2_0", "C3_0"),
                ("S3_0", "C2_1"),
                ("S3_1", "C2_1"),
                ("C2_1", "C3_0"),
            ],
            3,
            8,
        );
        assert_boundary(
            &[
                ("In", "C2_0"),
                ("C2_0", "S2_0"),
                ("S2_0", "C3_0"),
                ("C3_0", "S3_0"),
                ("S3_0", "C2_0"),
                ("S3_0", "S3_1"),
                ("S3_1", "C3_0"),
                ("S3_0", "C3_1"),
                ("S3_1", "C3_1"),
                ("S3_1", "C3_1"),
                ("C3_1", "C3_0"),
                ("S2_0", "Out"),
            ],
            4,
            7,
        );
        assert_boundary(
            &[
                ("In", "C3_0"),
                ("C3_0", "C3_1"),
                ("C3_1", "C3_2"),
                ("C3_2", "S3_2"),
                ("S3_2", "S3_1"),
                ("S3_1", "S3_0"),
                ("S3_0", "Out"),
                ("S3_0", "C3_0"),
                ("S3_1", "C3_1"),
                ("S3_2", "C3_2"),
                ("S3_0", "C3_1"),
                ("S3_1", "C3_2"),
                ("S3_2", "C3_0"),
            ],
            1,
            17,
        );
    }
}
