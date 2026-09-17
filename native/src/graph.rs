use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Source,
    Sink,
    Splitter,
    Merger,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub u: usize,
    pub v: usize,
}

#[derive(Clone, Debug)]
pub struct GraphData {
    pub names: Vec<String>,
    pub edges: Vec<Edge>,
    pub incoming: Vec<Vec<usize>>,
    pub outgoing: Vec<Vec<usize>>,
    pub kinds: Vec<Kind>,
    pub sources: Vec<usize>,
    pub sinks: Vec<usize>,
    pub component_of: Vec<usize>,
    pub component_count: usize,
}

impl GraphData {
    pub fn new(raw_edges: Vec<(String, String)>) -> Result<Self, String> {
        if raw_edges.is_empty() {
            return Err("graph has no edges".into());
        }
        let mut names = Vec::new();
        let mut ids = HashMap::new();
        for (u, v) in &raw_edges {
            for name in [u, v] {
                if !ids.contains_key(name) {
                    let id = names.len();
                    ids.insert(name.clone(), id);
                    names.push(name.clone());
                }
            }
        }
        let edges: Vec<Edge> = raw_edges
            .into_iter()
            .map(|(u, v)| Edge {
                u: ids[&u],
                v: ids[&v],
            })
            .collect();
        let mut incoming = vec![Vec::new(); names.len()];
        let mut outgoing = vec![Vec::new(); names.len()];
        for (id, edge) in edges.iter().enumerate() {
            if edge.u == edge.v {
                return Err(format!("self-loop at {}", names[edge.u]));
            }
            outgoing[edge.u].push(id);
            incoming[edge.v].push(id);
        }
        let mut kinds = Vec::with_capacity(names.len());
        let mut sources = Vec::new();
        let mut sinks = Vec::new();
        for node in 0..names.len() {
            let degree = (incoming[node].len(), outgoing[node].len());
            let kind = match degree {
                (0, 1) => Kind::Source,
                (1, 0) => Kind::Sink,
                (1, 1..=3) => Kind::Splitter,
                (2..=3, 1) => Kind::Merger,
                _ => {
                    return Err(format!(
                        "invalid degree {:?} for node {}",
                        degree, names[node]
                    ));
                }
            };
            if kind == Kind::Source {
                sources.push(node);
            }
            if kind == Kind::Sink {
                sinks.push(node);
            }
            kinds.push(kind);
        }
        if sources.is_empty() {
            return Err("graph has no source node (expected a node with degree (0, 1))".into());
        }
        if sinks.is_empty() {
            return Err("graph has no sink node (expected a node with degree (1, 0))".into());
        }

        let mut undirected = vec![Vec::new(); names.len()];
        for edge in &edges {
            undirected[edge.u].push(edge.v);
            undirected[edge.v].push(edge.u);
        }
        let mut component_of = vec![usize::MAX; names.len()];
        let mut component_count = 0;
        for start in 0..names.len() {
            if component_of[start] != usize::MAX {
                continue;
            }
            let mut stack = vec![start];
            component_of[start] = component_count;
            while let Some(node) = stack.pop() {
                for &neighbor in &undirected[node] {
                    if component_of[neighbor] == usize::MAX {
                        component_of[neighbor] = component_count;
                        stack.push(neighbor);
                    }
                }
            }
            component_count += 1;
        }
        let mut component_has_source = vec![false; component_count];
        let mut component_has_sink = vec![false; component_count];
        for &node in &sources {
            component_has_source[component_of[node]] = true;
        }
        for &node in &sinks {
            component_has_sink[component_of[node]] = true;
        }
        for component in 0..component_count {
            if component_has_source[component] && component_has_sink[component] {
                continue;
            }
            let node = component_of
                .iter()
                .position(|&value| value == component)
                .expect("component has a node");
            return Err(format!(
                "component containing {} must have at least one source and one sink",
                names[node]
            ));
        }
        Ok(Self {
            names,
            edges,
            incoming,
            outgoing,
            kinds,
            sources,
            sinks,
            component_of,
            component_count,
        })
    }

    /// Index pairs of parallel edges (same `u -> v`), grouped in one pass.
    pub fn parallel_pairs(&self) -> Vec<(usize, usize)> {
        let mut groups: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (id, edge) in self.edges.iter().enumerate() {
            groups.entry((edge.u, edge.v)).or_default().push(id);
        }
        let mut pairs = Vec::new();
        for ids in groups.values() {
            for a in 0..ids.len() {
                for b in (a + 1)..ids.len() {
                    pairs.push((ids[a], ids[b]));
                }
            }
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(raw: &[(&str, &str)]) -> GraphData {
        GraphData::new(
            raw.iter()
                .map(|(u, v)| (u.to_string(), v.to_string()))
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn groups_two_parallel_edges() {
        let g = graph(&[
            ("In", "S2_0"),
            ("S2_0", "C2_0"),
            ("S2_0", "C2_0"),
            ("C2_0", "Out"),
        ]);
        assert_eq!(g.parallel_pairs(), vec![(1, 2)]);
    }

    #[test]
    fn groups_three_parallel_edges() {
        let g = graph(&[
            ("In", "S2_0"),
            ("S2_0", "C2_0"),
            ("S2_0", "C2_0"),
            ("S2_0", "C2_0"),
            ("C2_0", "Out"),
        ]);
        assert_eq!(g.parallel_pairs(), vec![(1, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn accepts_multiple_boundary_nodes_and_components() {
        let g = graph(&[("InputA", "OutputA"), ("InputB", "OutputB")]);
        assert_eq!(g.sources.len(), 2);
        assert_eq!(g.sinks.len(), 2);
        assert_eq!(g.component_count, 2);
        assert_ne!(g.component_of[g.sources[0]], g.component_of[g.sources[1]]);
    }

    #[test]
    fn rejects_component_without_a_sink() {
        let error = GraphData::new(
            [
                ("GoodIn", "GoodOut"),
                ("In", "M"),
                ("S", "M"),
                ("M", "S"),
            ]
                .into_iter()
                .map(|(u, v)| (u.to_string(), v.to_string()))
                .collect(),
        )
        .expect_err("component without a sink must be rejected");
        assert!(error.contains("at least one source and one sink"));
    }
}
