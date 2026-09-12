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
    pub source: usize,
    pub sink: usize,
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
        let source = *ids.get("In").ok_or("missing In node")?;
        let sink = *ids.get("Out").ok_or("missing Out node")?;
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
        for node in 0..names.len() {
            let degree = (incoming[node].len(), outgoing[node].len());
            let kind = match degree {
                (0, 1) if node == source => Kind::Source,
                (1, 0) if node == sink => Kind::Sink,
                (1, 1..=3) => Kind::Splitter,
                (2..=3, 1) => Kind::Merger,
                _ => {
                    return Err(format!(
                        "invalid degree {:?} for node {}",
                        degree, names[node]
                    ));
                }
            };
            kinds.push(kind);
        }
        Ok(Self {
            names,
            edges,
            incoming,
            outgoing,
            kinds,
            source,
            sink,
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
}
