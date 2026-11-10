//! # The random forest: scikit-learn's `RandomForestRegressor`, as the original fits it
//!
//! The original model fits `RandomForestRegressor(max_depth, min_samples_leaf, max_features="sqrt")`
//! and leaves everything else at scikit-learn's defaults. This file is that forest:
//!
//! - **Bootstrap.** Each tree is grown on `n` rows drawn with replacement from the `n` training
//!   rows. As in scikit-learn, a row drawn `k` times counts with weight `k` (sums and means are
//!   weighted) and rows never drawn are left out.
//! - **Growing a tree** (CART, depth first). A node becomes a leaf when it is `max_depth` deep, has
//!   fewer than 2 rows or fewer than `2 * min_samples_leaf` rows, or all its targets are equal.
//!   Otherwise it looks for the best split among `max_features = floor(sqrt(n_features))`
//!   features drawn at random without replacement. A drawn feature that is constant in the node
//!   still counts as drawn, but if none of the drawn features varies, drawing goes on until one
//!   does (scikit-learn's rule). Features found constant stay excluded below that node.
//! - **The best split** of a feature: sort the node's rows by it; between every two neighbouring
//!   distinct values (more than 1e-7 apart) that leaves at least `min_samples_leaf` rows on each
//!   side, score the split by the reduction in squared error, which for a fixed node is largest
//!   when `sum_left^2 / w_left + sum_right^2 / w_right` is (sums of weighted targets, `w` the
//!   weights). The threshold is the midpoint of the two values; rows `<=` it go left. The first
//!   strictly best split in draw order wins.
//! - **Leaves** predict the weighted mean of their targets; the forest predicts the mean of its
//!   trees.
//! - **Missing values.** Training rows have none. A missing feature at prediction time goes to the
//!   child that had more (distinct) training rows, the right one on a tie (scikit-learn's rule when
//!   no values were missing in training).
//!
//! The draws come from a seeded generator, so a seed gives the same forest on every machine and
//! for any number of threads. The draws differ from scikit-learn's (its RNG is not reproduced), so
//! forests agree in distribution, not tree for tree.

use chapkit::rng::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Two values closer than this count as equal when looking for split points (scikit-learn's
/// `FEATURE_THRESHOLD`).
const FEATURE_THRESHOLD: f64 = 1e-7;

/// Hyperparameters of the forest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForestParams {
    /// Number of trees (`n_estimators`).
    pub n_estimators: usize,
    /// Maximum depth of a tree; the root is depth 0.
    pub max_depth: usize,
    /// Minimum number of (distinct) rows in a leaf.
    pub min_samples_leaf: usize,
}

/// Training rows, column by column: `columns[f][i]` is feature `f` of row `i`.
#[derive(Debug, Clone, Default)]
pub struct Columns {
    pub columns: Vec<Vec<f64>>,
    pub n_rows: usize,
}

impl Columns {
    /// Columns from rows of equal length.
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let n_features = rows.first().map_or(0, Vec::len);
        let columns = (0..n_features)
            .map(|f| {
                rows.iter()
                    .map(|r| r.get(f).copied().unwrap_or(f64::NAN))
                    .collect()
            })
            .collect();
        Self {
            columns,
            n_rows: rows.len(),
        }
    }

    fn value(&self, feature: usize, row: usize) -> f64 {
        self.columns
            .get(feature)
            .and_then(|c| c.get(row))
            .copied()
            .unwrap_or(f64::NAN)
    }
}

/// A split: rows whose `feature` is `<= threshold` go to `left`, the others to `right`, and rows
/// missing it to `left` when `missing_left`. Children always come after their parent in the node
/// list.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Split {
    pub feature: usize,
    pub threshold: f64,
    pub left: usize,
    pub right: usize,
    pub missing_left: bool,
}

/// A tree node: a split, or a leaf (no split).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub split: Option<Split>,
    /// The weighted mean target of the node's rows: the prediction when this is a leaf.
    pub value: f64,
}

/// One regression tree; node 0 is the root.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    /// The prediction for one row of features.
    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut i = 0;
        while let Some(node) = self.nodes.get(i) {
            match node.split {
                None => return node.value,
                Some(s) => {
                    let v = x.get(s.feature).copied().unwrap_or(f64::NAN);
                    let left = if v.is_nan() {
                        s.missing_left
                    } else {
                        v <= s.threshold
                    };
                    i = if left { s.left } else { s.right };
                }
            }
        }
        f64::NAN
    }

    /// True when every child comes after its parent and every index is in range, so
    /// [`Tree::predict`] ends; checked on trees read from a checkpoint.
    pub fn is_well_formed(&self, n_features: usize) -> bool {
        !self.nodes.is_empty()
            && self.nodes.iter().enumerate().all(|(i, n)| {
                n.split.is_none_or(|s| {
                    s.feature < n_features
                        && s.left > i
                        && s.right > i
                        && s.left < self.nodes.len()
                        && s.right < self.nodes.len()
                })
            })
    }
}

/// The forest: the mean of its trees.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Forest {
    pub trees: Vec<Tree>,
}

impl Forest {
    /// Fit `params.n_estimators` trees on `x` (one entry per row) and `y`, each on its own
    /// bootstrap sample, in parallel. The trees' seeds are drawn from `seed` up front, so the
    /// result does not depend on the number of threads.
    pub fn fit(x: &Columns, y: &[f64], params: ForestParams, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let seeds: Vec<u64> = (0..params.n_estimators).map(|_| rng.next_u64()).collect();
        let trees = seeds
            .into_par_iter()
            .map(|s| grow_tree(x, y, params, &mut Rng::new(s)))
            .collect();
        Self { trees }
    }

    /// The predictions for many rows at once (`rows` holds them one after the other, `width`
    /// features each), into `out`. Tree by tree, so each tree is read once for all rows.
    pub fn predict_rows(&self, rows: &[f64], width: usize, out: &mut Vec<f64>) {
        out.clear();
        out.resize(rows.len() / width.max(1), 0.0);
        for tree in &self.trees {
            for (sum, x) in out.iter_mut().zip(rows.chunks_exact(width.max(1))) {
                *sum += tree.predict(x);
            }
        }
        let n = self.trees.len() as f64;
        for v in out.iter_mut() {
            *v /= n;
        }
    }

    /// The mean of the trees' predictions for one row of features.
    pub fn predict(&self, x: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return f64::NAN;
        }
        self.trees.iter().map(|t| t.predict(x)).sum::<f64>() / self.trees.len() as f64
    }
}

/// scikit-learn's `max_features="sqrt"`: `max(1, floor(sqrt(n_features)))`.
pub fn sqrt_features(n_features: usize) -> usize {
    ((n_features as f64).sqrt().floor() as usize).max(1)
}

/// A uniform integer in `[0, n)` (`n > 0`).
fn below(rng: &mut Rng, n: usize) -> usize {
    (rng.next_u64() % n as u64) as usize
}

/// A node waiting to be grown: its rows, depth, the features already known to be constant in it,
/// and where its parent points to it.
struct Pending {
    rows: Vec<usize>,
    depth: usize,
    constant: Vec<bool>,
    node: usize,
}

/// One tree on a bootstrap sample of the rows.
fn grow_tree(x: &Columns, y: &[f64], params: ForestParams, rng: &mut Rng) -> Tree {
    let n = x.n_rows.min(y.len());
    // Bootstrap: n draws with replacement; a row's weight is how often it was drawn.
    let mut weight = vec![0.0; n];
    for _ in 0..n {
        if let Some(w) = weight.get_mut(below(rng, n.max(1))) {
            *w += 1.0;
        }
    }
    let rows: Vec<usize> = (0..n)
        .filter(|&i| weight.get(i).is_some_and(|w| *w > 0.0))
        .collect();
    let n_features = x.columns.len();
    let max_features = sqrt_features(n_features);

    let mut tree = Tree::default();
    tree.nodes.push(Node {
        split: None,
        value: f64::NAN,
    });
    let mut stack = vec![Pending {
        rows,
        depth: 0,
        constant: vec![false; n_features],
        node: 0,
    }];
    let mut sorted = Vec::new();
    while let Some(Pending {
        rows,
        depth,
        mut constant,
        node,
    }) = stack.pop()
    {
        let (w, sum, sum_sq) = rows.iter().fold((0.0, 0.0, 0.0), |(w, s, q), &i| {
            let (wi, yi) = (at(&weight, i), at(y, i));
            (w + wi, s + wi * yi, q + wi * yi * yi)
        });
        let mean = sum / w;
        let impurity = sum_sq / w - mean * mean;
        if let Some(slot) = tree.nodes.get_mut(node) {
            slot.value = mean;
        }
        let leaf = depth >= params.max_depth
            || rows.len() < 2
            || rows.len() < 2 * params.min_samples_leaf
            || impurity <= f64::EPSILON;
        if leaf {
            continue;
        }
        let Some((feature, threshold, missing_left)) = best_split(
            x,
            y,
            &weight,
            &rows,
            &mut constant,
            max_features,
            params.min_samples_leaf,
            rng,
            &mut sorted,
        ) else {
            continue;
        };
        let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
            .iter()
            .partition(|&&i| x.value(feature, i) <= threshold);
        let (left, right) = (tree.nodes.len(), tree.nodes.len() + 1);
        for _ in 0..2 {
            tree.nodes.push(Node {
                split: None,
                value: f64::NAN,
            });
        }
        if let Some(slot) = tree.nodes.get_mut(node) {
            slot.split = Some(Split {
                feature,
                threshold,
                left,
                right,
                missing_left,
            });
        }
        // Depth first, left child first (pushed last).
        for (rows, node) in [(right_rows, right), (left_rows, left)] {
            stack.push(Pending {
                rows,
                depth: depth + 1,
                constant: constant.clone(),
                node,
            });
        }
    }
    tree
}

fn at(v: &[f64], i: usize) -> f64 {
    v.get(i).copied().unwrap_or(f64::NAN)
}

/// The best split of a node's rows, as (feature, threshold, whether missing values go left);
/// `None` when no feature has a split that leaves `min_samples_leaf` rows on both sides. Marks
/// features found constant.
#[allow(clippy::too_many_arguments)]
fn best_split(
    features: &Columns,
    targets: &[f64],
    weights: &[f64],
    rows: &[usize],
    constant: &mut [bool],
    max_features: usize,
    min_samples_leaf: usize,
    rng: &mut Rng,
    sorted: &mut Vec<(f64, f64, f64)>,
) -> Option<(usize, f64, bool)> {
    let n_rows = rows.len();
    let mut order: Vec<usize> = (0..constant.len()).collect();
    let (mut visited, mut varying) = (0, 0);
    let mut best: Option<(f64, usize, f64, bool)> = None;
    // Draw features without replacement (a Fisher-Yates shuffle, one step per draw).
    for draw in 0..order.len() {
        if visited >= max_features && varying > 0 {
            break;
        }
        let pick = draw + below(rng, order.len() - draw);
        order.swap(draw, pick);
        let feature = order.get(draw).copied().unwrap_or(0);
        visited += 1;
        if constant.get(feature).copied().unwrap_or(true) {
            continue;
        }
        // The node's rows sorted by this feature, with their targets and weights.
        sorted.clear();
        sorted.extend(rows.iter().map(|&row| {
            (
                features.value(feature, row),
                at(targets, row),
                at(weights, row),
            )
        }));
        sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let (lowest, highest) = match (sorted.first(), sorted.last()) {
            (Some(first), Some(last)) => (first.0, last.0),
            _ => continue,
        };
        if highest <= lowest + FEATURE_THRESHOLD {
            if let Some(flag) = constant.get_mut(feature) {
                *flag = true;
            }
            continue;
        }
        varying += 1;
        let (w_total, s_total) = sorted
            .iter()
            .fold((0.0, 0.0), |(w, s), &(_, yi, wi)| (w + wi, s + wi * yi));
        let (mut w_left, mut s_left) = (0.0, 0.0);
        // A split at `pos` puts sorted[..pos] left and sorted[pos..] right.
        for pos in 1..n_rows {
            let (Some(&(prev, y_prev, w_prev)), Some(&(next, _, _))) =
                (sorted.get(pos - 1), sorted.get(pos))
            else {
                break;
            };
            w_left += w_prev;
            s_left += w_prev * y_prev;
            if next <= prev + FEATURE_THRESHOLD
                || pos < min_samples_leaf
                || n_rows - pos < min_samples_leaf
            {
                continue;
            }
            let (w_right, s_right) = (w_total - w_left, s_total - s_left);
            let proxy = s_left * s_left / w_left + s_right * s_right / w_right;
            if best.is_none_or(|(b, _, _, _)| proxy > b) {
                best = Some((proxy, feature, midpoint(prev, next), pos > n_rows - pos));
            }
        }
    }
    best.map(|(_, feature, threshold, missing_left)| (feature, threshold, missing_left))
}

/// The threshold between two neighbouring values: their midpoint, or the lower value when the
/// midpoint rounds to the upper one or overflows (scikit-learn's exact check).
#[allow(clippy::float_cmp)]
fn midpoint(prev: f64, next: f64) -> f64 {
    let mid = prev / 2.0 + next / 2.0;
    if mid == next || mid.is_infinite() {
        prev
    } else {
        mid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(n_estimators: usize, max_depth: usize, min_samples_leaf: usize) -> ForestParams {
        ForestParams {
            n_estimators,
            max_depth,
            min_samples_leaf,
        }
    }

    #[test]
    fn a_deep_tree_fits_a_step_exactly() {
        // y steps from 0 to 10 at x = 4.5; a single feature, so every node sees it.
        let rows: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i)]).collect();
        let y: Vec<f64> = (0..10).map(|i| if i < 5 { 0.0 } else { 10.0 }).collect();
        let forest = Forest::fit(&Columns::from_rows(&rows), &y, params(20, 5, 1), 1);
        assert!(forest.predict(&[0.0]) < 1.0);
        assert!(forest.predict(&[9.0]) > 9.0);
        assert!(forest.trees.iter().all(|t| t.is_well_formed(1)));
    }

    #[test]
    fn min_samples_leaf_and_depth_bound_the_tree() {
        let rows: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
        let y: Vec<f64> = (0..100).map(f64::from).collect();
        let forest = Forest::fit(&Columns::from_rows(&rows), &y, params(1, 2, 30), 3);
        let tree = forest.trees.first().unwrap();
        // Depth 2 at most: 1 + 2 + 4 nodes.
        assert!(tree.nodes.len() <= 7);
        // A leaf of 100 bootstrap rows holds at least 30 distinct rows, so at most 3 leaves.
        let leaves = tree.nodes.iter().filter(|n| n.split.is_none()).count();
        assert!(leaves <= 3, "{leaves} leaves");
    }

    #[test]
    fn constant_targets_give_a_single_leaf() {
        let rows: Vec<Vec<f64>> = (0..20).map(|i| vec![f64::from(i), 1.0]).collect();
        let forest = Forest::fit(&Columns::from_rows(&rows), &[4.0; 20], params(3, 10, 1), 5);
        for t in &forest.trees {
            assert_eq!(t.nodes.len(), 1);
            assert_eq!(t.predict(&[3.0, 1.0]), 4.0);
        }
    }

    #[test]
    fn same_seed_same_forest() {
        let rows: Vec<Vec<f64>> = (0..50)
            .map(|i| vec![f64::from(i % 7), f64::from(i % 3), f64::from(i)])
            .collect();
        let y: Vec<f64> = (0..50).map(|i| f64::from(i % 7) * 2.0).collect();
        let x = Columns::from_rows(&rows);
        let a = Forest::fit(&x, &y, params(10, 10, 1), 9);
        assert_eq!(a, Forest::fit(&x, &y, params(10, 10, 1), 9));
        assert_ne!(a, Forest::fit(&x, &y, params(10, 10, 1), 10));
    }

    #[test]
    fn missing_values_follow_the_larger_child() {
        // A split at x = 2.5 leaves 3 rows left and 7 right: a missing x goes right.
        let rows: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i)]).collect();
        let y: Vec<f64> = (0..10).map(|i| if i < 3 { 0.0 } else { 10.0 }).collect();
        let tree = Tree {
            nodes: vec![
                Node {
                    split: Some(Split {
                        feature: 0,
                        threshold: 2.5,
                        left: 1,
                        right: 2,
                        missing_left: false,
                    }),
                    value: 7.0,
                },
                Node {
                    split: None,
                    value: 0.0,
                },
                Node {
                    split: None,
                    value: 10.0,
                },
            ],
        };
        assert_eq!(tree.predict(&[f64::NAN]), 10.0);
        let forest = Forest::fit(&Columns::from_rows(&rows), &y, params(1, 1, 1), 4);
        let root = forest.trees[0].nodes[0].split.unwrap();
        let n_left = rows.iter().filter(|r| r[0] <= root.threshold).count();
        assert_eq!(
            root.missing_left,
            n_left > 10 - n_left || forest.trees[0].nodes.len() == 1
        );
    }

    #[test]
    fn predicting_many_rows_is_predicting_each() {
        let rows: Vec<Vec<f64>> = (0..40)
            .map(|i| vec![f64::from(i % 5), f64::from(i)])
            .collect();
        let y: Vec<f64> = (0..40)
            .map(|i| f64::from(i % 5) + f64::from(i) / 10.0)
            .collect();
        let forest = Forest::fit(&Columns::from_rows(&rows), &y, params(7, 6, 2), 11);
        let mut out = Vec::new();
        forest.predict_rows(&rows.concat(), 2, &mut out);
        let each: Vec<f64> = rows.iter().map(|r| forest.predict(r)).collect();
        for (a, b) in out.iter().zip(&each) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn sqrt_features_as_scikit_learn() {
        assert_eq!(sqrt_features(1), 1);
        assert_eq!(sqrt_features(24), 4);
        assert_eq!(sqrt_features(25), 5);
        assert_eq!(sqrt_features(33), 5);
    }
}
