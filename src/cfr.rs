//! External-sampling Monte Carlo CFR with linear weighting (Linear CFR)
//! and Pluribus-style negative-regret pruning.
//!
//! Training always uses button = 0; play mode rotates seats so the button is
//! seat 0 before building infoset keys, which keeps the infoset space aligned.
//!
//! Infoset key layout: [bucket_lo, bucket_hi, history tokens...].
//! History tokens come from AbsAction::token(), with TOKEN_STREET_SEP between
//! streets. Node action order == Abstraction::abstract_actions() order.

use crate::abstraction::{AbsAction, AbsConfig, Abstraction, Centroids, TOKEN_LEAF, TOKEN_STREET_SEP};
use crate::cards::fresh_deck;
use crate::engine::{Hand, HandConfig, PlayerAction, Street, MAX_PLAYERS};
use crate::record::{Arena, Record};
use dashmap::DashMap;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use hashbrown::HashTable;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub type InfoKey = Box<[u8]>;

/// An infoset's regrets and strategy sums as a checkpoint stores them.
#[derive(Clone, Serialize, Deserialize)]
pub struct Node {
    pub regret: Vec<f64>,
    pub strat: Vec<f64>,
}

/// Regrets and strategy sums by infoset, each node one [`Record`] with its key. A regret keeps
/// f32: in all but 0.002% of the 1.87B-iteration 6-max tree's 205M nodes, a next update of one big
/// blind stands over 1,000 times above f32's resolution of the regret it lands on. A strategy sum
/// keeps f64: in the most visited nodes, which hold 70% of preflop's and 35% of postflop's visits,
/// an update is 1 to 100 times f32's resolution of the sum, and later ones would round away. As a
/// boxed key beside two `Vec<f64>` that tree took ~150 B a node, past what a 48 GB machine holds
/// beside the live server.
pub struct Nodes {
    shards: Box<[RwLock<Shard>]>,
    hasher: ahash::RandomState,
}

/// A shard's table and the arena its records live in.
struct Shard {
    table: HashTable<NodeRecord>,
    arena: Arena,
}

/// `Nodes` has 2^SHARD_BITS shards.
const SHARD_BITS: u32 = 10;

fn is_postflop(key: &[u8]) -> bool {
    key.len() > 2 && key[2..].contains(&TOKEN_STREET_SEP)
}

/// A trainer node's values: its regrets as f32, then its strategy sums as f64.
type NodeRecord = Record<12>;

impl NodeRecord {
    fn node(arena: &mut Arena, key: &[u8], regret: &[f64], strat: &[f64]) -> NodeRecord {
        let mut r = Record::new_in(arena, key, regret.len());
        let values: Vec<f64> = regret.iter().chain(strat).copied().collect();
        r.write(&values);
        r
    }

    /// The regrets then the strategy sums into `out` (twice the action count long).
    fn read(&self, out: &mut [f64]) {
        let n = self.actions();
        let (regret, strat) = self.values().split_at(4 * n);
        let (regret_out, strat_out) = out.split_at_mut(n);
        for (o, c) in regret_out.iter_mut().zip(regret.as_chunks::<4>().0) {
            *o = f32::from_ne_bytes(*c) as f64;
        }
        for (o, c) in strat_out.iter_mut().zip(strat.as_chunks::<8>().0) {
            *o = f64::from_ne_bytes(*c);
        }
    }

    fn write(&mut self, values: &[f64]) {
        let n = self.actions();
        let (regret, strat) = values.split_at(n);
        let (regret_out, strat_out) = self.values_mut().split_at_mut(4 * n);
        for (c, &x) in regret_out.as_chunks_mut::<4>().0.iter_mut().zip(regret) {
            *c = (x as f32).to_ne_bytes();
        }
        for (c, &x) in strat_out.as_chunks_mut::<8>().0.iter_mut().zip(strat) {
            *c = x.to_ne_bytes();
        }
    }
}

impl Nodes {
    fn new() -> Self {
        Nodes {
            shards: (0..1usize << SHARD_BITS)
                .map(|_| RwLock::new(Shard { table: HashTable::new(), arena: Arena::new() }))
                .collect(),
            hasher: ahash::RandomState::new(),
        }
    }

    fn hash(&self, key: &[u8]) -> u64 {
        self.hasher.hash_one(key)
    }

    /// The shard below the hash's top 7 bits, which the table keeps for its control bytes.
    fn shard(&self, hash: u64) -> &RwLock<Shard> {
        &self.shards[((hash << 7) >> (64 - SHARD_BITS)) as usize]
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.read().unwrap().table.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Room for `n` nodes without a rehash (a checkpoint about to load).
    fn reserve(&self, n: usize) {
        let per_shard = n >> SHARD_BITS;
        for shard in self.shards.iter() {
            let mut s = shard.write().unwrap();
            s.table.reserve(per_shard + per_shard / 32 + 64, |r| self.hash(r.key()));
        }
    }

    /// `key`'s regrets into `out` (as long as its action count), the node created at zero on its
    /// first visit.
    fn regrets(&self, key: &[u8], out: &mut [f64]) {
        let n = out.len();
        let hash = self.hash(key);
        let shard = self.shard(hash);
        let mut values = [0.0f64; 2 * MAX_ACTS];
        if let Some(r) = shard.read().unwrap().table.find(hash, |r| r.key() == key) {
            r.read(&mut values[..2 * n]);
            out.copy_from_slice(&values[..n]);
            return;
        }
        let mut s = shard.write().unwrap();
        let Shard { table, arena } = &mut *s;
        match table.find(hash, |r| r.key() == key) {
            Some(r) => {
                r.read(&mut values[..2 * n]);
                out.copy_from_slice(&values[..n]);
            }
            None => {
                let zero = &values[..n];
                let fresh = NodeRecord::node(arena, key, zero, zero);
                table.insert_unique(hash, fresh, |r| self.hash(r.key()));
                out.fill(0.0);
            }
        }
    }

    /// Runs `f` on `key`'s regrets and strategy sums, when the node exists.
    fn update(&self, key: &[u8], f: impl FnOnce(&mut [f64], &mut [f64])) {
        let hash = self.hash(key);
        let mut s = self.shard(hash).write().unwrap();
        if let Some(r) = s.table.find_mut(hash, |r| r.key() == key) {
            let n = r.actions();
            let mut values = [0.0f64; 2 * MAX_ACTS];
            r.read(&mut values[..2 * n]);
            let (regret, strat) = values[..2 * n].split_at_mut(n);
            f(regret, strat);
            r.write(&values[..2 * n]);
        }
    }

    fn insert(&self, key: &[u8], regret: &[f64], strat: &[f64]) {
        let hash = self.hash(key);
        let mut s = self.shard(hash).write().unwrap();
        let Shard { table, arena } = &mut *s;
        let fresh = NodeRecord::node(arena, key, regret, strat);
        match table.find_mut(hash, |r| r.key() == key) {
            Some(r) => *r = fresh,
            None => {
                table.insert_unique(hash, fresh, |r| self.hash(r.key()));
            }
        }
    }

    /// `key`'s strategy sums, if visited.
    fn strat(&self, key: &[u8]) -> Option<Vec<f64>> {
        let hash = self.hash(key);
        let s = self.shard(hash).read().unwrap();
        let r = s.table.find(hash, |r| r.key() == key)?;
        let mut values = vec![0.0; 2 * r.actions()];
        r.read(&mut values);
        Some(values.split_off(r.actions()))
    }

    /// Every node's key, regrets and strategy sums, until `f` fails.
    fn try_for_each<E>(&self, mut f: impl FnMut(&[u8], &[f64], &[f64]) -> Result<(), E>) -> Result<(), E> {
        let mut values = [0.0f64; 2 * MAX_ACTS];
        for shard in self.shards.iter() {
            for r in shard.read().unwrap().table.iter() {
                let n = r.actions();
                r.read(&mut values[..2 * n]);
                let (regret, strat) = values[..2 * n].split_at(n);
                f(r.key(), regret, strat)?;
            }
        }
        Ok(())
    }

    pub fn for_each(&self, mut f: impl FnMut(&[u8], &[f64], &[f64])) {
        let _ = self.try_for_each(|key, regret, strat| {
            f(key, regret, strat);
            Ok::<(), std::convert::Infallible>(())
        });
    }

    /// Every node's regrets and strategy sums, for `f` to change.
    fn for_each_mut(&self, f: impl FnMut(&mut [f64], &mut [f64])) {
        self.for_each_mut_where(|_| true, f);
    }

    /// Every node whose key passes `keep`, its regrets and strategy sums for `f` to change.
    fn for_each_mut_where(&self, keep: impl Fn(&[u8]) -> bool, mut f: impl FnMut(&mut [f64], &mut [f64])) {
        let mut values = [0.0f64; 2 * MAX_ACTS];
        for shard in self.shards.iter() {
            for r in shard.write().unwrap().table.iter_mut() {
                if !keep(r.key()) {
                    continue;
                }
                let n = r.actions();
                r.read(&mut values[..2 * n]);
                let (regret, strat) = values[..2 * n].split_at_mut(n);
                f(regret, strat);
                r.write(&values[..2 * n]);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub hand: HandConfig,
    /// Regret below which an action may be skipped (chips, linearly weighted).
    pub prune_threshold: f64,
    /// Probability a prunable action is actually skipped.
    pub prune_prob: f64,
    /// Iterations before pruning activates.
    pub prune_after: u64,
    /// Traversal RNG seed — distinct seeds give independent self-play runs
    /// (equilibrium-selection studies).
    pub seed: u64,
    /// Pluribus's pruning rules (Science 2019 supplement, Algorithm 1):
    /// whether to prune is decided once per traversal (95% of traversals
    /// prune, 5% explore everything) rather than per action, and pruning
    /// never applies on the final betting round or to actions that lead
    /// directly to a terminal node — the river gains nothing from the
    /// abstraction-refining side effect of pruning, and terminal payoffs
    /// are cheap to examine. Default on: measured to halve the BR
    /// exploitability bound at equal iterations (BASELINES.md, Aug 2026).
    /// `false` restores per-action pruning everywhere.
    pub pluribus_prune: bool,
    /// Pluribus's postflop blueprint: no running average strategy after
    /// the first betting round. Instead `snapshot_strategy` periodically
    /// adds the current regret-matching strategy of every postflop infoset
    /// into `strat`, and the blueprint is the mean of those snapshots. The
    /// average has no convergence guarantee in 6-player poker anyway, and
    /// the current strategy has already zeroed actions the average still
    /// carries residual mass on.
    pub snapshot_avg: bool,
    /// Multiway-focused sampling: at a non-traverser's preflop node the
    /// action is drawn from (1-f)*sigma + f*(sigma renormalised over
    /// non-fold actions) and every update is importance-weighted by
    /// sigma/q, so multiway lines are reached more often while the
    /// fixed point is unchanged. 0 = plain external sampling.
    pub multiway_focus: f64,
    /// Linear CFR weight cap: the iteration weight is min(t, cap).
    /// Pluribus applied linear weighting only for the first 400 minutes;
    /// with an uncapped weight regrets grow ~t^2 over a long run and the
    /// fixed pruning threshold turns over-aggressive. u64::MAX = uncapped.
    pub linear_cap: u64,
    /// The house's cut of every contested pot, in basis points (0 = rake free).
    pub rake_bp: u32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            hand: HandConfig::default(),
            prune_threshold: -3.0e8,
            prune_prob: 0.95,
            prune_after: 200_000,
            seed: 0,
            pluribus_prune: true,
            snapshot_avg: false,
            multiway_focus: 0.0,
            linear_cap: u64::MAX,
            rake_bp: 0,
        }
    }
}

/// Depth-limited search: streets beyond `limit` are not solved; instead each
/// player picks one of four continuation strategies (blueprint as-is, or
/// fold-/call-/raise-biased) and the leaf is valued by a blueprint rollout.
/// This is Pluribus's guard against the blueprint's leaf values being
/// exploitable by a single fixed continuation.
pub struct LeafCfg {
    pub blueprint: Arc<Blueprint>,
    /// The blueprint's own abstraction (rollouts look infosets up in it;
    /// the solver's abstraction may be finer).
    pub abs: Arc<Abstraction>,
    pub limit: Street,
}

const LEAF_BIASES: usize = 4; // 0 = blueprint, 1 = fold-, 2 = call-, 3 = raise-biased
const BIAS_MULT: f64 = 5.0;

/// Re-randomizes hidden cards at a subgame root before each traversal.
pub type RootSampler<'a> = dyn Fn(&mut Hand, &mut SmallRng) + Sync + 'a;

/// Round-start rooting for online search (Pluribus): the subgame is rooted
/// at the start of the current betting round and the actions actually
/// taken since form the "spine". The hero's spine actions are held fixed
/// (already taken); every other player's strategy in the round is
/// re-solved, so their ranges at the hero's current decision are the
/// resolved reach-weighted ones rather than blueprint Bayes. Along the
/// spine the tree uses the real concrete actions (an off-tree bet is
/// solved at its actual size, replacing the nearest abstract size at
/// that node).
pub struct Spine {
    pub hero: usize,
    /// `hist` length at the round's start.
    pub base_len: usize,
    /// (seat, real action) for each step taken in the round so far.
    pub steps: Vec<(usize, PlayerAction)>,
    /// The abstract token recorded for each step (shadow mapping).
    pub tokens: Vec<u8>,
}

/// Mixture weight for spine-targeted sampling: at a non-traverser spine
/// node the actor's action is drawn from eps*spine + (1-eps)*sigma, with
/// the sample importance-weighted by sigma/q so regrets stay unbiased,
/// concentrating traversals on the line actually being played.
const SPINE_EPS: f64 = 0.5;

impl Spine {
    /// If the node (`hist` after `h`'s history, menu `acts`) lies on the
    /// spine: (index of the spine action in `acts`, its real concrete
    /// action, whether the actor is the hero).
    fn step_at(&self, hist: &[u8], h: &Hand, acts: &[AbsAction]) -> Option<(usize, PlayerAction, bool)> {
        let k = hist.len().checked_sub(self.base_len)?;
        if k >= self.steps.len() || hist[self.base_len..] != self.tokens[..k] {
            return None;
        }
        let (seat, act) = self.steps[k];
        if h.to_act() != seat {
            return None;
        }
        let si = acts.iter().position(|a| a.token() == self.tokens[k])?;
        Some((si, act, seat == self.hero))
    }
}

/// Restricted Nash response training (Johanson et al. 2008): with
/// probability `p`, decided once per traversal, every opponent plays the
/// fixed `model` for the whole hand (and contributes nothing to the average
/// strategy); otherwise the traversal is ordinary self-play. The learned
/// strategy maximally exploits the model subject to staying an equilibrium
/// against rational play with weight (1 − p) — an exploitation dial with a
/// bounded-exploitability knob.
#[derive(Debug, Clone, Copy)]
pub struct RnrCfg {
    pub model: crate::table::Baseline,
    pub p: f64,
}

pub struct Trainer {
    pub abs: Arc<Abstraction>,
    pub cfg: TrainConfig,
    nodes: Nodes,
    iters_done: AtomicU64,
    leaf: Option<LeafCfg>,
    /// CFR+ mode (regrets floored at 0). Used for online subgame solves;
    /// blueprint training keeps Pluribus's negative-regret scheme, which the
    /// pruning depends on.
    plus: bool,
    /// Model opponents as lambda-rational (logit QRE) instead of
    /// regret-matching at their nodes. Subgame-only exploitation knob.
    qre_lambda: Option<f64>,
    /// Restricted Nash response mixture: blueprint training, and resolves in
    /// the search's model mode (`SearchParams::model_p`).
    rnr: Option<RnrCfg>,
    /// Round-start rooting for subgame solves.
    spine: Option<Spine>,
    /// RNR opponent model as a fixed blueprint (e.g. a cloned static
    /// opponent). When set, model-opponent traversals sample from this
    /// strategy instead of `rnr.model`'s baseline. Must share this trainer's
    /// abstraction so its (bucket, history) keys line up.
    rnr_opp: Option<Arc<Blueprint>>,
    /// VR-MCCFR (Schmid et al. 2019): control-variate baselines at sampled
    /// opponent nodes. The sampled child's value is replaced by
    /// sum_a sigma(a) b(a) + (v_sampled - b(a_sampled)); unbiased for any
    /// b, and the variance of the traverser's regret updates shrinks as b
    /// learns the expected value. Baselines are keyed by the opponent's
    /// infoset plus the traverser's seat and card bucket (the traverser's
    /// hand drives most of the value's variance), learned as an
    /// exponential average with alpha = 0.5. Not checkpointed: a resumed
    /// run relearns them from zero, which costs variance, not bias.
    vr: bool,
    baselines: DashMap<InfoKey, Vec<f32>, ahash::RandomState>,
}

const BASELINE_ALPHA: f32 = 0.5;

/// Baseline key: opponent infoset key, then the traverser's seat and bucket.
fn baseline_key(key: &[u8], traverser: usize, bucket: u16) -> InfoKey {
    let mut k = Vec::with_capacity(key.len() + 3);
    k.extend_from_slice(key);
    k.push(traverser as u8);
    k.extend_from_slice(&bucket.to_le_bytes());
    k.into_boxed_slice()
}

/// Approximate logit quantal-response distribution over a node's actions:
/// softmax of cumulative regrets normalized to [-1, 1]. lambda = 0 is
/// uniform random; lambda -> infinity approaches the argmax (fully
/// rational). Used to model boundedly rational opponents in search.
pub(crate) fn qre_distribution(regrets: &[f64], lambda: f64, out: &mut Vec<f64>) {
    out.clear();
    let scale = regrets.iter().fold(0.0f64, |m, &r| m.max(r.abs()));
    if scale <= 0.0 || lambda <= 0.0 {
        out.extend(std::iter::repeat_n(
            1.0 / regrets.len() as f64,
            regrets.len(),
        ));
        return;
    }
    let logits: Vec<f64> = regrets.iter().map(|&r| lambda * r / scale).collect();
    let mx = logits.iter().cloned().fold(f64::MIN, f64::max);
    out.extend(logits.iter().map(|&l| (l - mx).exp()));
    let total: f64 = out.iter().sum();
    out.iter_mut().for_each(|p| *p /= total);
}

/// sigma = positive-regret matching; uniform when no positive regret.
pub fn regret_matching(regrets: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.resize(regrets.len(), 0.0);
    regret_matching_into(regrets, out);
}

/// Regret matching into a caller-provided slice (no allocation).
fn regret_matching_into(regrets: &[f64], out: &mut [f64]) {
    let total: f64 = regrets.iter().map(|&r| r.max(0.0)).sum();
    if total > 0.0 {
        for (o, &r) in out.iter_mut().zip(regrets) {
            *o = r.max(0.0) / total;
        }
    } else {
        out.fill(1.0 / regrets.len() as f64);
    }
}

/// Upper bound on the action menu size, for stack buffers in the
/// traversal (fold, check/call, up to 8 bet sizes, all-in).
const MAX_ACTS: usize = 16;
/// Stack buffer for an infoset key; histories longer than this fall back
/// to a heap key.
const KEY_BUF: usize = 96;

pub fn make_key(bucket: u16, hist: &[u8]) -> InfoKey {
    let mut k = Vec::with_capacity(2 + hist.len());
    k.push(bucket as u8);
    k.push((bucket >> 8) as u8);
    k.extend_from_slice(hist);
    k.into_boxed_slice()
}

impl Trainer {
    pub fn new(abs: Arc<Abstraction>, cfg: TrainConfig) -> Self {
        Trainer {
            abs,
            cfg,
            nodes: Nodes::new(),
            iters_done: AtomicU64::new(0),
            leaf: None,
            plus: false,
            qre_lambda: None,
            rnr: None,
            spine: None,
            rnr_opp: None,
            vr: false,
            baselines: DashMap::with_hasher(ahash::RandomState::new()),
        }
    }

    pub fn with_qre(mut self, lambda: Option<f64>) -> Self {
        self.qre_lambda = lambda;
        self
    }

    pub fn with_rnr(mut self, rnr: Option<RnrCfg>) -> Self {
        self.rnr = rnr;
        self
    }

    /// Exploit a fixed opponent given as a blueprint (e.g. a cloned static
    /// bot). The opponent must have been built with this trainer's
    /// abstraction so bucket/history keys align.
    pub fn with_vr(mut self, vr: bool) -> Self {
        self.vr = vr;
        self
    }

    pub fn with_rnr_opponent(mut self, opp: Option<Arc<Blueprint>>) -> Self {
        self.rnr_opp = opp;
        self
    }

    pub fn with_leaf(mut self, leaf: Option<LeafCfg>) -> Self {
        self.leaf = leaf;
        self
    }

    pub fn with_spine(mut self, spine: Option<Spine>) -> Self {
        self.spine = spine;
        self
    }

    /// Concrete action for child `i` of `h`: the real action on the spine
    /// step, the abstract mapping elsewhere.
    fn child_action(&self, h: &Hand, a: AbsAction, i: usize, step: Option<(usize, PlayerAction, bool)>) -> PlayerAction {
        match step {
            Some((si, real, _)) if si == i => real,
            _ => self.abs.concrete(h, a),
        }
    }

    pub fn with_plus(mut self, plus: bool) -> Self {
        self.plus = plus;
        self
    }

    fn regret_floor(&self) -> f64 {
        if self.plus {
            0.0
        } else {
            2.0 * self.cfg.prune_threshold
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn iterations(&self) -> u64 {
        self.iters_done.load(Ordering::Relaxed)
    }

    /// Run `iterations` external-sampling traversals in parallel.
    /// `progress` is called periodically with total completed iterations.
    pub fn run(&self, iterations: u64, progress: &(dyn Fn(u64) + Sync)) {
        let start = self.iters_done.load(Ordering::Relaxed);
        let n = self.cfg.hand.num_players;
        // Traversals are batched per rayon task: hundreds of millions of
        // one-traversal tasks pay measurable scheduler overhead. Each
        // traversal keeps its own t-derived rng, so results are identical
        // to unbatched scheduling.
        const BATCH: u64 = 256;
        let chunks = iterations.div_ceil(BATCH);
        (0..chunks).into_par_iter().for_each(|c| {
            let lo = c * BATCH;
            let hi = ((c + 1) * BATCH).min(iterations);
            for i in lo..hi {
                let t = start + i + 1;
                let mut rng =
                    SmallRng::seed_from_u64(t.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.cfg.seed);
                let traverser = (t % n as u64) as usize;
                let mut deck = fresh_deck();
                deck.shuffle(&mut rng);
                let mut hand = Hand::new(&self.cfg.hand, 0, deck);
                if hand.is_terminal() {
                    continue; // degenerate deal (e.g. blinds all-in, tiny stacks)
                }
                hand.set_rake(self.cfg.rake_bp);
                let weight = t.min(self.cfg.linear_cap) as f64;
                // Pluribus mode decides pruning once per traversal; the
                // per-action mode keeps prune_ok on and rolls per action.
                let prune_ok = t > self.cfg.prune_after
                    && (!self.cfg.pluribus_prune || rng.random::<f64>() < self.cfg.prune_prob);
                let model_opp = self
                    .rnr
                    .as_ref()
                    .is_some_and(|r| rng.random::<f64>() < r.p);
                let mut hist = Vec::with_capacity(32);
                self.traverse(&hand, &mut hist, traverser, weight, 1.0, prune_ok, model_opp, &mut rng);
                let done = self.iters_done.fetch_add(1, Ordering::Relaxed) + 1;
                if done.is_multiple_of(4096) {
                    progress(done - start);
                }
            }
        });
        // Make the final count exact even for the skipped terminal deals.
        self.iters_done.store(start + iterations, Ordering::Relaxed);
        progress(iterations);
    }

    /// Train a subgame rooted at `root` for a time/iteration budget.
    /// Hidden cards are resampled every traversal — uniformly by default, or
    /// by `sampler` (range-weighted sampling from a RangeTracker) — so the
    /// solver learns strategies for every bucket; callers query the bucket
    /// they actually hold afterwards.
    pub fn run_subgame(
        &self,
        root: &Hand,
        root_hist: &[u8],
        time_ms: u64,
        max_iters: u64,
        sampler: Option<&RootSampler<'_>>,
    ) {
        let start = std::time::Instant::now();
        let n = root.num_players();
        let hero = self.spine.as_ref().map_or(root.to_act(), |s| s.hero);
        while self.iterations() < max_iters
            && (start.elapsed().as_millis() as u64) < time_ms
        {
            let t0 = self.iters_done.load(Ordering::Relaxed);
            let batch = 1024.min(max_iters - t0);
            (0..batch).into_par_iter().for_each(|j| {
                let t = t0 + j + 1;
                let mut rng =
                    SmallRng::seed_from_u64(t.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xACE);
                // RNR subgame: against the model only the hero learns, so the hero traverses.
                let model_opp = self.rnr.as_ref().is_some_and(|r| rng.random::<f64>() < r.p);
                let traverser = if model_opp { hero } else { (t % n as u64) as usize };
                if root.folded(traverser) {
                    return;
                }
                let mut h = root.clone();
                match sampler {
                    Some(f) => f(&mut h, &mut rng),
                    None => h.resample_hidden(None, &mut rng),
                }
                let mut hist = root_hist.to_vec();
                self.traverse(&h, &mut hist, traverser, t as f64, 1.0, false, model_opp, &mut rng);
            });
            self.iters_done.fetch_add(batch, Ordering::Relaxed);
        }
    }

    /// One external-sampling traversal. Returns utility (chips) for `traverser`.
    /// `model_opp`: this traversal's opponents play the fixed RNR model.
    #[allow(clippy::too_many_arguments)]
    /// `own_reach`: the traverser's own probability of the path so far, which weighs its average
    /// strategy in an RNR traversal against the model.
    #[allow(clippy::too_many_arguments)]
    fn traverse(
        &self,
        h: &Hand,
        hist: &mut Vec<u8>,
        traverser: usize,
        weight: f64,
        own_reach: f64,
        prune_ok: bool,
        model_opp: bool,
        rng: &mut SmallRng,
    ) -> f64 {
        if h.is_terminal() {
            return h.utilities()[traverser] as f64;
        }
        if let Some(lc) = &self.leaf {
            if h.street() > lc.limit {
                return self.leaf_value(h, hist, traverser, weight, lc, model_opp, rng);
            }
        }
        let p = h.to_act();
        let acts = self.abs.abstract_actions(h);
        let n = acts.len();
        debug_assert!(n <= MAX_ACTS);
        let bucket = self.abs.bucket(h.hole(p), h.board(), rng);
        // Key built on the stack; the map is queried by slice and a heap
        // key is allocated only on the infoset's first visit.
        let mut kbuf = [0u8; KEY_BUF];
        let heap_key;
        let key: &[u8] = if hist.len() + 2 <= KEY_BUF {
            kbuf[0] = bucket as u8;
            kbuf[1] = (bucket >> 8) as u8;
            kbuf[2..2 + hist.len()].copy_from_slice(hist);
            &kbuf[..2 + hist.len()]
        } else {
            heap_key = make_key(bucket, hist);
            &heap_key
        };

        // Snapshot sigma without holding the shard lock during recursion.
        let mut sigma_buf = [0.0f64; MAX_ACTS];
        let mut regrets_buf = [0.0f64; MAX_ACTS];
        self.nodes.regrets(key, &mut regrets_buf[..n]);
        regret_matching_into(&regrets_buf[..n], &mut sigma_buf[..n]);
        let sigma = &sigma_buf[..n];
        let regrets = &regrets_buf[..n];

        let step = self.spine.as_ref().and_then(|sp| sp.step_at(hist, h, &acts));
        if let Some((si, real, true)) = step {
            // The hero's already-taken action this round: fixed, no
            // learning at this node.
            let mut child = h.clone();
            child.apply(real);
            let depth = hist.len();
            hist.push(acts[si].token());
            if !child.is_terminal() && child.street() != h.street() {
                hist.push(TOKEN_STREET_SEP);
            }
            let v = self.traverse(&child, hist, traverser, weight, own_reach, prune_ok, model_opp, rng);
            hist.truncate(depth);
            return v;
        }

        if p == traverser {
            // RNR (Johanson et al. 2008): the strategy kept is the average over every iteration,
            // those against the model included. Against the model no opponent is a learner to
            // accumulate it at its own node, so the traverser adds its strategy here, weighted by
            // its own reach. Self-play updates each seat at five nodes of every six traversals
            // (as a non-traverser), the model only in the one where it traverses: (players - 1)
            // keeps the two at the RNR mixture's p : 1 - p.
            if model_opp && !(self.cfg.snapshot_avg && h.street() != Street::Preflop) {
                let scale = weight * own_reach * (h.num_players() - 1) as f64;
                self.nodes.update(key, |_, strat| {
                    for (st, &s) in strat.iter_mut().zip(sigma) {
                        *st += scale * s;
                    }
                });
            }
            // Full-width over own actions, with negative-regret pruning.
            let pluribus = self.cfg.pluribus_prune;
            let mut explore = [true; MAX_ACTS];
            if prune_ok && !(pluribus && h.street() == Street::River) {
                for (e, &r) in explore.iter_mut().zip(regrets) {
                    *e = r >= self.cfg.prune_threshold
                        || (!pluribus && rng.random::<f64>() >= self.cfg.prune_prob);
                }
            }
            let explore = &mut explore[..n];
            let mut children: Vec<Hand> = Vec::with_capacity(acts.len());
            for (i, &a) in acts.iter().enumerate() {
                let mut child = h.clone();
                child.apply(self.child_action(h, a, i, step));
                if pluribus && child.is_terminal() {
                    explore[i] = true; // terminal payoffs are never pruned
                }
                children.push(child);
            }
            if !explore.iter().any(|&e| e) {
                explore.iter_mut().for_each(|e| *e = true);
            }

            let mut utils = [0.0f64; MAX_ACTS];
            let mut node_util = 0.0;
            for (i, &a) in acts.iter().enumerate() {
                if !explore[i] {
                    continue;
                }
                let child = &children[i];
                let depth = hist.len();
                hist.push(a.token());
                if !child.is_terminal() && child.street() != h.street() {
                    hist.push(TOKEN_STREET_SEP);
                }
                utils[i] = self.traverse(child, hist, traverser, weight, own_reach * sigma[i], prune_ok, model_opp, rng);
                hist.truncate(depth);
                node_util += sigma[i] * utils[i];
            }

            let floor = self.regret_floor();
            self.nodes.update(key, |regret, _| {
                for i in 0..acts.len() {
                    if explore[i] {
                        regret[i] = (regret[i] + weight * (utils[i] - node_util)).max(floor);
                    }
                }
            });
            node_util
        } else if model_opp {
            // RNR model traversal: the opponent plays the fixed model and
            // contributes nothing to the learned average strategy. A cloned
            // blueprint opponent (rnr_opp) takes precedence over the baseline
            // model; both are fixed, not learned.
            let a = if let Some(opp) = &self.rnr_opp {
                let p = h.to_act();
                let bucket = self.abs.bucket(h.hole(p), h.board(), rng);
                match opp.get(bucket, hist) {
                    Some(s) if s.len() == acts.len() && s.iter().sum::<f32>() > 0.0 => {
                        let probs: Vec<f64> = s.iter().map(|&x| x as f64).collect();
                        acts[sample_index(&probs, rng)]
                    }
                    // Unseen/mismatched: the clone's fallback is check/call.
                    _ => AbsAction::CheckCall,
                }
            } else {
                let model = self.rnr.as_ref().expect("model_opp without rnr").model;
                let a = crate::table::baseline_action(model, h, &self.abs, rng);
                if acts.contains(&a) {
                    a
                } else {
                    acts[0]
                }
            };
            let mut child = h.clone();
            child.apply(self.abs.concrete(h, a));
            hist.push(a.token());
            if !child.is_terminal() && child.street() != h.street() {
                hist.push(TOKEN_STREET_SEP);
            }
            self.traverse(&child, hist, traverser, weight, own_reach, prune_ok, model_opp, rng)
        } else {
            // Sample one opponent action from their modeled strategy
            // (regret matching, or logit QRE when exploiting) and
            // accumulate it into their average strategy.
            let qre_buf;
            let dist: &[f64] = match self.qre_lambda {
                Some(l) => {
                    let mut q = Vec::with_capacity(regrets.len());
                    qre_distribution(regrets, l, &mut q);
                    qre_buf = q;
                    &qre_buf
                }
                None => sigma,
            };
            if !(self.cfg.snapshot_avg && h.street() != Street::Preflop) {
                self.nodes.update(key, |_, strat| {
                    for (st, &d) in strat.iter_mut().zip(dist) {
                        *st += weight * d;
                    }
                });
            }
            // Spine-targeted sampling at an opponent's spine node.
            // Off-policy sampling (spine targeting, multiway focus) draws
            // the action from q instead of sigma. Two corrections keep
            // every update unbiased: descendants' updates carry the reach
            // ratio sigma/q in `weight`, and the value returned to the
            // ancestors is scaled by the same ratio (E_q[ratio * v] =
            // E_sigma[v]).
            let (idx, ratio) = match step {
                Some((si, _, false)) => {
                    let idx = if rng.random::<f64>() < SPINE_EPS {
                        si
                    } else {
                        sample_index(dist, rng)
                    };
                    let q = (1.0 - SPINE_EPS) * dist[idx] + if idx == si { SPINE_EPS } else { 0.0 };
                    (idx, dist[idx] / q)
                }
                _ if self.cfg.multiway_focus > 0.0
                    && h.street() == Street::Preflop
                    && acts[0] == AbsAction::Fold
                    && dist[0] < 1.0 =>
                {
                    // Multiway focus (fold suppression): q = (1-f)*sigma +
                    // f*sigma_nonfold, where sigma_nonfold is sigma
                    // renormalised over the non-fold actions. Never samples
                    // an action sigma gives zero mass (no wasted
                    // traversals) and keeps sigma/q within
                    // [1/(1-f+f/(1-p_fold)), 1/(1-f)]. Fold is always
                    // index 0 when legal.
                    let f = self.cfg.multiway_focus;
                    let z = 1.0 - dist[0];
                    let q: Vec<f64> = dist
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| (1.0 - f) * d + if i == 0 { 0.0 } else { f * d / z })
                        .collect();
                    let idx = sample_index(&q, rng);
                    (idx, dist[idx] / q[idx])
                }
                _ => (sample_index(dist, rng), 1.0),
            };
            let weight = weight * ratio;
            let a = acts[idx];
            let mut child = h.clone();
            child.apply(self.child_action(h, a, idx, step));
            hist.push(a.token());
            if !child.is_terminal() && child.street() != h.street() {
                hist.push(TOKEN_STREET_SEP);
            }
            // No truncation needed: the nearest traverser ancestor restores hist.
            let v = ratio * self.traverse(&child, hist, traverser, weight, own_reach, prune_ok, model_opp, rng);
            if !self.vr {
                return v;
            }
            let tb = self.abs.bucket(h.hole(traverser), h.board(), rng);
            let bkey = baseline_key(&key, traverser, tb);
            let mut b = self
                .baselines
                .entry(bkey)
                .or_insert_with(|| vec![0.0f32; acts.len()]);
            let expected: f64 = b.iter().zip(dist).map(|(&bi, &d)| bi as f64 * d).sum();
            let corrected = expected + (v - b[idx] as f64);
            b[idx] += BASELINE_ALPHA * (v as f32 - b[idx]);
            corrected
        }
    }

    /// Value of a depth-limit leaf: every live player simultaneously picks
    /// one of four continuation strategies (an extra 4-action infoset keyed
    /// by bucket + history + TOKEN_LEAF); the traverser explores all four
    /// with regret updates, opponents sample theirs; the outcome is a
    /// blueprint rollout to the end of the hand under the chosen biases.
    fn leaf_value(
        &self,
        h: &Hand,
        hist: &[u8],
        traverser: usize,
        weight: f64,
        lc: &LeafCfg,
        model_opp: bool,
        rng: &mut SmallRng,
    ) -> f64 {
        // Against the model, the opponents play it out in the rollout: no continuation to choose.
        let model = if model_opp { self.rnr.as_ref().map(|r| r.model) } else { None };
        let mut biases = [0usize; MAX_PLAYERS];
        for (p, bias) in biases.iter_mut().enumerate().take(h.num_players()) {
            if p == traverser || h.folded(p) || h.all_in(p) || model.is_some() {
                continue;
            }
            let bucket = self.abs.bucket(h.hole(p), h.board(), rng);
            let key = leaf_key(bucket, hist);
            let mut regret = [0.0f64; LEAF_BIASES];
            self.nodes.regrets(&key, &mut regret);
            let mut sigma = [0.0f64; LEAF_BIASES];
            regret_matching_into(&regret, &mut sigma);
            *bias = sample_index(&sigma, rng);
        }

        if h.folded(traverser) || h.all_in(traverser) {
            return self.leaf_rollout(h, hist, &biases, traverser, lc, model, rng);
        }

        let bucket = self.abs.bucket(h.hole(traverser), h.board(), rng);
        let key = leaf_key(bucket, hist);
        let mut regret = [0.0f64; LEAF_BIASES];
        self.nodes.regrets(&key, &mut regret);
        let mut sigma = [0.0f64; LEAF_BIASES];
        regret_matching_into(&regret, &mut sigma);
        let mut utils = [0.0f64; LEAF_BIASES];
        let mut node_util = 0.0;
        for (b, u) in utils.iter_mut().enumerate() {
            let mut bs = biases;
            bs[traverser] = b;
            *u = self.leaf_rollout(h, hist, &bs, traverser, lc, model, rng);
            node_util += sigma[b] * *u;
        }
        let floor = self.regret_floor();
        self.nodes.update(&key, |regret, _| {
            for (r, &u) in regret.iter_mut().zip(&utils) {
                *r = (*r + weight * (u - node_util)).max(floor);
            }
        });
        node_util
    }

    /// Play the hand out from `h` with every player following the blueprint
    /// under their continuation bias, or, given a `model`, the opponents
    /// playing it; returns the traverser's chip utility.
    #[allow(clippy::too_many_arguments)]
    fn leaf_rollout(
        &self,
        h: &Hand,
        hist: &[u8],
        biases: &[usize; MAX_PLAYERS],
        traverser: usize,
        lc: &LeafCfg,
        model: Option<crate::table::Baseline>,
        rng: &mut SmallRng,
    ) -> f64 {
        let mut h = h.clone();
        let mut hist = hist.to_vec();
        while !h.is_terminal() {
            let p = h.to_act();
            let acts = lc.abs.abstract_actions(&h);
            let a = match model {
                Some(m) if p != traverser => {
                    let a = crate::table::baseline_action(m, &h, &lc.abs, rng);
                    if acts.contains(&a) { a } else { acts[0] }
                }
                _ => {
                    let bucket = lc.abs.bucket(h.hole(p), h.board(), rng);
                    let mut probs: Vec<f64> = match lc.blueprint.get(bucket, &hist) {
                        Some(s) if s.len() == acts.len() => s.iter().map(|&x| x as f64).collect(),
                        _ => vec![1.0 / acts.len() as f64; acts.len()],
                    };
                    apply_bias(&mut probs, &acts, biases[p]);
                    acts[sample_index(&probs, rng)]
                }
            };
            let street_before = h.street();
            h.apply(lc.abs.concrete(&h, a));
            hist.push(a.token());
            if !h.is_terminal() && h.street() != street_before {
                hist.push(TOKEN_STREET_SEP);
            }
        }
        h.utilities()[traverser] as f64
    }

    /// Scales every infoset's strategy sums by `factor` (see `train --strat-discount`).
    pub fn discount_strategy(&self, factor: f64) {
        self.nodes.for_each_mut(|_, strat| strat.iter_mut().for_each(|s| *s *= factor));
    }

    /// Snapshot mode: add every postflop infoset's current regret-matching
    /// strategy into its `strat` accumulator. Called periodically after a
    /// warm-up; the exported blueprint is the mean of the snapshots.
    /// Returns the number of infosets snapshotted.
    pub fn snapshot_strategy(&self) -> usize {
        let mut n = 0;
        let mut s = [0.0f64; MAX_ACTS];
        self.nodes.for_each_mut_where(is_postflop, |regret, strat| {
            regret_matching_into(regret, &mut s[..regret.len()]);
            for (st, &p) in strat.iter_mut().zip(&s) {
                *st += p;
            }
            n += 1;
        });
        n
    }

    /// Normalized average strategy for an infoset, if visited.
    pub fn avg_strategy(&self, bucket: u16, hist: &[u8]) -> Option<Vec<f64>> {
        normalize(&self.nodes.strat(&make_key(bucket, hist))?)
    }

    /// Export the (much smaller) normalized average strategy for play.
    pub fn to_blueprint(&self) -> Blueprint {
        let mut strategies = StrategyMap::with_capacity(self.nodes.len());
        self.nodes.for_each(|key, _, strat| {
            if let Some(s) = normalize(strat) {
                strategies.insert(key, &s.iter().map(|&x| x as f32).collect::<Vec<f32>>());
            }
        });
        Blueprint {
            strategies,
            iterations: self.iterations(),
            num_players: self.cfg.hand.num_players,
            abs_cfg: self.abs.cfg.clone(),
            centroids: self.abs.centroids.clone(),
        }
    }

    /// Full checkpoint (regrets + strategy sums) for resuming training.
    ///
    /// The nodes stream straight from the table (a copy of a 1.5B-iteration
    /// tree doubles its memory, and the system killed the 2B run at 1.475B),
    /// into a temporary file renamed over `path` only once it is complete.
    pub fn save_checkpoint(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let ckpt = CheckpointRef {
            iterations: self.iterations(),
            abs_cfg: &self.abs.cfg,
            centroids: &self.abs.centroids,
            nodes: NodesRef(&self.nodes),
        };
        let tmp = format!("{path}.tmp");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        bincode::serialize_into(&mut f, &ckpt).map_err(std::io::Error::other)?;
        f.flush()?;
        drop(f);
        std::fs::rename(&tmp, path)
    }

    /// Resume from a checkpoint. The card abstraction (config + k-means
    /// centroids) is restored from the file so bucketing matches exactly.
    /// The nodes stream into the table one by one: the whole list in memory
    /// beside the table would double the peak.
    pub fn load_checkpoint(path: &str, cfg: TrainConfig) -> std::io::Result<Trainer> {
        fn next<T: serde::de::DeserializeOwned>(f: impl std::io::Read) -> std::io::Result<T> {
            bincode::deserialize_from(f).map_err(std::io::Error::other)
        }
        let mut f = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(path)?);
        let iterations: u64 = next(&mut f)?;
        let abs_cfg: AbsConfig = next(&mut f)?;
        let centroids: Option<Centroids> = next(&mut f)?;
        let len: u64 = next(&mut f)?;
        let t = Trainer::new(Arc::new(Abstraction::with_centroids(abs_cfg, centroids)), cfg);
        t.nodes.reserve(len as usize);
        for _ in 0..len {
            let (key, node): (Vec<u8>, Node) = next(&mut f)?;
            t.nodes.insert(&key, &node.regret, &node.strat);
        }
        t.iters_done.store(iterations, Ordering::Relaxed);
        Ok(t)
    }

    /// Writes the blueprint [`Trainer::to_blueprint`] builds, streamed from
    /// the nodes: built in memory, a 205M-infoset blueprint takes ~20 GB on
    /// top of the trainer. Returns the number of strategies.
    pub fn save_blueprint(&self, path: &str) -> std::io::Result<usize> {
        use std::io::Write;
        let mut strategies = 0;
        self.nodes.for_each(|_, _, strat| strategies += usize::from(strat.iter().sum::<f64>() > 0.0));
        let bp = BlueprintRef {
            strategies: StrategiesRef { nodes: &self.nodes, len: strategies },
            iterations: self.iterations(),
            num_players: self.cfg.hand.num_players,
            abs_cfg: &self.abs.cfg,
            centroids: &self.abs.centroids,
        };
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        bincode::serialize_into(&mut f, &bp).map_err(std::io::Error::other)?;
        f.flush()?;
        Ok(strategies)
    }
}

/// On-disk layout of blueprints written before `AbsConfig::menu`.
#[derive(Deserialize)]
struct LegacyAbsConfig {
    postflop_buckets: u16,
    equity_rollouts: u32,
    dist_runouts: u32,
    runout_rollouts: u32,
    cache_cap: usize,
}

#[derive(Deserialize)]
struct LegacyBlueprint {
    strategies: StrategyMap,
    iterations: u64,
    num_players: usize,
    abs_cfg: LegacyAbsConfig,
    centroids: Option<Centroids>,
}

/// Writes the blueprint of a checkpoint's nodes to `out`, reading the checkpoint twice instead of
/// loading it: a 1.92B-iteration RNR checkpoint holds ~480M nodes, past what the machine that
/// trained it holds in memory. With `fill_current`, a node with no strategy sums plays its current
/// strategy (regret matching) when some regret is positive: RNR traversals against the model add
/// to no average, so the lines only the model's opponents reach (300M of those nodes, against
/// Brasa's) have regrets alone. Returns the number of strategies.
pub fn blueprint_from_checkpoint(
    checkpoint: &str,
    num_players: usize,
    fill_current: bool,
    out: &str,
) -> std::io::Result<usize> {
    use std::io::Write;
    fn next<T: serde::de::DeserializeOwned>(f: impl std::io::Read) -> std::io::Result<T> {
        bincode::deserialize_from(f).map_err(std::io::Error::other)
    }
    fn put<T: Serialize + ?Sized>(f: impl Write, value: &T) -> std::io::Result<()> {
        bincode::serialize_into(f, value).map_err(std::io::Error::other)
    }
    let open = || -> std::io::Result<_> {
        let mut f = std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(checkpoint)?);
        let iterations: u64 = next(&mut f)?;
        let abs_cfg: AbsConfig = next(&mut f)?;
        let centroids: Option<Centroids> = next(&mut f)?;
        let len: u64 = next(&mut f)?;
        Ok((f, iterations, abs_cfg, centroids, len))
    };
    let strategy = |node: &Node| -> Option<Vec<f64>> {
        normalize(&node.strat).or_else(|| {
            let positive = fill_current && node.regret.iter().any(|&r| r > 0.0);
            positive.then(|| {
                let mut s = Vec::new();
                regret_matching(&node.regret, &mut s);
                s
            })
        })
    };
    let (mut f, _, _, _, len) = open()?;
    let mut strategies = 0usize;
    for _ in 0..len {
        let (_, node): (Vec<u8>, Node) = next(&mut f)?;
        strategies += usize::from(strategy(&node).is_some());
    }
    let (mut f, iterations, abs_cfg, centroids, len) = open()?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(out)?);
    put(&mut w, &(strategies as u64))?;
    for _ in 0..len {
        let (key, node): (Vec<u8>, Node) = next(&mut f)?;
        if let Some(p) = strategy(&node) {
            put(&mut w, &key)?;
            put(&mut w, &p.iter().map(|&x| x as f32).collect::<Vec<f32>>())?;
        }
    }
    put(&mut w, &iterations)?;
    put(&mut w, &num_players)?;
    put(&mut w, &abs_cfg)?;
    put(&mut w, &centroids)?;
    w.flush()?;
    Ok(strategies)
}

/// A checkpoint on disk: the iterations, the abstraction, then the nodes as a
/// list of (key, [`Node`]) pairs, every value f64 whatever the table holds.
#[derive(Serialize)]
struct CheckpointRef<'a> {
    iterations: u64,
    abs_cfg: &'a AbsConfig,
    centroids: &'a Option<Centroids>,
    nodes: NodesRef<'a>,
}

/// The node table as the list of (key, [`Node`]) pairs a checkpoint stores.
struct NodesRef<'a>(&'a Nodes);

impl Serialize for NodesRef<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        self.0.try_for_each(|key, regret, strat| seq.serialize_element(&(key, (regret, strat))))?;
        seq.end()
    }
}

/// Serializes exactly like [`Blueprint`], its strategies read off a trainer's nodes.
#[derive(Serialize)]
struct BlueprintRef<'a> {
    strategies: StrategiesRef<'a>,
    iterations: u64,
    num_players: usize,
    abs_cfg: &'a AbsConfig,
    centroids: &'a Option<Centroids>,
}

/// The [`StrategyMap`] of the `len` nodes with a strategy.
struct StrategiesRef<'a> {
    nodes: &'a Nodes,
    len: usize,
}

impl Serialize for StrategiesRef<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(self.len))?;
        self.nodes.try_for_each(|key, _, strat| match normalize(strat) {
            Some(p) => map.serialize_entry(key, &p.iter().map(|&x| x as f32).collect::<Vec<f32>>()),
            None => Ok(()),
        })?;
        map.end()
    }
}

fn leaf_key(bucket: u16, hist: &[u8]) -> InfoKey {
    let mut k = Vec::with_capacity(3 + hist.len());
    k.push(bucket as u8);
    k.push((bucket >> 8) as u8);
    k.extend_from_slice(hist);
    k.push(TOKEN_LEAF);
    k.into_boxed_slice()
}

/// Reweight a strategy toward folding / calling / raising (bias 1/2/3) by
/// multiplying the matching actions' probabilities by BIAS_MULT, then
/// renormalizing. Bias 0 leaves the blueprint strategy as-is.
pub(crate) fn apply_bias(probs: &mut [f64], acts: &[AbsAction], bias: usize) {
    for (p, a) in probs.iter_mut().zip(acts) {
        let boosted = match bias {
            1 => matches!(a, AbsAction::Fold),
            2 => matches!(a, AbsAction::CheckCall),
            3 => matches!(a, AbsAction::Bet(_) | AbsAction::AllIn),
            _ => false,
        };
        if boosted {
            *p *= BIAS_MULT;
        }
    }
    let total: f64 = probs.iter().sum();
    if total > 0.0 {
        for p in probs.iter_mut() {
            *p /= total;
        }
    } else {
        let u = 1.0 / probs.len() as f64;
        probs.iter_mut().for_each(|p| *p = u);
    }
}

fn normalize(v: &[f64]) -> Option<Vec<f64>> {
    let total: f64 = v.iter().sum();
    if total <= 0.0 {
        return None;
    }
    Some(v.iter().map(|&x| x / total).collect())
}

pub fn sample_index(probs: &[f64], rng: &mut SmallRng) -> usize {
    let mut r: f64 = rng.random();
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    probs.len() - 1
}

/// Strategy store keyed by (bucket, history) bytes: each infoset one [`Record`] of f32
/// probabilities behind a thin pointer. As a map of `Vec<u8>` to `Vec<f32>` the 172M-infoset
/// blueprint took 18 GB in the live server. It serializes as that map did (len + entries), so
/// existing .bin files load as they are. ahash: `get` runs on every decision of every hand.
pub struct StrategyMap {
    table: HashTable<Record<4>>,
    arena: Arena,
    hasher: ahash::RandomState,
}

impl StrategyMap {
    pub fn with_capacity(n: usize) -> Self {
        StrategyMap { table: HashTable::with_capacity(n), arena: Arena::new(), hasher: ahash::RandomState::new() }
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    pub fn get(&self, key: &[u8]) -> Option<&[f32]> {
        let hash = self.hasher.hash_one(key);
        self.table.find(hash, |r| r.key() == key).map(|r| r.floats())
    }

    /// `key`'s probabilities, in place of any it had.
    pub fn insert(&mut self, key: &[u8], probs: &[f32]) {
        let StrategyMap { table, arena, hasher } = self;
        let hash = hasher.hash_one(key);
        let mut fresh = Record::new_in(arena, key, probs.len());
        fresh.floats_mut().copy_from_slice(probs);
        match table.find_mut(hash, |r| r.key() == key) {
            Some(r) => *r = fresh,
            None => {
                table.insert_unique(hash, fresh, |r| hasher.hash_one(r.key()));
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &[f32])> {
        self.table.iter().map(|r| (r.key(), r.floats()))
    }

    pub fn keys(&self) -> impl Iterator<Item = &[u8]> {
        self.table.iter().map(|r| r.key())
    }

    pub fn values(&self) -> impl Iterator<Item = &[f32]> {
        self.table.iter().map(|r| r.floats())
    }
}

impl Default for StrategyMap {
    fn default() -> Self {
        StrategyMap::with_capacity(0)
    }
}

impl Clone for StrategyMap {
    fn clone(&self) -> Self {
        let mut copy = StrategyMap::with_capacity(self.len());
        for (key, probs) in self.iter() {
            copy.insert(key, probs);
        }
        copy
    }
}

impl PartialEq for StrategyMap {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().all(|(key, probs)| other.get(key) == Some(probs))
    }
}

impl std::fmt::Debug for StrategyMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StrategyMap({} infosets)", self.len())
    }
}

impl FromIterator<(Vec<u8>, Vec<f32>)> for StrategyMap {
    fn from_iter<I: IntoIterator<Item = (Vec<u8>, Vec<f32>)>>(entries: I) -> Self {
        let mut map = StrategyMap::default();
        for (key, probs) in entries {
            map.insert(&key, &probs);
        }
        map
    }
}

impl Serialize for StrategyMap {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = s.serialize_map(Some(self.len()))?;
        for (key, probs) in self.iter() {
            map.serialize_entry(key, probs)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for StrategyMap {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Entries;
        impl<'de> serde::de::Visitor<'de> for Entries {
            type Value = StrategyMap;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a map of infoset keys to probabilities")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut entries: A) -> Result<StrategyMap, A::Error> {
                let mut map = StrategyMap::with_capacity(entries.size_hint().unwrap_or(0));
                while let Some((key, probs)) = entries.next_entry::<Vec<u8>, Vec<f32>>()? {
                    map.insert(&key, &probs);
                }
                Ok(map)
            }
        }
        d.deserialize_map(Entries)
    }
}

/// The trained average strategy used at the table, together with the card
/// abstraction it was trained under (config + k-means centroids), so play
/// and benchmarking bucket cards exactly as training did.
#[derive(Clone, Serialize, Deserialize)]
pub struct Blueprint {
    pub strategies: StrategyMap,
    pub iterations: u64,
    pub num_players: usize,
    pub abs_cfg: AbsConfig,
    pub centroids: Option<Centroids>,
}

impl Blueprint {
    pub fn get(&self, bucket: u16, hist: &[u8]) -> Option<&[f32]> {
        self.strategies.get(&make_key(bucket, hist))
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let f = std::io::BufWriter::new(std::fs::File::create(path)?);
        bincode::serialize_into(f, self).map_err(std::io::Error::other)
    }

    pub fn load(path: &str) -> std::io::Result<Blueprint> {
        let f = std::io::BufReader::new(std::fs::File::open(path)?);
        match bincode::deserialize_from(f) {
            Ok(bp) => Ok(bp),
            Err(e) => {
                // Blueprints written before AbsConfig::menu existed (all
                // 2026-07/08 artifacts) were trained with the Wide menu.
                let f = std::io::BufReader::new(std::fs::File::open(path)?);
                let legacy: LegacyBlueprint =
                    bincode::deserialize_from(f).map_err(|_| std::io::Error::other(e))?;
                Ok(Blueprint {
                    strategies: legacy.strategies,
                    iterations: legacy.iterations,
                    num_players: legacy.num_players,
                    abs_cfg: AbsConfig {
                        postflop_buckets: legacy.abs_cfg.postflop_buckets,
                        menu: crate::abstraction::MenuShape::Wide,
                        equity_rollouts: legacy.abs_cfg.equity_rollouts,
                        dist_runouts: legacy.abs_cfg.dist_runouts,
                        runout_rollouts: legacy.abs_cfg.runout_rollouts,
                        cache_cap: legacy.abs_cfg.cache_cap,
                    },
                    centroids: legacy.centroids,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (written first, TDD)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{preflop_bucket, AbsAction, AbsConfig};
    use crate::cards::make_card;

    fn push_fold_trainer() -> Trainer {
        // 10bb heads-up: near push/fold; converges fast and has known properties.
        let abs_cfg = AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 50,
            dist_runouts: 12,
            runout_rollouts: 25,
            cache_cap: 1_000_000,
            menu: crate::abstraction::MenuShape::Wide,
        };
        let cents = Centroids::train(&abs_cfg, 400, 99);
        let abs = Abstraction::with_centroids(abs_cfg, Some(cents));
        let cfg = TrainConfig {
            hand: HandConfig {
                num_players: 2,
                stack: 1_000,
                sb: 50,
                bb: 100,
            },
            prune_after: u64::MAX, // no pruning in this small test
            ..TrainConfig::default()
        };
        Trainer::new(Arc::new(abs), cfg)
    }

    #[test]
    fn regret_matching_math() {
        let mut out = Vec::new();
        regret_matching(&[300.0, 100.0, -50.0], &mut out);
        assert_eq!(out, vec![0.75, 0.25, 0.0]);
        regret_matching(&[-5.0, -10.0], &mut out);
        assert_eq!(out, vec![0.5, 0.5]);
    }

    #[test]
    fn qre_distribution_interpolates_rationality() {
        let regrets = [900.0, 300.0, -600.0];
        let mut q = Vec::new();

        qre_distribution(&regrets, 0.0, &mut q);
        for &p in &q {
            assert!((p - 1.0 / 3.0).abs() < 1e-12, "lambda=0 must be uniform");
        }

        qre_distribution(&regrets, 2.0, &mut q);
        assert!(q[0] > q[1] && q[1] > q[2], "must order by regret: {:?}", q);
        assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        qre_distribution(&regrets, 50.0, &mut q);
        assert!(q[0] > 0.99, "large lambda approaches argmax: {:?}", q);

        // Degenerate all-zero regrets: uniform.
        qre_distribution(&[0.0, 0.0], 3.0, &mut q);
        assert_eq!(q, vec![0.5, 0.5]);
    }

    #[test]
    fn apply_bias_shifts_mass() {
        let acts = [
            AbsAction::Fold,
            AbsAction::CheckCall,
            AbsAction::Bet(2),
            AbsAction::AllIn,
        ];
        let base = [0.25f64, 0.25, 0.25, 0.25];

        let mut p = base;
        apply_bias(&mut p, &acts, 0);
        assert_eq!(p, base, "bias 0 is the blueprint as-is");

        let mut p = base;
        apply_bias(&mut p, &acts, 1);
        assert!(p[0] > 0.6, "fold bias must boost fold: {:?}", p);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let mut p = base;
        apply_bias(&mut p, &acts, 3);
        assert!(p[2] > 0.3 && p[3] > 0.3, "raise bias boosts bets+allin: {:?}", p);
        assert!(p[0] < 0.1);
    }

    /// RNR against an always-caller model: with no fold equity to exploit,
    /// the near-pure best response must beat a calling station by clearly
    /// more than the equilibrium strategy does at the same training budget.
    #[test]
    fn rnr_exploits_the_modeled_opponent() {
        use crate::bot::Policy;
        use crate::table::{run_eval, Baseline};

        let train = |rnr: Option<RnrCfg>| {
            let t = push_fold_trainer().with_rnr(rnr);
            t.run(120_000, &|_| {});
            let bp = t.to_blueprint();
            let abs = Abstraction::with_centroids(bp.abs_cfg.clone(), bp.centroids.clone());
            Policy::new(bp, Arc::new(abs))
        };
        let cfg = HandConfig {
            num_players: 2,
            stack: 1_000,
            sb: 50,
            bb: 100,
        };
        let nash = train(None);
        let rnr = train(Some(RnrCfg {
            model: Baseline::Caller,
            p: 0.9,
        }));
        let w_nash = run_eval(&nash, &cfg, Baseline::Caller, 40_000, 5);
        let w_rnr = run_eval(&rnr, &cfg, Baseline::Caller, 40_000, 5);
        assert!(
            w_rnr.mbb_per_hand > w_nash.mbb_per_hand + 50.0,
            "RNR(caller, 0.9) must exploit a caller more than equilibrium: \
             rnr {:+.0}±{:.0} vs nash {:+.0}±{:.0}",
            w_rnr.mbb_per_hand,
            w_rnr.ci95,
            w_nash.mbb_per_hand,
            w_nash.ci95
        );
    }

    /// RNR against a blueprint opponent: an empty-strategy blueprint plays
    /// pure check/call by fallback (a calling station), so exploiting it via
    /// the blueprint-opponent path must beat equilibrium by clearly more —
    /// the same result as exploiting Baseline::Caller, validating that the
    /// clone-opponent traversal drives the exploit.
    #[test]
    fn rnr_exploits_a_blueprint_opponent() {
        use crate::bot::Policy;
        use crate::table::{run_eval, Baseline};

        let policy_from = |t: Trainer| {
            t.run(120_000, &|_| {});
            let bp = t.to_blueprint();
            let abs = Abstraction::with_centroids(bp.abs_cfg.clone(), bp.centroids.clone());
            Policy::new(bp, Arc::new(abs))
        };
        let empty_opp = Arc::new(Blueprint {
            strategies: Default::default(),
            iterations: 0,
            num_players: 2,
            abs_cfg: AbsConfig::default(),
            centroids: None,
        });
        let nash = policy_from(push_fold_trainer());
        let rnr = policy_from(
            push_fold_trainer()
                .with_rnr(Some(RnrCfg {
                    model: Baseline::Caller, // ignored when rnr_opp is set
                    p: 0.9,
                }))
                .with_rnr_opponent(Some(empty_opp)),
        );
        let cfg = HandConfig {
            num_players: 2,
            stack: 1_000,
            sb: 50,
            bb: 100,
        };
        let w_nash = run_eval(&nash, &cfg, Baseline::Caller, 40_000, 5);
        let w_rnr = run_eval(&rnr, &cfg, Baseline::Caller, 40_000, 5);
        assert!(
            w_rnr.mbb_per_hand > w_nash.mbb_per_hand + 50.0,
            "RNR vs a check/call blueprint must exploit more than equilibrium: \
             rnr {:+.0}±{:.0} vs nash {:+.0}±{:.0}",
            w_rnr.mbb_per_hand,
            w_rnr.ci95,
            w_nash.mbb_per_hand,
            w_nash.ci95
        );
    }

    #[test]
    fn key_roundtrip_bucket_encoding() {
        let k = make_key(168, &[2, 15, 1]);
        assert_eq!(&*k, &[168, 0, 2, 15, 1]);
        let k = make_key(300, &[]);
        assert_eq!(&*k, &[300u16 as u8, 1]);
    }

    /// After training 10bb heads-up:
    /// - BB with AA facing an all-in must mostly call.
    /// - BB with 32o facing an all-in must mostly fold.
    /// - The button must not be folding AA.
    #[test]
    fn push_fold_convergence() {
        let t = push_fold_trainer();
        t.run(80_000, &|_| {});
        assert!(t.node_count() > 100);

        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        let junk = preflop_bucket([make_card(1, 0), make_card(0, 1)]); // 32o

        // Facing an all-in: history = [AllIn token]. Actions: [Fold, CheckCall].
        let shove_hist = [AbsAction::AllIn.token()];
        let aa_call = t.avg_strategy(aa, &shove_hist).expect("AA-vs-shove visited");
        assert_eq!(aa_call.len(), 2);
        assert!(
            aa_call[1] > 0.8,
            "AA should call a 10bb shove, got call prob {:.3}",
            aa_call[1]
        );
        let junk_call = t.avg_strategy(junk, &shove_hist).expect("32o-vs-shove visited");
        assert!(
            junk_call[0] > 0.6,
            "32o should fold to a 10bb shove, got fold prob {:.3}",
            junk_call[0]
        );

        // Button root with AA: actions [Fold, CheckCall, Bet, AllIn] (or without Bet
        // if deduped). Fold must be near zero.
        let btn_aa = t.avg_strategy(aa, &[]).expect("root AA visited");
        assert!(
            btn_aa[0] < 0.1,
            "button must not fold AA, got fold prob {:.3}",
            btn_aa[0]
        );
    }

    /// VR-MCCFR baselines are a control variate: they must be populated and
    /// finite after training, and must not move the fixed point. The
    /// push/fold checks that are stable across seeds at this budget (AA
    /// calls a shove, the button never folds AA) must still hold.
    #[test]
    fn vr_baselines_are_learned_and_unbiased() {
        let t = push_fold_trainer().with_vr(true);
        t.run(80_000, &|_| {});
        assert!(!t.baselines.is_empty(), "baselines never populated");
        assert!(t
            .baselines
            .iter()
            .all(|e| e.value().iter().all(|v| v.is_finite())));
        // Some baseline must have learned a non-zero value: the sampled
        // child values feeding the EMA are chip utilities.
        assert!(t.baselines.iter().any(|e| e.value().iter().any(|&v| v != 0.0)));

        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        let shove_hist = [AbsAction::AllIn.token()];
        let aa_call = t.avg_strategy(aa, &shove_hist).expect("AA-vs-shove visited");
        assert!(aa_call[1] > 0.8, "AA should call a shove, got {:.3}", aa_call[1]);
        let btn_aa = t.avg_strategy(aa, &[]).expect("root AA visited");
        assert!(btn_aa[0] < 0.1, "button must not fold AA, got {:.3}", btn_aa[0]);
    }

    /// Spine bookkeeping: a node is on the spine only while the history
    /// since the round start matches the recorded tokens and the recorded
    /// seat is the one to act; the hero flag follows the seat.
    #[test]
    fn spine_step_matches_history_and_seat() {
        let abs = Abstraction::new(AbsConfig::default());
        let cfg = HandConfig {
            num_players: 3,
            ..HandConfig::default()
        };
        let h0 = Hand::new(&cfg, 0, fresh_deck()); // p0 to act preflop
        let acts0 = abs.abstract_actions(&h0);
        let mut h1 = h0.clone();
        h1.apply(PlayerAction::CheckCall); // p1 to act
        let acts1 = abs.abstract_actions(&h1);
        let base = vec![7u8, 7, TOKEN_STREET_SEP];
        let bet = *acts1.iter().find(|a| matches!(a, AbsAction::Bet(_))).unwrap();
        let spine = Spine {
            hero: 1,
            base_len: base.len(),
            steps: vec![(0, PlayerAction::CheckCall), (1, PlayerAction::RaiseTo(333))],
            tokens: vec![AbsAction::CheckCall.token(), bet.token()],
        };
        let mut hist = base.clone();
        let (si, act, hero) = spine.step_at(&hist, &h0, &acts0).expect("root is on the spine");
        assert_eq!(acts0[si], AbsAction::CheckCall);
        assert_eq!(act, PlayerAction::CheckCall);
        assert!(!hero);
        hist.push(AbsAction::CheckCall.token());
        let (si, act, hero) = spine.step_at(&hist, &h1, &acts1).expect("second step");
        assert_eq!(acts1[si], bet);
        assert_eq!(act, PlayerAction::RaiseTo(333));
        assert!(hero, "seat 1 is the hero");
        // Wrong seat to act for the recorded step: off the spine.
        assert!(spine.step_at(&hist, &h0, &acts0).is_none());
        // Deviated history: off the spine.
        let mut dev = base.clone();
        dev.push(AbsAction::Fold.token());
        assert!(spine.step_at(&dev, &h1, &acts1).is_none());
        // Past the last step: off the spine (the current decision).
        hist.push(bet.token());
        let mut h2 = h1.clone();
        h2.apply(PlayerAction::RaiseTo(333));
        assert!(spine.step_at(&hist, &h2, &abs.abstract_actions(&h2)).is_none());
        // Shorter than the base: off the spine.
        assert!(spine.step_at(&base[..1], &h0, &acts0).is_none());
    }

    /// Pluribus pruning rules (per-traversal decision, river and terminal
    /// exemptions) must keep the push/fold fixed point under aggressive
    /// pruning.
    #[test]
    fn pluribus_prune_keeps_the_fixed_point() {
        let mut t = push_fold_trainer();
        t.cfg.prune_after = 1_000; // prune almost from the start
        t.cfg.prune_threshold = -1.0; // ...anything with negative regret
        t.cfg.pluribus_prune = true;
        t.run(60_000, &|_| {});
        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        let shove_hist = [AbsAction::AllIn.token()];
        let aa_call = t.avg_strategy(aa, &shove_hist).expect("AA-vs-shove visited");
        assert!(aa_call[1] > 0.8, "AA should call a shove, got {:.3}", aa_call[1]);
        let btn_aa = t.avg_strategy(aa, &[]).expect("root AA visited");
        assert!(btn_aa[0] < 0.1, "button must not fold AA, got {:.3}", btn_aa[0]);
    }

    /// Multiway-focused sampling is importance-weighted, so it must keep
    /// the push/fold fixed point (AA calls a shove, the button never
    /// folds AA) even at a strong focus weight.
    #[test]
    fn multiway_focus_keeps_the_fixed_point() {
        let mut t = push_fold_trainer();
        t.cfg.multiway_focus = 0.6;
        t.run(120_000, &|_| {}); // importance weights add variance: extra margin
        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        let shove_hist = [AbsAction::AllIn.token()];
        let aa_call = t.avg_strategy(aa, &shove_hist).expect("AA-vs-shove visited");
        assert!(aa_call[1] > 0.8, "AA should call a shove, got {:.3}", aa_call[1]);
        let btn_aa = t.avg_strategy(aa, &[]).expect("root AA visited");
        assert!(btn_aa[0] < 0.1, "button must not fold AA, got {:.3}", btn_aa[0]);
        // Junk must still fold to a shove: the focus changes which lines
        // are sampled, not what is learned on them.
        let junk = preflop_bucket([make_card(0, 0), make_card(5, 1)]);
        let junk_call = t.avg_strategy(junk, &shove_hist).expect("junk-vs-shove visited");
        assert!(junk_call[1] < 0.2, "72o must fold to a shove, got {:.3}", junk_call[1]);
    }

    /// Snapshot averaging: postflop strategies come only from snapshots
    /// (zero before the first one), preflop keeps the running average.
    #[test]
    fn snapshot_avg_fills_postflop_only_from_snapshots() {
        let mut t = push_fold_trainer();
        t.cfg.snapshot_avg = true;
        t.run(20_000, &|_| {});
        let mut postflop_before = 0;
        t.nodes.for_each(|key, _, strat| {
            postflop_before += usize::from(is_postflop(key) && strat.iter().any(|&x| x != 0.0));
        });
        assert_eq!(postflop_before, 0, "postflop strat accumulated without a snapshot");
        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        assert!(t.avg_strategy(aa, &[]).is_some(), "preflop running average missing");
        let n = t.snapshot_strategy();
        assert!(n > 0);
        let mut postflop_after = 0;
        t.nodes.for_each(|key, _, strat| {
            postflop_after += usize::from(is_postflop(key) && normalize(strat).is_some());
        });
        assert_eq!(postflop_after, n);
    }

    /// Subgame solving uses CFR+ (regret matching+): cumulative regrets are
    /// floored at zero, which converges faster on small trees.
    #[test]
    fn subgame_solver_keeps_regrets_nonnegative() {
        let t = push_fold_trainer().with_plus(true);
        let h = Hand::new(&t.cfg.hand, 0, fresh_deck());
        t.run_subgame(&h, &[], 5_000, 2_000, None);
        assert!(t.node_count() > 0);
        t.nodes.for_each(|_, regret, _| {
            assert!(regret.iter().all(|&r| r >= 0.0), "CFR+ regrets must be nonnegative, got {regret:?}");
        });
    }

    #[test]
    fn blueprint_and_checkpoint_roundtrip() {
        let t = push_fold_trainer();
        t.run(2_000, &|_| {});
        let n_before = t.node_count();
        let iters_before = t.iterations();
        assert!(n_before > 0);

        let dir = std::env::temp_dir();
        let ckpt = dir.join("pluribus_test_ckpt.bin");
        let bp_path = dir.join("pluribus_test_bp.bin");
        let ckpt_s = ckpt.to_str().unwrap();
        let bp_s = bp_path.to_str().unwrap();

        t.save_checkpoint(ckpt_s).unwrap();
        let t2 = Trainer::load_checkpoint(ckpt_s, t.cfg.clone()).unwrap();
        assert_eq!(t2.node_count(), n_before);
        assert_eq!(t2.iterations(), iters_before);
        // Abstraction restored from the checkpoint, not from caller flags.
        assert_eq!(t2.abs.cfg.postflop_buckets, 6);
        assert_eq!(t2.abs.centroids, t.abs.centroids);
        // Resumed trainer can keep training.
        t2.run(500, &|_| {});
        assert_eq!(t2.iterations(), iters_before + 500);

        let bp = t.to_blueprint();
        assert_eq!(t.save_blueprint(bp_s).unwrap(), bp.strategies.len());
        let bp2 = Blueprint::load(bp_s).unwrap();
        assert_eq!(bp.strategies, bp2.strategies);
        assert_eq!(blueprint_from_checkpoint(ckpt_s, 2, false, bp_s).unwrap(), bp.strategies.len());
        let bp3 = Blueprint::load(bp_s).unwrap();
        assert_eq!((&bp3.strategies, bp3.iterations, bp3.num_players), (&bp.strategies, iters_before, 2));
        assert_eq!(bp2.num_players, 2);
        assert_eq!(bp2.abs_cfg.postflop_buckets, 6);
        assert_eq!(bp2.centroids, t.abs.centroids);
        // Strategies are normalized distributions.
        for (_, s) in bp2.strategies.iter().take(50) {
            let sum: f32 = s.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "unnormalized strategy: {}", sum);
            assert!(s.iter().all(|&p| (0.0..=1.0).contains(&p)));
        }

        let _ = std::fs::remove_file(ckpt);
        let _ = std::fs::remove_file(bp_path);
    }

    #[test]
    fn a_strategy_discount_scales_every_sum_and_keeps_every_average() {
        let t = push_fold_trainer();
        t.run(2_000, &|_| {});
        let mut before = Vec::new();
        t.nodes.for_each(|key, _, strat| before.push((key.to_vec(), strat.to_vec())));
        t.discount_strategy(0.1);
        for (key, strat) in &before {
            let after = t.nodes.strat(key).unwrap();
            for (a, b) in after.iter().zip(strat) {
                assert!((a - 0.1 * b).abs() <= 1e-9 * b.abs().max(1.0), "{a} vs 0.1 x {b}");
            }
        }
    }

    #[test]
    fn rnr_traversals_against_the_model_build_the_traversers_average() {
        use crate::table::Baseline;
        let t = push_fold_trainer().with_rnr(Some(RnrCfg { model: Baseline::Caller, p: 1.0 }));
        t.run(5_000, &|_| {});
        let mut with_average = 0;
        t.nodes.for_each(|_, _, strat| with_average += usize::from(strat.iter().sum::<f64>() > 0.0));
        assert!(with_average > 0, "an all-model RNR run kept no average strategy");
        let aa = preflop_bucket([make_card(12, 0), make_card(12, 1)]);
        assert!(t.avg_strategy(aa, &[]).is_some(), "the root has no average after 5,000 model traversals");
    }

    #[test]
    fn parallel_training_smoke() {
        let t = push_fold_trainer();
        // Runs across rayon's default thread pool without panicking or deadlocking.
        t.run(10_000, &|_| {});
        assert_eq!(t.iterations(), 10_000);
        assert!(t.node_count() > 50);
    }
}
