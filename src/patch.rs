//! Training a blueprint's rare preflop histories from their own roots, to patch them into the blueprint.
//!
//! A history self-play reaches once in a million iterations keeps the strategy of a handful of noisy visits: on r3
//! the squeezes the live bot made with junk sat at 1 to 100 visits per hand and jammed up to 100% (rank correlation
//! of visits and jam share −0.82 over 82 live squeezes, 2026-11-28). [`train_root`] replays such a history on the
//! blueprint's tree, deals every player's hole from its reach along the history under the blueprint (the server's
//! range update), and runs the trainer from there: restricted Nash response against the Brasa model, no
//! negative-regret pruning, the rest of the hand past preflop frozen to the blueprint (`LeafCfg::frozen`). The
//! trained preflop infosets at and below the root replace the blueprint's (`StrategyMap::write_patched`).
use crate::abstraction::{preflop_bucket, Abstraction};
use crate::cfr::{Blueprint, LeafCfg, RnrCfg, TrainConfig, Trainer};
use crate::engine::{Hand, HandConfig, PlayerAction, Street};
use crate::search::{all_combos, RangeTracker};
use crate::table::{Baseline, RoundRoot, Table};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// The stack every seat holds in the blueprint's training, in chips.
pub const TRAINED_STACK: u32 = 10_000;

/// One action of a root's history, as the server's requests carry them.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RootAction {
    pub player: usize,
    pub action: String,
    #[serde(default)]
    pub raise_to: Option<u32>,
}

/// A preflop history to train from: a hand on the blueprint's seats (missing ones folded first), up to `hero`'s turn.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreflopRoot {
    pub name: String,
    pub stacks: Vec<u32>,
    pub button: usize,
    pub hero: usize,
    pub actions: Vec<RootAction>,
}

/// What training a root gave: its history on the tree, the trained infosets at and below it, and how long it took.
#[derive(Debug)]
pub struct RootResult {
    pub hist: Vec<u8>,
    pub iterations: u64,
    pub seconds: f64,
    pub strategies: Vec<(Vec<u8>, Vec<f32>)>,
}

fn concrete(a: &RootAction) -> PlayerAction {
    match a.action.as_str() {
        "fold" => PlayerAction::Fold,
        "check" | "call" => PlayerAction::CheckCall,
        _ => PlayerAction::RaiseTo(a.raise_to.unwrap_or(0)),
    }
}

/// The root's hand on the blueprint's tree (every seat at the trained stack) and its history, with every seat's range
/// narrowed by the blueprint along it (hard Bayes, as the server's tracker reads the blueprint).
pub fn replay(bp: &Blueprint, abs: &Abstraction, root: &PreflopRoot) -> Result<(Hand, Vec<u8>, RangeTracker), String> {
    let n = root.stacks.len();
    if n != bp.num_players {
        return Err(format!("{}: {n} seats, the blueprint plays {}", root.name, bp.num_players));
    }
    let cfg = HandConfig { num_players: n, stack: root.stacks.iter().copied().max().unwrap_or(TRAINED_STACK), sb: 50, bb: 100 };
    let deck: [u8; 52] = std::array::from_fn(|i| i as u8);
    let real = Hand::new_with_stacks(&cfg, root.button, deck, &root.stacks);
    let shadow = Hand::new(&HandConfig { stack: TRAINED_STACK, ..cfg }, root.button, deck);
    let mut t = Table { round: RoundRoot { real: real.clone(), hist_len: 0, actions: Vec::new() }, real, shadow, hist: Vec::new() };
    let mut tracker = RangeTracker::new(n);
    for a in &root.actions {
        if t.real.is_terminal() || t.real.to_act() != a.player {
            return Err(format!("{}: {} by player {} out of turn", root.name, a.action, a.player));
        }
        let c = concrete(a);
        let taken = t.map_concrete(c, abs);
        tracker.observe(a.player, taken, &t.shadow, &t.hist, bp, abs);
        t.apply_concrete(c, abs);
    }
    if t.shadow.street() != Street::Preflop || t.shadow.is_terminal() || t.shadow.to_act() != root.hero {
        return Err(format!("{}: the history does not end at the hero's preflop turn", root.name));
    }
    Ok((t.shadow, t.hist, tracker))
}

/// The trainer of a root: restricted Nash response against the Brasa model in a share `rnr_p` of the traversals, no
/// pruning, the hand past preflop frozen to the blueprint.
fn trainer_for(bp: &Arc<Blueprint>, abs: &Arc<Abstraction>, root: &Hand, hist: &[u8], seed: u64, rnr_p: f64) -> Trainer {
    let cfg = TrainConfig {
        hand: HandConfig { num_players: root.num_players(), stack: TRAINED_STACK, sb: 50, bb: 100 },
        prune_after: u64::MAX,
        seed,
        ..TrainConfig::default()
    };
    let leaf = LeafCfg { blueprint: Arc::clone(bp), abs: Arc::clone(abs), limit: Street::Preflop, root: Some((root.clone(), hist.len())), frozen: true };
    Trainer::new(Arc::clone(abs), cfg).with_leaf(Some(leaf)).with_rnr(Some(RnrCfg { model: Baseline::Brasa, p: rnr_p }))
}

/// Trains `root` for `iters` traversals (`seed` sets apart independent runs) under a `rake_bp` rake.
pub fn train_root(bp: &Arc<Blueprint>, abs: &Arc<Abstraction>, root: &PreflopRoot, iters: u64, seed: u64, rnr_p: f64, rake_bp: u32) -> Result<RootResult, String> {
    let (mut hand, hist, tracker) = replay(bp, abs, root)?;
    hand.set_rake(rake_bp);
    let trainer = trainer_for(bp, abs, &hand, &hist, seed, rnr_p);
    let sampler = |h: &mut Hand, rng: &mut SmallRng| {
        let holes = tracker.sample_holes(h, rng);
        h.resample_hidden_with(&holes, rng);
    };
    let t0 = std::time::Instant::now();
    trainer.run_subgame(&hand, &hist, u64::MAX, iters, Some(&sampler));
    let seconds = t0.elapsed().as_secs_f64();
    let strategies = trainer.to_blueprint().strategies.iter().filter(|(k, _)| k.len() >= 2 && k[2..].starts_with(&hist)).map(|(k, p)| (k, p.to_vec())).collect();
    Ok(RootResult { hist, iterations: trainer.iterations(), seconds, strategies })
}

/// The hero's EV at the root, per hand class bucket (`preflop_bucket`), in chips, playing `strategy` at its preflop
/// nodes against opponents on the Brasa model, the rest of the hand on the blueprint for the hero; holes dealt from
/// the replayed ranges, the hero's fixed to each class in turn. `strategy(bucket, hist)` gives the hero's probabilities
/// on the abstract menu, falling back to the blueprint when it has none.
pub fn model_ev(
    bp: &Blueprint,
    abs: &Abstraction,
    root: &PreflopRoot,
    rake_bp: u32,
    samples: usize,
    seed: u64,
    strategy: &(dyn Fn(u16, &[u8]) -> Option<Vec<f32>> + Sync),
) -> Result<Vec<(u16, f64, f64)>, String> {
    use rand::Rng;
    let (mut hand, hist, tracker) = replay(bp, abs, root)?;
    hand.set_rake(rake_bp);
    let hero = hand.to_act();
    let combos = all_combos();
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for c in &combos {
        let b = preflop_bucket(*c);
        if !seen.insert(b) {
            continue;
        }
        let members: Vec<[u8; 2]> = combos.iter().copied().filter(|x| preflop_bucket(*x) == b).collect();
        let mut rng = SmallRng::seed_from_u64(seed ^ u64::from(b).wrapping_mul(0x9E37_79B9));
        let (mut sum, mut sq) = (0.0, 0.0);
        for _ in 0..samples {
            let mut h = hand.clone();
            let mine = members[rng.random_range(0..members.len())];
            let mut holes = tracker.sample_holes(&h, &mut rng);
            holes[hero] = Some(mine);
            // redraw any opponent holding one of our cards
            for p in 0..h.num_players() {
                if p != hero && holes[p].is_some_and(|o| o.iter().any(|x| mine.contains(x))) {
                    holes[p] = None;
                }
            }
            h.resample_hidden_with(&holes, &mut rng);
            let mut hst = hist.clone();
            while !h.is_terminal() {
                let p = h.to_act();
                let acts = abs.abstract_actions(&h);
                let a = if p != hero {
                    let a = crate::brasa::action(&h, hero, abs, &mut rng);
                    if acts.contains(&a) { a } else { acts[0] }
                } else {
                    let bucket = abs.key_bucket(&h, h.hole(p), &mut rng);
                    let probs = if h.street() == Street::Preflop { strategy(bucket, &hst) } else { None }.or_else(|| bp.get(bucket, &hst).map(<[f32]>::to_vec));
                    match probs.filter(|s| s.len() == acts.len()) {
                        Some(s) => {
                            let total: f32 = s.iter().sum();
                            let mut x = rng.random::<f32>() * total;
                            let mut k = acts.len() - 1;
                            for (i, &q) in s.iter().enumerate() {
                                x -= q;
                                if x < 0.0 {
                                    k = i;
                                    break;
                                }
                            }
                            acts[k]
                        }
                        None => acts[rng.random_range(0..acts.len())],
                    }
                };
                let before = h.street();
                h.apply(abs.concrete(&h, a));
                hst.push(a.token());
                if !h.is_terminal() && h.street() != before {
                    hst.push(crate::abstraction::TOKEN_STREET_SEP);
                }
            }
            let u = h.utilities()[hero] as f64;
            sum += u;
            sq += u * u;
        }
        let n = samples as f64;
        let mean = sum / n;
        out.push((b, mean, ((sq / n - mean * mean) / n).max(0.0).sqrt()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::AbsConfig;
    use crate::cfr::StrategyMap;

    fn tiny() -> (Arc<Blueprint>, Arc<Abstraction>) {
        let cfg = AbsConfig { postflop_buckets: 4, equity_rollouts: 20, dist_runouts: 4, runout_rollouts: 8, cache_cap: 10_000, ..AbsConfig::default() };
        let bp = Blueprint { strategies: StrategyMap::default(), iterations: 0, num_players: 6, abs_cfg: cfg.clone(), centroids: None };
        (Arc::new(bp), Arc::new(Abstraction::new(cfg)))
    }

    /// An open and a call before us, six-handed.
    fn squeeze() -> PreflopRoot {
        let a = |player: usize, action: &str, raise_to: Option<u32>| RootAction { player, action: action.into(), raise_to };
        PreflopRoot { name: "squeeze".into(), stacks: vec![TRAINED_STACK; 6], button: 0, hero: 2, actions: vec![a(3, "raise", Some(250)), a(4, "call", None), a(5, "fold", None), a(0, "fold", None), a(1, "fold", None)] }
    }

    #[test]
    fn a_root_trains_its_own_preflop_infosets_and_a_seed_sets_a_run_apart() {
        let (bp, abs) = tiny();
        let root = squeeze();
        // one thread: parallel traversals update shared infosets in no set order
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("a pool");
        let train = |seed: u64| one.install(|| train_root(&bp, &abs, &root, 300, seed, 0.5, 500).expect("trains"));
        let (a, again, other) = (train(1), train(1), train(2));
        assert!(!a.strategies.is_empty() && a.iterations >= 300);
        // every trained infoset is at or below the root, and preflop: no street separator past the root
        assert!(a.strategies.iter().all(|(k, _)| k[2..].starts_with(&a.hist) && !k[2 + a.hist.len()..].contains(&crate::abstraction::TOKEN_STREET_SEP)));
        // our own decision at the root is among them
        assert!(a.strategies.iter().any(|(k, _)| &k[2..] == a.hist.as_slice()));
        let probs = |r: &RootResult| r.strategies.iter().find(|(k, _)| &k[2..] == r.hist.as_slice()).map(|(_, p)| p.clone());
        assert_eq!(probs(&a), probs(&again), "one seed on one thread, one result");
        assert_ne!(probs(&a), probs(&other), "another seed, another run");
    }

    #[test]
    fn a_history_that_leaves_the_hero_no_turn_is_refused() {
        let (bp, abs) = tiny();
        let mut root = squeeze();
        root.hero = 3;
        assert!(replay(&bp, &abs, &root).is_err());
    }
}
