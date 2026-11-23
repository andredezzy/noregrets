//! Brasa players as a fixed opponent model, for restricted Nash response training
//! (`train --rnr-model brasa`) and evaluation. Measured on the hands of two accounts at Brasa's
//! 0.05/0.10 tables (Nov 2026, about 16k preflop and 10k postflop actions of opponents): they
//! limp two thirds of their hands, call two thirds of small raises and fold most of large ones.
//! A player reads its hand strength with noise; the strongest share of the read raises, the next
//! share continues, at the rates Brasa players raise, call and fold by street, players in the
//! hand and size faced. The raise lands on the abstract size nearest to theirs.

use crate::abstraction::{preflop_bucket, AbsAction, Abstraction};
use crate::cards::Card;
use crate::engine::{Hand, PlayerAction, Street};
use crate::eval::eval7;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// The rates and read noises a Brasa player acts by. The defaults are calibrated: moment-matched
/// (four rounds, 20,000 simulated hands each) so a simulated table's raise, call, fold and bet-size
/// shares, by street, players in the hand and size faced, sit 0.76 points on average from those of
/// Brasa's players over 19,000 opponent actions in account1's and account2's hands (2026-11-16 to
/// 23); the rates measured as shares of all hands sat 5.0 points off. The trainer (`train --rnr-model brasa`) and
/// the server's simulator play the same model through [`intent`]; `BRASA_MODEL`, a JSON file of
/// this shape, replaces the defaults in both while it is calibrated against live hands.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    /// Unopened: (open, open or limp), as shares of hands.
    pub unopened: (f64, f64),
    /// Facing one raise, then two or more, by the raise's size (up to 3, 6, 12, 25, 50 big
    /// blinds and above): (re-raise, continue).
    pub facing_raise: [[(f64, f64); 6]; 2],
    /// The big blind's raise share when the pot comes to it unraised.
    pub bb_raise: f64,
    /// Postflop, unbet: bet shares by size (s, m, l); heads-up then multiway, by street.
    pub unbet: [[[f64; 3]; 3]; 2],
    /// Postflop, facing a bet of size s, m, l: (raise, continue); heads-up then multiway, by street.
    pub facing: [[[(f64, f64); 3]; 3]; 2],
    /// How far a player's read of its strength strays (0 strongest, 1 weakest).
    pub preflop_noise: f64,
    pub postflop_noise: f64,
}

impl Default for Model {
    fn default() -> Self {
        Model {
            unopened: (0.132, 0.784),
            facing_raise: [
                [(0.089, 0.75), (0.069, 0.685), (0.057, 0.538), (0.01, 0.33), (0.01, 0.27), (0.0, 0.26)],
                [(0.21, 0.93), (0.041, 0.69), (0.018, 0.546), (0.016, 0.428), (0.003, 0.315), (0.001, 0.192)],
            ],
            bb_raise: 0.144,
            unbet: [
                [[0.087, 0.101, 0.148], [0.138, 0.112, 0.058], [0.122, 0.089, 0.111]],
                [[0.105, 0.082, 0.064], [0.117, 0.072, 0.054], [0.128, 0.076, 0.11]],
            ],
            facing: [
                [
                    [(0.195, 0.716), (0.094, 0.471), (0.058, 0.47)],
                    [(0.118, 0.713), (0.022, 0.487), (0.02, 0.423)],
                    [(0.121, 0.634), (0.015, 0.442), (0.017, 0.344)],
                ],
                [
                    [(0.125, 0.575), (0.041, 0.439), (0.029, 0.412)],
                    [(0.058, 0.625), (0.033, 0.435), (0.012, 0.355)],
                    [(0.106, 0.496), (0.03, 0.369), (0.021, 0.265)],
                ],
            ],
            preflop_noise: 0.08,
            postflop_noise: 0.12,
        }
    }
}

/// The model in play: `BRASA_MODEL`'s, else the defaults.
pub fn model() -> &'static Model {
    static MODEL: OnceLock<Model> = OnceLock::new();
    MODEL.get_or_init(|| match std::env::var("BRASA_MODEL") {
        Ok(path) => {
            let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("BRASA_MODEL {path}: {e}"));
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("BRASA_MODEL {path}: {e}"))
        }
        Err(_) => Model::default(),
    })
}

pub const RAISE_SIZES_BB: [u32; 5] = [3, 6, 12, 25, 50];

/// The pot fractions their bets of size s, m and l stand for, and their opens in big blinds.
pub const BET_POTS: [f64; 3] = [0.33, 0.6, 1.0];
pub const OPEN_BB: [u32; 5] = [2, 3, 4, 5, 7];

const STRENGTH_RUNOUTS: usize = 16;

/// A bet's size against the pot before it: 0 up to 0.4 pots, 1 up to 0.8, 2 above.
pub fn size_bucket(pots: f64) -> usize {
    if pots <= 0.4 {
        0
    } else if pots <= 0.8 {
        1
    } else {
        2
    }
}

/// Share of preflop hands stronger than `hole` (by equity against one random hand).
pub fn preflop_rank(hole: [Card; 2]) -> f64 {
    static RANKS: OnceLock<Vec<f64>> = OnceLock::new();
    let ranks = RANKS.get_or_init(|| {
        let mut rng = SmallRng::seed_from_u64(0xB5A5_A000);
        let mut equity = vec![None; 169];
        let mut count = vec![0usize; 169];
        for a in 0..52u8 {
            for b in (a + 1)..52u8 {
                let class = preflop_bucket([a, b]) as usize;
                count[class] += 1;
                if equity[class].is_none() {
                    equity[class] = Some(equity_vs_random([a, b], &[], 4000, &mut rng));
                }
            }
        }
        let mut classes: Vec<usize> = (0..169).filter(|&c| count[c] > 0).collect();
        classes.sort_by(|&x, &y| equity[y].unwrap_or(0.0).total_cmp(&equity[x].unwrap_or(0.0)));
        let total: usize = count.iter().sum();
        let mut rank = vec![0.0; 169];
        let mut seen = 0usize;
        for c in classes {
            rank[c] = (seen as f64 + count[c] as f64 / 2.0) / total as f64;
            seen += count[c];
        }
        rank
    });
    ranks[preflop_bucket(hole) as usize]
}

/// Equity of `hole` on `board` against one random hand, over `runouts` random completions.
fn equity_vs_random(hole: [Card; 2], board: &[Card], runouts: usize, rng: &mut SmallRng) -> f64 {
    let mut used = [false; 52];
    for &c in hole.iter().chain(board) {
        used[c as usize] = true;
    }
    let live: Vec<Card> = (0..52u8).filter(|&c| !used[c as usize]).collect();
    let mut won = 0.0;
    let mut pick = [0 as Card; 7];
    for _ in 0..runouts {
        let need = 2 + 5 - board.len();
        let mut drawn: Vec<Card> = Vec::with_capacity(need);
        while drawn.len() < need {
            let c = live[rng.random_range(0..live.len())];
            if !drawn.contains(&c) {
                drawn.push(c);
            }
        }
        let opp = [drawn[0], drawn[1]];
        let mut full = [0 as Card; 5];
        full[..board.len()].copy_from_slice(board);
        full[board.len()..].copy_from_slice(&drawn[2..]);
        pick[..5].copy_from_slice(&full);
        pick[5..].copy_from_slice(&hole);
        let ours = eval7(&pick);
        pick[5..].copy_from_slice(&opp);
        let theirs = eval7(&pick);
        won += match ours.cmp(&theirs) {
            std::cmp::Ordering::Greater => 1.0,
            std::cmp::Ordering::Equal => 0.5,
            std::cmp::Ordering::Less => 0.0,
        };
    }
    won / runouts as f64
}

/// A noisy read of `rank` (0 strongest, 1 weakest).
pub fn read(rank: f64, noise: f64, rng: &mut SmallRng) -> f64 {
    let (u, v): (f64, f64) = (rng.random::<f64>().max(1e-12), rng.random());
    rank + noise * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// A postflop hand's strength as a rank: 1 minus its equity against one random hand, over a few
/// random completions.
pub fn postflop_rank(hole: [Card; 2], board: &[Card], rng: &mut SmallRng) -> f64 {
    1.0 - equity_vs_random(hole, board, STRENGTH_RUNOUTS, rng)
}

/// What a Brasa player at `h` (its seat `h.to_act()`) means to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    /// Fold facing a bet, check when it can.
    Fold,
    CheckCall,
    /// Raise or bet to this street total, in chips (`u32::MAX` all in).
    RaiseTo(u32),
}

/// The decision of a Brasa player at `h`: the strongest share of its noisy read raises, the next
/// share continues, at the model's rates for the street, the players in the hand and the size
/// faced.
pub fn intent(h: &Hand, rng: &mut SmallRng) -> Intent {
    let m = model();
    let p = h.to_act();
    let bb = h.big_blind();
    if h.street() == Street::Preflop {
        let r = read(preflop_rank(h.hole(p)), m.preflop_noise, rng);
        if h.can_check() {
            return if r < m.bb_raise {
                Intent::RaiseTo(OPEN_BB[rng.random_range(0..OPEN_BB.len())] * bb + h.current_bet())
            } else {
                Intent::CheckCall
            };
        }
        let raises = usize::from(h.n_raises());
        let (raise, cont) = if raises == 0 {
            m.unopened
        } else {
            let size = h.current_bet() / bb;
            let z = RAISE_SIZES_BB.iter().position(|&b| size < b).unwrap_or(RAISE_SIZES_BB.len());
            m.facing_raise[(raises - 1).min(1)][z]
        };
        return if r >= cont {
            Intent::Fold
        } else if r >= raise {
            Intent::CheckCall
        } else if raises == 0 {
            Intent::RaiseTo(OPEN_BB[rng.random_range(0..OPEN_BB.len())] * bb)
        } else if raises == 1 {
            Intent::RaiseTo(h.current_bet() * [20, 25, 30, 35][rng.random_range(0..4)] / 10)
        } else {
            Intent::RaiseTo(u32::MAX)
        };
    }
    let table = usize::from(h.live_count() > 2);
    let st = match h.street() {
        Street::Flop => 0,
        Street::Turn => 1,
        _ => 2,
    };
    let r = read(postflop_rank(h.hole(p), h.board(), rng), m.postflop_noise, rng);
    let pot = f64::from(h.pot());
    if h.to_call() == 0 {
        let rates = m.unbet[table][st];
        if r >= rates.iter().sum::<f64>() {
            return Intent::CheckCall;
        }
        let z = if r < rates[2] {
            2
        } else if r < rates[2] + rates[1] {
            1
        } else {
            0
        };
        return Intent::RaiseTo((BET_POTS[z] * pot).round() as u32);
    }
    let bet = f64::from(h.to_call());
    let (raise, cont) = m.facing[table][st][size_bucket(bet / (pot - bet).max(1.0))];
    if r >= cont {
        Intent::Fold
    } else if r >= raise {
        Intent::CheckCall
    } else {
        Intent::RaiseTo(h.current_bet() * 3)
    }
}

/// The abstract raise on `acts` whose concrete total lands nearest `target`, or check/call when
/// the menu has none.
fn nearest_raise(h: &Hand, acts: &[AbsAction], abs: &Abstraction, target: u32) -> AbsAction {
    acts.iter()
        .filter_map(|&a| match abs.concrete(h, a) {
            PlayerAction::RaiseTo(to) => Some((a, to.abs_diff(target))),
            _ => None,
        })
        .min_by_key(|&(_, d)| d)
        .map_or(AbsAction::CheckCall, |(a, _)| a)
}

fn fold_or_check(h: &Hand, acts: &[AbsAction]) -> AbsAction {
    if h.can_check() || !acts.contains(&AbsAction::Fold) {
        AbsAction::CheckCall
    } else {
        AbsAction::Fold
    }
}

/// The action a Brasa player takes at `h` on the abstraction's menu: its [`intent`], a raise
/// landing on the abstract size nearest to its own.
pub fn action(h: &Hand, abs: &Abstraction, rng: &mut SmallRng) -> AbsAction {
    let acts = abs.abstract_actions(h);
    match intent(h, rng) {
        Intent::Fold => fold_or_check(h, &acts),
        Intent::CheckCall => AbsAction::CheckCall,
        Intent::RaiseTo(target) => nearest_raise(h, &acts, abs, target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::AbsConfig;
    use crate::cards::parse_cards;
    use crate::engine::HandConfig;

    fn abs() -> Abstraction {
        Abstraction::new(AbsConfig {
            postflop_buckets: 6,
            equity_rollouts: 50,
            cache_cap: 100_000,
            ..AbsConfig::default()
        })
    }

    /// A six-max deal where seat 3 (first to act preflop) holds `hole`.
    fn deal(hole: &str) -> Hand {
        let h = parse_cards(hole).expect("hole parses");
        let mut deck: Vec<u8> = (0..52u8).filter(|c| !h.contains(c)).collect();
        deck.insert(6, h[0]);
        deck.insert(7, h[1]);
        let deck: [u8; 52] = deck.try_into().expect("52 cards");
        Hand::new(&HandConfig::default(), 0, deck)
    }

    fn shares(h: &Hand, n: usize) -> (f64, f64, f64) {
        let abs = abs();
        let mut rng = SmallRng::seed_from_u64(7);
        let (mut fold, mut call, mut raise) = (0, 0, 0);
        for _ in 0..n {
            match action(h, &abs, &mut rng) {
                AbsAction::Fold => fold += 1,
                AbsAction::CheckCall => call += 1,
                _ => raise += 1,
            }
        }
        let n = n as f64;
        (f64::from(fold) / n, f64::from(call) / n, f64::from(raise) / n)
    }

    #[test]
    fn aces_raise_and_seven_deuce_folds_unopened_as_brasa_players_do() {
        let aces = deal("As Ah");
        assert_eq!(aces.to_act(), 3);
        let (_, _, raise) = shares(&aces, 400);
        assert!(raise > 0.8, "aces raise {raise}");
        let (fold, call, _) = shares(&deal("7d 2c"), 400);
        assert!(fold > 0.6, "72o folds {fold}");
        assert!(call < 0.4);
    }

    #[test]
    fn a_middling_hand_limps_a_small_raise_and_folds_a_huge_one() {
        let mut small = deal("Ts 8d");
        small.apply(PlayerAction::RaiseTo(250));
        for _ in 0..2 {
            small.apply(PlayerAction::Fold);
        }
        let (_, call, _) = shares(&small, 400);
        let mut huge = deal("Ts 8d");
        huge.apply(PlayerAction::RaiseTo(6000));
        for _ in 0..2 {
            huge.apply(PlayerAction::Fold);
        }
        let (fold, _, _) = shares(&huge, 400);
        assert!(call > 0.5, "calls a 2.5 bb raise {call}");
        assert!(fold > 0.8, "folds a 60 bb raise {fold}");
    }
}
