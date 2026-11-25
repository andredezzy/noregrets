//! Brasa players as a fixed opponent model, for restricted Nash response training
//! (`train --rnr-model brasa`) and evaluation. Measured on the hands of three accounts at Brasa's
//! 0.05/0.10 tables (Nov 2026, about 21k preflop and 30k postflop actions of opponents): they
//! limp two thirds of their hands, call two thirds of small raises and fold most of large ones.
//! A player reads its hand strength with noise; the strongest share of the read raises, the next
//! share continues, at the rates Brasa players raise, call and fold by street, players in the
//! hand and size faced. The raise lands on the abstract size nearest to theirs.

use crate::abstraction::{preflop_bucket, AbsAction, Abstraction};
use crate::cards::{rank, suit, Card};
use crate::engine::{Hand, PlayerAction, Street};
use crate::eval::eval7;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// The rates and read noises a Brasa player acts by. Preflop, they are moment-matched in the simulator: a simulated
/// table's raise, call and fold shares by position and raise faced sit within a point of those of Brasa's players.
/// Postflop, they are fitted on Brasa's live hands read through the model (3,609 hands of account1, account2 and
/// account3 with an opponent past the flop, 2026-11-16 to 24): each opponent's possible holes are weighted by the
/// model's chance of the actions it took. Every context's thresholds (street, heads-up or multiway, unbet or the size
/// faced) are set so the model's chance of each action, given what each player did before in the hand, averages the
/// live share there; a context with under 30 live decisions takes its size's shares. The read's noise, power, share
/// of random reads and what a hand is ranked against are those under which the 4,149 holes the players showed are
/// likeliest. Brasa's players read their hands far more loosely than a pure strength order: they check and call with
/// strong hands and bet with nothing; facing a bet, a tenth to a fifth of them continue whatever they hold, and above
/// 0.8 pots the rest judge their hand against hands that make a pair. Their callers of bets above 3 pots still hold
/// two pairs or better more often than the model's (36% of 121 shown hands, against 26%), a gap no read here closes. Raise thresholds below zero let a strong hand
/// raise only when its read strays low. The trainer (`train --rnr-model brasa`) and the server's simulator play the
/// same model through [`intent`]; `BRASA_MODEL`, a JSON file of this shape, replaces the defaults in both.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    /// Unopened: (open, open or limp), as shares of hands.
    pub unopened: (f64, f64),
    /// Facing one raise, then two or more, by the raise's size (up to 3, 6, 12, 25, 50 big
    /// blinds and above): (re-raise, continue).
    pub facing_raise: [[(f64, f64); 6]; 2],
    /// The big blind's raise share when the pot comes to it unraised.
    pub bb_raise: f64,
    /// Postflop, unbet: the read thresholds of the bet sizes (s, m, l); heads-up then multiway, by street.
    pub unbet: [[[f64; 3]; 3]; 2],
    /// Postflop, facing a bet, by its size against the pot before it ([`size_bucket`]: up to 0.4,
    /// 0.8, 1.5, 3 and 6 pots, and above): the read thresholds (raise, continue); heads-up then multiway, by street.
    pub facing: [[[(f64, f64); 6]; 3]; 2],
    /// How far a read strays (0 strongest, 1 weakest), as the sd of a normal noise: preflop; postflop unbet; postflop
    /// facing each of the six sizes.
    pub preflop_noise: f64,
    pub unbet_noise: f64,
    pub facing_noise: [f64; 6],
    /// A postflop read is the hand's rank raised to this power before its noise: below 1, the strongest hands' reads
    /// spread apart and the middling ones' crowd together, as if a player knew a monster but not a medium hand.
    pub postflop_power: f64,
    /// The share of players who, facing a postflop bet of each of the six sizes, read their hand at random.
    pub facing_guess: [f64; 6],
    /// From this size faced on, a player ranks its hand against the hands that make a pair or better, the ones that
    /// bet that big, instead of against any hand.
    pub range_from: usize,
}

impl Default for Model {
    fn default() -> Self {
        Model {
            unopened: (0.13, 0.783),
            facing_raise: [
                [(0.083, 0.753), (0.068, 0.685), (0.059, 0.534), (0.01, 0.33), (0.01, 0.27), (0.0, 0.248)],
                [(0.21, 0.93), (0.032, 0.688), (0.012, 0.543), (0.011, 0.424), (0.001, 0.313), (0.001, 0.186)],
            ],
            bb_raise: 0.145,
            unbet: [
                [[0.103, 0.115, 0.171], [0.185, 0.152, 0.07], [0.156, 0.128, 0.153]],
                [[0.105, 0.081, 0.066], [0.124, 0.076, 0.056], [0.129, 0.081, 0.108]],
            ],
            facing: [
                [
                    [(0.194, 0.81), (0.163, 0.622), (0.172, 0.753), (0.218, 0.723), (-0.017, 0.644), (0.031, 0.526)],
                    [(0.155, 0.853), (0.055, 0.63), (0.151, 0.735), (-0.08, 0.769), (0.078, 0.721), (-0.04, 0.485)],
                    [(0.172, 0.752), (0.039, 0.61), (0.114, 0.705), (-0.096, 0.457), (0.106, 0.558), (-0.023, 0.439)],
                ],
                [
                    [(0.143, 0.681), (0.06, 0.537), (0.133, 0.72), (0.184, 0.691), (0.078, 0.584), (-0.022, 0.513)],
                    [(0.071, 0.724), (0.057, 0.522), (0.079, 0.659), (0.07, 0.727), (0.157, 0.65), (0.027, 0.526)],
                    [(0.142, 0.588), (0.039, 0.437), (0.106, 0.521), (0.102, 0.545), (-0.047, 0.436), (0.035, 0.362)],
                ],
            ],
            preflop_noise: 0.08,
            unbet_noise: 0.3,
            facing_noise: [0.2; 6],
            postflop_power: 0.7,
            facing_guess: [0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
            range_from: 2,
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

// WARNING: with every bet above 0.8 pots one size, a model player continued against a 17-pot shove
// as often as against a pot bet, and the search shoved 130 bb into 7.5 with top pair, weak kicker
// (account1, 2026-11-24 03:02Z); Brasa's players fold 53% of pot bets and 79% of bets above 6 pots.
// And with a strict read, a model player who checked held almost no strong hand, so a shove into
// checks looked safe: it still shoved 94 bb into 4 with top pair, weak kicker, into a four-way
// limped pot (account2, 2026-11-23), where real players who check still hold two pairs and better.
/// The upper bounds of the sizes a bet is faced at, in pots before it; a bet above the last is
/// the sixth size.
pub const FACING_POTS: [f64; 5] = [0.4, 0.8, 1.5, 3.0, 6.0];

/// A bet's size against the pot before it, as the index of the first [`FACING_POTS`] bound it is
/// within (5 above them all).
pub fn size_bucket(pots: f64) -> usize {
    FACING_POTS.iter().position(|&b| pots <= b).unwrap_or(FACING_POTS.len())
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

/// A hand category (`eval`'s: 0 high card to 8 straight flush) of 3 to 7 cards.
fn category(cards: &[Card]) -> u32 {
    if let Ok(seven) = <[Card; 7]>::try_from(cards) {
        return eval7(&seven) >> 20;
    }
    let mut counts = [0u8; 13];
    let mut suits = [0u8; 4];
    let mut mask = 0u16;
    for &c in cards {
        counts[usize::from(rank(c))] += 1;
        suits[usize::from(suit(c))] += 1;
        mask |= 1 << rank(c);
    }
    let flush = suits.iter().any(|&n| n >= 5);
    let straight = mask & 0b1_0000_0000_1111 == 0b1_0000_0000_1111 || (0..=8).any(|lo| (mask >> lo) & 0b11111 == 0b11111);
    let trips = counts.iter().filter(|&&n| n == 3).count();
    let pairs = counts.iter().filter(|&&n| n == 2).count();
    match () {
        _ if straight && flush => 8,
        _ if counts.contains(&4) => 7,
        _ if trips >= 1 && (pairs >= 1 || trips >= 2) => 6,
        _ if flush => 5,
        _ if straight => 4,
        _ if trips >= 1 => 3,
        _ if pairs >= 2 => 2,
        _ if pairs == 1 => 1,
        _ => 0,
    }
}

/// Whether `hole` makes a pair or better of its own on `board`: the board's own pairs, trips, straight or flush don't
/// count.
fn makes_a_pair(hole: [Card; 2], board: &[Card]) -> bool {
    let mut all = [0 as Card; 7];
    all[..board.len()].copy_from_slice(board);
    all[board.len()..board.len() + 2].copy_from_slice(&hole);
    let cat = category(&all[..board.len() + 2]);
    let board_cat = category(board);
    let pocket = rank(hole[0]) == rank(hole[1]);
    let hits = hole.iter().any(|&c| board.iter().any(|&b| rank(b) == rank(c)));
    match cat {
        0 => false,
        1 | 2 => pocket || hits || board_cat == 0,
        3 => pocket || hits,
        _ => !(board.len() == 5 && board_cat >= 4 && cat == board_cat),
    }
}

/// Random holes drawn for one that makes a pair before [`postflop_rank_vs_pairs`] gives up on the board.
const PAIR_DRAWS: usize = 400;

/// [`postflop_rank`] against a hand that makes a pair or better on the board instead of any hand; against any hand
/// on a board where hardly any does (a straight or flush on the board that few holes beat with a better category).
pub fn postflop_rank_vs_pairs(hole: [Card; 2], board: &[Card], rng: &mut SmallRng) -> f64 {
    let mut used = [false; 52];
    for &c in hole.iter().chain(board) {
        used[c as usize] = true;
    }
    let live: Vec<Card> = (0..52u8).filter(|&c| !used[c as usize]).collect();
    let mut won = 0.0;
    let mut pick = [0 as Card; 7];
    for _ in 0..STRENGTH_RUNOUTS {
        let drawn = (0..PAIR_DRAWS).find_map(|_| {
            let (a, b) = (live[rng.random_range(0..live.len())], live[rng.random_range(0..live.len())]);
            (a != b && makes_a_pair([a, b], board)).then_some([a, b])
        });
        let Some(opp) = drawn else {
            return postflop_rank(hole, board, rng);
        };
        let mut full = [0 as Card; 5];
        full[..board.len()].copy_from_slice(board);
        let mut n = board.len();
        while n < 5 {
            let c = live[rng.random_range(0..live.len())];
            if !opp.contains(&c) && !full[board.len()..n].contains(&c) {
                full[n] = c;
                n += 1;
            }
        }
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
    1.0 - won / STRENGTH_RUNOUTS as f64
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
    intent_by(model(), h, rng)
}

/// [`intent`] by the rates of `m`: a player who acts unlike the model in play (the simulator's
/// opponents, to test what the model's errors cost).
pub fn intent_by(m: &Model, h: &Hand, rng: &mut SmallRng) -> Intent {
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
    let pot = f64::from(h.pot());
    if h.to_call() == 0 {
        let r = read(postflop_rank(h.hole(p), h.board(), rng).max(0.0).powf(m.postflop_power), m.unbet_noise, rng);
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
    let size = size_bucket(bet / (pot - bet).max(1.0));
    let (raise, cont) = m.facing[table][st][size];
    let r = if rng.random::<f64>() < m.facing_guess[size] {
        rng.random::<f64>()
    } else {
        let strength = if size >= m.range_from { postflop_rank_vs_pairs } else { postflop_rank };
        read(strength(h.hole(p), h.board(), rng).max(0.0).powf(m.postflop_power), m.facing_noise[size], rng)
    };
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

    /// A heads-up flop K-9-2 where the big blind (seat 1) bets `bet` into the 200-chip pot and the
    /// button (seat 0, 9-8: middle pair) faces it, 1000 big blinds deep.
    fn facing_flop_bet(bet: u32) -> Hand {
        facing_flop_bet_with("9h 8h", bet)
    }

    /// [`facing_flop_bet`] with the button holding `hole`.
    fn facing_flop_bet_with(hole: &str, bet: u32) -> Hand {
        let cards = parse_cards(&format!("{hole} Ad Qd Kc 9c 2d")).expect("cards parse");
        let mut deck: Vec<u8> = (0..52u8).filter(|c| !cards.contains(c)).collect();
        for (i, &c) in cards.iter().enumerate() {
            deck.insert(i, c);
        }
        let deck: [u8; 52] = deck.try_into().expect("52 cards");
        let cfg = HandConfig { num_players: 2, stack: 100_000, ..HandConfig::default() };
        let mut h = Hand::new(&cfg, 0, deck);
        h.apply(PlayerAction::CheckCall);
        h.apply(PlayerAction::CheckCall);
        assert_eq!((h.street(), h.to_act(), h.pot()), (Street::Flop, 1, 200));
        h.apply(PlayerAction::RaiseTo(bet));
        h
    }

    fn fold_share(h: &Hand, n: usize) -> f64 {
        let mut rng = SmallRng::seed_from_u64(11);
        let folds = (0..n).filter(|_| intent(h, &mut rng) == Intent::Fold).count();
        folds as f64 / n as f64
    }

    #[test]
    fn a_middle_pair_folds_a_ten_pot_shove_more_often_than_a_pot_bet() {
        let pot_bet = fold_share(&facing_flop_bet(200), 2000);
        let shove = fold_share(&facing_flop_bet(2000), 2000);
        assert!(shove > pot_bet + 0.1, "folds a pot bet {pot_bet}, a ten-pot shove {shove}");
    }

    fn continue_share(m: &Model, h: &Hand, n: usize) -> f64 {
        let mut rng = SmallRng::seed_from_u64(13);
        let goes_on = (0..n).filter(|_| intent_by(m, h, &mut rng) != Intent::Fold).count();
        goes_on as f64 / n as f64
    }

    #[test]
    fn each_size_faced_is_read_with_its_own_noise() {
        // 7-3 offsuit on K-9-2: no pair and no draw.
        let quiet = Model { facing_noise: [0.0; 6], facing_guess: [0.0; 6], ..Model::default() };
        let loose = Model { facing_noise: [0.0, 0.0, 0.0, 0.6, 0.0, 0.0], ..quiet.clone() };
        let half_pot = facing_flop_bet_with("7s 3h", 100);
        let two_pots = facing_flop_bet_with("7s 3h", 400);
        assert_eq!(continue_share(&loose, &half_pot, 2000), continue_share(&quiet, &half_pot, 2000));
        let (q, l) = (continue_share(&quiet, &two_pots, 2000), continue_share(&loose, &two_pots, 2000));
        assert!(l > q + 0.1, "continues two pots read quietly {q}, loosely {l}");
    }

    #[test]
    fn a_share_of_players_facing_a_bet_continue_whatever_they_hold() {
        let quiet = Model { facing_noise: [0.0; 6], facing_guess: [0.0; 6], ..Model::default() };
        let guessing = Model { facing_guess: [0.5; 6], ..quiet.clone() };
        let shove = facing_flop_bet_with("7s 3h", 2000);
        let (q, g) = (continue_share(&quiet, &shove, 2000), continue_share(&guessing, &shove, 2000));
        assert!(q < 0.02 && g > 0.1, "continues a ten-pot shove with air read strictly {q}, half at random {g}");
    }

    #[test]
    fn top_pair_weak_kicker_ranks_lower_against_hands_that_make_a_pair() {
        let cards = parse_cards("Ah 3s As Js 2d").expect("cards parse");
        let (hole, board) = ([cards[0], cards[1]], &cards[2..]);
        let mut rng = SmallRng::seed_from_u64(17);
        let mean = |f: fn([Card; 2], &[Card], &mut SmallRng) -> f64, rng: &mut SmallRng| (0..400).map(|_| f(hole, board, rng)).sum::<f64>() / 400.0;
        let (any, pairs) = (mean(postflop_rank, &mut rng), mean(postflop_rank_vs_pairs, &mut rng));
        assert!(any < 0.2 && pairs > any + 0.1, "A3 on A-J-2 ranks {any} against any hand, {pairs} against a pair");
        let set = parse_cards("2h 2s").expect("cards parse");
        assert!(makes_a_pair([set[0], set[1]], board) && !makes_a_pair([cards[1], parse_cards("Kd").expect("card")[0]], board));
    }

    // WARNING: on 5-6-7-8-9 offsuit no hole makes a pair of its own, and drawing until one did never
    // ended: a 20,000-hand simulation hung at hand 13,800 (2026-11-24 22:52).
    #[test]
    fn a_board_no_hole_pairs_ranks_against_any_hand() {
        let cards = parse_cards("Ah Kd 5s 6h 7d 8c 9s").expect("cards parse");
        let mut rng = SmallRng::seed_from_u64(19);
        let r = postflop_rank_vs_pairs([cards[0], cards[1]], &cards[2..], &mut rng);
        assert!((0.0..=1.0).contains(&r));
    }

    #[test]
    fn bets_are_faced_at_six_sizes_by_their_share_of_the_pot() {
        let sizes: Vec<usize> = [0.3, 0.5, 1.0, 2.0, 4.0, 17.0].iter().map(|&p| size_bucket(p)).collect();
        assert_eq!(sizes, [0, 1, 2, 3, 4, 5]);
        assert_eq!((size_bucket(0.8), size_bucket(1.5), size_bucket(6.0)), (1, 2, 4));
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
