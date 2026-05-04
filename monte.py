#!/usr/bin/env python3
"""
By: Miller Nguyen.
Monte Carlo Weighted Lottery Simulation  —  Aggressive Multi-Strategy Edition
==============================================================================
GPU-accelerated (CuPy / CUDA) with NumPy fallback, chunked to never crash.

Sets produced every run
------------------------
  A  HOT          top-N most-frequent balls in Gumbel simulation
  B  COLD         top-N least-frequent balls (contrarian)
  C  MIXED        ½ hot + ½ cold
  D  RECENCY      Gumbel sim with exponential recency decay (~1-yr half-life)
  E  CO-OCCUR     greedy clique: balls that historically appear together most
  F  GENETIC ALG  evolves a combo maximising recency score + pairwise co-freq
  G  CONSENSUS    balls hot across last-100 / last-300 / last-500 / all-time
  ★  VOTE CHAMP   balls that appear in the most strategies above
  H  LEAST VOTE   balls that appear in the fewest strategies (opposite of ★)
  I  MIX 4×3      4 least-vote balls  +  3 vote-champion balls

Usage
-----
  python monte.py                        # 1 M sims
  python monte.py -n 100_000_000         # 100 M sims
  python monte.py -n 1_000_000_000       # 1 B sims (GPU)
  python monte.py -n 5000000 --seed 42   # reproducible
  python monte.py --cpu                  # force CPU
  python monte.py --no-save              # skip disk write
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ── GPU / CPU backend ──────────────────────────────────────────────────────────
try:
    import cupy as cp
    _GPU_AVAILABLE = True
except ImportError:
    cp = None
    _GPU_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_CSV         = "ToTo.csv"
NUM_BALLS           = 49
PICK                = 7           # 6 main + 1 additional  (ToTo format)
DEFAULT_SIMULATIONS = 100_000_000
RESULTS_DIR         = "results"
# ───────────────────────────────────────────────────────────────────────────────


# =============================================================================
# System info & auto-sizing
# =============================================================================

def _get_memory_info():
    gpu_free = None
    if _GPU_AVAILABLE:
        try:
            gpu_free = cp.cuda.runtime.memGetInfo()[0]
        except Exception:
            pass
    try:
        import psutil
        cpu_free = psutil.virtual_memory().available
    except ImportError:
        cpu_free = 8 * 1024 ** 3
    return gpu_free, cpu_free


def auto_chunk_size(n_balls: int = NUM_BALLS) -> int:
    gpu_free, cpu_free = _get_memory_info()
    safe = int(gpu_free * 0.60) if gpu_free is not None else int(cpu_free * 0.40)
    chunk = max(100_000, safe // (n_balls * 4 * 4))
    return min(chunk, 20_000_000)


def max_recommended_sims() -> int:
    gpu_free, _ = _get_memory_info()
    return 1_000_000_000 if gpu_free is not None else 200_000_000


def print_system_info():
    sep = "-" * 62
    print(f"\n{sep}")
    print("  Hardware Summary")
    print(sep)
    try:
        import multiprocessing
        print(f"  CPU cores        : {multiprocessing.cpu_count()}")
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  RAM total        : {vm.total  / 1024**3:.1f} GB")
        print(f"  RAM available    : {vm.available / 1024**3:.1f} GB")
    except ImportError:
        print("  RAM              : (install psutil for details)")
    if _GPU_AVAILABLE:
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            ver   = cp.cuda.runtime.runtimeGetVersion()
            major = ver // 1000
            minor = (ver % 1000) // 10
            print(f"  GPU              : NVIDIA (CUDA {major}.{minor})")
            print(f"  VRAM total       : {total / 1024**3:.1f} GB")
            print(f"  VRAM free        : {free  / 1024**3:.1f} GB")
        except Exception:
            print("  GPU              : CuPy available")
    else:
        print("  GPU              : Not available -- using CPU")
    chunk = auto_chunk_size()
    max_s = max_recommended_sims()
    tag   = "GPU" if _GPU_AVAILABLE else "CPU"
    print(f"\n  Auto chunk size  : {chunk:>12,}  sims")
    print(f"  Max recommended  : {max_s:>12,}  sims  ({tag}, time-bound)")
    print(sep)


# =============================================================================
# Data loading
# =============================================================================

def load_historical_numbers(csv_path: str) -> np.ndarray:
    """Flat int array of every number ever drawn (1-indexed)."""
    df       = pd.read_csv(csv_path)
    win_cols = ["Winning Number 1", "2", "3", "4", "5", "6", "Additional Number"]
    raw      = df[win_cols].values.flatten().astype(float)
    valid    = raw[~np.isnan(raw)].astype(int)
    return valid[(valid >= 1) & (valid <= NUM_BALLS)]


def load_draws_structured(csv_path: str):
    """
    Returns (draws, days_ago).
      draws    : int array (N, 7), 0-indexed ball numbers, newest draw first.
      days_ago : float array (N,), days before today each draw was held.
    """
    df        = pd.read_csv(csv_path)
    win_cols  = ["Winning Number 1", "2", "3", "4", "5", "6", "Additional Number"]
    draws_raw = df[win_cols].values.astype(float)
    mask      = ~np.isnan(draws_raw).any(axis=1)
    draws     = np.clip(draws_raw[mask].astype(int), 1, NUM_BALLS) - 1  # 0-indexed
    today     = pd.Timestamp.now()
    dates     = pd.to_datetime(df["Date"].values[mask])
    days_ago  = np.maximum(0.0, np.array([(today - d).days for d in dates],
                                          dtype=np.float64))
    return draws, days_ago


# =============================================================================
# Weight computation  (flat frequency)
# =============================================================================

def compute_weights(numbers: np.ndarray, num_balls: int = NUM_BALLS) -> np.ndarray:
    counts = np.zeros(num_balls, dtype=np.float64)
    np.add.at(counts, numbers - 1, 1.0)
    counts += 1.0       # Laplace smoothing
    return counts / counts.sum()


# =============================================================================
# Gumbel-max simulation  (GPU / CPU, chunked)
# =============================================================================

def _chunk_gpu(log_w_gpu, chunk_size: int, pick: int, n_balls: int) -> np.ndarray:
    U      = cp.random.uniform(1e-6, 1.0 - 1e-6, size=(chunk_size, n_balls), dtype=cp.float32)
    gumbel = -cp.log(-cp.log(U));  del U
    keys   = log_w_gpu + gumbel;   del gumbel
    top    = cp.argpartition(keys, -pick, axis=1)[:, -pick:];  del keys
    flat   = top.ravel().astype(cp.int32);                     del top
    counts = cp.bincount(flat, minlength=n_balls)
    return cp.asnumpy(counts)[:n_balls].astype(np.int64)


def _chunk_cpu(log_w, chunk_size, pick, n_balls, rng):
    U      = rng.random(size=(chunk_size, n_balls), dtype=np.float32)
    np.clip(U, 1e-6, 1.0 - 1e-6, out=U)
    gumbel = -np.log(-np.log(U));  del U
    keys   = log_w + gumbel;       del gumbel
    top    = np.argpartition(keys, -pick, axis=1)[:, -pick:];  del keys
    flat   = top.ravel().astype(np.int32);                     del top
    return np.bincount(flat, minlength=n_balls)[:n_balls].astype(np.int64)


def run_simulations(weights, n_sims, pick=PICK, chunk_size=None, seed=None,
                    label="Simulating") -> np.ndarray:
    """Chunked Gumbel-max; returns accumulated count array (NUM_BALLS,)."""
    n_balls   = len(weights)
    chunk_size = chunk_size or auto_chunk_size(n_balls)
    log_w     = np.log(weights).astype(np.float32)
    log_w_gpu = cp.asarray(log_w[np.newaxis, :]) if _GPU_AVAILABLE else None
    rng       = np.random.default_rng(seed)
    if _GPU_AVAILABLE and seed is not None:
        cp.random.seed(seed)

    accumulated = np.zeros(n_balls, dtype=np.int64)
    done  = 0
    t0    = time.perf_counter()
    BAR_W = 28

    while done < n_sims:
        this = min(chunk_size, n_sims - done)
        accumulated += (_chunk_gpu(log_w_gpu, this, pick, n_balls)
                        if _GPU_AVAILABLE
                        else _chunk_cpu(log_w, this, pick, n_balls, rng))
        done += this

        elapsed = time.perf_counter() - t0
        pct     = done / n_sims * 100
        rate    = done / elapsed if elapsed > 1e-9 else 0.0
        remain  = (n_sims - done) / rate if rate > 0 else 0.0
        eta     = f"{remain:.0f}s" if remain < 3600 else f"{remain/3600:.1f}h"
        bar     = "█" * int(BAR_W * done / n_sims) + "░" * int(BAR_W * (1 - done/n_sims))
        print(f"\r  [{bar}] {pct:5.1f}%  {done:>13,}/{n_sims:,}"
              f"  {rate/1e6:.2f} M/s  ETA:{eta}   ", end="", flush=True)

    total_time = time.perf_counter() - t0
    print(f"\r  [{'█'*BAR_W}] 100.0%  {n_sims:>13,}/{n_sims:,}"
          f"  done in {total_time:.2f}s{' '*28}")
    return accumulated


# =============================================================================
# Core result analysis
# =============================================================================

def analyse(counts: np.ndarray, pick: int = PICK):
    freq        = {i + 1: int(counts[i]) for i in range(len(counts))}
    recommended = sorted(sorted(freq, key=freq.__getitem__, reverse=True)[:pick])
    return freq, recommended


# =============================================================================
# Sets A / B / C  (classical hot / cold / mixed)
# =============================================================================

def compute_ABC(freq: dict, recommended: list, pick: int = PICK):
    """
    Returns (anti_set, mixed_set) where each list is ordered
    STRONGEST-FIRST so the 7th element is the weakest signal
    (used as the 'additional' number).
    """
    by_asc   = sorted(freq, key=freq.__getitem__)         # cold -> hot
    by_desc  = list(reversed(by_asc))                     # hot  -> cold
    rec_set  = set(recommended)

    # B (cold): coldest first => strongest contrarian first
    anti_pool = [b for b in by_asc if b not in rec_set]
    anti_set  = anti_pool[:pick]

    # C (mixed): hottest hots first, then coldest colds
    hot_n     = math.ceil(pick / 2)
    cold_n    = pick - hot_n
    hot_balls = [b for b in by_desc if b in rec_set][:hot_n]
    mixed_set = hot_balls + anti_set[:cold_n]

    return anti_set, mixed_set


# =============================================================================
# Advanced aggressive strategies  (D / E / F / G)
# =============================================================================

def _co_matrix(draws: np.ndarray, num_balls: int) -> np.ndarray:
    """(num_balls × num_balls) symmetric co-occurrence count matrix."""
    co = np.zeros((num_balls, num_balls), dtype=np.int64)
    k  = draws.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            np.add.at(co, (draws[:, i], draws[:, j]), 1)
            np.add.at(co, (draws[:, j], draws[:, i]), 1)
    return co


def _recency_weights(draws: np.ndarray, days_ago: np.ndarray,
                     num_balls: int, half_life: float = 365.0) -> np.ndarray:
    """Per-ball probability weighted by exponential recency decay."""
    decay   = np.exp(-np.log(2) * days_ago / half_life)          # (N,)
    one_hot = np.zeros((len(draws), num_balls), dtype=np.float32)
    rows    = np.repeat(np.arange(len(draws)), draws.shape[1])
    one_hot[rows, draws.ravel()] = 1.0
    counts  = (one_hot.T @ decay.astype(np.float32)).astype(np.float64) + 1.0
    return counts / counts.sum()


def set_D_recency(draws, days_ago, num_balls, pick,
                  n_sims=2_000_000, seed=None, half_life=365.0) -> list:
    """
    SET D — Gumbel-max simulation but with recency-decayed weights.
    Draws from ~1 year ago count half as much as today's draws.
    """
    w     = _recency_weights(draws, days_ago, num_balls, half_life)
    log_w = np.log(w).astype(np.float32)
    rng   = np.random.default_rng(seed)
    chunk = auto_chunk_size(num_balls)
    acc   = np.zeros(num_balls, dtype=np.int64)
    done  = 0
    while done < n_sims:
        this = min(chunk, n_sims - done)
        if _GPU_AVAILABLE:
            log_w_gpu = cp.asarray(log_w[np.newaxis, :])
            acc += _chunk_gpu(log_w_gpu, this, pick, num_balls)
        else:
            acc += _chunk_cpu(log_w, this, pick, num_balls, rng)
        done += this
    # ordered strongest-first (highest sim count first)
    top_idx = np.argsort(acc)[-pick:][::-1]
    return [int(b) + 1 for b in top_idx]


def set_E_cooccurrence(draws, num_balls, pick) -> list:
    """
    SET E — Greedy co-occurrence clique.
    Finds the pick-ball set whose members appeared together most in history.
    Start with the globally best-connected ball; greedily add the ball
    that maximises sum of co-occurrences with already-selected members.
    """
    co       = _co_matrix(draws, num_balls)
    selected = [int(np.argmax(co.sum(axis=1)))]
    for _ in range(pick - 1):
        mask  = np.ones(num_balls, dtype=bool)
        mask[selected] = False
        cands = np.where(mask)[0]
        scores = co[np.ix_(cands, selected)].sum(axis=1)
        selected.append(int(cands[np.argmax(scores)]))
    # `selected` already strongest-first (greedy picks best link each round)
    return [b + 1 for b in selected]


def set_F_genetic(draws, days_ago, num_balls, pick,
                  n_pop=1_000, n_gen=400, mut_rate=0.15,
                  half_life=365.0, seed=None) -> list:
    """
    SET F — Genetic Algorithm.
    Maximises blended fitness:
        fitness = 0.5 × normalised_recency_score
                + 0.5 × normalised_co_occurrence_score

    Population of n_pop combos, evolved for up to n_gen generations with
    rank-proportional selection, union-crossover, and point mutation.
    Early-stops after 50 stagnant generations.
    """
    rng     = np.random.default_rng(seed)
    rec_w   = _recency_weights(draws, days_ago, num_balls, half_life)
    co_mat  = _co_matrix(draws, num_balls).astype(np.float64)
    co_max  = max(co_mat.max() * pick * (pick - 1) / 2, 1.0)
    max_pairs = pick * (pick - 1) // 2

    def _fit(pop: np.ndarray) -> np.ndarray:   # (M, pick) -> (M,)
        f = rec_w[pop].sum(axis=1)
        c = np.zeros(len(pop))
        for i in range(pick):
            for j in range(i + 1, pick):
                c += co_mat[pop[:, i], pop[:, j]]
        return 0.5 * f + 0.5 * (c / co_max)

    probs = rec_w / rec_w.sum()
    pop   = np.array([np.sort(rng.choice(num_balls, pick, replace=False, p=probs))
                      for _ in range(n_pop)], dtype=np.int32)

    best_fit = -np.inf
    best_set = pop[0].copy()
    stagnant = 0

    for _ in range(n_gen):
        fits = _fit(pop)
        top  = int(fits.argmax())
        if fits[top] > best_fit:
            best_fit = fits[top]
            best_set = pop[top].copy()
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= 50:
                break

        # Rank-proportional selection
        ranks = np.argsort(np.argsort(fits)) + 1.0
        sel_p = ranks / ranks.sum()
        n_el  = max(1, n_pop // 10)
        new_pop = list(pop[np.argsort(fits)[-n_el:]])

        pidx = rng.choice(n_pop, size=(n_pop - n_el, 2), replace=True, p=sel_p)
        for pi, qi in pidx:
            union = np.union1d(pop[pi], pop[qi])
            if len(union) >= pick:
                child = union[np.argsort(rec_w[union])[-pick:]]
            else:
                avail = np.setdiff1d(np.arange(num_balls), union)
                child = np.sort(np.concatenate(
                    [union, rng.choice(avail, pick - len(union), replace=False)]))
            if rng.random() < mut_rate:
                pos        = rng.integers(0, pick)
                avail      = np.setdiff1d(np.arange(num_balls), child)
                child      = child.copy()
                child[pos] = rng.choice(avail)
                child.sort()
            new_pop.append(child.astype(np.int32))

        pop = np.array(new_pop[:n_pop])

    # Rank balls inside best_set by their personal contribution to fitness:
    # individual recency weight + sum of co-occurrence with the other 6 picks.
    contrib = np.array([
        rec_w[b] + sum(co_mat[b, o] for o in best_set if o != b) / max(co_max, 1.0)
        for b in best_set
    ])
    order = np.argsort(contrib)[::-1]   # strongest first
    return [int(best_set[i]) + 1 for i in order]


def set_G_consensus(draws, num_balls, pick, windows=(100, 300, 500)) -> list:
    """
    SET G — Multi-window consensus.
    Score each ball by how many time-windows it ranks in the top-pick.
    Windows: last 100, last 300, last 500, and all-time draws.
    Break ties by all-time frequency.
    """
    all_wins   = list(windows) + [len(draws)]
    scores     = np.zeros(num_balls, dtype=np.int32)
    all_counts = np.bincount(draws.ravel(), minlength=num_balls)[:num_balls]
    for w in all_wins:
        cnt     = np.bincount(draws[:w].ravel(), minlength=num_balls)[:num_balls]
        top_set = set(np.argsort(cnt)[-pick:])
        for b in top_set:
            scores[b] += 1
    key = scores.astype(np.float64) + all_counts / (all_counts.max() + 1)
    # strongest-first: highest consensus score first
    top_idx = np.argsort(key)[-pick:][::-1]
    return [int(b) + 1 for b in top_idx]


def _dedupe_preserve_order(values) -> list:
    seen = set()
    ordered = []
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _set_key(balls) -> tuple:
    return tuple(sorted(int(ball) for ball in balls))


def _pick_distinct_ranked_set(ranked_balls, pick, excluded_keys, pool_size=14) -> list:
    """
    Pick the best-ranked `pick`-ball combo that is not already present in
    `excluded_keys`. Ranking is determined by the input order (best first).
    """
    ranked = _dedupe_preserve_order(ranked_balls)
    if len(ranked) < pick:
        raise ValueError("Not enough ranked balls to build a distinct set")

    pool = ranked[:max(pick, min(pool_size, len(ranked)))]
    positions = {ball: idx for idx, ball in enumerate(pool)}
    best = None
    best_rank = None

    for combo in itertools.combinations(pool, pick):
        key = _set_key(combo)
        if key in excluded_keys:
            continue
        rank = tuple(sorted(positions[ball] for ball in combo))
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = [ball for ball in pool if ball in combo]

    if best is None:
        for combo in itertools.combinations(ranked, pick):
            key = _set_key(combo)
            if key not in excluded_keys:
                combo_set = set(combo)
                return [ball for ball in ranked if ball in combo_set]
        raise ValueError("Unable to build a distinct ranked set")

    return best


def _pick_distinct_mixed_set(primary_ranked, primary_count,
                            secondary_ranked, secondary_count,
                            excluded_keys, primary_pool=10, secondary_pool=10) -> list:
    """
    Pick a distinct mixed set while preserving the intended primary/secondary
    ranking order inside the returned list.
    """
    primary = _dedupe_preserve_order(primary_ranked)
    secondary = _dedupe_preserve_order(secondary_ranked)
    primary = primary[:max(primary_count, min(primary_pool, len(primary)))]
    secondary = secondary[:max(secondary_count, min(secondary_pool, len(secondary)))]

    primary_pos = {ball: idx for idx, ball in enumerate(primary)}
    secondary_pos = {ball: idx for idx, ball in enumerate(secondary)}
    best = None
    best_rank = None

    for p_combo in itertools.combinations(primary, primary_count):
        p_set = set(p_combo)
        p_rank = tuple(sorted(primary_pos[ball] for ball in p_combo))
        for s_combo in itertools.combinations(secondary, secondary_count):
            union = p_set | set(s_combo)
            if len(union) != primary_count + secondary_count:
                continue
            key = _set_key(union)
            if key in excluded_keys:
                continue
            s_rank = tuple(sorted(secondary_pos[ball] for ball in s_combo))
            rank = (p_rank, s_rank)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = ([ball for ball in primary if ball in p_set] +
                        [ball for ball in secondary if ball in union and ball not in p_set])

    if best is None:
        raise ValueError("Unable to build a distinct mixed set")

    return best


def compute_all_sets(freq, counts_arr, draws, days_ago, pick, seed=None):
    """
    Compute sets A-G + the vote champion ★.
    Returns a dict keyed by single-letter label.
    """
    print("\n[INFO] Computing advanced strategies ...")
    t0 = time.perf_counter()

    recommended = sorted(sorted(freq, key=freq.__getitem__, reverse=True)[:pick])
    anti_set, mixed_set = compute_ABC(freq, recommended, pick)

    print("  [D] Recency-weighted simulation ...", end=" ", flush=True)
    sd = set_D_recency(draws, days_ago, NUM_BALLS, pick, seed=seed)
    print("done")

    print("  [E] Co-occurrence greedy search  ...", end=" ", flush=True)
    se = set_E_cooccurrence(draws, NUM_BALLS, pick)
    print("done")

    print("  [F] Genetic algorithm            ...", end=" ", flush=True)
    sf = set_F_genetic(draws, days_ago, NUM_BALLS, pick, seed=seed)
    print("done")

    print("  [G] Consensus windows            ...", end=" ", flush=True)
    sg = set_G_consensus(draws, NUM_BALLS, pick)
    print("done")

    used_keys = {
        _set_key(recommended),
        _set_key(anti_set),
        _set_key(mixed_set),
        _set_key(sd),
        _set_key(se),
        _set_key(sf),
        _set_key(sg),
    }

    # Vote champion: balls in the most strategies
    all_7 = [recommended, anti_set, mixed_set, sd, se, sf, sg]
    votes = np.zeros(NUM_BALLS + 1, dtype=np.int32)
    for s in all_7:
        for b in s:
            votes[b] += 1
    # Sort by vote count desc, break ties by sim frequency
    raw_freq = np.array([freq.get(b, 0) for b in range(NUM_BALLS + 1)], dtype=np.float64)
    sort_key_desc = votes.astype(np.float64) + raw_freq / (raw_freq.max() + 1)
    # strongest-first (highest votes first); slot[6] = weakest of the 7 = additional
    ranked_desc = [int(b) for b in np.argsort(sort_key_desc)[::-1] if b != 0]
    star = _pick_distinct_ranked_set(ranked_desc, pick, used_keys)
    used_keys.add(_set_key(star))

    # Set H — Least-vote: balls that appeared in the FEWEST strategies
    # Ties broken by lowest sim frequency (most contrarian overall)
    sort_key_asc = votes.astype(np.float64) - raw_freq / (raw_freq.max() + 1)
    # already strongest-first (least-voted = strongest contrarian signal first)
    ranked_asc = [int(b) for b in np.argsort(sort_key_asc) if b != 0]
    least_vote = _pick_distinct_ranked_set(ranked_asc, pick, used_keys)
    used_keys.add(_set_key(least_vote))

    # Set I — Mix 4×3: 3 vote-champion (strongest support)
    #                 + 4 least-vote (strongest contrarian first)
    # The 7th slot (additional) is the weakest of the four LV picks.
    star_strongest = sorted(star, key=lambda b: (-votes[b], -raw_freq[b]))
    lv_strongest = sorted(least_vote, key=lambda b: (votes[b], raw_freq[b]))
    mix_4x3 = _pick_distinct_mixed_set(star_strongest, 3,
                                       lv_strongest, 4,
                                       used_keys)

    elapsed = time.perf_counter() - t0
    print(f"[INFO] Advanced strategies done in {elapsed:.2f}s\n")

    return {
        "A": recommended,
        "B": anti_set,
        "C": mixed_set,
        "D": sd,
        "E": se,
        "F": sf,
        "G": sg,
        "star": star,
        "H": least_vote,
        "I": mix_4x3,
    }


# =============================================================================
# Reporting
# =============================================================================

def _fmt_set(label: str, tag: str, balls: list, bar: str):
    """
    `balls` MUST be ordered strongest-first.
    The 6 strongest become the main; the 7th (weakest) becomes the additional.
    """
    main = sorted(balls[:6])
    rest = balls[6:]
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"\n  Main numbers  : {main}")
    if rest:
        addn_label = "Additional    " if len(rest) == 1 else "Extra numbers "
        print(f"  {addn_label}: {rest[0] if len(rest) == 1 else rest}")
    print(f"  Full set      : {sorted(balls)}")
    print(f"  ({tag})")


def print_report(freq, sets, weights, n_sims, elapsed):
    pick        = len(sets["A"])
    total_picks = n_sims * pick
    backend     = "GPU -- CuPy / CUDA" if _GPU_AVAILABLE else "CPU -- NumPy"
    bar         = "=" * 62
    hot_n       = math.ceil(pick / 2)
    cold_n      = pick - hot_n

    print(f"\n{bar}")
    print("  Monte Carlo Weighted ToTo Simulation -- Results")
    print(bar)
    print(f"  Simulations : {n_sims:>12,}")
    print(f"  Backend     : {backend}")
    print(f"  Elapsed     : {elapsed:>11.3f} s")
    print(f"  Throughput  : {n_sims / elapsed / 1e6:>10.2f} M sims/s")
    print(bar)

    # Tag lookup
    tag_map = {
        "A:hot":      set(sets["A"]),
        "B:cold":     set(sets["B"]),
        "C:mixed":    set(sets["C"]),
        "D:recency":  set(sets["D"]),
        "E:co-occ":   set(sets["E"]),
        "F:genetic":  set(sets["F"]),
        "G:consensus":set(sets["G"]),
        "STAR":       set(sets["star"]),
        "H:least":    set(sets["H"]),
        "I:mix4x3":   set(sets["I"]),
    }
    # votes only counts the 7 independent strategies (not derived sets H/I)
    vote_keys = {"A:hot","B:cold","C:mixed","D:recency","E:co-occ","F:genetic","G:consensus","STAR"}
    all_special = set().union(*tag_map.values())

    print("\n  Ball frequency table (top 20 by sim count + all cold balls):")
    print("  " + "-" * 76)
    print(f"  {'Ball':>5}  {'Hist.%':>7}  {'Sim.Count':>13}  {'Sim.%':>7}  Sets")
    print("  " + "-" * 76)

    by_desc   = sorted(freq, key=freq.__getitem__, reverse=True)
    cold_balls = [b for b in sets["B"] if b not in by_desc[:20]]
    for ball in by_desc[:20] + cold_balls:
        hw    = weights[ball - 1] * 100
        scnt  = freq[ball]
        spct  = scnt / total_picks * 100
        tags  = [t for t, s in tag_map.items() if ball in s]
        print(f"  {ball:>5}  {hw:>6.2f}%  {scnt:>13,}  {spct:>6.2f}%  {' '.join(tags)}")

    # NOTE: pass each set RAW (strongest-first); _fmt_set picks the weakest as 'additional'.
    _fmt_set(f"SET A — HOT      (top {pick} most-frequent in sim)",
             "pure hot — highest Gumbel-sim frequency", sets["A"], bar)
    _fmt_set(f"SET B — COLD     (top {pick} least-frequent, not in A)",
             "contrarian — lowest sim frequency", sets["B"], bar)
    _fmt_set(f"SET C — MIXED    ({hot_n} hottest from A  +  {cold_n} coldest from B)",
             f"balanced — {hot_n} hot + {cold_n} cold", sets["C"], bar)
    _fmt_set("SET D — RECENCY  (sim weighted by recency decay ~1yr half-life)",
             "recent draws count 2x more per year", sets["D"], bar)
    _fmt_set("SET E — CO-OCCUR (greedy pairwise co-occurrence clique)",
             "balls that appear together most in history", sets["E"], bar)
    _fmt_set("SET F — GENETIC  (GA maximising recency + co-occurrence fitness)",
             "evolved combo — best blended score after 400 generations", sets["F"], bar)
    _fmt_set("SET G — CONSENSUS (hot across last-100/300/500/all-time windows)",
             "stable picks across all time-window views", sets["G"], bar)

    # Vote counts per ball — only from the 7 independent strategies
    votes = {b: sum(1 for k, s in tag_map.items() if k in vote_keys and b in s)
             for b in range(1, NUM_BALLS+1)}

    print(f"\n{bar}")
    print(f"  SET ★ — VOTE CHAMPION  (top {pick} balls across all 7 strategies)")
    print(bar)
    print(f"\n  Votes per ball in top-{pick*2} (most → least):")
    champ_balls_sorted = sorted(range(1, NUM_BALLS + 1),
                                key=lambda b: votes[b], reverse=True)[:pick * 2]
    vote_parts = [f"ball {b:>2} → {votes[b]}" for b in champ_balls_sorted]
    for line in [vote_parts[i:i+5] for i in range(0, len(vote_parts), 5)]:
        print("    " + "   ".join(f"{p:<16}" for p in line))
    star_raw = sets["star"]                  # already strongest-first
    main = sorted(star_raw[:6])
    addl = star_raw[6]
    print(f"\n  Main numbers  : {main}")
    print(f"  Additional    : {addl}")
    print(f"  Full set      : {sorted(star_raw)}")
    print(f"  (cross-strategy consensus — strongest overall signal)")

    _fmt_set(f"SET H — LEAST VOTE  (top {pick} balls in the FEWEST strategies)",
             "contrarian consensus — ignored by most strategies",
             sets["H"], bar)

    set_i_balls = set(sets["I"])
    star_top3 = [b for b in sets["star"] if b in set_i_balls][:3]
    lv_bot4 = [b for b in sets["H"] if b in set_i_balls][:4]
    _fmt_set("SET I — MIX 4×3  (4 least-vote  +  3 vote-champion)",
             f"4 from Set H {sorted(lv_bot4)}  +  3 from Set ★ {sorted(star_top3)}",
             sets["I"], bar)

    print("\n" + bar + "\n")


# =============================================================================
# Save results
# =============================================================================

def save_results(freq, sets, weights, n_sims, elapsed, output_dir=RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend     = "GPU (CuPy/CUDA)" if _GPU_AVAILABLE else "CPU (NumPy)"
    total_picks = n_sims * len(sets["A"])

    def _entry(key, balls):
        # `balls` is strongest-first; weakest (slot 6) is the additional number.
        return {
            f"set_{key}":            sorted(balls),
            f"set_{key}_main":       sorted(balls[:6]),
            f"set_{key}_additional": balls[6] if len(balls) > 6 else None,
        }

    summary = {
        "timestamp":       ts,
        "simulations":     n_sims,
        "elapsed_seconds": round(elapsed, 3),
        "sims_per_second": round(n_sims / elapsed) if elapsed > 0 else 0,
        "backend":         backend,
        **_entry("A_hot", sets["A"]),
        **_entry("B_cold", sets["B"]),
        **_entry("C_mixed", sets["C"]),
        **_entry("D_recency", sets["D"]),
        **_entry("E_cooccurrence", sets["E"]),
        **_entry("F_genetic", sets["F"]),
        **_entry("G_consensus", sets["G"]),
        **_entry("STAR_champion", sets["star"]),
        **_entry("H_least_vote", sets["H"]),
        **_entry("I_mix4x3", sets["I"]),
        "ball_frequencies": {str(k): v for k, v in freq.items()},
    }
    json_path = os.path.join(output_dir, f"sim_{ts}_n{n_sims}.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    ball_sets = {f"set_{k}": set(v) for k, v in sets.items()}
    rows = [
        {
            "ball":                  b,
            "historical_weight_pct": round(float(weights[b - 1]) * 100, 4),
            "sim_count":             freq[b],
            "sim_pct":               round(freq[b] / total_picks * 100, 4),
            **{col: b in s for col, s in ball_sets.items()},
        }
        for b in range(1, NUM_BALLS + 1)
    ]
    csv_path = os.path.join(output_dir, f"sim_{ts}_n{n_sims}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"[INFO] Saved JSON -> {json_path}")
    print(f"[INFO] Saved CSV  -> {csv_path}")
    return json_path, csv_path


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo ToTo simulation — 8 strategies, GPU-accelerated",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--simulations", "-n", type=int, default=DEFAULT_SIMULATIONS,
                        metavar="N", help="Gumbel simulation count")
    parser.add_argument("--csv",     type=str,  default=DEFAULT_CSV,  metavar="PATH")
    parser.add_argument("--seed",    type=int,  default=None,         metavar="INT")
    parser.add_argument("--top",     type=int,  default=PICK,         metavar="K",
                        help="How many numbers to pick per set (default 7)")
    parser.add_argument("--cpu",     action="store_true")
    parser.add_argument("--chunk",   type=int,  default=None,         metavar="C")
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR, metavar="DIR")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--half-life", type=float, default=10*365.0, metavar="DAYS",
                        help="Recency half-life in days for sets D and F (default 3650)")
    return parser.parse_args()


def main():
    global _GPU_AVAILABLE
    args = parse_args()
    if args.cpu:
        _GPU_AVAILABLE = False

    print_system_info()

    backend_label = "GPU (CuPy / CUDA)" if _GPU_AVAILABLE else "CPU (NumPy)"
    print(f"\n[INFO] Backend     : {backend_label}")
    print(f"[INFO] Simulations : {args.simulations:,}")
    print(f"[INFO] CSV         : {args.csv}")

    try:
        numbers        = load_historical_numbers(args.csv)
        draws, days_ago = load_draws_structured(args.csv)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Historical  : {len(numbers):,} occurrences  "
          f"({len(draws):,} draws)")

    weights = compute_weights(numbers, NUM_BALLS)
    chunk   = args.chunk or auto_chunk_size(NUM_BALLS)
    print(f"[INFO] Chunk size  : {chunk:,}")
    print(f"\n[INFO] Running Gumbel simulation ...\n")

    t0      = time.perf_counter()
    counts  = run_simulations(weights, args.simulations,
                              pick=args.top, chunk_size=chunk, seed=args.seed)
    elapsed = time.perf_counter() - t0

    freq, _ = analyse(counts, args.top)

    sets = compute_all_sets(freq, counts, draws, days_ago,
                            pick=args.top, seed=args.seed)

    print_report(freq, sets, weights, args.simulations, elapsed)

    if not args.no_save:
        save_results(freq, sets, weights, args.simulations, elapsed,
                     output_dir=args.output_dir)


if __name__ == "__main__":
    main()
