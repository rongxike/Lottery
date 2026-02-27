#!/usr/bin/env python3
"""
Monte Carlo Weighted Lottery Simulation  (Chunked + Progress Edition)
======================================================================
Uses historical ToTo draw frequencies as probability weights.
GPU-accelerated via CuPy (NVIDIA CUDA) with automatic NumPy fallback.

Key improvements over the original
------------------------------------
* Memory-safe chunked execution — no single (N × 49) allocation that
  could crash large runs.  Intermediate arrays are deleted immediately
  after use so peak VRAM / RAM stays at one-chunk size.
* Ball counts are *accumulated* across chunks → final memory is O(49),
  not O(n_sims × 7).
* Live progress bar: percentage, throughput (M sims/s) and ETA.
* Results are automatically saved to the results/ folder (JSON + CSV).
* System-info block at startup with auto-detected safe chunk size and
  the maximum recommended simulation count for this hardware.
* CPU path uses float32 throughout (halves RAM vs the default float64).

Algorithm — Gumbel-max trick (weighted sampling WITHOUT replacement)
--------------------------------------------------------------------
  key_i = log(w_i) + Gumbel(0,1)_i
  The top-PICK indices by key give an unbiased weighted draw.

Usage
-----
  python monte.py                             # 1 000 000 sims (default)
  python monte.py -n 100_000_000             # 100 M sims
  python monte.py -n 1_000_000_000           # 1 B sims (GPU recommended)
  python monte.py -n 50000 --seed 42         # reproducible run
  python monte.py --csv path/to/data.csv     # custom CSV
  python monte.py --cpu                      # force CPU mode
  python monte.py --chunk 5000000            # manual chunk override
  python monte.py --no-save                  # skip writing to results/
"""

import argparse
import json
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
NUM_BALLS           = 49          # lottery pool 1–49
PICK                = 7           # 6 main numbers + 1 additional
DEFAULT_SIMULATIONS = 100_000_000
RESULTS_DIR         = "results"
# ───────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# System info & auto-sizing
# ─────────────────────────────────────────────────────────────────────────────

def _get_memory_info():
    """Return (gpu_free_bytes_or_None, cpu_free_bytes)."""
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
        cpu_free = 8 * 1024 ** 3   # conservative 8 GB fallback
    return gpu_free, cpu_free


def auto_chunk_size(n_balls: int = NUM_BALLS) -> int:
    """
    Determine a safe chunk size from available memory.

    Peak usage per sim in a chunk:
        4 float32 arrays x n_balls x 4 bytes
        (U, gumbel, keys, argpartition output briefly overlapping)
    We use at most 60% of free GPU VRAM (or 40% of free RAM on CPU).
    Chunk is capped at 20 M to keep progress updates frequent.
    """
    gpu_free, cpu_free = _get_memory_info()
    if gpu_free is not None:
        safe_bytes = int(gpu_free * 0.60)
    else:
        safe_bytes = int(cpu_free * 0.40)

    bytes_per_sim = n_balls * 4 * 4          # 4 float32 arrays
    chunk = max(100_000, safe_bytes // bytes_per_sim)
    return min(chunk, 20_000_000)


def max_recommended_sims() -> int:
    """
    Maximum simulation count recommended for this hardware.

    With chunked execution the hard limit is TIME, not memory:
      GPU  (NVIDIA GB10, ~112 GB VRAM) -> 1 B  (~10-30 s typical)
      CPU  (20-core ARM, ~110 GB RAM)  -> 200 M (~30-120 s typical)
    Both figures assume the auto chunk size.
    """
    gpu_free, _ = _get_memory_info()
    return 1_000_000_000 if gpu_free is not None else 200_000_000


def print_system_info():
    """Print a hardware summary block including recommended max sims."""
    sep = "-" * 62
    print(f"\n{sep}")
    print("  Hardware Summary")
    print(sep)

    # CPU
    try:
        import multiprocessing
        cores = multiprocessing.cpu_count()
        print(f"  CPU cores        : {cores}")
    except Exception:
        pass

    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        print(f"  RAM total        : {vm.total  / 1024**3:.1f} GB")
        print(f"  RAM available    : {vm.available / 1024**3:.1f} GB")
    except ImportError:
        print("  RAM              : (install psutil for details)")

    # GPU
    if _GPU_AVAILABLE:
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            ver = cp.cuda.runtime.runtimeGetVersion()
            cuda_major = ver // 1000
            cuda_minor = (ver % 1000) // 10
            print(f"  GPU              : NVIDIA (CUDA {cuda_major}.{cuda_minor})")
            print(f"  VRAM total       : {total / 1024**3:.1f} GB")
            print(f"  VRAM free        : {free  / 1024**3:.1f} GB")
        except Exception:
            print("  GPU              : CuPy available (details unavailable)")
    else:
        print("  GPU              : Not available -- using CPU")

    chunk = auto_chunk_size()
    max_s = max_recommended_sims()
    backend_tag = "GPU" if _GPU_AVAILABLE else "CPU"
    print(f"\n  Auto chunk size  : {chunk:>12,}  sims  (memory-safe, no crash)")
    print(f"  Max recommended  : {max_s:>12,}  sims  ({backend_tag}, time-bound)")
    print(f"    -> Larger counts are fine too -- just run longer.")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_historical_numbers(csv_path: str) -> np.ndarray:
    """Read CSV -> flat int array of every number ever drawn."""
    df = pd.read_csv(csv_path)
    win_cols = [
        "Winning Number 1", "2", "3", "4", "5", "6", "Additional Number"
    ]
    raw   = df[win_cols].values.flatten().astype(float)
    valid = raw[~np.isnan(raw)].astype(int)
    valid = valid[(valid >= 1) & (valid <= NUM_BALLS)]
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Weight computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_weights(numbers: np.ndarray, num_balls: int = NUM_BALLS) -> np.ndarray:
    """
    Build a probability vector of length num_balls.
    Proportional to historical appearance frequency.
    Laplace smoothing (+1) prevents zero-probability balls.
    """
    counts = np.zeros(num_balls, dtype=np.float64)
    np.add.at(counts, numbers - 1, 1.0)
    counts += 1.0
    return counts / counts.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Single-chunk simulation kernels
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_gpu(log_w_gpu, chunk_size: int, pick: int, n_balls: int) -> np.ndarray:
    """
    GPU path -- one chunk.

    Intermediate large arrays (U, gumbel, keys, top_idx) are deleted
    immediately after use so peak VRAM is minimised.

    Returns numpy int64 count vector of length n_balls.
    """
    U      = cp.random.uniform(1e-6, 1.0 - 1e-6,
                               size=(chunk_size, n_balls), dtype=cp.float32)
    gumbel = -cp.log(-cp.log(U));  del U
    keys   = log_w_gpu + gumbel;   del gumbel
    top_idx = cp.argpartition(keys, -pick, axis=1)[:, -pick:];  del keys
    flat   = top_idx.ravel().astype(cp.int32);                  del top_idx
    counts = cp.bincount(flat, minlength=n_balls)
    return cp.asnumpy(counts)[:n_balls].astype(np.int64)


def _chunk_cpu(
    log_w: np.ndarray,
    chunk_size: int,
    pick: int,
    n_balls: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    CPU fallback -- one chunk.  float32 throughout to halve RAM usage.

    Returns numpy int64 count vector of length n_balls.
    """
    U      = rng.random(size=(chunk_size, n_balls), dtype=np.float32)
    np.clip(U, 1e-6, 1.0 - 1e-6, out=U)
    gumbel = -np.log(-np.log(U));  del U
    keys   = log_w + gumbel;       del gumbel       # broadcasts (n_balls,)
    top_idx = np.argpartition(keys, -pick, axis=1)[:, -pick:];  del keys
    flat   = top_idx.ravel().astype(np.int32);                  del top_idx
    return np.bincount(flat, minlength=n_balls)[:n_balls].astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop -- chunked, with live progress bar
# ─────────────────────────────────────────────────────────────────────────────

def run_simulations(
    weights:    np.ndarray,
    n_sims:     int,
    pick:       int  = PICK,
    chunk_size: int  = None,
    seed:       int  = None,
) -> np.ndarray:
    """
    Run n_sims Monte Carlo draws in memory-safe chunks.

    Ball counts are accumulated across chunks so the full
    (n_sims x pick) result matrix is NEVER held in memory.

    Returns accumulated count array of shape (NUM_BALLS,), dtype int64.
    """
    n_balls = len(weights)

    if chunk_size is None:
        chunk_size = auto_chunk_size(n_balls)

    log_w = np.log(weights).astype(np.float32)

    # Pre-load to GPU once (broadcast-ready row vector)
    log_w_gpu = cp.asarray(log_w[np.newaxis, :]) if _GPU_AVAILABLE else None

    # Seeding
    rng = np.random.default_rng(seed)
    if _GPU_AVAILABLE and seed is not None:
        cp.random.seed(seed)

    accumulated = np.zeros(n_balls, dtype=np.int64)
    done  = 0
    t0    = time.perf_counter()
    BAR_W = 28

    while done < n_sims:
        this_chunk = min(chunk_size, n_sims - done)

        if _GPU_AVAILABLE:
            chunk_counts = _chunk_gpu(log_w_gpu, this_chunk, pick, n_balls)
        else:
            chunk_counts = _chunk_cpu(log_w, this_chunk, pick, n_balls, rng)

        accumulated += chunk_counts
        done        += this_chunk

        # ── live progress ────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        pct     = done / n_sims * 100
        rate    = done / elapsed if elapsed > 1e-9 else 0.0
        remain  = (n_sims - done) / rate if rate > 0 else 0.0
        eta_str = (f"{remain:.0f}s" if remain < 3600 else f"{remain / 3600:.1f}h")

        filled = int(BAR_W * done / n_sims)
        bar    = "█" * filled + "░" * (BAR_W - filled)

        print(
            f"\r  [{bar}] {pct:5.1f}%"
            f"  {done:>13,} / {n_sims:,}"
            f"  {rate / 1_000_000:.2f} M/s"
            f"  ETA: {eta_str}   ",
            end="", flush=True,
        )

    # Final "done" line
    elapsed_total = time.perf_counter() - t0
    print(
        f"\r  [{'█' * BAR_W}] 100.0%"
        f"  {n_sims:>13,} / {n_sims:,}"
        f"  done in {elapsed_total:.2f}s{' ' * 28}"
    )

    return accumulated


# ─────────────────────────────────────────────────────────────────────────────
# Result analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse(counts: np.ndarray, pick: int = PICK):
    """
    Accept the accumulated count array (shape NUM_BALLS,).
    Returns:
        freq        -- dict {ball: int count}
        recommended -- sorted list of top-pick balls
    """
    freq        = {i + 1: int(counts[i]) for i in range(len(counts))}
    recommended = sorted(sorted(freq, key=freq.__getitem__, reverse=True)[:pick])
    return freq, recommended


# ─────────────────────────────────────────────────────────────────────────────
# Extra set computation (anti / cold  +  mixed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_extra_sets(freq: dict, recommended: list, pick: int = PICK):
    """
    Derive two alternative pick sets from simulation frequencies.

    anti_set  -- the `pick` LEAST-frequent balls not already in recommended
                 (contrarian / cold play).
    mixed_set -- ceil(pick/2) HOTTEST balls from recommended
                 + floor(pick/2) COLDEST balls from anti_set.

    Both lists are sorted ascending (ball number) for display.
    Breakdown for pick=7: 4 hot + 3 cold.
    """
    import math

    by_freq_asc  = sorted(freq, key=freq.__getitem__)         # coldest first
    by_freq_desc = list(reversed(by_freq_asc))                # hottest first
    rec_set      = set(recommended)

    # Anti: coldest balls that are NOT already recommended
    anti_pool = [b for b in by_freq_asc if b not in rec_set]
    anti_set  = sorted(anti_pool[:pick])

    # Mixed: hottest ceil(pick/2) from recommended
    #      + coldest floor(pick/2) from anti
    hot_n      = math.ceil(pick / 2)
    cold_n     = pick - hot_n
    hot_balls  = [b for b in by_freq_desc if b in rec_set][:hot_n]
    cold_balls = anti_set[:cold_n]
    mixed_set  = sorted(hot_balls + cold_balls)

    return anti_set, mixed_set


# ─────────────────────────────────────────────────────────────────────────────
# Save results to disk
# ─────────────────────────────────────────────────────────────────────────────

def save_results(
    freq:        dict,
    recommended: list,
    anti_set:    list,
    mixed_set:   list,
    weights:     np.ndarray,
    n_sims:      int,
    elapsed:     float,
    output_dir:  str = RESULTS_DIR,
):
    """Write JSON summary and CSV frequency table to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend     = "GPU (CuPy/CUDA)" if _GPU_AVAILABLE else "CPU (NumPy)"
    total_picks = n_sims * PICK

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = {
        "timestamp":         ts,
        "simulations":       n_sims,
        "elapsed_seconds":   round(elapsed, 3),
        "sims_per_second":   round(n_sims / elapsed) if elapsed > 0 else 0,
        "backend":           backend,
        # Set A -- hot (most frequent)
        "set_A_hot":         sorted(recommended),
        "set_A_main":        sorted(recommended[:6]),
        "set_A_additional":  recommended[6] if len(recommended) > 6 else None,
        # Set B -- cold (least frequent, not in A)
        "set_B_cold":        sorted(anti_set),
        "set_B_main":        sorted(anti_set[:6]),
        "set_B_additional":  anti_set[6] if len(anti_set) > 6 else None,
        # Set C -- mixed (half hot + half cold)
        "set_C_mixed":       sorted(mixed_set),
        "set_C_main":        sorted(mixed_set[:6]),
        "set_C_additional":  mixed_set[6] if len(mixed_set) > 6 else None,
        "ball_frequencies":  {str(k): v for k, v in freq.items()},
    }
    json_path = os.path.join(output_dir, f"sim_{ts}_n{n_sims}.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # ── CSV frequency table ───────────────────────────────────────────────────
    rec_set   = set(recommended)
    anti_set_ = set(anti_set)
    mix_set_  = set(mixed_set)
    rows = [
        {
            "ball":                  b,
            "historical_weight_pct": round(float(weights[b - 1]) * 100, 4),
            "sim_count":             freq[b],
            "sim_pct":               round(freq[b] / total_picks * 100, 4),
            "set_A_hot":             b in rec_set,
            "set_B_cold":            b in anti_set_,
            "set_C_mixed":           b in mix_set_,
        }
        for b in range(1, NUM_BALLS + 1)
    ]
    csv_path = os.path.join(output_dir, f"sim_{ts}_n{n_sims}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n[INFO] Saved JSON -> {json_path}")
    print(f"[INFO] Saved CSV  -> {csv_path}")
    return json_path, csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable report
# ─────────────────────────────────────────────────────────────────────────────

def _format_set(label: str, tag: str, balls: list, pick: int, bar: str):
    """Print one pick set (main + additional) with a labelled header."""
    main = sorted(balls[:6])
    addn = balls[6] if len(balls) > 6 else "—"
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"\n  Main numbers  : {main}")
    print(f"  Additional    : [{addn}]")
    print(f"  Full set      : {sorted(balls)}")
    print(f"  ({tag})")


def print_report(freq, recommended, anti_set, mixed_set, weights, n_sims, elapsed):
    total_picks = n_sims * PICK
    pick        = len(recommended)
    backend     = "GPU -- CuPy / CUDA" if _GPU_AVAILABLE else "CPU -- NumPy"
    bar         = "=" * 62

    print(f"\n{bar}")
    print("  Monte Carlo Weighted ToTo Simulation -- Results")
    print(bar)
    print(f"  Simulations : {n_sims:>12,}")
    print(f"  Backend     : {backend}")
    print(f"  Elapsed     : {elapsed:>11.3f} s")
    print(f"  Throughput  : {n_sims / elapsed / 1e6:>10.2f} M sims/s")
    print(bar)

    import math
    hot_n  = math.ceil(pick / 2)
    cold_n = pick - hot_n

    all_sets = set(recommended) | set(anti_set) | set(mixed_set)
    print("\n  Historical weight  x  Simulation appearances (top 20 + cold balls):")
    print("  " + "-" * 70)
    print(f"  {'Ball':>5}  {'Hist.%':>7}  {'Sim.Count':>13}  {'Sim.%':>7}  {'Sets':>14}")
    print("  " + "-" * 70)

    top20 = sorted(freq, key=freq.__getitem__, reverse=True)[:20]
    # also include any cold balls not already in top20
    cold_extra = [b for b in anti_set if b not in top20]
    display_balls = top20 + cold_extra

    for ball in display_balls:
        hw   = weights[ball - 1] * 100
        scnt = freq[ball]
        spct = scnt / total_picks * 100
        tags = []
        if ball in set(recommended): tags.append("A:hot")
        if ball in set(anti_set):    tags.append("B:cold")
        if ball in set(mixed_set):   tags.append("C:mixed")
        tag_str = " ".join(tags)
        print(f"  {ball:>5}  {hw:>6.2f}%  {scnt:>13,}  {spct:>6.2f}%  {tag_str:>14}")

    _format_set(
        f"SET A — HOT   (top {pick} most-frequent balls)",
        f"pure hot — highest sim frequency",
        sorted(recommended), pick, bar,
    )
    _format_set(
        f"SET B — COLD  (top {pick} least-frequent, not in A)",
        f"pure cold — contrarian / lowest sim frequency",
        sorted(anti_set), pick, bar,
    )
    _format_set(
        f"SET C — MIXED ({hot_n} hottest from A  +  {cold_n} coldest from B)",
        f"balanced — {hot_n} hot + {cold_n} cold",
        sorted(mixed_set), pick, bar,
    )
    print("\n" + bar + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo weighted ToTo lottery simulation "
                    "(GPU-accelerated, chunked, crash-proof)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--simulations", "-n",
        type=int, default=DEFAULT_SIMULATIONS, metavar="N",
        help="Number of Monte Carlo simulations",
    )
    parser.add_argument(
        "--csv",
        type=str, default=DEFAULT_CSV, metavar="PATH",
        help="Path to the lottery results CSV",
    )
    parser.add_argument(
        "--seed",
        type=int, default=None, metavar="INT",
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--top",
        type=int, default=PICK, metavar="K",
        help="How many numbers to recommend (default 7)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode even if GPU is available",
    )
    parser.add_argument(
        "--chunk",
        type=int, default=None, metavar="C",
        help="Override auto chunk size (sims per chunk)",
    )
    parser.add_argument(
        "--output-dir",
        type=str, default=RESULTS_DIR, metavar="DIR",
        help="Directory to save result files",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving results to disk",
    )
    return parser.parse_args()


def main():
    global _GPU_AVAILABLE

    args = parse_args()

    if args.cpu:
        _GPU_AVAILABLE = False

    # Print system info (includes max recommended sims)
    print_system_info()

    backend_label = "GPU (CuPy / CUDA)" if _GPU_AVAILABLE else "CPU (NumPy)"
    print(f"\n[INFO] Backend     : {backend_label}")
    print(f"[INFO] Simulations : {args.simulations:,}")
    print(f"[INFO] CSV         : {args.csv}")

    # Load historical data
    try:
        numbers = load_historical_numbers(args.csv)
    except FileNotFoundError:
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Historical  : {len(numbers):,} occurrences "
          f"({len(numbers) // PICK:,} draws)")

    weights = compute_weights(numbers, NUM_BALLS)

    # Determine chunk size
    chunk = args.chunk if args.chunk else auto_chunk_size(NUM_BALLS)
    print(f"[INFO] Chunk size  : {chunk:,}")
    print(f"\n[INFO] Simulating ...\n")

    # Run
    t0     = time.perf_counter()
    counts = run_simulations(
        weights,
        args.simulations,
        pick       = args.top,
        chunk_size = chunk,
        seed       = args.seed,
    )
    elapsed = time.perf_counter() - t0

    # Analyse & report
    freq, recommended          = analyse(counts, args.top)
    anti_set, mixed_set        = compute_extra_sets(freq, recommended, args.top)
    print_report(freq, recommended, anti_set, mixed_set, weights, args.simulations, elapsed)

    # Save
    if not args.no_save:
        save_results(
            freq, recommended, anti_set, mixed_set,
            weights, args.simulations, elapsed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
