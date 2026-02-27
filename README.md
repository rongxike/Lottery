# 🎱 ToTo Monte Carlo Lottery Simulator

A **GPU-accelerated Monte Carlo simulation engine** for Singapore ToTo lottery analysis.  
Runs up to **1 billion simulations** using NVIDIA CUDA, applying 9 distinct mathematical strategies to surface the strongest number combinations from historical draw data.

---

## ✨ Features

| Capability         | Detail                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------- |
| 🚀 GPU acceleration | CuPy / CUDA — automatic fallback to NumPy CPU                                               |
| 🧠 9 strategies     | Hot, Cold, Mixed, Recency, Co-occurrence, Genetic Algorithm, Consensus, Least-vote, Mix 4×3 |
| 🗳️ Vote champion    | Cross-strategy consensus signal (★)                                                         |
| 🔒 Memory-safe      | Chunked execution — never crashes regardless of simulation count                            |
| 📊 Live progress    | Real-time progress bar with throughput (M sims/s) and ETA                                   |
| 💾 Auto-save        | Every run saved to `results/` as JSON + CSV                                                 |

---

## 🖥️ Hardware Requirements

| Component       | Minimum      | Tested on                  |
| --------------- | ------------ | -------------------------- |
| Python          | 3.9+         | 3.12.3                     |
| RAM             | 4 GB         | 119.6 GB                   |
| GPU (optional)  | Any CUDA GPU | NVIDIA GB10, 119.6 GB VRAM |
| CUDA (optional) | 11.x+        | 13.0                       |

> **No GPU?** The simulator falls back to NumPy automatically. All 9 strategies still work — just slower.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/rongxike/Lottery.git
cd Lottery
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# or
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

**CPU only:**
```bash
pip install numpy pandas psutil
```

**GPU (NVIDIA CUDA):**
```bash
pip install numpy pandas psutil
pip install cupy-cuda12x          # match your CUDA version (11x / 12x / 13x)
```

> Check your CUDA version with `nvidia-smi`. Use `cupy-cuda11x` for CUDA 11, `cupy-cuda12x` for CUDA 12/13.

### 4. Prepare data

Place your ToTo historical CSV as `ToTo.csv` in the project root.  
Expected columns: `Draw`, `Date`, `Winning Number 1`, `2`, `3`, `4`, `5`, `6`, `Additional Number`

---

## 🚀 Usage

```bash
# Default — 1 million simulations
python monte.py

# 100 million simulations (recommended for best accuracy)
python monte.py -n 100_000_000

# 1 billion simulations (GPU, ~30–60s)
python monte.py -n 1_000_000_000

# Reproducible run with fixed seed
python monte.py -n 10_000_000 --seed 42

# Emphasise recent draws more (6-month half-life instead of 1 year)
python monte.py -n 50_000_000 --half-life 180

# Force CPU mode
python monte.py --cpu

# Custom CSV path
python monte.py --csv path/to/data.csv

# Skip saving results to disk
python monte.py --no-save

# Override chunk size (advanced)
python monte.py --chunk 5000000
```

### All options

```
usage: monte.py [-h] [--simulations N] [--csv PATH] [--seed INT]
                [--top K] [--cpu] [--chunk C] [--output-dir DIR]
                [--no-save] [--half-life DAYS]

  -n, --simulations N    Number of Monte Carlo simulations (default: 1000000)
  --csv PATH             Path to ToTo results CSV (default: ToTo.csv)
  --seed INT             Random seed for reproducibility
  --top K                Numbers to pick per set (default: 7)
  --cpu                  Force CPU mode
  --chunk C              Override auto chunk size
  --output-dir DIR       Results output directory (default: results/)
  --no-save              Skip saving results to disk
  --half-life DAYS       Recency decay half-life in days (default: 365)
```

---

## 🧮 Strategy Guide

| Set                | Name                        | Algorithm                                                                                      |
| ------------------ | --------------------------- | ---------------------------------------------------------------------------------------------- |
| **A — HOT**        | Gumbel-max simulation       | Top-7 most-frequent balls across all simulated draws                                           |
| **B — COLD**       | Contrarian                  | Top-7 least-frequent balls not in Set A                                                        |
| **C — MIXED**      | Balanced                    | 4 hottest from A + 3 coldest from B                                                            |
| **D — RECENCY**    | Decay-weighted simulation   | Same as A but recent draws count exponentially more (~1yr half-life)                           |
| **E — CO-OCCUR**   | Greedy co-occurrence clique | 7 balls that have appeared together in the same draw most often historically                   |
| **F — GENETIC**    | Genetic Algorithm           | 1,000-population GA over 400 generations, maximising `0.5×recency + 0.5×co-occurrence` fitness |
| **G — CONSENSUS**  | Multi-window                | Balls hot across last-100, last-300, last-500, and all-time draw windows                       |
| **★ — VOTE CHAMP** | Cross-strategy vote         | Balls appearing in the most strategies above                                                   |
| **H — LEAST VOTE** | Anti-consensus              | Balls appearing in the fewest strategies (ignored by all algorithms)                           |
| **I — MIX 4×3**    | Vote split                  | 4 least-vote balls + 3 vote-champion balls                                                     |

### Why chunked simulation is equally accurate

Ball count accumulation is linear — `bincount(all rows) = Σ bincount(chunk_i)` — so chunking is mathematically identical to a single monolithic run.  
Chunking only reduces peak VRAM from `n_sims × 49 × 4 bytes` to `chunk_size × 49 × 4 bytes`, preventing out-of-memory crashes at 100M+ simulations.

---

## 📈 Sample Output — 100 Million Simulations

```bash
--------------------------------------------------------------
  Hardware Summary
--------------------------------------------------------------
  CPU cores        : 20
  RAM total        : 119.6 GB
  RAM available    : 107.4 GB
  GPU              : NVIDIA (CUDA 13.0)
  VRAM total       : 119.6 GB
  VRAM free        : 100.1 GB

  Auto chunk size  :   20,000,000  sims
  Max recommended  : 1,000,000,000  sims  (GPU, time-bound)
--------------------------------------------------------------

[INFO] Backend     : GPU (CuPy / CUDA)
[INFO] Simulations : 100,000,000
[INFO] CSV         : ToTo.csv
[INFO] Historical  : 238 occurrences  (34 draws)
[INFO] Chunk size  : 20,000,000

[INFO] Running Gumbel simulation ...

  [████████████████████████████] 100.0%    100,000,000/100,000,000  done in 36.92s                            

[INFO] Computing advanced strategies ...
  [D] Recency-weighted simulation ... done
  [E] Co-occurrence greedy search  ... done
  [F] Genetic algorithm            ... done
  [G] Consensus windows            ... done
[INFO] Advanced strategies done in 1.03s


==============================================================
  Monte Carlo Weighted ToTo Simulation -- Results
==============================================================
  Simulations :  100,000,000
  Backend     : GPU -- CuPy / CUDA
  Elapsed     :      36.920 s
  Throughput  :       2.71 M sims/s
==============================================================

  Ball frequency table (top 20 by sim count + all cold balls):
  ----------------------------------------------------------------------------
   Ball   Hist.%      Sim.Count    Sim.%  Sets
  ----------------------------------------------------------------------------
     24    3.48%     23,403,794    3.34%  A:hot C:mixed D:recency E:co-occ G:consensus STAR I:mix4x3
     34    3.48%     23,399,779    3.34%  A:hot C:mixed D:recency F:genetic G:consensus STAR I:mix4x3
     35    3.14%     21,304,587    3.04%  A:hot C:mixed D:recency F:genetic G:consensus STAR I:mix4x3
     11    2.79%     19,161,670    2.74%  A:hot C:mixed F:genetic STAR
     31    2.79%     19,161,209    2.74%  A:hot D:recency F:genetic G:consensus STAR
     49    2.79%     19,161,072    2.74%  A:hot D:recency E:co-occ G:consensus STAR
     16    2.79%     19,156,697    2.74%  A:hot D:recency
     22    2.79%     19,154,159    2.74%  E:co-occ F:genetic G:consensus STAR
     32    2.79%     19,153,825    2.74%  D:recency G:consensus
     39    2.44%     16,959,929    2.42%  F:genetic
     27    2.44%     16,958,173    2.42%  H:least
     13    2.44%     16,956,493    2.42%  F:genetic
     15    2.44%     16,956,379    2.42%  H:least
     43    2.44%     16,954,528    2.42%  H:least
     37    2.44%     16,951,774    2.42%  E:co-occ
     19    2.44%     16,949,194    2.42%  H:least I:mix4x3
     36    2.09%     14,705,992    2.10%  H:least I:mix4x3
      5    2.09%     14,705,313    2.10%  H:least I:mix4x3
     45    2.09%     14,704,038    2.10%  H:least I:mix4x3
      2    2.09%     14,701,793    2.10%  E:co-occ
      7    0.35%      2,588,209    0.37%  B:cold C:mixed
     14    1.39%     10,020,129    1.43%  B:cold C:mixed
     23    1.05%      7,606,620    1.09%  B:cold C:mixed
     38    1.39%     10,021,703    1.43%  B:cold
     40    1.39%     10,018,355    1.43%  B:cold
     41    1.39%     10,019,414    1.43%  B:cold
     42    1.05%      7,605,757    1.09%  B:cold

==============================================================
  SET A — HOT      (top 7 most-frequent in sim)
==============================================================

  Main numbers  : [11, 16, 24, 31, 34, 35]
  Additional    : 49
  Full set      : [11, 16, 24, 31, 34, 35, 49]
  (pure hot — highest Gumbel-sim frequency)

==============================================================
  SET B — COLD     (top 7 least-frequent, not in A)
==============================================================

  Main numbers  : [7, 14, 23, 38, 40, 41]
  Additional    : 42
  Full set      : [7, 14, 23, 38, 40, 41, 42]
  (contrarian — lowest sim frequency)

==============================================================
  SET C — MIXED    (4 hottest from A  +  3 coldest from B)
==============================================================

  Main numbers  : [7, 11, 14, 23, 24, 34]
  Additional    : 35
  Full set      : [7, 11, 14, 23, 24, 34, 35]
  (balanced — 4 hot + 3 cold)

==============================================================
  SET D — RECENCY  (sim weighted by recency decay ~1yr half-life)
==============================================================

  Main numbers  : [16, 24, 31, 32, 34, 35]
  Additional    : 49
  Full set      : [16, 24, 31, 32, 34, 35, 49]
  (recent draws count 2x more per year)

==============================================================
  SET E — CO-OCCUR (greedy pairwise co-occurrence clique)
==============================================================

  Main numbers  : [2, 4, 22, 24, 30, 37]
  Additional    : 49
  Full set      : [2, 4, 22, 24, 30, 37, 49]
  (balls that appear together most in history)

==============================================================
  SET F — GENETIC  (GA maximising recency + co-occurrence fitness)
==============================================================

  Main numbers  : [11, 13, 22, 31, 34, 35]
  Additional    : 39
  Full set      : [11, 13, 22, 31, 34, 35, 39]
  (evolved combo — best blended score after 400 generations)

==============================================================
  SET G — CONSENSUS (hot across last-100/300/500/all-time windows)
==============================================================

  Main numbers  : [22, 24, 31, 32, 34, 35]
  Additional    : 49
  Full set      : [22, 24, 31, 32, 34, 35, 49]
  (stable picks across all time-window views)

==============================================================
  SET ★ — VOTE CHAMPION  (top 7 balls across all 7 strategies)
==============================================================

  Votes per ball in top-14 (most → least):
    ball 24 → 6        ball 34 → 6        ball 35 → 6        ball 31 → 5        ball 49 → 5     
    ball 11 → 4        ball 22 → 4        ball  7 → 2        ball 14 → 2        ball 16 → 2     
    ball 23 → 2        ball 32 → 2        ball  2 → 1        ball  4 → 1     

  Main numbers  : [11, 22, 24, 31, 34, 35]
  Additional    : 49
  Full set      : [11, 22, 24, 31, 34, 35, 49]
  (cross-strategy consensus — strongest overall signal)

==============================================================
  SET H — LEAST VOTE  (top 7 balls in the FEWEST strategies)
==============================================================

  Main numbers  : [5, 15, 19, 27, 36, 43]
  Additional    : 45
  Full set      : [5, 15, 19, 27, 36, 43, 45]
  (contrarian consensus — ignored by most strategies)

==============================================================
  SET I — MIX 4×3  (4 least-vote  +  3 vote-champion)
==============================================================

  Main numbers  : [5, 19, 24, 34, 35, 36]
  Additional    : 45
  Full set      : [5, 19, 24, 34, 35, 36, 45]
  (4 from Set H [5, 15, 19, 27]  +  3 from Set ★ [24, 34, 35])

==============================================================
```

---

## 📁 Output Files

Every run writes two files to `results/` (or your `--output-dir`):

**`sim_<timestamp>_n<count>.json`** — full summary:
```json
{
  "timestamp": "20260227_155802",
  "simulations": 100000000,
  "elapsed_seconds": 37.497,
  "sims_per_second": 2667093,
  "backend": "GPU (CuPy/CUDA)",
  "set_A_hot":           [1, 10, 12, 15, 17, 22, 31],
  "set_B_cold":          [3, 25, 45, 46, 47, 48, 49],
  "set_C_mixed":         [1, 3, 15, 22, 25, 31, 45],
  "set_D_recency":       [10, 15, 22, 31, 34, 37, 49],
  "set_E_cooccurrence":  [1, 6, 22, 31, 32, 34, 36],
  "set_F_genetic":       [1, 6, 22, 31, 32, 34, 37],
  "set_G_consensus":     [1, 10, 15, 22, 28, 31, 34],
  "set_STAR_champion":   [1, 6, 10, 15, 22, 31, 34],
  "set_H_least_vote":    [8, 9, 20, 23, 27, 30, 44],
  "set_I_mix4x3":        [1, 9, 20, 22, 27, 30, 31],
  "ball_frequencies": { "1": 15781086, "2": ... }
}
```

**`sim_<timestamp>_n<count>.csv`** — per-ball frequency table with set membership flags:
```
ball,historical_weight_pct,sim_count,sim_pct,set_A,set_B,set_C,...,set_STAR,set_H,set_I
1,2.2688,15781086,2.25,True,False,True,...,True,False,True
...
```

---

## 🤔 Disclaimer

This tool is for **analytical and educational purposes only**.  
Lottery outcomes are random — no statistical model can predict future draws. Play responsibly. 🍀
