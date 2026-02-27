# FractalRL (Discrete) — Long-Horizon RL Stress Tests with Fractal Structure

This folder contains a small, reproducible RL benchmark where the environment’s structure scales in a fractal-like way. The point is not “pretty visuals.” The point is to create **controlled long-horizon difficulty** where small policy mistakes compound, and where you can **measure stability as horizon and structure depth increase**.

If you care about agent reliability over extended interactions (not just short episodic wins), this is meant to be a clean testbed.

## What this is

- A **discrete RL environment** whose transition structure is generated from a fractal-style construction.
- A simple pipeline for running training / evaluation and comparing behavior as you increase:
  - horizon length
  - fractal depth (structure complexity)
  - stochasticity / noise (if enabled in your config)

## Why it matters

Most RL benchmarks are either:
- short-horizon, where errors do not compound much, or
- messy, where it is hard to isolate why things broke.

This benchmark is designed to make long-horizon behavior fail in a way that is:
- **repeatable**
- **measurable**
- **easy to scale up** by turning a small number of knobs

Typical failure modes you can surface here:
- instability as horizon increases
- brittle policies that look fine at low depth but collapse at higher depth
- exploration methods that diverge sharply as structure complexity rises

## What to look at first

If you only read one thing:
- Start with the environment definition and how “depth” changes the task.
- Then look at the evaluation script (how it sweeps depth / horizon and logs results).

## Quick start

### 1) Create an environment
'''bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt'''

If you do not have a requirements.txt here, install the basics:

pip install numpy matplotlib
2) Run a minimal sanity check

Run the simplest script you have in this folder that:

instantiates the env

steps randomly for a few episodes

prints rewards and episode length

Examples (rename to match your files):

python sanity_check_env.py
# or
python -m fractal_rl_discrete.sanity_check
3) Run training

Examples (rename to match your files):

python train.py --depth 3 --horizon 200 --seed 0
python train.py --depth 5 --horizon 400 --seed 0
4) Run evaluation sweeps

The most important output is a sweep across increasing horizon/depth.

Examples (rename to match your files):

python eval.py --depths 2 3 4 5 --horizons 100 200 400 800 --seeds 0 1 2
# or
python sweep.py --config configs/sweep.yaml
Metrics to log (recommended)

At minimum:

average return

success rate (if you have a success definition)

variance across seeds (stability)

failure rate (crashes, timeouts, or degenerate behavior)

If you want one clean “reliability” scalar:

performance drop as horizon increases (slope)

performance drop as depth increases (slope)

seed sensitivity (std dev across seeds)

Reproducibility

Recommended defaults:

always log: seed, depth, horizon, and any noise parameters

write results to CSV so comparisons are easy

save plots with filenames that include depth/horizon/seed

Example output layout:

results/
  runs.csv
  plots/
    return_vs_horizon_depth3.png
    return_vs_depth_horizon400.png
Repository structure (typical)

This folder is intended to stay small. A clean structure looks like:

fractal_rl_discrete/
  env/                  # environment definition
  agents/               # baseline agents (if included)
  configs/              # sweep configs
  train.py              # training entrypoint
  eval.py               # evaluation entrypoint
  utils.py              # logging, seeding, helpers
  results/              # generated outputs (optional)
How to extend this (ideas that map to real agent work)

If you want to push this toward “frontier agent reliability” questions:

Add long-horizon distractors (more ways to fail without immediate penalty)

Add partial observability so memory matters

Add a multi-agent variant where coordination is required

Define an explicit “compounding error” metric (how quickly policies drift off track)

Contact

If you’re reading this because you care about long-horizon RL evaluation, feel free to reach out.

Rohan Nagaram, rohannagaram@gmail.com
