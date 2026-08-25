#!/usr/bin/env python3
"""
run_er_chunk.py
---------------
Generate ER events with FUSE 
One Slurm job = one ER chunk.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import fuse


# ---------------- CONFIG ----------------
#BASE_DIR = Path("/home/nreus/workspace/dark_matter_sbi/bg/fuse_er_band")
BASE_DIR = Path("/scratch/midway3/nreus/ers")

CSV_IN     = BASE_DIR/ "csv_input"
FUSE_OUT   = BASE_DIR / "fuse_output"
EVENTS_OUT = BASE_DIR / "events"

for d in [CSV_IN, FUSE_OUT, EVENTS_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# Actual ERs per job
N_EVENTS = 100_000

# FUSE internal chunking (leave small!)
FUSE_CHUNKSIZE = 200

# Max gamma energy (keVee)
GAMMA_MAX = 15.0

# ---------------- ER INPUT ----------------
def prepare_er_input(n_events, seed):
    rng = np.random.default_rng(seed)

    # Uniform in TPC
    r = np.sqrt(rng.uniform(0, 435600, n_events))
    theta = rng.uniform(-np.pi, np.pi, n_events)
    xp = r * np.cos(theta)
    yp = r * np.sin(theta)
    zp = rng.uniform(-1500, 0, n_events)

    # Flat ER energy spectrum (keVee)
    energies = rng.uniform(0.0, GAMMA_MAX, n_events)

    return pd.DataFrame({
        "xp": xp, "yp": yp, "zp": zp,
        "xp_pri": xp, "yp_pri": yp, "zp_pri": zp,
        "ed": energies,
        "type": "gamma",
        "edproc": "None",
        "time": np.arange(n_events) * 1e9,
        "eventid": np.arange(n_events),
        "trackid": np.ones(n_events),
        "parentid": np.zeros(n_events),
        "parenttype": "None",
        "creaproc": "None",
    })


# ---------------- MAIN ----------------
def main(chunk_id):
    seed = 10_000 + chunk_id

    df = prepare_er_input(N_EVENTS, seed)
    csv_path = CSV_IN / f"er_chunk_{chunk_id:04d}.csv"
    df.to_csv(csv_path, index=False)

    out_chunk = FUSE_OUT / f"chunk_{chunk_id:04d}"
    out_chunk.mkdir(exist_ok=True)

    st = fuse.context.xenonnt_fuse_full_chain_simulation(
        output_folder=str(out_chunk),
        simulation_config="sr2_dev",
        corrections_version="global_v18",
    )

    st.set_config({
        "path": str(CSV_IN),
        "file_name": csv_path.name,
        "n_interactions_per_chunk": FUSE_CHUNKSIZE,
    })

    run_number = f"999{chunk_id:04d}"
    st.make(run_number, "event_info", progress_bar=True)

    event_df = st.get_df(run_number, "event_info")

    # Save only what you need
    s1s2 = event_df[["cs1", "cs2"]]
    out_file = EVENTS_OUT / f"s1s2_chunk_{chunk_id:04d}.csv"
    s1s2.to_csv(out_file, index=False)

    print(f"[chunk {chunk_id}] saved {len(s1s2)} ER events → {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", type=int, required=True)
    args = parser.parse_args()

    main(args.chunk_id)
