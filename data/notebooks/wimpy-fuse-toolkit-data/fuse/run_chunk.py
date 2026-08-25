#!/usr/bin/env python3
"""
run_chunk.py
------------
Run FUSE full-chain simulation for a single chunk of recoil spectra.
Intended for Slurm array execution, where each job handles one chunk.
"""

import os
import re
import shutil
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import fuse

import time
import random


# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

# Dataset-specific setup
FILE = "sbi_n300000_low_shm.csv"  # --------------------------------------- adjust for dataset
DATASET_NAME = Path(FILE).stem  # Extract name without extension

# Main dataset directory (permanent outputs)
DATASET_DIR = DATA_DIR / DATASET_NAME
CHUNKS_DIR = DATASET_DIR / "chunks"      # input chunk CSVs
S1S2_DIR   = DATASET_DIR / "s1s2"        # processed detector outputs
        
# Scratch directories (temporary simulation data per dataset)
SCRATCH_BASE = Path("/scratch/midway3/nreus/dark_matter_sbi") / DATASET_NAME
CSV_IN    = SCRATCH_BASE / "csv_input"     # one input CSV per chunk
FUSE_OUT  = SCRATCH_BASE / "fuse_data"     # FUSE output per chunk

# Create all dataset-level directories
for d in [RAW_DIR, CHUNKS_DIR, S1S2_DIR, CSV_IN, FUSE_OUT]:
    os.makedirs(d, exist_ok=True)

# -------------- CHUNKSIZE ---------------
FUSE_CHUNKSIZE = 100  # ------------------------------- adjust if wanted (e.g. to 50, 100, 200)


# --- FUSE input preparation ---
def prepare_fuse_input_dataframe(energies_all):
    n = len(energies_all)
    r = np.sqrt(np.random.uniform(0, 435600, n))
    theta = np.random.uniform(-np.pi, np.pi, n)
    xp, yp = r * np.cos(theta), r * np.sin(theta)
    zp = np.random.uniform(-1500, 0, n)
    times = np.arange(n) * 1e9

    return pd.DataFrame({
        "xp": xp, "yp": yp, "zp": zp,
        "xp_pri": xp, "yp_pri": yp, "zp_pri": zp,
        "ed": energies_all,
        "type": "neutron",
        "edproc": "hadElastic",
        "time": times,
        "eventid": np.arange(n),
        "trackid": np.ones(n),
        "parentid": np.zeros(n),
        "parenttype": "None",
        "creaproc": "None",
    })


# --- Run FUSE ---
def run_fuse_simulation(csv_input, fuse_chunksize, chunk_id):

    FUSE_OUT_CHUNK = FUSE_OUT / f"chunk_{chunk_id:04d}"
    os.makedirs(FUSE_OUT_CHUNK, exist_ok=True)

    st = fuse.context.xenonnt_fuse_full_chain_simulation(
        output_folder=str(FUSE_OUT_CHUNK),
        simulation_config="sr2_dev",
        corrections_version="global_v18",
    )
    st.set_config({
        "path": os.path.dirname(str(csv_input)),
        "file_name": os.path.basename(str(csv_input)),
        "n_interactions_per_chunk": fuse_chunksize,
    })
    run_number = re.sub(r"\D", "", os.path.basename(str(csv_input)))
    st.make(run_number, "microphysics_summary", progress_bar=True)
    st.make(run_number, "event_info", progress_bar=True)
    return st.get_df(run_number, "microphysics_summary"), st.get_df(run_number, "event_info")


# --- Matching microphysics ↔ event_info ---
def match_and_merge_events(micro_df, event_df, spectrum_lengths, chunk_id):
    micro_df = micro_df.sort_values("endtime").reset_index(drop=True)
    event_df = event_df.sort_values("endtime").reset_index(drop=True)
    micro_end, event_end = micro_df["endtime"].to_numpy(), event_df["endtime"].to_numpy()

    idxs = np.searchsorted(micro_end, event_end)
    idxs = np.clip(idxs, 1, len(micro_end) - 1)
    left_diff = np.abs(event_end - micro_end[idxs - 1])
    right_diff = np.abs(event_end - micro_end[idxs])
    nearest_micro = np.where(left_diff < right_diff, idxs - 1, idxs)

    combined = pd.DataFrame({
        "ed": micro_df["ed"].to_numpy(),
        "cs1": np.nan,
        "cs2": np.nan,
        "micro_endtime": micro_end,
        "event_endtime": np.nan,
        "time_diff_ns": np.nan,
        "event_index": np.nan,
    })

    matched = pd.DataFrame({
        "micro_index": nearest_micro,
        "event_index": np.arange(len(event_end)),
        "event_endtime": event_end,
        "time_diff_ns": event_end - micro_end[nearest_micro],
        "cs1": event_df["cs1"].to_numpy(),
        "cs2": event_df["cs2"].to_numpy(),
    })

    combined.loc[matched["micro_index"], ["cs1", "cs2", "event_endtime", "time_diff_ns", "event_index"]] = \
        matched[["cs1", "cs2", "event_endtime", "time_diff_ns", "event_index"]].to_numpy()

    # Add identifiers
    n_micro = len(micro_df)  
    cum_lengths = np.cumsum([0] + list(spectrum_lengths))
    micro_indices = np.arange(n_micro)

    spectrum_id = np.searchsorted(cum_lengths, micro_indices, side="right") - 1
    local_id = micro_indices - cum_lengths[spectrum_id]

    combined["chunk_id"] = chunk_id
    combined["spectrum_id"] = spectrum_id
    combined["local_id"] = local_id

    return combined


# --- Process one chunk ---
def process_chunk(chunk_id):
    chunk_path = CHUNKS_DIR / f"chunk_{chunk_id:04d}.csv"
    FUSE_OUT_CHUNK = FUSE_OUT / f"chunk_{chunk_id:04d}"
    
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk file {chunk_path} not found")

    # Load spectra
    spectra = []
    with open(chunk_path, "r") as f:
        for line in f:
            s = line.strip()
            spectra.append(np.fromstring(s, sep=",") if s else np.array([]))
    spectrum_lengths = [len(s) for s in spectra]
    merged_energies = np.concatenate([s for s in spectra if len(s) > 0])

    df_input = prepare_fuse_input_dataframe(merged_energies)
    csv_input = CSV_IN / f"chunk_{chunk_id:04d}.csv"
    df_input.to_csv(csv_input, index=False)

    micro_df, event_df = run_fuse_simulation(csv_input, FUSE_CHUNKSIZE, chunk_id)
    combined = match_and_merge_events(micro_df, event_df, spectrum_lengths, chunk_id)

    out_path = S1S2_DIR / f"s1s2_chunk_{chunk_id:04d}.csv"
    combined.to_csv(out_path, index=False)

    print(f"[Chunk {chunk_id}] Saved {len(combined)} rows → {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", type=int, required=True)
    args = parser.parse_args()
    process_chunk(args.chunk_id)
    # Delete trash files (e.g. fuse_output) manually after the simulation
    
