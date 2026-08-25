#!/usr/bin/env python3
"""
resubmit_missing_chunks.py
--------------------------
Checks which FUSE S1S2 chunk files are missing and automatically
resubmits jobs for those chunks to Slurm.


Usage:
1. Check file and expected number of chunks in this script
2. Check file (and potentially fuse chunksize) in run_chunk.py
3. ./py_submit_chunks.py

"""

import os
from time import sleep
from pathlib import Path

# --- CONFIGURATION ---
DATASET = "sbi_n300000_low_shm" # ------------------------------------- adjust for dataset
BASE_DIR = Path("/home/nreus/workspace/dark_matter_sbi")
DATASET_DIR = BASE_DIR / "data" / DATASET
S1S2_DIR = DATASET_DIR / "s1s2"

# Where temp submit script is written
TEMPFILE = BASE_DIR / "scripts" / "temp_submit.sh"
LOG_DIR = Path("/home/nreus/workspace/cluster/slurmlogs")
SCRIPT_PATH = BASE_DIR / "scripts" / "run_chunk.py"
SING_IMG = "/project2/lgrandi/xenonnt/singularity-images/xenonnt-el7.2025.07.2.simg"

# SLURM SETTINGS
PARTITION = "caslake"
QOS = "caslake"
ACCOUNT = "pi-lgrandi"
TIME = "1-12:00:00"
MEM_PER_CPU = 16000
CPUS_PER_TASK = 1
DELAY = 10   # ---------------------------------------------------------- adjust if wanted

# --- Find missing chunks ---
expected_chunks = 200  # ------------------------------------------------- adjust for dataset
missing = []

for chunk_id in range(expected_chunks):
    out_file = S1S2_DIR / f"s1s2_chunk_{chunk_id:04d}.csv"
    if not out_file.exists():
        missing.append(chunk_id)

print(f"Found {len(missing)} missing chunks: {missing}\n")

# --- Submit missing chunks ---
for chunk_id in missing:
    with open(TEMPFILE, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=fuse_chunk_{chunk_id}\n")
        f.write(f"#SBATCH --output={LOG_DIR}/fuse_chunk_{chunk_id}-%j.out\n")
        f.write(f"#SBATCH --ntasks=1\n")
        f.write(f"#SBATCH --cpus-per-task={CPUS_PER_TASK}\n")
        f.write(f"#SBATCH --mem-per-cpu={MEM_PER_CPU}\n")
        f.write(f"#SBATCH --time={TIME}\n")
        f.write(f"#SBATCH --account={ACCOUNT}\n")
        f.write(f"#SBATCH --partition={PARTITION}\n")
        f.write(f"#SBATCH --qos={QOS}\n")
        f.write("\n")
        f.write("module load singularity\n")
        f.write("\n")
        f.write(f"singularity exec --bind /project2,/project {SING_IMG} "
                f"python3 -u {SCRIPT_PATH} --chunk_id {chunk_id}\n")
        
    os.system(f"sbatch {TEMPFILE}")
    print(f"Resubmitted missing chunk {chunk_id}")
    sleep(DELAY)

print("✅ Done resubmitting missing chunks.")
