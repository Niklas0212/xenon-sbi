#!/usr/bin/env python3
"""
submit_er_chunks.py
-------------------
Checks which ER FUSE chunk files are missing and submits jobs to Slurm.
"""

import os
from time import sleep
from pathlib import Path

# ---------------- CONFIG ----------------

BASE_DIR = Path("/home/nreus/workspace/dark_matter_sbi/bg")

#DATASET_DIR = BASE_DIR / "fuse_er_band"
#EVENTS_DIR = DATASET_DIR / "events"

SCRIPT_PATH = BASE_DIR / "run_er_chunk.py"
TEMPFILE = BASE_DIR / "temp_submit_er.sh"

LOG_DIR = Path("/home/nreus/workspace/cluster/slurmlogs")

# Singularity
SING_IMG = "/project2/lgrandi/xenonnt/singularity-images/xenonnt-el7.2025.07.2.simg"

# SLURM SETTINGS
PARTITION = "caslake"
QOS = "caslake"
ACCOUNT = "pi-lgrandi"
TIME = "1-12:00:00"
MEM_PER_CPU = 16000
CPUS_PER_TASK = 1
DELAY = 1

# ER production size
EXPECTED_CHUNKS = 100   # 100 × 10k = 1M ERs

# ---------------- FIND MISSING ----------------
#missing = []
#
#for chunk_id in range(EXPECTED_CHUNKS):
#    out_file = EVENTS_DIR / f"s1s2_chunk_{chunk_id:04d}.csv"
#    if not out_file.exists():
#        missing.append(chunk_id)
#
#print(f"Found {len(missing)} missing ER chunks:")
#print(missing, "\n")

# ---------------- SUBMIT ----------------
for chunk_id in range(EXPECTED_CHUNKS):
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
    print(f"Submitting chunk {chunk_id}")
    sleep(DELAY)

print("✅ Done submitting ER chunks.")
