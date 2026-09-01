#!/usr/bin/env python3
"""Energy-vs-sweeps falloff for protMinRep.

For each structure, run a ladder of sweep budgets from the same input and
record ending energy and wall time. Each run restarts from the input PDB, so
E(sweeps) is a genuine curve over the deliverable rather than an internal
trace. replicas is held at 1: protMinRep's own header notes a population is
mildly harmful for minimisation.
"""
import os, re, subprocess, sys, csv

ROOT = "/home/doughp1/protcad"
BIN = f"{ROOT}/bin/protMinRep"
WORK = "/var/tmp/ct"
SWEEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000]
SEEDS = [1, 2, 3]
PROTS = ["1CRN", "1UBQ", "1LYZ", "2LZM", "1AKE"]

pat = {k: re.compile(rf"{v}: *([-\d.e+]+)") for k, v in
       {"start": "Starting Energy", "end": "Ending Energy"}.items()}
wpat = re.compile(r"Wall: *([-\d.e+]+)")


def run(prot, sweeps, seed):
    env = dict(os.environ, PROTCADDIR=ROOT, PROTCAD_MC_SEED=str(seed))
    out = subprocess.run(
        [BIN, str(sweeps), f"{WORK}/{prot}.pdb", f"{WORK}/o_{prot}_{sweeps}_{seed}.pdb"],
        cwd=WORK, env=env, capture_output=True, text=True, timeout=3600).stdout
    return (float(pat["start"].search(out).group(1)),
            float(pat["end"].search(out).group(1)),
            float(wpat.search(out).group(1)))


fh = open(f"{WORK}/ladder.csv", "w", buffering=1)
w = csv.writer(fh)
w.writerow(["protein", "sweeps", "seed", "startE", "endE", "wall_s"])
for p in PROTS:
    for s in SWEEPS:
        for sd in SEEDS:
            s0, s1, wall = run(p, s, sd)
            w.writerow([p, s, sd, s0, s1, wall])
            print(f"{p:6s} {s:6d} seed{sd} E={s1:14.4f} wall={wall:8.3f}s", flush=True)
