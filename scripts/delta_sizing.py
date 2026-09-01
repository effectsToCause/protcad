#!/usr/bin/env python3
"""Size the affected set of an incremental (delta) energy evaluation.

A single-chi Monte Carlo move moves only the atoms distal to one bond, so in
principle its energy change can be evaluated without touching the whole
protein.  The question this script answers is how much of the protein a move
actually reaches once the local-dielectric solvation model is accounted for,
because that decides whether a delta path is worth building.

Two sets matter:

  A  the atoms the move displaces.  Small: about 11 atoms.

  B  the atoms whose hydration-shell occupancy changes, and therefore whose
     local dielectric changes.  Every atom within shellRadius = r + 4.35 A of
     a displaced atom qualifies.  A changed dielectric changes the screening
     of every electrostatic pair that atom takes part in, so B is not a
     correction to A, it is the dominant cost: about 148 atoms, some 13x
     larger than A.

The changed-pair work is therefore |A u B| * (average neighbours within the
12 A cutoff), against N * avgNbr / 2 for a full evaluation.  Since |A u B| is
essentially independent of N, the speedup grows linearly with system size.

Also compared is a static conservative alternative: instead of finding B per
move, precompute one sphere per rotation group that covers the whole sidechain
sweep.  A superset is still exact, and it needs no per-move neighbour search.
It is reported here because it looked attractive and is not: it roughly
doubles the affected set and gives back over half the speedup.

Input files are PDBs as protcad loads them, i.e. with hydrogens present, which
matters because hydrogens are most of the atom count and all of them occlude.
Generate them with:  protMinRep 0 in.pdb out_h.pdb

Usage:  delta_sizing.py <structure_h.pdb> [...]
"""

import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

CUTOFF = 12.0          # nonbonded outer cutoff, energyParams::cutoff
SHELL = 1.7 + 4.35     # typical heavy-atom shell radius, r + effectiveWaterDiameter

# Backbone atoms are not moved by a sidechain chi rotation.
BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA", "1HA", "2HA", "3HA", "HN"}


def load(path):
    xyz, res, name = [], [], []
    for line in open(path):
        if line.startswith("ATOM"):
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            res.append((line[21], int(line[22:26])))
            name.append(line[12:16].strip())
    return np.array(xyz), res, name


def analyse(path):
    X, res, name = load(path)
    n = len(X)
    tree = cKDTree(X)

    total_pairs = (tree.count_neighbors(tree, CUTOFF) - n) // 2
    avg_nbr = 2.0 * total_pairs / n

    sidechain = defaultdict(list)
    index_of = {}
    for i, r in enumerate(res):
        index_of[(r, name[i])] = i
        if name[i] not in BACKBONE:
            sidechain[r].append(i)

    moved, dynamic, static = [], [], []
    for r, members in sidechain.items():
        # Fewer than three sidechain atoms means nothing worth rotating.
        if len(members) < 3:
            continue
        ca = index_of.get((r, "CA"))
        if ca is None:
            continue

        a = np.array(members)
        b = set()
        for hit in tree.query_ball_point(X[a], SHELL):
            b.update(hit)
        moved.append(len(a))
        dynamic.append(len(b | set(members)))

        sweep = np.linalg.norm(X[a] - X[ca], axis=1).max()
        static.append(len(tree.query_ball_point(X[ca], sweep + SHELL)))

    a_mean = float(np.mean(moved))
    d_mean = float(np.mean(dynamic))
    s_mean = float(np.mean(static))
    return {
        "atoms": n,
        "pairs": total_pairs,
        "avgNbr": avg_nbr,
        "movedA": a_mean,
        "dynamicC": d_mean,
        "staticC": s_mean,
        "dynamicSpeedup": total_pairs / (d_mean * avg_nbr),
        "staticSpeedup": total_pairs / (s_mean * avg_nbr),
    }


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    print("structure,atoms,pairs,avgNbr,movedA,dynamicC,staticC,dynamicSpeedup,staticSpeedup")
    for path in argv[1:]:
        r = analyse(path)
        label = path.split("/")[-1].replace("_h.pdb", "")
        print(f"{label},{r['atoms']},{r['pairs']},{r['avgNbr']:.1f},{r['movedA']:.1f},"
              f"{r['dynamicC']:.0f},{r['staticC']:.0f},"
              f"{r['dynamicSpeedup']:.1f},{r['staticSpeedup']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
