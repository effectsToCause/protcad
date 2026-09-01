#!/usr/bin/env python3
"""Measure the residue interaction graph that any packing search must solve.

Side-chain packing is MAP inference on a pairwise Markov random field: each
flexible residue is a variable, its rotamers are the labels, and the energy
decomposes into self terms and residue-pair terms.  The problem is NP-hard in
general but is solved routinely in practice, so the question is not whether it
is tractable but which structural property makes it tractable here.

Two properties decide that, and this script measures both.

  treewidth   Exact dynamic programming over a tree decomposition -- the
              TreePack approach -- costs O(n * R^(treewidth+1)).  It is the
              canonical exact method, so it is worth knowing the actual number
              rather than assuming protein graphs are sparse.

  R           Rotamers per residue, read from protcad's own library.  It enters
              the DP cost as the base of that exponent, so a fine chi grid is
              far more damaging than it looks.

The treewidth reported is an upper bound from the min-fill heuristic.  Exact
treewidth is itself NP-hard, but min-fill is the standard bound and is more
than good enough to tell a tractable graph from an intractable one.

The interaction cutoff is swept because the answer depends entirely on it: a
12 A nonbonded cutoff makes the residue graph nearly dense, while a tight
contact criterion sparsifies it.  Sweeping shows whether sparsification can
rescue exact DP.  It cannot, which is the point.

Input files are PDBs as protcad loads them, with hydrogens.  Generate with:
    protMinRep 0 in.pdb out_h.pdb

Usage:  packing_graph.py <structure_h.pdb> [...]
"""

import itertools
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

# Backbone atoms do not move when a sidechain rotamer changes.
BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA", "1HA", "2HA", "3HA", "HN"}

# Rotamer counts from data/rotamerLib.  ALA, GLY and PRO carry no rotatable
# chi in this library and so are not variables in the packing problem.
ROTAMERS = {
    "SER": 24, "VAL": 24, "THR": 24, "CYS": 24,
    "LEU": 60, "ILE": 60, "ASP": 60, "ASN": 60, "PHE": 60, "TRP": 60, "HIS": 60,
    "TYR": 78,
    "MET": 168, "GLU": 168, "GLN": 168,
    "LYS": 492, "ARG": 492,
    "ALA": 1, "GLY": 1, "PRO": 1,
}

CUTOFFS = (4.0, 5.0, 6.0, 8.0, 12.0)


def min_fill_treewidth(adj, nodes):
    """Upper bound on treewidth by min-fill elimination ordering."""
    work = {v: set(adj[v]) for v in nodes}
    live = set(nodes)
    width = 0
    while live:
        best, best_fill = None, None
        for v in live:
            nb = work[v] & live
            fill = sum(1 for a, b in itertools.combinations(nb, 2) if b not in work[a])
            if best_fill is None or fill < best_fill:
                best, best_fill = v, fill
        nb = work[best] & live
        width = max(width, len(nb))
        for a, b in itertools.combinations(nb, 2):
            work[a].add(b)
            work[b].add(a)
        live.discard(best)
    return width


def load(path):
    xyz, res, name, restype = [], [], [], {}
    for line in open(path):
        if line.startswith("ATOM"):
            key = (line[21], int(line[22:26]))
            restype[key] = line[17:20].strip()
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            res.append(key)
            name.append(line[12:16].strip())
    return np.array(xyz), res, name, restype


def analyse(path):
    X, res, name, restype = load(path)
    sidechain = defaultdict(list)
    for i, key in enumerate(res):
        if name[i] not in BACKBONE:
            sidechain[key].append(i)

    flexible = [k for k in sidechain
                if ROTAMERS.get(restype[k], 1) > 1 and len(sidechain[k]) >= 2]
    tree = cKDTree(X)
    counts = [ROTAMERS[restype[k]] for k in flexible]

    flexible_set = set(flexible)
    rows = []
    for cutoff in CUTOFFS:
        adj = defaultdict(set)
        for k in flexible:
            hits = set()
            for h in tree.query_ball_point(X[sidechain[k]], cutoff):
                hits.update(h)
            for j in hits:
                other = res[j]
                if other != k and name[j] not in BACKBONE and other in flexible_set:
                    adj[k].add(other)
                    adj[other].add(k)
        degree = np.mean([len(adj[k]) for k in flexible]) if flexible else 0.0
        width = min_fill_treewidth(adj, flexible)
        rows.append((cutoff, degree, width))

    return flexible, counts, rows


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    print("structure,flexRes,avgRotamers,maxRotamers,cutoff,avgDegree,treewidthBound,log10DPcost")
    for path in argv[1:]:
        flexible, counts, rows = analyse(path)
        label = path.split("/")[-1].replace("_h.pdb", "")
        r_mean, r_max = float(np.mean(counts)), max(counts)
        for cutoff, degree, width in rows:
            cost = (width + 1) * np.log10(r_mean)
            print(f"{label},{len(flexible)},{r_mean:.0f},{r_max},{cutoff:.0f},"
                  f"{degree:.1f},{width},{cost:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
