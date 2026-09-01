#!/usr/bin/env python3
"""Measure how many Monte Carlo moves can be made simultaneously and exactly.

The Markov chain in protMinCU and protMinReplicaCU is serial: one chi angle per
energy evaluation.  The textbook way to parallelise it without approximation is
the same trick used for lattice models: partition the variables into classes
that do not interact, then update a whole class at once.  Moves within a class
are exactly additive, so detailed balance is preserved and no accuracy is
given up.  Finding those classes is graph colouring on the residue interaction
graph.

The question this script answers is how large the colour classes actually are,
because that is the available parallelism.

Everything turns on which separation makes two moves independent, and three are
reported:

  contact  (6 A)     what independence would require if only direct sidechain
                     contact mattered.  Not a valid criterion, shown as the
                     optimistic bound.

  vdW-only (12 A)    what independence would require if the energy were purely
                     pairwise with the nonbonded cutoff.  This is the criterion
                     that would apply under a frozen dielectric.

  exact    (24.1 A)  what independence actually requires for protcad's energy.
                     A move perturbs the dielectric of everything within 6.05 A
                     of a displaced atom, and a perturbed dielectric rescreens
                     pairs out to the full 12 A cutoff, so two moves interact
                     unless their sidechains are more than 12 + 2 * 6.05 A
                     apart.

The gap between the last two is the cost of the many-body solvation term,
expressed as lost parallelism rather than as arithmetic.  Globular proteins are
not much larger than 24 A across, so under the exact criterion almost nothing
is independent.

Colouring is greedy in descending degree order, which gives an upper bound on
the chromatic number and therefore a conservative estimate of class size.

Usage:  move_parallelism.py <structure_h.pdb> [...]
"""

import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA", "1HA", "2HA", "3HA", "HN"}

FLEXIBLE = {"SER", "VAL", "THR", "CYS", "LEU", "ILE", "ASP", "ASN", "PHE",
            "TYR", "TRP", "HIS", "MET", "GLU", "GLN", "LYS", "ARG"}

SHELL = 1.7 + 4.35     # r + effectiveWaterDiameter
CUTOFF = 12.0

CRITERIA = (
    ("contact", 6.0),
    ("vdWonly", CUTOFF),
    ("exact", CUTOFF + 2 * SHELL),
)


def greedy_colouring(adj, nodes):
    colour = {}
    for v in sorted(nodes, key=lambda x: -len(adj[x])):
        used = {colour[u] for u in adj[v] if u in colour}
        c = 0
        while c in used:
            c += 1
        colour[v] = c
    return colour


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
                if restype[k] in FLEXIBLE and len(sidechain[k]) >= 2]
    flexible_set = set(flexible)
    tree = cKDTree(X)

    rows = []
    for label, sep in CRITERIA:
        adj = defaultdict(set)
        for k in flexible:
            hits = set()
            for h in tree.query_ball_point(X[sidechain[k]], sep):
                hits.update(h)
            for j in hits:
                other = res[j]
                if other != k and other in flexible_set and name[j] not in BACKBONE:
                    adj[k].add(other)
                    adj[other].add(k)
        colour = greedy_colouring(adj, flexible)
        n_colours = max(colour.values()) + 1 if colour else 0
        rows.append((label, sep, n_colours, len(flexible) / n_colours))
    return flexible, rows


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    print("structure,flexRes,criterion,separation,colours,avgClassSize")
    for path in argv[1:]:
        flexible, rows = analyse(path)
        label = path.split("/")[-1].replace("_h.pdb", "")
        for name_, sep, n, size in rows:
            print(f"{label},{len(flexible)},{name_},{sep:.1f},{n},{size:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
