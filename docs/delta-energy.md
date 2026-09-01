# Incremental (delta) energy for single-chi Monte Carlo moves

Status: designed and sized, not yet implemented. The measurements below are
real; the kernel work is not started.

## Why

`protMinReplicaCU` and `protMinCU` both advance the structure one chi angle at
a time, and both pay a full-protein energy evaluation for each such move.
Nothing in the sweep ladder plateaued at 8000 sweeps, and 1AKE has roughly 816
rotatable chi angles, so 8000 sweeps is about ten passes over its degrees of
freedom. The minimiser is evaluation-bound, and the evaluations are almost
entirely redundant: a single-chi move displaces about eleven atoms out of
several thousand.

This is a larger and safer win than tuning the energy kernel. The batched
reduction work bought 2.8% and did not scale with N. This scales with N by
construction.

## What a move actually reaches

Two sets matter, and the second is the whole story.

**A**, the displaced atoms: the members of one rotation group, about 11 atoms.

**B**, the atoms whose local dielectric changes. ProtCAD screens electrostatics
with a per-atom dielectric derived from hydration-shell occupancy, so moving an
atom changes `eps` for every atom within `r + 4.35 A` of it, and a changed
`eps` changes the screening of *every* electrostatic pair that atom belongs to,
out to the full 12 A cutoff. B is about 148 atoms, some 13x larger than A.

So solvation is roughly 93% of the delta's cost. A delta that tracked only the
moved atoms would be fast and wrong. This is the constraint that shapes the
whole design.

Measured by `scripts/delta_sizing.py`, results in `bench/delta_sizing.csv`:

| structure | atoms | pairs < 12 A | moved A | affected C | speedup |
|---|---|---|---|---|---|
| 1CRN | 648  | 96,243    | 8.9  | 107 | 3.0x  |
| 1UBQ | 1404 | 284,249   | 11.0 | 140 | 5.0x  |
| 2LZM | 2996 | 685,783   | 10.8 | 145 | 10.3x |
| 1AKE | 7814 | 1,899,584 | 10.6 | 148 | 26.4x |

`|C|` is essentially independent of N, so the speedup grows linearly with
system size. Note the honest shape of this: it is 3x on a small peptide, not
the orders of magnitude a naive moved-atoms-only argument suggests. It is worth
building because the interesting targets are large, not because it is dramatic
everywhere.

A caveat on small structures: at 648 atoms the GPU is already far from
saturated -- the kernel comment notes a 600-atom protein is about 19 warps on
1280 cores -- so removing arithmetic there may return little wall clock. The
delta should be validated for correctness on 1CRN and for speed on 1AKE.

## Formulation

Let `C = A u B` be the atoms whose energy inputs changed. Restricting the sum
to `i in C` counts pairs with both endpoints in C twice and pairs with one
endpoint in C once. With

    S1 = sum over i in C, all j != i
    S2 = sum over i in C, j in C, j != i

each changed pair is counted exactly once by `S1 - S2/2`. Both accumulators can
be produced by one pass of the existing tile kernel with an `inC` byte mask,
since it already emits per-atom partials. Then

    E_new = E_old - delta_old + delta_new

where each `delta` is `S1 - S2/2` plus the per-atom solvation terms for `i in
C`, which need no double-count correction.

Occupancy must be recomputed for `i in C` only. B is defined precisely so that
occupancy outside C is unchanged, so the resident occupancy array stays valid
for the untouched atoms and supplies `eps_j` when `j` is outside C.

## Build C dynamically, not statically

A conservative static set per rotation group -- one sphere about CA covering
the sidechain's whole sweep -- is tempting because it needs no per-move search,
and a superset is still exact. It was measured and rejected: it roughly doubles
the affected set (2.4x bloat) and gives back over half the speedup, 26.4x to
10.2x on 1AKE. The per-move search is cheap by comparison, about `|A|` times a
6 A neighbourhood, and can reuse the existing tile bounding boxes.

## Numerical hygiene

`E_new = E_old - delta_old + delta_new` accumulates rounding over thousands of
accepted moves. The minimiser must re-anchor with a full evaluation
periodically and assert the drift is small. This is also the natural
correctness test: run a delta chain and a full-evaluation chain from the same
seed and require they agree to tolerance at every re-anchor.

Exactness is "exact in exact arithmetic", the same standard `energyComputeBatch`
already documents, not bitwise: the restricted sum accumulates pairs in a
different order.

## Order of work

1. Device kernel to build C from a rotation group's members.
2. Restricted-sum variant of `kOccupancy` and `kEnergy` over an `inC` mask,
   emitting S1 and S2.
3. `energyComputeRotamerBatchDelta`, matching the existing rotamer batch API so
   `bestSidechainCandidateCU` can adopt it without restructuring.
4. Re-anchoring and the paired-chain drift test.
5. Only then wire it into `protMinReplicaCU`.
