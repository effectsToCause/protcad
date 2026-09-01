# Search strategy for sidechain packing

Status: analysis and measurement only. Nothing implemented.

## The problem has a name

What `protMinCU` and `protMinReplicaCU` do is sidechain packing, and it is not
a travelling salesman problem. TSP is a sequencing problem over one permutation
variable. Packing is a *labeling* problem: each flexible residue is a variable,
its rotamers are the labels, and the objective decomposes into self terms and
residue-pair terms. That makes it MAP inference on a pairwise Markov random
field, equivalently a weighted constraint satisfaction problem.

It is NP-hard, and it is also solved in practice many times a day by SCWRL4,
Rosetta and OSPREY. So the useful question is which of the standard machinery
transfers to protcad, and which does not.

## The part that is genuinely solved, and that protcad is not using

The single largest structural inefficiency is not the kernel and not the
redundant recomputation. It is that protcad evaluates *physics* on every move
when the field's answer is to evaluate physics once, into a table, and then
search over the table.

Given a rotamer library, precompute

    Eself(i, r)          rotamer r of residue i against backbone and fixed atoms
    Epair(i, r; j, s)    rotamer r of i against rotamer s of j

and every subsequent move costs a handful of table lookups instead of a
full-protein energy evaluation. For a single-residue rotamer change the cost is
one lookup per graph neighbour, about 37 on 1AKE, against 1.9 million pair
terms for a full evaluation. That is roughly five orders of magnitude per move,
and it dwarfs both the 2.8% kernel win and the 3x to 26x delta path.

Building the table is exactly the kind of work the GPU is for: it is
embarrassingly parallel and done once. Sizing it from `bench/packing_graph.csv`,
with edges = flexRes * avgDegree / 2 at the 12 A cutoff:

| structure | flexible residues | edges | table entries | fp32 size |
|---|---|---|---|---|
| 1CRN | 32  | 334  | 1.8M   | 7 MB   |
| 1UBQ | 64  | 1034 | 23M    | 92 MB  |
| 2LZM | 134 | 2345 | 54M    | 217 MB |
| 1AKE | 324 | 5929 | 146M   | 585 MB |

That fits the 3090 comfortably, and DEE pruning shrinks it further.

## Exact dynamic programming: right instinct, measured, and dead

Tree decomposition DP is the canonical *exact* packing method, TreePack being
the reference implementation, and it is the natural thing to reach for. Its
cost is `O(n * R^(treewidth+1))`, so it lives or dies on two numbers, both
measured by `scripts/packing_graph.py` into `bench/packing_graph.csv`.

Treewidth upper bounds by min-fill, swept over the interaction cutoff:

| structure | 4 A | 5 A | 6 A | 8 A | 12 A |
|---|---|---|---|---|---|
| 1CRN | 4  | 6  | 8  | 14 | 19 |
| 1UBQ | 10 | 13 | 17 | 27 | 39 |
| 2LZM | 14 | 23 | 23 | 38 | 56 |
| 1AKE | 12 | 26 | 37 | 48 | 82 |

At the 12 A nonbonded cutoff the residue graph has average degree 37. It is
nearly dense, and treewidth 82 with R = 157 is 10^182. Tightening the
interaction criterion to a 4 A contact still leaves treewidth 10 to 14, which
with this library is 10^24 to 10^33.

The reason the literature reports small treewidth is that those graphs are
built *after* aggressive rotamer pruning and with a contact criterion far
tighter than a nonbonded cutoff. protcad's library is a chi grid running to 492
rotamers for Arg and Lys, and R is the base of the exponent, so a fine grid is
much more damaging than it appears.

Conclusion: exact DP is not available at this rotamer resolution. It only
becomes worth revisiting if DEE cuts R to single digits, and even then the
treewidth has to be re-measured, not assumed.

## What survives

In order of payoff:

1. **Precomputed pair table.** The prerequisite for everything else, and the
   dominant win on its own.
2. **Dead-end elimination.** Provable, cheap, and it prunes rotamers that
   cannot appear in the global minimum. Routinely removes most of the library,
   which shrinks the table and may make stronger methods viable.
3. **Stochastic search over the table**, simulated annealing as in Rosetta, or
   belief propagation as in SCWRL4. Not provably optimal, but on a table these
   run in seconds where protcad currently spends minutes.

## What blocks it, and it is the same thing as before

The table formulation assumes the energy is a sum of self and pair terms.
protcad's is not. The local-dielectric model makes `eps_i` a function of the
hydration-shell occupancy of atom i, which depends on every neighbour at once,
and that `eps_i` then rescreens every electrostatic pair atom i belongs to.
This is the same many-body coupling measured in `docs/delta-energy.md`, where
it accounted for 93% of the delta's cost. It is not a detail; it is the reason
protcad cannot simply adopt the standard pipeline.

Note what is and is not decomposable. The occupancy itself *is* a sum over
neighbours, so it tabulates and updates incrementally without difficulty. What
breaks is the nonlinear `eps = f(occ)` and the pair mixing `mix(eps_i, eps_j)`
in the denominator, which prevent an electrostatic pair energy from being a
single number independent of the rest of the structure.

The field's standard answer is to use a pairwise-decomposable solvation model
for the search, which is precisely why Rosetta uses Lazaridis-Karplus, and to
rescore with the expensive model afterwards.

The answer better suited to protcad is self-consistent mean field. The
dielectric is a slowly varying field, so: freeze `eps`, build the table, solve
the now genuinely pairwise problem, recompute `eps` from the solution, repeat.
This is a classic packing method and it preserves protcad's own energy model
rather than substituting someone else's.

Under that scheme the delta work in `docs/delta-energy.md` is not wasted. It
becomes the exact rescoring and refinement pass that runs after the tabulated
search has done the heavy lifting, and it is what handles anything the table
cannot represent.

## Filling the GPU rather than reducing the work

A fair objection to all of the above: a 600-atom protein occupies about 19
warps on 1280 cores, so the GPU is mostly idle and the problem may be
under-decomposition rather than excess work. Three ways to fill it, in
increasing order of how well they survive scrutiny.

**More candidates per move.** Already done: `bestSidechainCandidateCU`
evaluates 32, and a 64-batch on 1ubq costs only 4.3x a single evaluation, so
throughput is roughly 15x. But best-of-K spends K evaluations to advance one
degree of freedom, and returns diminish quickly: for the 24-rotamer residues a
32-candidate batch is already close to exhaustive. This axis is saturated.

**More replicas.** Measured and rejected; see `protMinReplicaCU`. A population
advances K chains one step each rather than one chain K steps, and for
minimisation that is a loss.

**Simultaneous moves at different residues.** This is the textbook way to
parallelise a Markov chain exactly: colour the interaction graph and update a
whole colour class at once, since moves within a class are additive and
detailed balance is preserved. Measured by `scripts/move_parallelism.py` into
`bench/move_parallelism.csv`, as average residues per colour class:

| structure | contact 6 A | vdW only 12 A | exact 24.1 A |
|---|---|---|---|
| 1CRN | 5.3  | 2.0  | 1.1 |
| 1UBQ | 8.0  | 3.2  | 1.2 |
| 2LZM | 19.1 | 6.7  | 2.0 |
| 1AKE | 40.5 | 14.7 | 4.4 |

The exact criterion is 24.1 A because a move perturbs the dielectric within
6.05 A of every displaced atom, and a perturbed dielectric rescreens pairs out
to the full 12 A cutoff. Globular proteins are barely larger than that, so
under protcad's real energy almost nothing is independent: 1.1 residues per
class on 1CRN. Exact move parallelism does not exist here.

## The same term blocks all three routes

It is worth stating plainly, because it turned up independently three times:

* it is 93% of the cost of an incremental evaluation (`docs/delta-energy.md`),
* it makes the rotamer pair table inexact,
* and it collapses parallel move classes from 14.7 to 4.4 on 1ake, and to
  essentially 1 on small proteins.

All three are the local-dielectric solvation term. It is the single thing
standing between protcad and the standard packing machinery.

Which means freezing it does not buy one win, it buys all of them at once. With
`eps` held fixed over an outer iteration, the delta's affected set collapses
from about 148 atoms to the roughly 11 that actually moved:

| structure | delta, dielectric coupled | delta, dielectric frozen |
|---|---|---|
| 1CRN | 3.0x  | 36.4x  |
| 1UBQ | 5.0x  | 63.8x  |
| 2LZM | 10.3x | 138.7x |
| 1AKE | 26.4x | 368.6x |

and simultaneously the pair table becomes exact and the colour classes widen to
the 12 A column above. The self-consistent outer loop is therefore not a
concession to make one method work; it is the hinge the whole acceleration
strategy turns on, and its cost is a handful of outer iterations.

The one thing that must be established first is whether a frozen dielectric
ranks rotamers consistently with the full model over an outer iteration. That
is cheap to test and everything else depends on it.

## Suggested order

1. DONE, see above. Frozen far field with a recomputed near shell, not a global
   freeze; and do not drop solvation, which misranks in the near-native band.
2. Build the pair table on the GPU for a fixed dielectric.
3. DEE prune, then simulated annealing over the table.
4. Wrap in the self-consistent outer loop.
5. Exact refinement using the delta path.

## Measured: how much of the model can be frozen

`projects/rotamerRank.cc` answers step 1. For each flexible residue it samples
chi uniformly, evaluates the full per-term breakdown for every sample, and asks
how well a reduced score reproduces the full model's ordering. It also exports
the per-atom dielectric field via `updateDielectricsCU` and measures how far
that field actually moves when one sidechain rotates.

Two different claims have to be kept apart, because they are not the same
strength and they do not get the same answer:

* **Drop solvation** and rank on van der Waals. Much stronger.
* **Freeze the dielectric** across an outer iteration. Every solvation and
  electrostatic term is still evaluated; only `eps` lags. Much milder.

### Ranking on van der Waals alone does not work

`rotamerRank <pdb> 400 1`, seed 1:

| protein | regime | mean rho | top-1 | top-5 |
| --- | --- | ---: | ---: | ---: |
| 1UBQ | all samples | +0.991 | 20.0% | 53.8% |
| 1UBQ | near-native | +0.667 | 15.0% | 50.0% |
| 1CRN | all samples | +0.983 | 46.9% | 87.5% |
| 1CRN | near-native | +0.765 | 34.5% | 65.5% |
| 2LZM | all samples | +0.995 | 51.1% | 82.2% |
| 2LZM | near-native | +0.685 | 38.2% | 70.9% |

The rank correlation over all samples looks superb and means nothing. Uniform
chi sampling produces a mean van der Waals spread of about 1.2e7 kcal/mol, so
almost every sample is a catastrophic clash and every scoring function agrees
that garbage is garbage. That inflates rho without conferring any ability to
pack.

Restricted to the band within 10 kcal/mol of the best sampled van der Waals --
the conformations among which the packing is actually decided -- rho falls to
0.67 to 0.77 and the correct rotamer is picked 15 to 38 percent of the time.

So the intuition that van der Waals dominates is right about *where* it is
right: clashes are an overwhelming signal no other term can offset. But that is
the regime a minimiser leaves in its first few sweeps. In the near-native regime
where the answer is settled, the electrostatic spread (33 kcal/mol on 1UBQ)
exceeds the residual van der Waals spread, and dropping solvation misranks.
Adding the torsion term helps a little and is free, but does not rescue it.

### Freezing the dielectric does work, if the near shell is exempt

The weaker claim survives. Under a near-native rotamer change:

| protein | atoms | mean drift, environment | moved residue | env atoms >1% | >5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1CRN | 648 | 0.71% | 14.9% | 50.9 | 18.1 |
| 1UBQ | 1404 | 0.37% | 16.3% | 65.0 | 24.4 |
| 2LZM | 2996 | 0.19% | 22.5% | 65.7 | 24.2 |

The mean environment drift is a fraction of a percent and falls as the protein
grows, simply because the far field dilutes. But the worst environment atom
moves 38 to 46 percent, so a naive global freeze is not safe: it would be wrong
precisely at the contacts that decide the packing.

The useful result is the last two columns. **The set of environment atoms whose
dielectric moves at all is essentially constant in N** -- about 65 atoms above
one percent and 24 above five percent, unchanged from 648 to 2996 atoms. The
dielectric response is local, and it does not grow with the protein.

That reshapes the design. The right scheme is not "coupled" or "frozen" but
frozen far field with a recomputed near shell: hold `eps` fixed everywhere
except a small neighbourhood of the moved sidechain, and refresh that
neighbourhood exactly. The exempt set is roughly 65 atoms, well under half the
148-atom conservative dielectric-affected set B measured in
`docs/delta-energy.md`, and unlike set B it does not scale.

This lands between the two columns of the unlock table above rather than at the
frozen extreme. The frozen 368.6x on 1AKE was always an upper bound that assumed
`eps` never had to be touched; the honest expectation is roughly half the
dielectric work of the coupled path with the far field exact by construction.
The pair table is exact for every pair outside the near shell, which is almost
all of them, and the colour classes widen because move independence is now set
by the near shell rather than by the full 24.1 A dielectric radius.

Status: measured, not implemented.

## Implemented: the frozen dielectric is exact, not an approximation

`energyFreezeDielectric` / `energySetDielectricThaw` / `energyReleaseDielectric`
in `energy.cu`, with `protein::protFreezeDielectricCU` and
`protThawDielectricNearCU` as the callable front end, and
`tests/frozenDielectricTest.cc` as the guard.

What is frozen is the **occupancy** field, not `eps`. Every part of the solvent
model -- shell water count, water fraction, local dielectric, polar and
nonpolar solvation -- is a function of that one number, so freezing occupancy
freezes all of them consistently. Freezing `eps` alone would leave the
solvation terms disagreeing with the dielectric that screens them.

The measured section above framed this as a speed-for-accuracy trade. It is
not. An atom's occupancy is a sum over the atoms inside a hydration shell of
radius `r_i + effectiveWaterDiameter`, about 6.4 A for a heavy atom. A move can
therefore only perturb the occupancy of atoms within roughly that distance of a
moved atom's old or new position. Exempt those atoms and the held far field is
not an estimate: those atoms kept exactly the value the coupled model would
have computed, because nothing in their neighbourhood changed.

Measured, rotating one sidechain by 120 degrees on every chi:

| protein | atoms | 4 A | 6 A | 7 A | 8 A | 10 A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1CRN | 648 | -2.2e-1 | +3.6e-1 | -1.7e-3 | exact (185) | exact (268) |
| 1UBQ | 1404 | -1.9e0 | +1.7e-2 | -2.7e-2 | exact (351) | exact (497) |
| 2LZM | 2996 | +3.1e0 | -5.2e-3 | +3.4e-2 | exact (259) | exact (396) |
| 1AKE | 7814 | -5.0e0 | -2.5e-1 | -1.2e-2 | exact (324) | exact (536) |

Error is against the fully coupled model in kcal/mol; the exempt set size is in
parentheses. Freezing with no exemption at all costs 3.8 to 19.7 kcal/mol,
which is the size of the error a naive global freeze would have introduced, and
is consistent with the worst-atom drift of 38 to 46 percent measured above.

"Exact" here means exact in exact arithmetic, the standard `energyComputeBatch`
already documents. The residual is 1e-11 relative and comes from association,
not modelling: the spatial sort is rebuilt after a move, so a held occupancy and
a freshly computed one for the same undisturbed atom sum their neighbours in a
different order. The residual does not grow with the size of the held region.

Two consequences for the plan above.

First, **the self-consistent outer loop is not needed for correctness.** It was
introduced to make a frozen dielectric defensible; with a correct exemption it
is defensible without iterating. The frozen field is not a mean field being
converged, it is a cache of values that provably did not change.

Second, the exempt set is constant in N: 185, 351, 259, 324 atoms across
proteins from 648 to 7814 atoms. On 1AKE that is 4.1% of the structure. This is
the same locality the drift measurement found and it is what makes the delta
path worth building, since the dielectric-affected set was 93% of the delta cost
in `docs/delta-energy.md` and was sized there at 148 atoms *conservatively but
without exactness*. The honest exact figure is larger per move but N-independent
and, unlike set B, it buys exactness rather than trading it away.

The thaw set must be the union over the conformation before the move and the
one after it, since an atom matters if the move brings a neighbour into its
shell or takes one out. `protThawDielectricNearCU` accumulates for this reason.

Status: implemented and tested. Not yet wired into any minimiser -- the full
occupancy pass still runs, so this currently buys exactness and a correct
exemption set, not wall clock. Consuming it is the delta path's job.
