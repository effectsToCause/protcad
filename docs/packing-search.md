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

1. Confirm on one protein that a frozen-dielectric pairwise surrogate ranks
   rotamers consistently with the full model. If it does not, the whole
   approach needs rethinking and it is better to learn that first.
2. Build the pair table on the GPU for a fixed dielectric.
3. DEE prune, then simulated annealing over the table.
4. Wrap in the self-consistent outer loop.
5. Exact refinement using the delta path.
