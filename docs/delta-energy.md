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

## Where this landed

Steps 1-5 are done, as `energyComputeBatchDelta`, `bestSidechainCandidateDeltaCU`
and the chain in `protMinCU`. Three things came out differently from the plan.

The restricted-sum kernels were not written. `kEnergy` and `kOccupancy` were
already both candidate-aware and changed-set aware, and `kTorsion` already took
a torsion list and a candidate count together, so the batch delta is those
kernels launched over `nCand x C` rather than new machinery.

The chain needs `energyRefreezeDielectric`, which is not in the plan above and
without which none of it reaches wall clock. Carrying a held field across
accepted moves means refreshing it after each one, and freezing again from
coordinates recomputes occupancy over every pair -- most of a full evaluation,
capping the whole approach near 2x. It does not need recomputing: after a delta
the resident field already is the new conformation's, so the refresh is a
snapshot.

The bloat estimate above was for a static superset over a full sidechain sweep,
and it turns out to describe a 32-candidate batch too, because a batch of random
conformations covers that sweep. Measured union changed set: 37% of 1crn, 30% of
1ubq, 21% of 2lzm, 6.3% of 1ake. That is the whole story of the result. Over 200
moves the chain runs 1.11x on 1crn, 1.43x on 1ubq, 2.19x on 2lzm and 3.49x on
1ake, and the delta is only worth having above roughly two thousand atoms.

The host-side floor bit again, in the same way and for the same reason as the
single-move delta. Rebuilding all N coordinates from the atom pointers once per
candidate grows with the whole structure while a sidechain move does not, and it
was the dominant cost of both the delta and the full path. Building candidates
by refreshing only the moved residue took 1ake from 2.33x to 3.94x per batch.

Drift over 64 accepted moves is 5.5e-4 to 2.8e-3 kcal/mol against totals of
several hundred to a thousand, and the carried energy finishes within 8e-4 of a
coupled evaluation. The re-anchor interval of 64 is comfortable, not tight.

`protMin` cannot be used to time any of this. Its plateau counter does not
survive a batched move -- steepest descent over 32 candidates almost always
finds some improvement, so the counter resets on nearly every trial and the loop
runs far past where a single-candidate move would stop. `minChainBench` drives
the same path over a fixed move count instead. Fixing the termination criterion
for batched moves is a separate open problem.

## Closing the plateau problem

The "Where this landed" section left the termination criterion open. It is now
closed, and the fix is not in this file's subject matter at all -- it is in what
the criterion was measuring.

`nobetter` counted consecutive trials with no improvement better than KT. That
is a hitting time on the *proposal mechanism*, and it is only meaningful while
the per-trial success probability falls towards zero as the structure
converges. Steepest descent over 32 candidates keeps that probability bounded
away from zero indefinitely, so the counter resets almost every iteration and
the required run of consecutive failures never occurs. This is why 1crn, 13 s
under the old single-candidate move, ran past 6.5 minutes here and 83 minutes
in the run `protMinReplicaCU` records.

`protMinReplicaCU` answered by taking a fixed sweep budget and arguing a
sampler's cost should be chosen rather than discovered. That is right for
sampling and wrong for minimising, because a fixed budget spends the same wall
clock on a structure that is already good as on one that is far from it.

The criterion that works measures the *trajectory*. Descent here is roughly
log-linear -- about 59 kcal/mol per e-fold of budget on 1ubq -- so improvement
per fixed window decays like 1/B by construction and has no plateau to detect.
Improvement per *doubling* is the quantity that is flat under that law. So:
checkpoint at geometrically spaced trial counts, compare best-so-far against
best-so-far one doubling earlier, and stop when a doubling of the whole budget
spent so far buys less than a threshold. Overshoot is bounded at 2x by
construction, and the unit of work is a sweep -- one trial per residue -- so
nothing needs retuning per structure size.

The threshold is absolute, not a fraction of the gain so far. At the checkpoint
where 1crn has 1.4 kcal/mol left to win and 1ubq has 58, both show a last
doubling worth about 1.8% of their cumulative gain, so a relative rule cannot
separate them and stops 1ubq far too early.

Measured, running `protMinCU` repeatedly in one process so each pass starts from
the conformation the last one left (`projects/minStopProbe.cc` -- a PDB round
trip will not do, it reintroduces clashes and comes back at 373592 kcal/mol):

|       | pass 1            | pass 2           | pass 3          |
|-------|-------------------|------------------|-----------------|
| 1crn  | 492.8 -> 440.0, 30.7 s | 440.0 -> 442.4, 0.97 s | 442.4 -> 441.3, 1.9 s |
| 1ubq  | 987.2 -> 204.6, 171 s  | 204.6 -> 189.4, 86 s   | 189.4 -> 191.8, 2.7 s |

That is the adaptivity a budget cannot express: 1ubq gets a second full-cost
pass because it is still buying 15 kcal/mol, and both structures fall to a
couple of seconds once they are spent.

Two things this does not fix, both visible above. Passes 2 and 3 can end
slightly *worse* than they started, because the walk accepts uphill moves at KT
and returns its final conformation rather than the best one it saw; the excursions
are a few KT, but a minimiser that can hand back a worse structure than it was
given is a defect worth closing separately -- `saveCurrentState`/`undoState`
would do it, at the cost of reasoning carefully about the delta chain's device
coordinates. And on 1crn the descent never actually flattens below 1 kcal/mol
per doubling, so the 256-sweep cap is doing the stopping there, not the
criterion. The knob is real; there is no free lunch on a structure that keeps
paying a little forever.

## Where the time actually goes

Prompted by asking how well the GPU is used on the smallest structure. The
answer was 19% kernel-resident wall on 1crn, which raised the obvious proposal:
keep coordinates on the device for the whole minimisation and do the torsion
transforms there too. Measured, that proposal is aimed at the wrong 81%.

Per-trial phase timing (`PROTCAD_PROFILE=1`, `projects/hostMoveProbe.cc` for the
move-generation half):

| phase                          | 1crn    | 1ake    |
|--------------------------------|---------|---------|
| `energyComputeBatchDelta`      | 1777 us | 6761 us |
| `energyComputeDelta` (pOld)    |  223 us |  450 us |
| candidate generation, K x transform | 187 us | 243 us |
| setup, thaw, fill-in, ranking  |  115 us |  922 us |
| **trial**                      | 2302 us | 8377 us |

The host torsion transform is 8% on 1crn and 3% on 1ake, so moving it to the
device buys almost nothing, and coordinates were never the problem: uploads are
already restricted to the changed set. Of `energyComputeBatchDelta`'s 1777 us,
only about 450 us was kernel time.

The gap was the *return* path. `kGatherTermsBatch` wrote 7 values per changed
atom per candidate and all of it came home to be Kahan-summed on the host --
58k double adds behind a 233 kB transfer per trial on 1crn, 137k on 1ake. The
changed set is the one quantity in this path that grows with the structure, so
it was the worst possible thing to move across the bus.

The seven terms combine linearly, so each atom's contribution collapses to one
scalar and each candidate to one number. `kReduceBatchPart` does that on the
device and returns `nCand` doubles instead of `7 * count * nCand`. The host
version's determinism guarantee is the binding constraint and is preserved:
each thread sums one contiguous index range in order and thread zero combines
partials in thread order, so a candidate's energy is a function of `count` alone
and does not depend on the spatial sort or the batch size. A strided or tree
reduction would be faster and would quietly break that. `PROTCAD_BATCH_HOSTREDUCE=1`
restores the host path.

Worst relative error is unchanged at 2.39e-07 on 1crn. Per trial: 2317 -> 1676 us
(1.38x) on 1crn, 3666 -> 2639 us (1.39x) on 1ubq, 8286 -> 7057 us (1.17x) on
1ake. A 1crn minimisation goes 30.7 s -> 25.4 s, and the cheap repeat passes
0.97 s -> 0.19 s.

This does not finish the job. `energyComputeBatchDelta` is still 68% of a trial
against ~450 us of kernel time, so roughly 700 us remains in staging, the
now-pointless `7 * count * nCand` device write, and ~31 launches per trial. The
next rung is fusing the reduction into `kGatherTermsBatch` so the intermediate
is never materialised. Duty cycle is still about 27%, and the kernels themselves
are latency-bound: 12x the atoms costs 1.7x the `kEnergy` time, so the device is
not close to saturated at any size in this corpus.

## Following the profile down: the torsion half

The device reduction left `energyComputeBatchDelta` at 68% of a trial against
about 450 us of kernel time, so the obvious next move was to fuse the gather
into the reduction and stop materialising `7 * count * nCand` elements only to
read them straight back. That was done, and it is a wash: 1683 vs 1676 us per
trial on 1crn. It removes a buffer, an allocation and a launch, so it stays,
but the intermediate write was not the cost either.

Two wrong guesses in a row is a sign to stop guessing, so the inside of the
call got its own sync'd phase timers (`PROTCAD_PROFILE=1`). The first run
accounted for only 564 us of the 1152 us the caller measured, and the missing
half was the one phase with no timer on it -- the torsion term, which earlier
notes had written off as separate and cheap. With the mark added:

| phase (1crn / 1ake, us/call) | before | after |
|------------------------------|--------|-------|
| stage fill (host)            | 15 / 60    | 15 / 61   |
| stage H2D + scatter          | 181 / 351  | 181 / 353 |
| tile bounds                  | 12 / 21    | 12 / 21   |
| occupancy                    | 53 / 346   | 53 / 349  |
| energy                       | 257 / 1082 | 258 / 1088|
| reduce + D2H                 | 45 / 48    | 45 / 49   |
| **torsion**                  | **597 / 3744** | **67 / 239** |

`torsionBatchDelta` gave every candidate a full copy of the primed per-torsion
energies, overwrote the entries the move touched, and Kahan-summed over every
torsion in the structure -- `nCand * nTor` host adds per trial behind an
`nL * nCand` readback.

The full ascending sum is not the mistake. It is what keeps the delta
bit-comparable to a full evaluation instead of a drifting running total, and
replacing it with base-plus-correction would trade that away for arithmetic
that is only cheap because it ignores cancellation. The mistake was doing it on
the host, once per candidate. The primed baseline is now broadcast into the
candidate rows on the device, the listed torsions overwrite it exactly as
before, and the ascending sum runs there under the same determinism rule as the
nonbonded half: order fixed by torsion index alone, so a candidate's total does
not depend on its batch or on how many torsions the move touched.
`PROTCAD_TORSION_HOSTREDUCE=1` restores the host path, and the two produce
identical descent trajectories.

Per trial across the three changes:

| structure | host reduce | + device reduce | + fused | + device torsion |
|-----------|-------------|-----------------|---------|------------------|
| 1crn      | 2317 us     | 1676            | 1683    | **1197**  (1.94x) |
| 1ubq      | 3666 us     | 2639            | 2708    | **1830**  (2.00x) |
| 1ake      | 8286 us     | 7057            | 7355    | **3887**  (2.13x) |

The shape of the profile has changed, which matters more than the factor. The
batch delta is now 54-57% of a trial and `kEnergy` is the largest line in it at
41-50%, so what remains is mostly work the GPU is actually there to do. The
next candidates are the H2D staging and scatter, which is flat at ~180 us on
1crn regardless of size and so is launch and transfer latency rather than
volume, and the per-trial `3 * K * N` zeroed allocation in
`bestSidechainCandidateDeltaCU`, which is still worth ~1.7 ms on 1ake. The
kernels themselves remain latency-bound and the device is still not close to
saturated.

## The zeroed allocation was a mismeasurement

The note above listed the per-trial `3 * K * N` zeroed allocation in
`bestSidechainCandidateDeltaCU` as worth about 1.7 ms on 1ake. It is not. It is
worth roughly 30-80 us, and the 1.7 ms came from reading a microbenchmark as if
it were the real loop.

`hostMoveProbe` timed the host move path with the energy evaluation removed. In
that setting the 6 MB buffers are allocated and freed with nothing in between,
which on glibc means an mmap and munmap per iteration and a page fault on every
page of the zeroing. In the real loop the same block is recycled and the memset
lands on pages that are already resident, so almost all of the measured cost was
an artifact of the isolation.

The buffers are still worth holding across trials, just not for the stated
reason. The consumers read only the moved residue (the thaw builder, via its
`_support` argument) and the thaw set (the batch delta), and both are written in
full every trial before either runs, so the buffer needs to be large enough
rather than clean. Reusing it removes a recurring multi-megabyte allocation and
6 MB of pointless stores per trial. Measured: 2.72 -> 2.68 s over 512 trials on
1ake, descent trajectories unchanged.

The general lesson is the one this session keeps re-learning. Every estimate
here that came from a component measured in isolation -- the host transforms,
the intermediate gather buffer, this allocation -- has been wrong about its
share of the real loop, and the one phase nobody instrumented at all turned out
to be the largest. Profile in place, or do not profile.

Current per-trial shape on 1ake: batch delta 57%, `energyComputeDelta` for the
pre-move part 12%, thaw set construction 10%, setup 7%, candidate generation 6%.

## Folding the pre-move evaluation into the batch

With the batch delta down to a reasonable shape, the next line in the profile
was the pre-move part: `energyComputeDelta` for P(old), at 19% of a trial on
1crn and 12% on 1ake. It was evaluating a single conformation with the
machinery built for thirty-two, paying a full set of launches and a staging
round trip for a thirty-second of the work. Widening the existing batch by one
slot costs a few percent of a call that already runs.

The stronger argument is correctness rather than cost. P(old) and P(new) have
to be taken over the same changed set for their difference to mean anything,
and they were -- the thaw set is installed before either runs -- but they were
computed by two different code paths with two different reduction orders, so
the subtraction carried the gap between the paths as well as the effect of the
move. Both now come out of one call, over one changed set, in one reduction
order. `batchDeltaTest`'s worst relative error moves from 2.39e-07 to 3.21e-07,
which is P(old) changing paths, not accuracy being lost; the gate is 5e-6.

| structure | before | after |
|-----------|--------|-------|
| 1crn | 1198 us/trial | **1050** |
| 1ubq | 1847 us/trial | **1673** |
| 1ake | 3891 us/trial | **3675** |

1ake over 512 trials: 2.68 -> 2.47 s wall.

Cumulative for the session, against the host-reduction baseline: 2317 -> 1050
us/trial on 1crn (2.21x), 3666 -> 1673 on 1ubq (2.19x), 8286 -> 3675 on 1ake
(2.25x).

What is left on 1ake is batch delta 66%, thaw set construction 12%, setup 8%,
candidate generation 7%. Thaw set construction is host work that rebuilds a
bounding-sphere set per trial, and `setup` is still the full-N
`updateDeviceCoords()` at the top of the function. Neither is the dominant term
any more, and the batch delta's remaining cost is now mostly `kEnergy` plus H2D
staging latency that does not scale with structure size -- which is where the
question of keeping coordinates device-resident finally becomes the right one
to ask.

## Staging the moved set instead of the changed set

`kSeedBatch` gives every candidate row the resident device conformation before
anything is scattered into it.  The staging step then overwrote the *entire*
changed set from host memory -- but the changed set is the union of the atoms
the move displaced and every atom whose dielectric environment it disturbed,
and the second group is by construction unchanged.  Those values were already
sitting on the device.  We were shipping them back.

On 1crn the changed set averages 44.9% of 648 atoms, about 291, while the moved
residue is 10-20 atoms: roughly a fourteenth of the volume was doing all the
work.  `energyComputeBatchDelta` now takes an optional `moved`/`nMoved` pair and
stages and scatters only that subset; passing 0 keeps the old behaviour for any
caller whose moves lack a known support.

The host-side thaw fill-in in `bestSidechainCandidateDeltaCU` went away with it.
It existed only to stop the batch shipping uninitialised coordinates for thawed
atoms outside the moved residue; since those now come from the seed, the loop
was writing O(K * |thaw|) doubles per trial that nothing read.

Measured, same seed and trial count, `energyComputeBatchDelta` per trial:

| structure | before | after |
|---|---|---|
| 1crn  |  763 us | 580 us |
| 1ubq  | 1100 us | 837 us |
| 1ake  | 2243 us | 1884 us |

The staging sub-phase itself went 181 -> 16 us on 1crn and 353 -> 18 us on 1ake,
and is now flat in structure size -- which is the signature of a transfer that
has stopped being about volume and is purely launch latency.  The separate thaw
fill-in phase went from 57 us (1ubq) and 89 us (1ake) to zero.  1ake over 512
trials, best of three: 2.50 -> 2.24 s wall.

Worst relative error in `batchDeltaTest` is unchanged at 3.21e-07 and descent
trajectories are identical to the host-reduction reference, as they must be:
the atoms that stopped being staged were being written with the values they
already held.

## The candidate transform, and why it did not go to the GPU

With the batch delta down, candidate generation had risen to 19% of a trial on
1crn, so it was the next thing to look at.  Splitting it three ways showed the
whole cost in one place: of 193 us per trial, the random draw was 4.8 us, the
readback and copy 14 us, and `setSidechainDihedralAngles` 167 us.

Instrumenting `residue::setChi` split that again: 0.94 us measuring the current
dihedral and 1.77 us applying the rotation, over roughly 63 calls a trial.  That
is about 2.7 us to rotate a handful of atoms about one bond, which is far too
much to be arithmetic.  It is not arithmetic.  `getChi` builds a `UIntVec` by
value and recomputes a dihedral we already know -- we set it ourselves on the
previous candidate -- and `rotate` walks the atom subtree three times, once to
translate to the origin, once to transform, once to translate back, allocating
a `dblVec` per atom and a `dblMat` per call along the way.

So the obvious move was to put the transform on the GPU, and the obvious move
was wrong.  The work is about 480 atoms per trial.  A kernel launch plus the
angles up and the coordinates back is 20-30 us before any arithmetic happens,
and the arithmetic here is a few hundred multiply-adds.  The device was never
the fix; the allocator was.  `chiRotateFlat` performs exactly the rotation
`residue::rotate` performs -- same translate-rotate-translate, same matrix,
including CMath's truncated `0.017453293` -- over a flat block of residue
coordinates, with no tree walk and no allocation:

| phase (us/trial) | 1crn | 1ubq | 1ake |
|---|---|---|---|
| candidate generation, before |  193 |  238 |  285 |
| candidate generation, after  |   16 |   19 |   22 |
| of which the transform       |  4.3 |  5.7 |  6.1 |

The transform itself went 167 -> 4.3 us, a factor of 39, and the phase is now
2.3% of a trial on 1crn and 0.8% on 1ake.  A device port would now be competing
against 4 us with a 10 us launch.  There is nothing left here to move.

### It is also more accurate than what it replaced

The old loop reached each candidate from the last one's geometry, because
`setChi` is a delta operation: it measures where the residue currently sits and
rotates by the difference.  Thirty-two of those in a row accumulate drift.  The
flat transform reaches every candidate from the entry conformation instead --
which is what the caller does when it adopts the winner, since it applies the
chosen chis to the restored entry residue.  Evaluated geometry and adopted
geometry are now reached the same way.

`batchDeltaTest` was chaining its reference the same way the generator did, so
the two drifted together and the test was partly comparing a path to itself.
Against a reference that reaches each candidate from entry:

| generation path | worst relative error |
|---|---|
| flat transform from entry | 1.88e-06 |
| residue object, chained   | 7.25e-06 |

The old path is nearly four times further from a clean full evaluation than the
new one.  The reported 3.21e-07 was flattering rather than accurate.  The gate
stays at 5e-6 and the suite is 12/12.

1ake over 512 trials: 2.24 -> 2.10 s wall.  1crn: 0.91 -> 0.82 s.
`PROTCAD_HOSTCHI=1` restores the residue-object generator for A/B.

## Two full-N host passes that did not need to happen

With the transform dealt with, the two largest host phases on 1ake were setup
at 286 us and thaw set construction at 423 us.  Both were paying for the same
two mistakes.

**`getCoords` returns a `dblVec` by value.**  `updateDeviceCoords` pulled all N
atoms through it, so reading three doubles per atom cost a heap allocation per
atom -- 7814 of them a trial on 1ake.  `point` already exposes `getX/getY/getZ`,
which read the same stored values with no copy.  Same numbers, no allocator.

**The thaw builder called `updateDeviceCoords` again.**  The batch caller
refreshes every coordinate immediately before generating candidates, and in the
flat-transform path the residue object is never moved at all, so the second
full-N pass was reproducing what `itsCoord*` already held.  It now takes a
`_coordsCurrent` flag from the one caller that has just done the work, and
falls back to the refresh otherwise.

**The sphere scan tested every atom against every moved atom's sphere.**  All of
those spheres sit inside a single residue, so nearly every atom in the structure
misses all of them, and O(N * nMoved) was spent proving it.  One enclosing
sphere, computed from the others, rejects an atom in a single test.  It only
ever rejects -- whatever it admits is still checked sphere by sphere -- so the
thaw set is exactly the set the plain scan produced, which the unchanged 44.9%
changed-set fraction and the bit-identical trajectories confirm.

| phase (us/trial) | 1crn before | 1crn after | 1ake before | 1ake after |
|---|---|---|---|---|
| setup                  | 21 | 6.9 | 286 |  90 |
| thaw set construction  | 30 | 12  | 423 |  45 |

Per trial: 1crn 684 -> 654 us, 1ake 2748 -> 2206 us.  1ake over 512 trials went
2.10 -> 1.80 s wall, 1crn 0.82 -> 0.78 s.

Host bookkeeping on 1ake is now about 14% of a trial -- setup 84 us,
`energySetCoords` 154 us, thaw 43 us, candidate generation 23 us -- against 85%
in the batch delta itself, which is 58% `kEnergy`, 18% occupancy and 18%
torsion.  The remaining host cost is the two full-N passes that are left, and
neither can go away without deciding when the coordinate arrays may be trusted
to be stale.  That is a correctness question, not a performance one.

## Two correctness defects in the minimiser

Both of these were invisible in the energies the minimiser reported and showed up
only in what it did across repeated passes.

### protMinCU returned its last conformation, not its best

The loop accepts uphill moves, so the conformation it happens to stop on is not
in general the best one it visited.  Running `minStopProbe` on 1crn for six
passes in one process -- the only valid way to test this, since a PDB round trip
reintroduces clashes and hands the second pass a different problem -- showed the
structure drifting *upward* once it was near the minimum:

    pass 1  492.791 -> 447.469
    pass 2  447.469 -> 448.881
    pass 3  448.881 -> 449.082
    pass 4  449.082 -> 449.387
    pass 6  447.650 -> 449.335

Four of six passes ended worse than they started and there was no fixed point.

The fix keeps a copy of every atom position whenever the accepted energy sets a
new best, and restores it on exit.  Coordinates are the right thing to snapshot
rather than dihedrals: `getChi` and the energy both read them, they capture
backbone and cofactor moves as well as sidechain ones, and they do not go
through the mutation buffers that `undoState` works on -- which exist to reverse
mutations, not to rewind a search.  The bonding tree is untouched, so every
dihedral accessor keeps working and reports the angles the snapshot was taken
at.

Restoring has to put the device back too, and there is a trap: `energySetCoords`
does *not* clear `torPrimed`, deliberately, because the minimiser re-uploads
coordinates every trial and would otherwise never hold a torsion baseline at
all.  So a restore that only uploads coordinates leaves `h_torE` describing the
conformation that was just abandoned.  `energyInvalidateTorsionBaseline` exists
for exactly this and the restore path calls it.

The snapshot fires only on improvement and costs nothing measurable: 1ake stays
at 2199 us/trial.

### boltzmannEnergyCriteria was a factor of two too cold

    double Entropy = 1000000/((rand() % 1000000)+1);
    double PiPj = pow(EU,(_deltaEnergy/residue::getKT()));
    if (PiPj < Entropy){acceptance = true;}

The positive exponent and the reciprocal random variable are two sign inversions
that cancel, and the large-dE tail is right, so this looks worse than it is at
first glance.  The actual defect is that `1000000/(...)` is **integer
division**.  `Entropy` was `floor(1e6/U)`, which is never below 1 and is exactly
1 for half of all draws.  Against a strict `<`, that gives

    P(accept) = 1 / (floor(e^(dE/KT)) + 1)      instead of      e^(-dE/KT)

Measured over 2e6 draws at KT = 0.6, the model is exact:

    dE     old     correct
    0.00   0.500   1.000
    0.25   0.500   0.659
    0.50   0.333   0.435
    1.00   0.167   0.189
    2.00   0.034   0.036

Downhill moves were unaffected, which is why this never showed up as a bad
energy.  It biased the walk instead: every uphill move up to dE = KT*ln2 was
under-accepted, by a factor of two in the limit dE -> 0.  Those are most of what
a Metropolis walk near a minimum proposes, so the search was substantially
greedier than the temperature claimed.

The replacement draws u uniform on (0, 1] and tests `exp(-dE/KT) >= u`.  The
draw happens first and unconditionally so that the position of the RNG stream
does not depend on the sign of the move.

Attribution on 1crn, six passes, same seed:

    variant             pass-1 wall   fixed point   monotone
    baseline               16.67 s    none (~449)   no
    best-conformation       8.22 s    446.474       yes
    both                   16.64 s    444.344       yes

Monotonicity comes entirely from the best-conformation fix.  The acceptance fix
is worth a further 2.13 kcal, and the wall time tells you why: the too-cold
sampler was converging falsely at 8.2 s.  Corrected, the pass costs what it cost
before the best-conformation fix was added and now keeps what it finds.

## Widening the batch does not help kEnergy

Widening the candidate batch past 32 has been the standing suggestion for
kEnergy, which is 58% of the batch delta and the largest single cost left.  The
determinism constraint turns out not to bind: candidate k is `blockIdx.y` and
owns its own slice of every buffer, `chooseSplit` keys off `nTiles` only, and
the batch path sizes `dSplit` against `nList * nCand`.  A candidate's energy
does not depend on how many candidates share its batch, so the batch may be
widened freely.  It just does not pay.

1ake, 7814 atoms, 64 trials, `PROTCAD_CANDIDATES`:

    nCand   kEnergy      per cand     trial       per cand
        8    328.1 us     41.01 us    1201.1 us   150.14 us
       16    556.3        34.77       1525.7       95.36
       32   1037.3        32.42       2227.0       69.59
       64   2103.9        32.87       3784.6       59.13
      128   4409.9        34.45       7224.2       56.44

kEnergy is linear in nCand from 32 upward -- 32.4, 32.9, 34.5 us per candidate.
The knee is between 16 and 32, and 32 is already past it.  There is no occupancy
left to buy: at 32 candidates the grid is saturated, which is also why
`PROTCAD_ENERGY_JSPLIT` moves nothing in this path.  The batch dSplit rule
already collapses to 1 for exactly this reason.

The earlier "kEnergy is latency-bound, 12x atoms gives 1.7x time" holds, but it
is a statement about scaling in *atoms* at fixed nCand, where the thawed set and
the tiles are underfilled.  In the *candidate* dimension the device is full.
The two are not in tension and only the first one was ever measured.

Per-candidate *trial* cost does keep falling (150 -> 56 us) because the fixed
per-trial overhead amortises, so a wider batch remains defensible on grounds of
move quality per unit time.  That is an argument about the search, not about
kEnergy, and it should be settled on minimisation outcome rather than on
throughput.

What is left for kEnergy is algorithmic -- fewer or better-pruned candidates --
or precision.  Nothing remains at the launch-configuration level.

### A measurement that was almost reported

The first sweep showed kEnergy flat at 1035-1037 us across a 32x change in
nCand, which would have been a spectacular result.  `minStopBench` has its own
hardcoded `nCand = 32` and never read the environment override, so the sweep ran
the same configuration six times.  The knob is now wired through to the bench.

## Precision is already spent

`ereal` is `float` unless the build defines `PROTCAD_ENERGY_FP64`, and nothing
defines it.  The pair terms have been evaluated in single precision all along,
so there is no precision lever left to pull.  Going below float is not
available: `batchDeltaTest` gates at 5e-6 relative and currently sits at
1.88e-06, which is roughly float epsilon accumulated over the sum.  Half
precision would not survive the gate, and the 3090's TF32 path applies to
tensor-core matmul, not to a pairwise kernel like this one.

## What kEnergy actually spends its time on

The assumption worth checking was that kEnergy is dominated by the physics --
the LJ term, the electrostatics, the exclusion lookup, the square root.  It is
not.  Ablating each in turn with a uniform branch on 1ake, 64 trials:

    ablation                    kEnergy
    none (real kernel)          1102.1 us
    no exclusion lookup          917.0
    no LJ                       1041.5
    no electrostatics           1006.7
    no sqrt                     1079.6
    no LJ + no electrostatics    948.0
    all of the above             772.0

Deleting **every** arithmetic term in the pair loop buys 30%.  The other 70% is
neighbour traversal: staging each j tile into shared memory, `boxBoxDist2`
tile tests, and the `r2 >= cutSq` reject.  Counting directly, one batch delta
call on 1ake examines ~56M candidate pairs and keeps ~7.5M of them -- **14.5%**.
Six of every seven trips through the inner loop exist only to discover that two
atoms are too far apart.

(The ablation branches themselves cost about 6%: the real kernel measures 1032
us without the harness and 1102 with it.  The harness is measurement-only and
is not in the shipped kernel.  Only the loop-external `PROTCAD_SETDIAG` counter
was kept.)

## The pair list this points at

The traversal is the cost, and the traversal is *candidate-invariant*.
`PROTCAD_SETDIAG` on 1ake:

    N=7814  changed=468  (6.0%)   moved=14  (0.18%)
    N=7814  changed=920 (11.8%)   moved=24  (0.31%)
    N=7814  changed=446  (5.7%)   moved=19  (0.24%)

Nearly every atom in the changed set is there because its *dielectric* moved,
not because it did: 14-24 atoms out of 450-920.  And staging only scatters the
moved set, so every other atom holds bitwise identical coordinates in all 32
candidate rows.  For a pair with neither endpoint moved -- which, at 0.2% moved,
is essentially every pair -- these are all invariant across the batch:

    r2, d, the exclusion code, the switch factor, and the entire LJ term

and the electrostatic term varies only through `epsPair`, since

    ele = (c_kc * qi * qj / d) / epsPair

with the bracketed factor invariant.  So the batch currently rediscovers the
same neighbour topology 32 times and recomputes the same LJ energy 32 times.

The shape of the fix is to build, once per trial, a compacted list of surviving
pairs holding the j index, the invariant electrostatic numerator, and the
invariant LJ contribution, then have each candidate stream that dense list and
do only a dielectric mix, a divide and an add.  This removes exactly the 70%
that is traversal and the 85% of inner-loop trips that are rejects, because a
compacted list has no rejects in it at all.

Rough sizing on 1ake: ~227K surviving pairs per candidate row, so ~2.7 MB at
three 4-byte fields per pair, built once and read 32 times.  The per-candidate
pass becomes bandwidth-bound at roughly 2.7 MB, and the j-side dielectrics it
gathers against are a few tens of KB and will sit in cache.  That projects to
order 150 us against the present 1032, with the invariant LJ sum hoisted out of
the candidate loop entirely.

Determinism is not threatened and arguably improves: the list order is fixed for
the trial and does not depend on nCand, so each thread still sums a contiguous
range in a fixed order.  The `2*V1 - V2` double-count correction survives by
storing the j-side cmask bit in the list entry.

This has not been built.  It is the largest remaining lever in the delta path
and it is a real kernel redesign, not a tuning change.

## Hoisting the dielectric out of tile staging

The ablation in the previous section measured where kEnergy's time goes by
switching off arithmetic *inside the pair loop*. It missed something. The
j-side dielectric is not computed in the pair loop; it is computed in the
**staging** loop, once per neighbour atom per i-tile:

    shD[base + lane] = shellFromOcclusion(occ[js], srad[js], p, v43).eps;

`shellFromOcclusion` carries two cubes, three divides and a truncation. An
atom's dielectric depends only on its own radius and its settled occupancy, so
this produces the same number for every i-tile that stages it as a neighbour.
On 1ake that is `nList` = ~37 recomputations of an identical value.

`kPrecomputeEps` now evaluates it once per (candidate, atom) after the
occupancy settles, and kEnergy reads `epsAll[js]` instead. Same function, same
inputs, same order, so the result is bit-identical -- `batchDeltaTest` stayed at
exactly 1.88e-06, unchanged to every digit.

Measured on 1ake, 64 trials, best of three:

| | before | after |
|---|---|---|
| kEnergy | 1032.5 us | 967.0 us |
| batch delta | 1877 us | 1811 us |
| per trial | 2201.9 us | 2127 us |

Kept, but read the size honestly. Removing a 37x redundant block of divides
bought **6.3%**. That is the useful finding: kEnergy is not ALU-bound. Roughly
300 lane-cycles are being spent per examined pair, which is two orders of
magnitude more than the reject path's arithmetic can account for. Whatever the
kernel is waiting on, it is not the physics.

### What this does to the pair-list plan

It argues against it, and the pair counts already collected say the same thing
once they are converted to bytes.

The current staging moves 32 neighbours x 10 arrays x 4 B = 1280 B into shared
memory and 32 i-lanes each consume all 32 of them: **1.25 B per examined pair**,
or ~8.6 B per *surviving* pair at the measured 14.5% survival. The whole sorted
attribute set is ~313 KB and fits in L2, so almost none of that reaches DRAM.

A compacted pair list costs ~12 B per surviving entry, and then needs a
**random** gather of the j-side dielectric per entry, which is uncoalesced and
pulls a 32 B sector for every 4 B used. It replaces ~8.6 B of L2-resident,
perfectly coalesced, broadcast-shared traffic with ~12 B of streamed traffic
plus a scattered gather. On bandwidth it is a downgrade, and the eps hoist just
demonstrated that the arithmetic it would save is not what we are paying for.

So the pair list is withdrawn as designed. The tile structure is not the
problem; it is a good structure. The open question is why a tile pair costs what
it does, and that points at staging *instructions* and occupancy rather than at
the pair loop -- ten separate scalar loads and a `__syncwarp` per j-tile per
i-tile. Packing the sort-invariant attributes into wide vector loads is the
next thing to measure, not a new traversal scheme.

## Lifting i's exclusion row out of the pair loop

`exclusionCode` took `origI` and indexed three things with it: `exclSpan[origI]`,
`exclCount[origI]`, and the row base `exclList + origI*stride`. `origI` is
`sorig[s]` -- fixed for the life of the thread. All three were being issued as
global loads on every one of the ~56M pairs a batch delta examines.

The loads look trivially hoistable, and there are no stores in the pair loop, so
nvcc was expected to lift them. It did not. `exclusionCodeHoisted` takes the
three values as arguments and the caller computes them once in its prologue:

| | kEnergy |
|---|---|
| baseline | 1032.5 us |
| + dielectric hoist | 967.0 us |
| + exclusion hoist | 890.0 us |

13.8% off kEnergy, in two changes that alter no arithmetic and no summation
order. `batchDeltaTest` is unchanged at exactly 1.88e-06 through both, and
`minStopBench` on 1ake reaches bit-identical results at 64 trials (37.9784) and
512 trials (-310.4738).

Padding lanes carry `sorig == -1`. The old code was safe because the call sat
behind the `active` test; the hoist runs in the prologue ahead of it, so it
guards on `origI >= 0` explicitly.

The pattern is now the recurring one on this kernel: both wins came from work
that was *loop-invariant and being redone*, not from work that was expensive.
Look for redundancy before looking for cheaper physics.

## minStopProbe's end energy is a random variate, not a fixed point

Earlier notes treated 1crn settling at 444.344 as a reproducible value to check
against. It is not. Three runs of the same unmodified binary give 436.228,
437.733 and 444.959 -- the probe does not seed deterministically, so its end
energy is a draw from a distribution several kcal wide.

Only two things from `minStopProbe` are meaningful as regression evidence:
that successive passes are **monotone**, and that the search reaches a **fixed
point** (each pass's `start` exactly equals the previous pass's `end`). For a
reproducible energy, use `minStopBench`, which takes an explicit seed and is
deterministic run to run.
