// energy.h
//
// GPU interface for ProtCAD's atom-atom interaction energy, including the
// local-effective-dielectric electrostatic model.
//
// Design notes
// ------------
// * All device state lives in an opaque energyContext.  Callers hold a handle
//   and never see CUDA types, so this header is safe to include from plain C++
//   translation units compiled without nvcc.
// * Per-atom exclusions are supplied as a compressed list (count + indices)
//   rather than the old dense N*N byte matrix, which cost O(N^2) host time and
//   O(N^2) device memory.
// * Every physical choice that used to be hardcoded inside the kernel is now an
//   explicit field of energyParams, so the CPU and GPU paths can be held to the
//   same model and so model variants can be A/B tested without editing kernels.
//
// Precision: build energy.cu with -DPROTCAD_ENERGY_FP64 to evaluate pair terms
// in double.  The default is float, which is roughly 25x faster on consumer
// GeForce parts.  Final sums are always accumulated in double.
//
// This is deliberately an implementation detail of energy.cu and does NOT
// appear in any type crossing this header: every public struct uses plain
// double.  An earlier revision exposed `ereal` in energyParams, which made the
// struct layout depend on the flag, so a kernel built for FP64 and a caller
// built without it disagreed about the ABI and smashed the stack.

#ifndef ENERGY_H
#define ENERGY_H

#include <cstddef>

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

// How the shell occupancy of an atom is converted to a local dielectric.
enum dielectricModel
{
    // eps = epsProtein + (4*pi/3) * polarizability.
    //
    // This is what the shipped code actually evaluates.  The source reads like
    // a Clausius-Mossotti expression but a stray "/1" makes the intended
    // denominator a no-op, so the result is linear in polarizability.  Retained
    // verbatim for backward compatibility with existing calibrations.
    DIELECTRIC_LEGACY_LINEAR = 0,

    // eps = epsProtein + (epsWater - epsProtein) * f, where f is the fractional
    // water occupancy of the hydration shell in [0,1].
    //
    // Identical in shape to the legacy form but honest about what it is: an
    // empirical interpolation between the buried and fully solvated limits,
    // with endpoints that are physical by construction.  Recommended.
    DIELECTRIC_OCCUPANCY = 1,

    // True Clausius-Mossotti with a proper number density.  Provided for
    // reference and testing only.  This recovers the OPTICAL dielectric of
    // water (~1.8), not the static one (~78), because CM accounts only for
    // electronic polarizability and omits the orientational response that
    // dominates the static constant.  Do not use this for solvation.
    DIELECTRIC_CLAUSIUS_MOSSOTTI = 2
};

// How two per-atom dielectrics are combined into a screening factor for a pair.
enum pairMixingModel
{
    PAIRMIX_ARITHMETIC = 0,   // (epsI + epsJ) / 2   -- legacy behaviour
    PAIRMIX_HARMONIC   = 1    // 2*epsI*epsJ / (epsI + epsJ)
};

// How the volume of the hydration shell occluded by neighbours is estimated.
enum occupancyModel
{
    // Add each neighbour's whole van der Waals volume whenever its centre lies
    // within the shell radius.  Legacy behaviour: double counts overlapping
    // neighbours and ignores the fraction of a neighbour lying outside the
    // shell, so the occupied volume can exceed the shell volume.
    OCCUPANCY_LEGACY_FULLVOLUME = 0,

    // Add only the volume of the lens formed by intersecting the neighbour
    // sphere with the shell sphere.  Bounded and continuous in position.
    OCCUPANCY_LENS = 1
};

// Geometric test used to declare a steric clash.
enum clashModel
{
    // |dx|,|dy|,|dz| all below (radI+radJ)/sqrt(2): the cube inscribed in the
    // contact sphere.  This is what residue::isClash uses on the CPU path, so
    // it is the definition every clash threshold in ProtCAD was tuned against.
    CLASH_INSCRIBED_CUBE = 0,

    // d^2 < (radI+radJ)^2: the full contact sphere.  What the shipped CUDA
    // kernel used.  Reports roughly 1.5x more contacts than the cube test.
    CLASH_SPHERE = 1
};

// Shape of the repulsive wall.  All three share the same minimum: depth
// -sqrt(epsI*epsJ) at r = radI + radJ.  They differ only inside it.
enum vdwWallModel
{
    // eps*((rm/r)^12 - 2*(rm/r)^6).  r^-12 was chosen because it is the square
    // of r^-6, not for physics, and it has no shoulder near its zero crossing:
    // the only inflection is at 1.109*rm, on the attractive side beyond the
    // minimum.  Measured consequence is a surface 256x stiffer than harmonic at
    // 0.3 A displacement, while 1crn's B-factors put its RMSF at 0.296 A.
    VDW_LJ_12_6 = 0,

    // Buckingham exp-6, the physical form: exchange repulsion is exponential.
    //   eps*( 6/(a-6)*exp(a*(1 - r/rm)) - a/(a-6)*(rm/r)^6 )
    // Softer than r^-12 at contact.  It inverts at small r, where the -r^-6
    // term outruns the saturating exponential, so it is clamped at its inner
    // maximum; see vdwEnergy.
    VDW_EXP_6 = 1,

    // Beutler soft-core, the standard alchemical endpoint regulariser:
    //   eps*( 1/(d + (r/rm)^6)^2 - 2/(d + (r/rm)^6) )
    // Finite at r=0, monotone, no inversion to guard.  d=0 recovers 12-6
    // exactly.  X->Ala is an endpoint deletion, which is the problem this form
    // was designed for.
    VDW_SOFTCORE = 2
};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct energyParams
{
    // --- solvent and shell geometry ---
    double waterRadius;           // 1.4  A, for Born radii
    double effectiveWaterDiameter;// 4.35 A, shell thickness for occupancy
    double waterVolume;           // volume occupied per shell water, A^3
    // Compensation for the fact that occlusion is summed pairwise while
    // neighbouring atoms also overlap each other, so their contributions are
    // multiply counted.  The shipped model folds a hard 0.5 into the legacy
    // occupancy branch; it is exposed here so the occlusion geometry and the
    // double-counting correction can be varied independently.
    double occlusionScale;

    // Hard-clash threshold as a fraction of (radiusI + radiusJ), used by
    // CLASH_SPHERE.  A full-radius test (1.0) flags roughly 2.2x more pairs
    // than the legacy cube on a native structure, because normal packing puts
    // many contacts just inside the sum of the van der Waals radii; a hard
    // clash needs to mean something tighter than "in contact".
    //
    // The legacy CLASH_INSCRIBED_CUBE test used a cube of half-side
    // (rI+rJ)/sqrt(2).  That cube is not inscribed in the contact sphere -- an
    // inscribed cube has half-side R/sqrt(3) -- so it reaches only 0.707*R
    // along the coordinate axes but 0.707*sqrt(3) = 1.225*R along the body
    // diagonals.  The criterion is therefore orientation dependent: rigidly
    // rotating a structure changes its clash count by several percent, which
    // no physical property may do.  CLASH_SPHERE with this tolerance is the
    // isotropic replacement.  The default 0.905 is calibrated to reproduce the
    // legacy cube counts on native structures: 1crn 327 vs 332, 1ubq 704 vs
    // 706, i.e. within 1.5% on two independent structures, while being exactly
    // invariant under rigid rotation.  (The volume-matched equivalent of the
    // cube is 0.877, but the cube's corners reach out to 1.225*R where pair
    // density is higher, so matching volume undercounts by about 30%.)
    // Recalibrate if the radius source changes.
    double clashTolerance;
    double waterPolarizability;   // 1.47 A^3, electronic polarizability
    double waterEpsilon;          // vdW well depth of water
    double waterCharge;           // effective charge magnitude for polar solvation

    // --- dielectric model ---
    int   dielectric;            // dielectricModel
    int   pairMixing;            // pairMixingModel
    int   occupancy;             // occupancyModel
    double epsProtein;            // 2.0,  buried limit
    double epsWater;              // 78.4, fully solvated limit
    int   quantizeWaters;        // 1 = truncate shell waters to an integer

    // --- nonbonded truncation ---
    double cutoff;                // 12.0 A, outer cutoff for vdW and electrostatics
    double switchStart;           // 10.0 A, where the switching function begins
    int   useSwitching;          // 1 = smoothly taper to zero over [switchStart,cutoff]
    double minSeparation;         // floor on interatomic distance, guards 1/r blowup

    // --- solvation ---
    // 1 = weight polar solvation by shell occupancy fraction rather than by the
    // raw shell water count.
    //
    // Worth stating plainly what the shell-water weight is doing, because it
    // looks ad hoc and is not.  The polar term is q^2/(2a) divided by the local
    // dielectric and multiplied by the shell water count w.  Whenever the local
    // dielectric is affine in the water count, eps = e0 + b*w -- which is true
    // of both the occupancy model and the legacy linear one -- then
    //
    //     w/eps = (e0/b) * (1/e0 - 1/eps)
    //
    // identically.  So multiplying by the water count and dividing by the local
    // dielectric *is* the Born free energy of transferring the charge from a
    // uniform medium of dielectric e0 into its actual local environment.  The
    // model is a local-dielectric Born transfer term, exactly, not an
    // approximation of one, and the exposure dependence needs no correction.
    //
    // What does need correcting is the prefactor.  The raw water count carries
    // the shell capacity, which grows as (r+4.35)^3, so the raw form multiplies
    // the Born prefactor q^2/(2a) by a spurious extra power of atomic size:
    // e0/b picks up the capacity per atom.  Weighting by occupancy fraction
    // instead makes that constant the same for every atom, leaving q^2/(2a) as
    // the sole radius dependence, which is what Born actually says.  The change
    // is therefore a per-element reweighting -- roughly 0.175 to 0.249 relative
    // across the radii in use -- and not a change in solvent-exposure response.
    int   bornNormalize;
    // Reference shell capacity, used so the corrected prefactor lands on the
    // same scale as the calibrated one and eSolvationFactor -- which comes from
    // the parameter database -- does not need refitting.  It is the shell
    // capacity of a standard 1.7 A heavy atom, (4/3)pi(1.7+4.35)^3 / 107.31 =
    // 8.644.  It sets the overall scale only; it cannot affect any energy
    // difference between conformations beyond a uniform factor.
    double bornReferenceCapacity;
    double eSolvationFactor;      // scale on polar (electrostatic) solvation
    double hSolvationFactor;      // scale on nonpolar (hydrophobic) solvation
    double entropyFactor;         // scale on the shell-water entropy term

    // --- global scaling, mirrors the CPU path ---
    double vdwScale;
    double elecScale;

    // --- repulsive wall shape ---
    int    vdwWall;        // vdwWallModel; VDW_LJ_12_6 is the default
    double vdwAlpha;       // exp-6 steepness, 12-16 typical, 14 default
    double vdwSoftDelta;   // soft-core offset; 0 reproduces 12-6 exactly
    // Inner maximum of the exp-6 form in units of rm, the root of
    // -7*ln(x) = alpha*(1-x) below x=1.  Solved on the host whenever vdwAlpha
    // is set so the kernel never has to iterate.
    double vdwExp6Xmax;

    // --- bonded ---
    // Scale on the whole dihedral term.  Zero reproduces the behaviour before
    // torsions existed, which is what legacy mode wants.
    double torsionScale;

    // 1-4 nonbonded scaling.  Amber does not exclude atoms three bonds apart,
    // it damps them: 1/1.2 on electrostatics and 1/2.0 on van der Waals.  The
    // dihedral barriers in ff14SB were fitted with exactly those factors, so
    // the two are not separable -- applying the torsion parameters on top of a
    // full 1-4 exclusion uses them outside the fit that produced them.
    //
    // Zero here means "exclude 1-4 outright", which is what protcad did
    // before, so legacy mode sets both to zero along with torsionScale.
    double elec14Scale;
    double vdw14Scale;

    // --- clash detection ---
    int   clash;                 // clashModel

    // --- exclusions ---
    // Two atoms are considered for exclusion only when their residue indices
    // differ by at most this much; beyond it the exclusion list is skipped
    // entirely.  Set to a large value to always consult the list.
    int   exclusionResidueSpan;
};

// Physically recommended defaults: occupancy dielectric, lens occupancy,
// continuous water count, switched nonbonded.
energyParams defaultEnergyParams();

// Bit-compatible with the shipped kernel: linear dielectric, full-volume
// occupancy, quantized waters, hard cutoff.  Use for regression testing and
// for reproducing results generated with existing calibrations.
energyParams legacyEnergyParams();

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

struct energyBreakdown
{
    double vdw;
    double torsion;
    double electrostatic;
    double solvationPolar;
    double solvationNonpolar;
    double solvationEntropy;
    double total;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct energyContext;

// Per-atom static properties.  Arrays are of length numAtoms and are copied,
// so the caller may free them immediately.
//
//   radius        van der Waals radius
//   epsilon       van der Waals well depth
//   charge        partial charge
//   residueIndex  monotonically nondecreasing residue id, used to skip
//                 exclusion lookups for distant pairs
//   silent        nonzero to exclude the atom from all energy terms
//   exclusionCount / exclusionList
//                 exclusionList is a ragged array flattened to a fixed stride:
//                 the neighbours excluded from atom i occupy
//                 exclusionList[i*exclusionStride .. +exclusionCount[i]).
struct energyTopology
{
    int                  numAtoms;
    const double*        radius;
    const double*        epsilon;
    const double*        charge;
    const int*           residueIndex;
    const unsigned char* silent;
    const int*           exclusionCount;
    const int*           exclusionList;
    int                  exclusionStride;

    // Dihedrals, flattened one entry per Fourier term rather than per quartet,
    // so a four-term series is four entries sharing the same atom indices.
    // Costs a little memory and removes all branching from the kernel.
    //
    //   torsionAtoms   4*torsionCount original atom indices, i-j-k-l
    //   torsionParams  3*torsionCount: barrier already divided by IDIVF,
    //                  phase in radians, periodicity
    //
    // Impropers are carried in the same arrays; they have the same functional
    // form and differ only in how the quartet was chosen.
    int                  torsionCount;
    const int*           torsionAtoms;
    const double*        torsionParams;

    // Zeroed by default so a caller that predates the torsion term, or a test
    // that fills only the nonbonded fields, gets no torsions rather than a
    // count read out of uninitialised stack memory.
    energyTopology()
        : numAtoms(0), radius(0), epsilon(0), charge(0), residueIndex(0),
          silent(0), exclusionCount(0), exclusionList(0), exclusionStride(0),
          torsionCount(0), torsionAtoms(0), torsionParams(0) {}
};

// Create a context for a fixed topology.  Returns null on failure.
energyContext* energyCreate(const energyTopology& topology, const energyParams& params);

// Release all device and host resources.
void energyDestroy(energyContext* ctx);

// Change parameters without reallocating.  Topology must not change.
void energySetParams(energyContext* ctx, const energyParams& params);
// Read back what the context is currently using, so a caller can vary one
// field without having to reconstruct the rest from the defaults.
const energyParams& energyGetParams(energyContext* ctx);

// Mark atoms as silent (excluded) without rebuilding the context.
void energySetSilent(energyContext* ctx, const unsigned char* silent);

// Evaluate the total energy for the supplied coordinates.
// x, y, z are host arrays of length numAtoms.
// Returns 0 on success, nonzero on CUDA error.  If breakdown is non-null it
// receives the per-term decomposition.
// Batched candidate evaluation.
//
// Evaluates nCand complete conformations in one set of kernel launches and
// writes nCand total energies to `totals`.  x, y and z each hold nCand * N
// doubles, candidate k occupying [k*N, (k+1)*N).
//
// A protein leaves the GPU nearly idle -- a 600-atom system is ~19 warps on
// 1280 cores -- so many candidates cost little more than one.  All candidates
// share the spatial order built from the context's current resident
// coordinates, which is exact because the order only drives bounding-box
// culling.  Set the base conformation with energySetCoords first so that order
// reflects a representative geometry.
//
// Agreement with energyCompute is exact in exact arithmetic, and bitwise in
// FP64.  In FP32 it is ~1e-9 relative: candidates share the base spatial order
// while a lone evaluation sorts on its own coordinates, so pairs accumulate in
// a different order.  That is ~1e-6 kcal/mol on a typical protein energy, six
// orders below KT.  Within a batch the ranking is exactly self-consistent,
// since every candidate uses the same order.
//
// Returns 0 on success, -1 on failure.
// Register the sidechain rotation groups once, so candidate conformations can be
// generated on the device instead of being built on the host and uploaded.
//
// Group g rotates members[memberStart[g] .. memberStart[g+1]) about the axis
// running from atom axisA[g] to atom axisB[g].  Members must be the atoms
// strictly distal to axisB in the bonded tree -- the same set the host transform
// moves -- and must not include the axis atoms themselves.  Indices are into the
// original atom order.  Groups belonging to one residue must be consecutive and
// ordered chi_1, chi_2, ..., because they are applied as a kinematic chain.
//
// Passing numGroups == 0 clears the table.
// Read back candidate k's generated coordinates, original atom order. For
// verifying the device transform against the host one directly, without the
// r^-12 conditioning of the energy in between.
int energyGetBatchCoords(energyContext* ctx, int k, double* x, double* y, double* z);

int energySetRotationGroups(energyContext* ctx, int numGroups,
                            const int* axisA, const int* axisB,
                            const int* memberStart, const int* members);

// Generate nCand conformations on the device and evaluate them in one launch.
//
// Each candidate starts from the resident coordinates and applies its own angle
// deltas, in degrees, to groups [groupBegin, groupBegin + nGroups).  anglesDeg is
// row-major [nCand][nGroups].  This is the same convention as
// residue::setChiByDelta, and reproduces it to within rounding: measured max
// coordinate deviation from the host transform is ~6e-6 A in FP32 and ~1e-14 A
// in FP64.
//
// It is deliberately not bitwise, because it cannot be. CUDA's double sin/cos
// differ from glibc's by up to 1 ulp on some arguments, so the device and host
// rotation matrices differ before any coordinate is touched. tests/rotamerTest.cc
// pins the agreement at a few ulp rather than asserting equality.
//
// The host cost per candidate is nGroups angles rather than 3N coordinates, and
// no host-side dihedral transform is needed at all.  That is what makes large
// candidate counts -- enumerating a rotamer grid rather than sampling it --
// affordable.
//
// Accuracy against energyCompute is as for energyComputeBatch.
int energyComputeRotamerBatch(energyContext* ctx, int nCand,
                              int groupBegin, int nGroups,
                              const double* anglesDeg, double* totals);

int energyComputeBatch(energyContext* ctx, int nCand,
                       const double* x, const double* y, const double* z,
                       double* totals);

// ---------------------------------------------------------------------------
// Population Monte Carlo over replica states
// ---------------------------------------------------------------------------
//
// A replica is an independent Metropolis walker holding its own accepted
// conformation on the device.  The distinction from energyComputeBatch matters
// for search efficiency rather than for speed: best-of-K spends K evaluations
// to advance one chain by a single step, while K replicas spend the same K
// evaluations to advance K chains by one step each.  The GPU cost is identical.
// Note that "identical GPU cost" means identical to best-of-K at the same K --
// not free.  A 64-batch on 1ubq costs 4.3x the wall time of a single evaluation,
// so a given chain advances 4.3x more slowly than it would running alone.  For
// minimisation that trade is a loss: measured energy at a fixed budget is
// monotone in sweeps-per-chain, and small K wins.  See the comment on
// protMinReplicaCU in protein.cc for the numbers.
//
// Best-of-K followed by a Metropolis test is also not a valid sampler -- the
// proposal is biased toward low energy and the acceptance carries no
// compensating term, so the chain does not converge to the Boltzmann
// distribution of the energy function.  Replicas restore that, because each
// proposal is drawn from a symmetric kernel and tested on its own.
//
// Usage per sweep:
//   energySetReplicas      once, seeding every replica from resident coords
//   energyComputeReplicaBatch   propose and evaluate all replicas
//   energyCommitReplicas   write back only the accepted proposals
//
// energySetReplicas seeds all nRepl replicas from the resident coordinates.
// Replicas share the spatial order built from those coordinates; see the note
// in energy.cu on divergence drag.
int energySetReplicas(energyContext* ctx, int nRepl);

// Propose one move per replica and evaluate all of them in a single launch.
// Replica k rotates chi groups [groupBegin[k], groupBegin[k] + nGroups[k]) of
// its own state by anglesDeg[k * angleStride + g].  angleStride must be at
// least max(nGroups[k]); the ragged tail is ignored, so a mixed set of residue
// types needs no prefix sum.  Totals are returned in energy units, one per
// replica, and the proposed coordinates remain in the batch buffer for
// energyCommitReplicas.
int energyComputeReplicaBatch(energyContext* ctx, int nRepl,
                              const int* groupBegin, const int* nGroups,
                              const double* anglesDeg, int angleStride,
                              double* totals);

// Commit the proposals for replicas with accept[k] != 0.  Rejected replicas
// retain their previous state at no additional cost.  Must follow a
// energyComputeReplicaBatch call with the same nRepl.
int energyCommitReplicas(energyContext* ctx, int nRepl, const int* accept);

// Read back replica k's accepted coordinates in original atom order.
int energyGetReplicaCoords(energyContext* ctx, int k, double* x, double* y, double* z);

int energyCompute(energyContext* ctx,
                  const double* x, const double* y, const double* z,
                  double* totalOut, energyBreakdown* breakdown);

// As energyCompute, but additionally writes the per-atom energy (the sum of
// all five nonbonded terms attributed to that atom, plus a quarter of each
// torsion the atom takes part in) into perAtomOut, which must have room for
// topology.numAtoms doubles and is indexed in the caller's original atom
// order.  Every pair interaction is split evenly between its two atoms and
// every torsion evenly between its four, so the per-atom values sum to the
// total.  This is what lets callers derive per-residue energies without a
// second implementation of the model.
int energyComputeAtoms(energyContext* ctx,
                       const double* x, const double* y, const double* z,
                       double* totalOut, energyBreakdown* breakdown,
                       double* perAtomOut);

// ---------------------------------------------------------------------------
// Frozen dielectric
// ---------------------------------------------------------------------------
//
// The local dielectric is the one many-body piece of this model, and it is what
// makes a sidechain move expensive: moving one chi perturbs eps on far more
// atoms than it moves, so every pair touching those atoms has to be re-summed.
// See docs/packing-search.md for the measurement.
//
// These calls hold the occupancy field -- and therefore the shell water count,
// the water fraction, the local dielectric and both solvation terms, which are
// all functions of it -- fixed at a reference conformation, while allowing a
// named set of atoms to be recomputed exactly.  Freezing occupancy rather than
// eps keeps the solvation terms consistent with the dielectric screening them.
//
// Snapshot the occupancy field at the given coordinates and hold it.  Pass null
// coordinates to use whatever is already resident on the device.  Thaws nothing
// by default.  Returns 0 on success.
int energyFreezeDielectric(energyContext* ctx,
                           const double* x, const double* y, const double* z);

// Choose which atoms are recomputed rather than held, by original-order index.
// Passing a null list or a negative count thaws every atom, which must
// reproduce the fully coupled energy exactly.
//
// With accumulate nonzero the atoms are added to the current set rather than
// replacing it.  That matters for correctness, not convenience: a move changes
// the occupancy of every atom whose hydration shell the moved atoms enter OR
// leave, so an exact thaw set is the union taken over the conformation before
// the move and the conformation after it.  Building it from one conformation
// alone is exact only when the move is small.
int energySetDielectricThaw(energyContext* ctx, const int* atoms, int count,
                            int accumulate);

// How many atoms are currently thawed.  Reports the union, so it is the number
// to quote when sizing the exemption; a single call's own list is not it.
double energyDielectricInfluenceRadius(energyContext* ctx);

// Evaluate only the part of the energy a move can have changed: every pair with
// at least one end in `atoms`, plus the per-atom solvation of those atoms.  The
// dielectric must be frozen and `atoms` must be exactly the thaw set, since
// those are the atoms whose occupancy is allowed to differ from the snapshot.
// Track the total as E_new = E_old - P(old) + P(new) + torsion; that identity is
// exact, not an approximation.  Pass atoms sorted by index for reproducibility.
int energyComputeDelta(energyContext* ctx,
                       const double* x, const double* y, const double* z,
                       const int* atoms, int count,
                       double* partOut, double* torsionOut);

// Refresh the held field from the occupancy currently resident on the device,
// which after a delta is exactly the field of the conformation just evaluated.
// This is what makes a chain of deltas affordable: the alternative, freezing
// again from coordinates, recomputes occupancy over every atom pair.
int energyRefreezeDielectric(energyContext* ctx);

int energyDielectricThawCount(const energyContext* ctx);

// As energyComputeDelta, but for nCand candidate conformations of the same
// move at once.  x, y and z are nCand * numAtoms arrays in original atom order;
// only the entries named by `atoms` are read, since every candidate is seeded
// from the resident conformation and differs from it nowhere else.  `atoms`
// must be exactly the thaw set, and must be the union over all candidates --
// each candidate's own moved atoms are a subset of it.  partOut and torsionOut
// receive nCand values, and candidate k's total is
// E_old - P(old) + partOut[k] + torsionOut[k], the same identity the
// single-candidate delta uses.
// `moved`, when given, is the subset of `atoms` that actually differs between
// candidates -- in practice the one residue a sidechain move touches.  The rest
// of the thaw set is identical in every candidate and identical to the resident
// conformation the batch is already seeded from, so staging it is redundant.
// Passing it shrinks the host-to-device transfer from the changed set to the
// moved set, which on 1crn is roughly a fourteenth of the volume.  Pass 0 to
// stage the whole changed set, which is what the caller must do if a move can
// displace atoms outside a known support.
int energyComputeBatchDelta(energyContext* ctx, int nCand,
                            const double* x, const double* y, const double* z,
                            const int* atoms, int count,
                            double* partOut, double* torsionOut,
                            const int* moved = 0, int nMoved = 0);

// Return to the fully coupled model.
int energyReleaseDielectric(energyContext* ctx);

// Nonzero if a frozen field is currently in force.
int energyDielectricFrozen(const energyContext* ctx);

// Export the per-atom local dielectric and shell water count for the supplied
// coordinates.  Both arrays are indexed in the caller's original atom order and
// must have room for topology.numAtoms doubles; either may be null.  Silent
// atoms report zero.  These are read from the same occupancy field the energy
// pass uses, so a reported dielectric always matches the one the energy was
// computed with.
int shellCompute(energyContext* ctx,
                 const double* x, const double* y, const double* z,
                 double* dielectricOut, double* watersOut);

// Count steric clashes for the supplied coordinates using params.clash.
// Returns 0 on success, nonzero on CUDA error.
int clashCompute(energyContext* ctx,
                 const double* x, const double* y, const double* z,
                 int* clashCountOut);

// As clashCompute, but also reports participation counts.  perAtomOut[i] is
// the number of clashing pairs atom i takes part in, so each pair is recorded
// once by each of its two atoms and sum(perAtomOut) == 2 * clashCountOut.
// The array is indexed in the caller's original atom order and must have room
// for topology.numAtoms ints.
int clashComputeAtoms(energyContext* ctx,
                      const double* x, const double* y, const double* z,
                      int* clashCountOut, int* perAtomOut);

// ---------------------------------------------------------------------------
// Residency
// ---------------------------------------------------------------------------
//
// The context owns a canonical copy of the coordinates on the device.  A
// caller may upload once, mutate them in place across many evaluations, and
// read them back only when it actually needs them on the host.  This is what a
// minimisation loop wants: the structure stays resident and the host sees only
// scalars.
//
// The *Resident entry points below evaluate whatever is currently in that
// device state.  They are exactly equivalent to their host-array counterparts
// called with the same coordinates; the host-array forms are now thin wrappers
// that upload and then delegate.

// Upload host coordinates into the resident device state.
void energyInvalidateTorsionBaseline(energyContext* ctx);

int energySetCoords(energyContext* ctx,
                    const double* x, const double* y, const double* z);

// Read the resident coordinates back.  Any of x, y, z may be null.
int energyGetCoords(energyContext* ctx, double* x, double* y, double* z);

// Save / restore the resident coordinates entirely on the device, for cheap
// accept/reject in a Monte Carlo loop.  No host transfer is involved.
int energySnapshot(energyContext* ctx);
int energyRestore(energyContext* ctx);

// Evaluate using the resident coordinates.  perAtomOut, breakdown and the
// per-atom arrays may be null.
int energyComputeResident(energyContext* ctx, double* totalOut,
                          energyBreakdown* breakdown, double* perAtomOut);
int clashComputeResident(energyContext* ctx, int* clashCountOut, int* perAtomOut);
int shellComputeResident(energyContext* ctx, double* dielectricOut, double* watersOut);

// Human-readable description of the last error, or null if none.
const char* energyLastError(energyContext* ctx);

#endif
