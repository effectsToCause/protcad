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
};

// Create a context for a fixed topology.  Returns null on failure.
energyContext* energyCreate(const energyTopology& topology, const energyParams& params);

// Release all device and host resources.
void energyDestroy(energyContext* ctx);

// Change parameters without reallocating.  Topology must not change.
void energySetParams(energyContext* ctx, const energyParams& params);

// Mark atoms as silent (excluded) without rebuilding the context.
void energySetSilent(energyContext* ctx, const unsigned char* silent);

// Evaluate the total energy for the supplied coordinates.
// x, y, z are host arrays of length numAtoms.
// Returns 0 on success, nonzero on CUDA error.  If breakdown is non-null it
// receives the per-term decomposition.
int energyCompute(energyContext* ctx,
                  const double* x, const double* y, const double* z,
                  double* totalOut, energyBreakdown* breakdown);

// As energyCompute, but additionally writes the per-atom energy (the sum of
// all five terms attributed to that atom) into perAtomOut, which must have
// room for topology.numAtoms doubles and is indexed in the caller's original
// atom order.  Every pair interaction is split evenly between its two atoms,
// so the per-atom values sum to the total.  This is what lets callers derive
// per-residue energies without a second implementation of the model.
int energyComputeAtoms(energyContext* ctx,
                       const double* x, const double* y, const double* z,
                       double* totalOut, energyBreakdown* breakdown,
                       double* perAtomOut);

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

// Human-readable description of the last error, or null if none.
const char* energyLastError(energyContext* ctx);

#endif
