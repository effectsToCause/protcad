// energy.cu
//
// GPU evaluation of ProtCAD's atom-atom interaction energy with a local
// effective dielectric.
//
// Structure
// ---------
//   1. Atoms are binned into a uniform grid and reordered so that 32 spatially
//      adjacent atoms occupy one "tile".  The sort is a counting sort followed
//      by a per-cell insertion sort on the original atom index, which makes the
//      resulting order a deterministic function of the coordinates alone.
//   2. Each tile gets an axis-aligned bounding box.  Both the occupancy pass
//      and the energy pass walk tiles and reject an entire 32x32 block of pairs
//      with a single box-box distance test.  This replaces the old dense O(N^2)
//      sweep without the over-inclusion of a fixed cell stencil.
//   3. Every thread owns one atom and accumulates into its own register, then
//      writes one value per term.  The host performs a Kahan sum in original
//      atom order.  There is no atomicAdd anywhere in the energy path, so the
//      result is bitwise reproducible.
//
// See energy.h for the model options.  legacyEnergyParams() reproduces the
// physics of the original implementation for regression purposes.

#include "energy.h"
#include <sys/time.h>

// Internal working precision.  Kept out of energy.h so the public ABI is
// independent of this flag.
#ifdef PROTCAD_ENERGY_FP64
typedef double ereal;
#else
typedef float ereal;
#endif

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

namespace
{

// 32 atoms per tile matches the warp width: one warp owns an i-tile and walks
// j-tiles, so the box-box rejection test is warp-uniform and free of divergence.
const int TILE = 32;
const int WARPS_PER_BLOCK = 4;
const int BLOCK = TILE * WARPS_PER_BLOCK;

// Target atoms per grid cell.  Sized to TILE so that a tile of 32 consecutive
// sorted atoms tends to come from a single cell and therefore has a compact,
// roughly cubic bounding box.
const int ATOMS_PER_CELL_TARGET = TILE;

__device__ __constant__ ereal c_kc  = ereal(332.0636);   // Coulomb constant, kcal*A/(mol*e^2)
__device__ __constant__ ereal c_kb  = ereal(0.0019872041); // Boltzmann, kcal/(mol*K)
__device__ __constant__ ereal c_T   = ereal(300.0);
__device__ __constant__ ereal c_pi  = ereal(3.1415926535);
// 4*pi/3 is passed in as v43 rather than held in constant memory: the original
// used a truncated 4.188 instead of 4.18879, a 0.02% bias on every atomic
// volume.  Legacy mode keeps the truncated value so regression comparisons stay
// clean; default mode uses the correct one.

#define CUDA_OK(ctx, expr)                                                    \
    do {                                                                      \
        cudaError_t _e = (expr);                                              \
        if (_e != cudaSuccess) {                                              \
            setError((ctx), #expr, cudaGetErrorString(_e), __FILE__, __LINE__);\
            return -1;                                                        \
        }                                                                     \
    } while (0)

} // namespace

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct energyContext
{
    int N;          // real atoms
    int nPad;       // padded to a multiple of TILE
    int nTiles;

    energyParams p;

    // Static per-atom properties, original order.
    ereal *d_rad, *d_sqrtEps, *d_chg, *d_selfVol;
    ereal maxRadius;   // largest atomic radius, sets the exemption radius
    // Delta evaluation scratch.  All sized once, so a move costs no allocation.
    unsigned char* d_cmask;   // changed set, sorted order
    int*   d_tileList;        // i-tiles holding at least one changed atom
    int*   d_tileCount;
    int*   d_inv;             // original index -> sorted slot
    int*   d_atoms;           // changed set as original indices
    ereal* d_eVdw2;
    ereal* d_eEle2;
    ereal* d_gather;          // 7 terms x count, compacted for the host sum
    ereal* d_dpart;           // delta partials, four blocks of dSplitCap * nPad
    int    dSplitCap;
    ereal* d_stage;           // 3 x count staging for the scattered coord upload
    ereal* h_stage;
    int*   d_torList;         // torsions touched by the changed set
    ereal* d_torSub;
    ereal* h_torSub;
    std::vector<int> h_torList;
    std::vector<int> torStart, torIdx;   // CSR: atom -> torsions containing it
    int    torPrimed;         // h_torE holds every torsion at the current coords
    int    deltaSetValid;     // cmask / inv / tileList match d_thaw and the sort
    int    deltaNList;
    int   *d_resIndex;
    unsigned char *d_silent;
    int   *d_exclCount, *d_exclList, *d_exclSpan;
    int    exclStride;

    // Dihedrals, one entry per Fourier term.  Fixed topology, so these are
    // uploaded once and never touched again.
    int    nTor;
    int   *d_torAtoms;
    ereal *d_torBarrier, *d_torPhase, *d_torPeriod;
    ereal *d_torE;                // nTor * torCap, per-term energies
    int    torCap;                // candidates d_torE is sized for
    std::vector<ereal> h_torE;

    // Coordinates, original order.  These are the canonical state: callers may
    // upload once and then mutate them on the device across many evaluations.
    ereal *d_x, *d_y, *d_z;

    // Snapshot for cheap accept/reject in a Monte Carlo loop.
    ereal *d_xSave, *d_ySave, *d_zSave;
    bool   haveSnapshot;
    bool   coordsValid;

    // Device-side coordinate bounds, so a resident evaluation never has to
    // read coordinates back to the host just to size the grid.
    ereal *d_bounds;     // 6: minX,minY,minZ,maxX,maxY,maxZ

    // Sorted (tile) order.
    int   *d_order;      // sorted slot -> original index
    ereal *d_sx, *d_sy, *d_sz;
    ereal *d_srad, *d_ssqrtEps, *d_schg, *d_sselfVol;
    int   *d_sresIndex, *d_sorig;
    unsigned char *d_ssilent;
    ereal *d_tileLo, *d_tileHi;   // 3 * nTiles each

    // Intermediates and outputs, sorted order.
    ereal *d_occ;                 // occluded shell volume

    // Frozen dielectric field.  The whole solvent model -- shell waters, water
    // fraction, local dielectric, and both solvation terms -- is a function of
    // this one occupancy number, so freezing occupancy freezes all of them
    // mutually consistently.  Freezing eps alone would leave the solvation
    // terms disagreeing with the dielectric that screens them.
    //
    // Held in ORIGINAL atom order, not sorted order: the spatial sort is
    // rebuilt whenever coordinates move, so a sorted-order snapshot would be
    // silently rebound to different atoms on the next evaluation.
    ereal *d_occFrozen;
    unsigned char *d_thaw;        // 1 = recompute this atom, 0 = use frozen
    std::vector<unsigned char> h_thaw;   // host mirror, so thaw sets can be unioned
    int    freezeActive;
    ereal *d_eVdw, *d_eEle, *d_eSolvP, *d_eSolvN, *d_eSolvS;
    int   *d_clash;

    // Grid.
    int   *d_cellOf, *d_cellCount, *d_cellStart, *d_cellCursor;
    int    cellCapacity;

    // Batched candidate evaluation.  Grown on demand; batchCap is the number of
    // candidates the buffers currently hold.
    int    batchCap;
    ereal *d_bx, *d_by, *d_bz;
    ereal *d_bsx, *d_bsy, *d_bsz;
    ereal *d_bocc, *d_btileLo, *d_btileHi;
    ereal *d_bterms;

    // Batched delta evaluation.  The batch buffers above hold the candidate
    // conformations; these hold what the restricted sum needs on top of them --
    // the second pair accumulator that removes the double count, the per-chunk
    // partials at the delta's own j split, and the scatter/gather staging.
    // Sized by candidate count as well as by changed-set size, so they are
    // separate from the single-move delta's buffers rather than shared with
    // them.
    ereal *d_bdpart;
    int    bdSplitCap, bdCandCap;
    ereal *d_bterms2;
    ereal *d_bstage, *d_bgather, *d_btgat;
    size_t bdStageCap, bdGatherCap, btgatCap;
    int    *d_moved;          // atoms that differ between candidates
    size_t  movedCap;
    ereal  *d_torBase;        // primed per-torsion energies, device copy
    double *d_torPart;        // one torsion total per candidate
    size_t torBaseCap, torPartCap;
    // Per-candidate reduced part, so the changed set is summed on the device
    // rather than shipped home one atom at a time.
    double *d_bpart;
    size_t bdPartCap;

    // Per-chunk pair-term slices for the j-split energy launch, and the split
    // factor itself.  jSplit is a function of the tile count alone, never of
    // the candidate count, so a batched evaluation and a resident one group
    // their sums identically and stay bit-comparable.
    ereal *d_part;
    int    partCap;      // candidates the partial buffer is sized for
    int    jSplit;
    std::vector<ereal> h_bx, h_by, h_bz, h_bterms;
    std::vector<ereal> h_bstage, h_bgather;

    // Rotation groups for on-device candidate generation.  CSR: group g rotates
    // members[memberStart[g] .. memberStart[g+1]) about the axis axisA->axisB.
    int    numGroups;
    int   *d_axisA, *d_axisB, *d_memberStart, *d_members;
    int    angleCap;
    ereal *d_angles;

    // Replica states for population Monte Carlo.  Each replica is an
    // independent walker holding its own accepted conformation.  A step
    // proposes into the batch buffer, evaluates all replicas in one launch,
    // and commits only the accepted ones back here.
    int    replCap;
    ereal *d_rx, *d_ry, *d_rz;
    int   *d_groupBegin, *d_nGroups, *d_accept;

    // Host staging (pinned).
    ereal *h_x, *h_y, *h_z;
    ereal *h_terms;               // 5 * nPad
    int   *h_order;

    std::vector<double> perAtom;  // scratch for deterministic host reduction
    // invOrder[i] is the sorted slot holding original atom i, i.e. the inverse
    // of h_order restricted to the non-padding slots.  h_order is a property of
    // the spatial sort and is therefore fixed for a whole batch, so building
    // this once lets the reduction read the device buffer directly in original
    // atom order instead of scattering into perAtom for every term of every
    // candidate.  The summation order is unchanged, so results stay bitwise
    // identical to the scatter-then-sum version.
    std::vector<int> invOrder;
    // Host copy of the torsion quartets, used to attribute each torsion's
    // energy back to its four atoms for the per-atom export.
    std::vector<int> h_torAtoms;

    std::string lastError;
};
static std::string gLastError;


static void setError(energyContext* ctx, const char* expr, const char* msg,
                     const char* file, int line)
{
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s failed: %s (%s:%d)", expr, msg, file, line);
    if (ctx) ctx->lastError = buf;
    // Also recorded globally so failures during energyCreate -- when there is
    // no context to attach the message to -- are still reportable.
    gLastError = buf;
    fprintf(stderr, "energy.cu: %s\n", buf);
}

// Never returns NULL: callers stream this straight into an ostream, where a
// null char* would be undefined behaviour.
const char* energyLastError(energyContext* ctx)
{
    if (ctx && !ctx->lastError.empty()) return ctx->lastError.c_str();
    if (!gLastError.empty()) return gLastError.c_str();
    return "no error recorded";
}

// ---------------------------------------------------------------------------
// Parameter presets
// ---------------------------------------------------------------------------

energyParams defaultEnergyParams()
{
    energyParams p;
    p.waterRadius            = ereal(1.4);
    p.effectiveWaterDiameter = ereal(4.35);
    p.waterVolume            = ereal(107.31);
    p.occlusionScale         = ereal(0.5);
    p.waterPolarizability    = ereal(1.47);
    p.waterEpsilon           = ereal(0.152);
    p.waterCharge            = ereal(0.0);

    p.dielectric     = DIELECTRIC_OCCUPANCY;
    p.pairMixing     = PAIRMIX_HARMONIC;
    p.occupancy      = OCCUPANCY_LENS;
    p.epsProtein     = ereal(2.0);
    p.epsWater       = ereal(78.4);
    p.quantizeWaters = 0;

    p.cutoff        = ereal(12.0);
    p.switchStart   = ereal(10.0);
    p.useSwitching  = 1;
    p.minSeparation = ereal(0.8);

    p.bornNormalize    = 1;
    p.bornReferenceCapacity = ereal(8.644);
    p.eSolvationFactor = ereal(1.0);
    p.hSolvationFactor = ereal(1.0);
    p.entropyFactor    = ereal(1.0);

    p.vdwScale  = ereal(1.0);
    p.elecScale = ereal(1.0);

    p.torsionScale = 1.0;
    p.elec14Scale  = 1.0 / 1.2;   // Amber SCEE
    p.vdw14Scale   = 1.0 / 2.0;   // Amber SCNB

    // Single-lever off switch for the bonded term.  Torsions and 1-4 scaling
    // move together and never separately: the ff14SB barriers were fitted with
    // the 1-4 terms damped by exactly SCEE/SCNB, so turning off one and not the
    // other uses the parameters outside the fit that produced them.  Zeroing
    // the 1-4 scales reverts those pairs to full exclusion, which is what the
    // potential did before this term existed.
    if (getenv("PROTCAD_TORSION_OFF"))
    {
        p.torsionScale = 0.0;
        p.elec14Scale  = 0.0;
        p.vdw14Scale   = 0.0;
    }

    // Isotropic hard-clash test.  The legacy cube criterion is retained as
    // CLASH_INSCRIBED_CUBE for CPU-parity checks, but it is orientation
    // dependent and so unusable as a physical quantity; see energy.h.
    p.clash          = CLASH_SPHERE;
    p.clashTolerance = 0.905;

    p.exclusionResidueSpan = 2;
    return p;
}

energyParams legacyEnergyParams()
{
    energyParams p = defaultEnergyParams();
    p.dielectric     = DIELECTRIC_LEGACY_LINEAR;
    p.pairMixing     = PAIRMIX_ARITHMETIC;
    p.occupancy      = OCCUPANCY_LEGACY_FULLVOLUME;
    p.quantizeWaters = 1;
    p.useSwitching   = 0;
    p.minSeparation  = ereal(1e-6);
    p.bornNormalize  = 0;   // legacy scales the Born term by the raw water count
    p.clash          = CLASH_SPHERE;
    p.clashTolerance = 1.0;   // legacy GPU flagged any pair inside rI+rJ

    // No dihedral term, and 1-4 pairs excluded outright rather than damped.
    // These two go together: the ff14SB barriers were fitted against scaled
    // 1-4 interactions, so turning one off without the other is not a
    // reproduction of anything.
    p.torsionScale = 0.0;
    p.elec14Scale  = 0.0;
    p.vdw14Scale   = 0.0;
    return p;
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

namespace
{

// Type-generic math wrappers.  Calling sqrtf/fmaxf directly would silently
// demote to single precision in a -DPROTCAD_ENERGY_FP64 build and quietly
// destroy the very reference the FP64 build exists to provide.
#ifdef PROTCAD_ENERGY_FP64
__device__ __forceinline__ ereal esqrt(ereal a)  { return sqrt(a); }
__device__ __forceinline__ ereal emax(ereal a, ereal b) { return fmax(a, b); }
__device__ __forceinline__ ereal emin(ereal a, ereal b) { return fmin(a, b); }
__device__ __forceinline__ ereal eabs(ereal a)   { return fabs(a); }
__device__ __forceinline__ ereal etrunc(ereal a) { return trunc(a); }
__device__ __forceinline__ ereal ecos(ereal a)   { return cos(a); }
__device__ __forceinline__ ereal eatan2(ereal a, ereal b) { return atan2(a, b); }
#else
__device__ __forceinline__ ereal esqrt(ereal a)  { return sqrtf(a); }
__device__ __forceinline__ ereal emax(ereal a, ereal b) { return fmaxf(a, b); }
__device__ __forceinline__ ereal emin(ereal a, ereal b) { return fminf(a, b); }
__device__ __forceinline__ ereal eabs(ereal a)   { return fabsf(a); }
__device__ __forceinline__ ereal etrunc(ereal a) { return truncf(a); }
__device__ __forceinline__ ereal ecos(ereal a)   { return cosf(a); }
__device__ __forceinline__ ereal eatan2(ereal a, ereal b) { return atan2f(a, b); }
#endif

__device__ __forceinline__ ereal cube(ereal a) { return a * a * a; }

// Volume of the region of sphere (radius r, centre distance d) that lies inside
// sphere (radius R).  Standard two-sphere lens formula, with the containment
// and disjoint cases handled exactly so the result is continuous in d.
__device__ __forceinline__ ereal lensVolume(ereal d, ereal R, ereal r, ereal v43)
{
    if (d >= R + r)  return ereal(0);
    if (d <= R - r)  return v43 * cube(r);        // small sphere fully inside
    if (d <= r - R)  return v43 * cube(R);        // shell fully inside neighbour
    if (d < ereal(1e-6)) return v43 * cube(emin(R, r));

    // pi * (R + r - d)^2 * (d^2 + 2*d*r - 3*r*r + 2*d*R + 6*r*R - 3*R*R) / (12*d)
    ereal t = R + r - d;
    ereal poly = d * d + ereal(2) * d * r - ereal(3) * r * r
               + ereal(2) * d * R + ereal(6) * r * R - ereal(3) * R * R;
    ereal vol = ereal(3.1415926535) * t * t * poly / (ereal(12) * d);
    return vol > ereal(0) ? vol : ereal(0);
}

// Smooth switching function S(d): 1 below d0, 0 above d1, with S'=0 at both
// ends.  Removes the energy discontinuity at the cutoff, which otherwise puts a
// step of order 0.01-0.1 kcal/mol into every Monte Carlo move that pushes a
// charged pair across 12 A.
__device__ __forceinline__ ereal switchFn(ereal d2, ereal d0sq, ereal d1sq)
{
    if (d2 <= d0sq) return ereal(1);
    if (d2 >= d1sq) return ereal(0);
    ereal u = (d1sq - d2) / (d1sq - d0sq);
    return u * u * (ereal(3) - ereal(2) * u);
}

// Convert free shell volume to a local dielectric.  Shared by the pair loop and
// the solvation term so the two can never drift apart.
struct shellState
{
    ereal waters;      // number of waters in the free shell volume
    ereal fraction;    // free volume / shell volume, in [0,1]
    ereal eps;         // local dielectric
    ereal capacity;    // waters that would fit in a fully free shell
};

__device__ __forceinline__ shellState shellFromOcclusion(
    ereal occ, ereal rad, const energyParams p, ereal v43)
{
    shellState s;
    ereal shellVol = v43 * cube(rad + p.effectiveWaterDiameter);
    ereal selfVol  = v43 * cube(rad);

    ereal envVol = (occ + selfVol) * ereal(p.occlusionScale);

    ereal freeVol = shellVol - envVol;
    if (freeVol < ereal(0)) freeVol = ereal(0);
    if (freeVol > shellVol) freeVol = shellVol;

    s.capacity = shellVol / p.waterVolume;
    s.waters   = freeVol / p.waterVolume;
    if (p.quantizeWaters) s.waters = etrunc(s.waters);
    s.fraction = freeVol / shellVol;

    ereal pol = s.waters * p.waterPolarizability;

    if (p.dielectric == DIELECTRIC_LEGACY_LINEAR)
    {
        // What the shipped code actually computes.  The "(8*pi/3)*pol/1" term
        // minus "(4*pi/3)*pol" collapses to a single linear term because the
        // "/1" makes the intended Clausius-Mossotti denominator inert.
        s.eps = p.epsProtein + (ereal(4) * c_pi / ereal(3)) * pol;
    }
    else if (p.dielectric == DIELECTRIC_OCCUPANCY)
    {
        // Same shape, physical endpoints, no false mechanistic claim.
        s.eps = p.epsProtein + (p.epsWater - p.epsProtein) * s.fraction;
    }
    else // DIELECTRIC_CLAUSIUS_MOSSOTTI
    {
        ereal n = s.waters / shellVol;                       // number density
        ereal y = (ereal(4) * c_pi / ereal(3)) * n * p.waterPolarizability;
        if (y > ereal(0.95)) y = ereal(0.95);                // keep 1-y positive
        s.eps = p.epsProtein * (ereal(1) + ereal(2) * y) / (ereal(1) - y);
    }

    if (s.eps < ereal(1)) s.eps = ereal(1);
    return s;
}

__device__ __forceinline__ ereal mixDielectric(ereal a, ereal b, int model)
{
    if (model == PAIRMIX_HARMONIC)
        return ereal(2) * a * b / (a + b);
    return (a + b) * ereal(0.5);
}

// Squared distance from a point to an axis-aligned box, and box-to-box.
__device__ __forceinline__ ereal boxBoxDist2(
    ereal aLoX, ereal aLoY, ereal aLoZ, ereal aHiX, ereal aHiY, ereal aHiZ,
    ereal bLoX, ereal bLoY, ereal bLoZ, ereal bHiX, ereal bHiY, ereal bHiZ)
{
    ereal dx = emax(emax(bLoX - aHiX, aLoX - bHiX), ereal(0));
    ereal dy = emax(emax(bLoY - aHiY, aLoY - bHiY), ereal(0));
    ereal dz = emax(emax(bLoZ - aHiZ, aLoZ - bHiZ), ereal(0));
    return dx * dx + dy * dy + dz * dz;
}

// Returns 0 for a normal pair, 1 for a fully excluded one (1-2 or 1-3), and 2
// for a 1-4 pair, which Amber damps rather than removes.
//
// The two cases share one list.  A 1-4 partner is stored as -(j+1), which is
// unambiguous because atom indices are non-negative, and costs nothing: the
// scan already had to compare every entry.  Keeping one list preserves the
// stride, the memory traffic and the per-atom span bound exactly as they were.
__device__ __forceinline__ int exclusionCode(
    int origI, int origJ, int resI, int resJ, const int* exclSpan,
    const int* exclCount, const int* exclList, int stride)
{
    // Fast reject on sequence separation.  The bound is per atom and exact --
    // it is the largest residue separation actually present in that atom's
    // exclusion list -- rather than a global constant.  A global constant is
    // wrong: a disulfide excludes two cysteines that can be any distance apart
    // in sequence, and a fixed span silently discards the exclusion and leaves
    // an ~2.0 A S-S pair in the nonbonded sum.  Atoms with no exclusions get
    // -1 and reject immediately, so the common case is as cheap as before.
    int dr = resI - resJ;
    if (dr < 0) dr = -dr;
    if (dr > exclSpan[origI]) return 0;

    int n = exclCount[origI];
    const int* list = exclList + (size_t)origI * stride;
    for (int k = 0; k < n; ++k) {
        int e = list[k];
        if (e == origJ) return 1;
        if (e == -(origJ + 1)) return 2;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Dihedrals
// ---------------------------------------------------------------------------

// One thread per (term, candidate).  Terms are stored one per Fourier
// component, so a four-term series is four independent threads over the same
// quartet and the kernel has no inner loop and no divergence.
//
// The energy is written per term rather than accumulated, because accumulating
// would need atomics and atomics reorder floating point additions.  Every
// reduction in this file is ordered so that two evaluations of the same
// conformation agree bit for bit; that property is what lets the batch and
// resident paths be compared at all, and it is not worth trading for the
// memory this costs (one float per term per candidate).
__global__ void kTorsion(const ereal* __restrict__ x,
                         const ereal* __restrict__ y,
                         const ereal* __restrict__ z,
                         int N, int nCand, int nTor,
                         const int* __restrict__ tAtoms,
                         const ereal* __restrict__ tBarrier,
                         const ereal* __restrict__ tPhase,
                         const ereal* __restrict__ tPeriod,
                         const unsigned char* __restrict__ silent,
                         ereal scale,
                         ereal* __restrict__ out,
                         const int* __restrict__ torList = 0, int nList = 0)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (torList) { if (t >= nList) return; t = torList[t]; }
    if (t >= nTor || k >= nCand) return;

    const ereal* cx = x + (size_t)k * N;
    const ereal* cy = y + (size_t)k * N;
    const ereal* cz = z + (size_t)k * N;

    int ia = tAtoms[4 * t + 0], ib = tAtoms[4 * t + 1];
    int ic = tAtoms[4 * t + 2], id = tAtoms[4 * t + 3];

    // A silent atom is excluded from *all* energy terms, which has to include
    // the bonded ones: a torsion is a property of its four atoms jointly, so
    // if any of them is masked out the term has no owner left.  Without this
    // the mask silently left the entire torsion energy in place -- on a
    // two-chain system that was ~1036 of a ~1036 kcal/mol total, so an
    // "isolated chain" energy came back as the whole complex.
    if (silent && (silent[ia] || silent[ib] || silent[ic] || silent[id])) {
        out[(size_t)k * nTor + t] = ereal(0);
        return;
    }

    // b1 x b2 and b2 x b3 give the two half-plane normals; the torsion is the
    // angle between them, signed by b2.  Using atan2 of (n1 x n2).b2hat and
    // n1.n2 rather than acos of a normalised dot product keeps the full range
    // and stays well conditioned near 0 and pi, where acos loses most of its
    // significant digits to the vanishing derivative.
    ereal b1x = cx[ib] - cx[ia], b1y = cy[ib] - cy[ia], b1z = cz[ib] - cz[ia];
    ereal b2x = cx[ic] - cx[ib], b2y = cy[ic] - cy[ib], b2z = cz[ic] - cz[ib];
    ereal b3x = cx[id] - cx[ic], b3y = cy[id] - cy[ic], b3z = cz[id] - cz[ic];

    ereal n1x = b1y * b2z - b1z * b2y;
    ereal n1y = b1z * b2x - b1x * b2z;
    ereal n1z = b1x * b2y - b1y * b2x;

    ereal n2x = b2y * b3z - b2z * b3y;
    ereal n2y = b2z * b3x - b2x * b3z;
    ereal n2z = b2x * b3y - b2y * b3x;

    ereal b2len = esqrt(b2x * b2x + b2y * b2y + b2z * b2z);
    if (b2len <= ereal(0)) { out[(size_t)k * nTor + t] = ereal(0); return; }

    ereal mx = n1y * n2z - n1z * n2y;
    ereal my = n1z * n2x - n1x * n2z;
    ereal mz = n1x * n2y - n1y * n2x;

    ereal sy = (mx * b2x + my * b2y + mz * b2z) / b2len;
    ereal sxv = n1x * n2x + n1y * n2y + n1z * n2z;

    ereal phi = eatan2(sy, sxv);

    // E = (PK/IDIVF) * (1 + cos(n*phi - phase)); the divisor is already folded
    // into the barrier on the host.
    ereal e = tBarrier[t] * (ereal(1) + ecos(tPeriod[t] * phi - tPhase[t]));
    out[(size_t)k * nTor + t] = scale * e;
}

// ---------------------------------------------------------------------------
// Grid construction kernels
// ---------------------------------------------------------------------------

__global__ void kCellOf(const ereal* x, const ereal* y, const ereal* z, int N,
                        ereal oxr, ereal oyr, ereal ozr, ereal invCell,
                        int nx, int ny, int nz,
                        int* cellOf, int* cellCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int cx = int((x[i] - oxr) * invCell); cx = min(max(cx, 0), nx - 1);
    int cy = int((y[i] - oyr) * invCell); cy = min(max(cy, 0), ny - 1);
    int cz = int((z[i] - ozr) * invCell); cz = min(max(cz, 0), nz - 1);

    int c = (cz * ny + cy) * nx + cx;
    cellOf[i] = c;
    atomicAdd(&cellCount[c], 1);
}

__global__ void kPlace(const int* cellOf, int N, int* cursor, int* order)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int slot = atomicAdd(&cursor[cellOf[i]], 1);
    order[slot] = i;
}

// Sort each cell's members by original atom index.  This is what makes the whole
// pipeline deterministic despite the atomicAdd placement above; it is pure
// bookkeeping, not physics, so it must not cost more than the physics.
//
// One BLOCK per cell, not one thread.  The earlier thread-per-cell insertion
// sort was the single most expensive kernel in the library -- at the target
// occupancy of ~32 atoms/cell there are only N/32 cells, so a 600-atom protein
// launched ~20 active threads on a 1280-core GPU, each running a serial chain of
// several hundred dependent, uncoalesced global-memory read-modify-writes with
// no occupancy to hide the latency.  It measured 806us, three times the cost of
// the clash kernel and 40% of the energy kernel.
//
// Members of a cell are distinct atom indices, so there are no ties and a rank
// sort is exact: each element's destination is simply the number of members
// smaller than it.  That is O(n) work per thread out of shared memory, and the
// ranks are a permutation so the scatter is race-free.  Cells are not capacity
// bounded, so a cell too large for the staging buffer falls back to the old
// serial path rather than silently truncating.
#define SORT_CELL_SMEM 1024

__global__ void kSortCells(const int* cellStart, const int* cellCount,
                           int nCells, int* order)
{
    __shared__ int buf[SORT_CELL_SMEM];

    int c = blockIdx.x;
    if (c >= nCells) return;
    const int s = cellStart[c], n = cellCount[c];
    if (n < 2) return;

    if (n > SORT_CELL_SMEM) {
        if (threadIdx.x == 0) {
            for (int a = 1; a < n; ++a) {
                int key = order[s + a];
                int b = a - 1;
                while (b >= 0 && order[s + b] > key) { order[s + b + 1] = order[s + b]; --b; }
                order[s + b + 1] = key;
            }
        }
        return;
    }

    for (int a = threadIdx.x; a < n; a += blockDim.x) buf[a] = order[s + a];
    __syncthreads();

    for (int a = threadIdx.x; a < n; a += blockDim.x) {
        const int key = buf[a];
        int rank = 0;
        for (int b = 0; b < n; ++b) rank += (buf[b] < key);
        order[s + rank] = key;
    }
}

// Gather coordinates only, for a batch of candidate conformations that all
// share one spatial order.  The order is an AABB-culling optimization, not
// physics, so reusing the base conformation's order for every candidate is
// exact; it also keeps radii, charges, residue indices and silent flags
// batch-invariant, which is what makes the batch cheap.
__global__ void kGatherCoords(int nPad, int N, const int* __restrict__ order,
                              const ereal* __restrict__ bx,
                              const ereal* __restrict__ by,
                              const ereal* __restrict__ bz,
                              ereal* __restrict__ sx,
                              ereal* __restrict__ sy,
                              ereal* __restrict__ sz)
{
    const size_t k = blockIdx.y;
    bx += k * (size_t)N;    by += k * (size_t)N;    bz += k * (size_t)N;
    sx += k * (size_t)nPad; sy += k * (size_t)nPad; sz += k * (size_t)nPad;

    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nPad) return;
    if (s >= N) { sx[s] = ereal(1e9); sy[s] = ereal(1e9); sz[s] = ereal(1e9); return; }
    int i = order[s];
    sx[s] = bx[i]; sy[s] = by[i]; sz[s] = bz[i];
}

__global__ void kGather(int nPad, int N, const int* order,
                        const ereal* x, const ereal* y, const ereal* z,
                        const ereal* rad, const ereal* sqrtEps, const ereal* chg,
                        const ereal* selfVol, const int* resIndex,
                        const unsigned char* silent,
                        ereal* sx, ereal* sy, ereal* sz,
                        ereal* srad, ereal* ssqrtEps, ereal* schg,
                        ereal* sselfVol, int* sresIndex, int* sorig,
                        unsigned char* ssilent)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nPad) return;

    if (s >= N) {
        // Pad slots: silent, zero-valued, and parked far away so they can never
        // pass a bounding-box test against real atoms.
        sx[s] = ereal(1e9); sy[s] = ereal(1e9); sz[s] = ereal(1e9);
        srad[s] = ereal(0); ssqrtEps[s] = ereal(0); schg[s] = ereal(0);
        sselfVol[s] = ereal(0); sresIndex[s] = -1000000; sorig[s] = -1;
        ssilent[s] = 1;
        return;
    }

    int i = order[s];
    sx[s] = x[i]; sy[s] = y[i]; sz[s] = z[i];
    srad[s] = rad[i]; ssqrtEps[s] = sqrtEps[i]; schg[s] = chg[i];
    sselfVol[s] = selfVol[i]; sresIndex[s] = resIndex[i]; sorig[s] = i;
    ssilent[s] = silent[i];
}

__global__ void kTileBounds(int nTiles, const ereal* sx, const ereal* sy,
                            const ereal* sz, const ereal* srad,
                            const unsigned char* ssilent,
                            ereal* lo, ereal* hi)
{
    // Batch dimension.  gridDim.y is the candidate count; single-candidate
    // launches leave blockIdx.y at 0, so every offset below is zero and the
    // kernel is byte-identical to the unbatched path.  Only quantities that
    // depend on the conformation are strided; radii and silent flags are shared.
    {
        const size_t k = blockIdx.y, nPad = (size_t)nTiles * TILE;
        sx += k * nPad; sy += k * nPad; sz += k * nPad;
        lo += k * (size_t)nTiles * 3; hi += k * (size_t)nTiles * 3;
    }
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nTiles) return;

    ereal lx = ereal(1e30), ly = ereal(1e30), lz = ereal(1e30);
    ereal hx = ereal(-1e30), hy = ereal(-1e30), hz = ereal(-1e30);
    bool any = false;
    for (int k = 0; k < TILE; ++k) {
        int s = t * TILE + k;
        if (ssilent[s]) continue;
        any = true;
        // Inflate by the radius so a box test on centres is still conservative
        // for any criterion expressed as (d < cutoff + radI + radJ).
        ereal r = srad[s];
        lx = emin(lx, sx[s] - r); hx = emax(hx, sx[s] + r);
        ly = emin(ly, sy[s] - r); hy = emax(hy, sy[s] + r);
        lz = emin(lz, sz[s] - r); hz = emax(hz, sz[s] + r);
    }
    if (!any) { lx = ly = lz = ereal(1e29); hx = hy = hz = ereal(1e29); }
    lo[t * 3 + 0] = lx; lo[t * 3 + 1] = ly; lo[t * 3 + 2] = lz;
    hi[t * 3 + 0] = hx; hi[t * 3 + 1] = hy; hi[t * 3 + 2] = hz;
}

// ---------------------------------------------------------------------------
// Pass 1: hydration shell occlusion
// ---------------------------------------------------------------------------

__global__ void kOccupancy(int nTiles,
                           const ereal* __restrict__ sx,
                           const ereal* __restrict__ sy,
                           const ereal* __restrict__ sz,
                           const ereal* __restrict__ srad,
                           const ereal* __restrict__ sselfVol,
                           const unsigned char* __restrict__ ssilent,
                           const ereal* __restrict__ tileLo,
                           const ereal* __restrict__ tileHi,
                           energyParams p, ereal v43, int occModel,
                           int nCand, int jSplit,
                           ereal* __restrict__ occ,
                           const int* __restrict__ iTileList, int nList)
{
    // Same j-split as kEnergy; see chooseSplit.  The occlusion sum has no
    // epilogue at all, so every chunk simply owns a stride of j.
    const int chunk = blockIdx.z;
    {
        const size_t k = blockIdx.y, nPad = (size_t)nTiles * TILE;
        sx += k * nPad; sy += k * nPad; sz += k * nPad;
        occ += ((size_t)chunk * nCand + k) * nPad;
        tileLo += k * (size_t)nTiles * 3; tileHi += k * (size_t)nTiles * 3;
    }
    __shared__ ereal shX[BLOCK], shY[BLOCK], shZ[BLOCK], shR[BLOCK], shV[BLOCK];

    const int warp = threadIdx.x / TILE;
    const int lane = threadIdx.x % TILE;
    int iTile = blockIdx.x * WARPS_PER_BLOCK + warp;
    // A delta evaluation touches only the tiles holding thawed atoms.  The
    // j loop is unchanged: a restricted atom still needs every neighbour.
    if (iTileList) { if (iTile >= nList) return; iTile = iTileList[iTile]; }
    else if (iTile >= nTiles) return;

    const int s = iTile * TILE + lane;
    const ereal xi = sx[s], yi = sy[s], zi = sz[s], ri = srad[s];
    const bool active = !ssilent[s];

    // i's shell radius; the widest shell in the tile bounds the tile test.
    const ereal shellI = ri + p.effectiveWaterDiameter;

    const ereal aLoX = tileLo[iTile * 3 + 0], aLoY = tileLo[iTile * 3 + 1], aLoZ = tileLo[iTile * 3 + 2];
    const ereal aHiX = tileHi[iTile * 3 + 0], aHiY = tileHi[iTile * 3 + 1], aHiZ = tileHi[iTile * 3 + 2];

    // Largest shell radius in this tile, used for the warp-uniform reject.
    ereal maxShell = shellI;
    for (int off = 16; off > 0; off >>= 1)
        maxShell = emax(maxShell, __shfl_xor_sync(0xffffffff, maxShell, off));

    ereal acc = ereal(0);
    const int base = warp * TILE;

    for (int jTile = chunk; jTile < nTiles; jTile += jSplit)
    {
        ereal d2 = boxBoxDist2(aLoX, aLoY, aLoZ, aHiX, aHiY, aHiZ,
                               tileLo[jTile * 3 + 0], tileLo[jTile * 3 + 1], tileLo[jTile * 3 + 2],
                               tileHi[jTile * 3 + 0], tileHi[jTile * 3 + 1], tileHi[jTile * 3 + 2]);
        if (d2 > maxShell * maxShell) continue;

        int js = jTile * TILE + lane;
        shX[base + lane] = sx[js]; shY[base + lane] = sy[js]; shZ[base + lane] = sz[js];
        shR[base + lane] = srad[js]; shV[base + lane] = sselfVol[js];
        // Park silent neighbours where no distance test can reach them.
        if (ssilent[js]) shX[base + lane] = ereal(1e9);
        __syncwarp();

        if (active)
        {
            #pragma unroll 8
            for (int k = 0; k < TILE; ++k)
            {
                int jsl = jTile * TILE + k;
                if (jsl == s) continue;                 // never self-occlude
                ereal dx = xi - shX[base + k];
                ereal dy = yi - shY[base + k];
                ereal dz = zi - shZ[base + k];
                ereal r2 = dx * dx + dy * dy + dz * dz;
                ereal rj = shR[base + k];

                if (occModel == OCCUPANCY_LEGACY_FULLVOLUME) {
                    if (r2 < shellI * shellI) acc += shV[base + k];
                } else {
                    if (r2 < (shellI + rj) * (shellI + rj))
                        acc += lensVolume(esqrt(r2), shellI, rj, v43);
                }
            }
        }
        __syncwarp();
    }

    if (active) occ[s] = acc;
    else        occ[s] = ereal(0);
}

// ---------------------------------------------------------------------------
// Pass 2: pair energies plus per-atom solvation
// ---------------------------------------------------------------------------

__global__ void kEnergy(int nTiles,
                        const ereal* __restrict__ sx,
                        const ereal* __restrict__ sy,
                        const ereal* __restrict__ sz,
                        const ereal* __restrict__ srad,
                        const ereal* __restrict__ ssqrtEps,
                        const ereal* __restrict__ schg,
                        const int*   __restrict__ sresIndex,
                        const int*   __restrict__ sorig,
                        const unsigned char* __restrict__ ssilent,
                        const ereal* __restrict__ occ,
                        const ereal* __restrict__ tileLo,
                        const ereal* __restrict__ tileHi,
                        const int* __restrict__ exclCount,
                        const int* __restrict__ exclList, int exclStride,
                        const int* __restrict__ exclSpan,
                        energyParams p, ereal v43, int nCand, int jSplit,
                        ereal* __restrict__ eVdw, ereal* __restrict__ eEle,
                        ereal* __restrict__ eSolvP, ereal* __restrict__ eSolvN,
                        ereal* __restrict__ eSolvS,
                        const int* __restrict__ iTileList, int nList,
                        const unsigned char* __restrict__ cmask,
                        ereal* __restrict__ eVdw2, ereal* __restrict__ eEle2)
{
    // blockIdx.z partitions the j-tile loop. Each chunk accumulates the pair
    // terms for its own stride of j into a separate slice, and a following
    // reduction sums the slices in fixed chunk order, so the result stays
    // deterministic. Only the pair terms need this; the solvation terms are
    // functions of the atom's own occupancy, so chunk 0 writes them once.
    const int chunk = blockIdx.z;
    {
        const size_t k = blockIdx.y, nPad = (size_t)nTiles * TILE;
        sx += k * nPad; sy += k * nPad; sz += k * nPad; occ += k * nPad;
        tileLo += k * (size_t)nTiles * 3; tileHi += k * (size_t)nTiles * 3;
        eVdw += ((size_t)chunk * nCand + k) * nPad;
        eEle += ((size_t)chunk * nCand + k) * nPad;
        if (eVdw2) eVdw2 += ((size_t)chunk * nCand + k) * nPad;
        if (eEle2) eEle2 += ((size_t)chunk * nCand + k) * nPad;
        eSolvP += k * nPad; eSolvN += k * nPad; eSolvS += k * nPad;
    }
    __shared__ ereal shX[BLOCK], shY[BLOCK], shZ[BLOCK];
    __shared__ ereal shR[BLOCK], shE[BLOCK], shQ[BLOCK], shD[BLOCK];
    __shared__ int   shRes[BLOCK], shOrig[BLOCK];
    __shared__ unsigned char shC[BLOCK];

    const int warp = threadIdx.x / TILE;
    const int lane = threadIdx.x % TILE;
    int iTile = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (iTileList) { if (iTile >= nList) return; iTile = iTileList[iTile]; }
    else if (iTile >= nTiles) return;

    const int s = iTile * TILE + lane;
    // cmask marks the atoms whose position or dielectric a move changed.  A
    // tile is launched if any of its lanes is in that set, so the rest of the
    // lanes fall out here rather than at tile granularity.
    const bool active = !ssilent[s] && (!cmask || cmask[s]);

    const ereal xi = sx[s], yi = sy[s], zi = sz[s];
    const ereal ri = srad[s], ei = ssqrtEps[s], qi = schg[s];
    const int   resI = sresIndex[s], origI = sorig[s];

    // The dielectric of i is a property of i alone, so compute it once here
    // rather than twice inside every pair as the original did.
    const shellState si = shellFromOcclusion(occ[s], ri, p, v43);

    const ereal cutSq   = p.cutoff * p.cutoff;
    const ereal swStart = p.switchStart * p.switchStart;
    const ereal minSep  = p.minSeparation;

    const ereal aLoX = tileLo[iTile * 3 + 0], aLoY = tileLo[iTile * 3 + 1], aLoZ = tileLo[iTile * 3 + 2];
    const ereal aHiX = tileHi[iTile * 3 + 0], aHiY = tileHi[iTile * 3 + 1], aHiZ = tileHi[iTile * 3 + 2];

    ereal accVdw = ereal(0), accEle = ereal(0);
    ereal accVdw2 = ereal(0), accEle2 = ereal(0);
    const int base = warp * TILE;

    for (int jTile = chunk; jTile < nTiles; jTile += jSplit)
    {
        ereal d2 = boxBoxDist2(aLoX, aLoY, aLoZ, aHiX, aHiY, aHiZ,
                               tileLo[jTile * 3 + 0], tileLo[jTile * 3 + 1], tileLo[jTile * 3 + 2],
                               tileHi[jTile * 3 + 0], tileHi[jTile * 3 + 1], tileHi[jTile * 3 + 2]);
        if (d2 > cutSq) continue;

        int js = jTile * TILE + lane;
        shX[base + lane] = sx[js]; shY[base + lane] = sy[js]; shZ[base + lane] = sz[js];
        shR[base + lane] = srad[js]; shE[base + lane] = ssqrtEps[js];
        shQ[base + lane] = schg[js]; shD[base + lane] = shellFromOcclusion(occ[js], srad[js], p, v43).eps;
        shRes[base + lane] = sresIndex[js]; shOrig[base + lane] = sorig[js];
        shC[base + lane] = cmask ? cmask[js] : (unsigned char)0;
        if (ssilent[js]) shX[base + lane] = ereal(1e9);
        __syncwarp();

        if (active)
        {
            for (int k = 0; k < TILE; ++k)
            {
                int jsl = jTile * TILE + k;
                if (jsl == s) continue;                 // no self interaction

                ereal dx = xi - shX[base + k];
                ereal dy = yi - shY[base + k];
                ereal dz = zi - shZ[base + k];
                ereal r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= cutSq) continue;

                int xcode = exclusionCode(origI, shOrig[base + k], resI,
                                          shRes[base + k], exclSpan, exclCount,
                                          exclList, exclStride);
                if (xcode == 1) continue;

                // 1-4 pairs survive at reduced weight.  In legacy mode both
                // factors are zero, which reproduces the old outright
                // exclusion exactly rather than approximately.
                ereal s14v = (xcode == 2) ? ereal(p.vdw14Scale)  : ereal(1);
                ereal s14e = (xcode == 2) ? ereal(p.elec14Scale) : ereal(1);
                if (xcode == 2 && s14v == ereal(0) && s14e == ereal(0)) continue;

                ereal d = esqrt(r2);
                if (d < minSep) { d = minSep; r2 = d * d; }

                ereal sw = p.useSwitching ? switchFn(r2, swStart, cutSq) : ereal(1);

                // Lennard-Jones 12-6.  pow() replaced with three multiplies on
                // a squared ratio: same value, roughly an order of magnitude
                // cheaper than two calls to pow().
                ereal rsum = ri + shR[base + k];
                ereal ratio2 = (rsum * rsum) / r2;
                ereal r6 = ratio2 * ratio2 * ratio2;
                ereal vdw = ei * shE[base + k] * (r6 * r6 - ereal(2) * r6);

                ereal epsPair = mixDielectric(si.eps, shD[base + k], p.pairMixing);
                ereal ele = c_kc * qi * shQ[base + k] / (d * epsPair);

                // Each pair is visited from both endpoints, so take half here.
                // Exact in binary arithmetic, and it yields a per-atom energy
                // decomposition for free.
                const ereal hv = ereal(0.5) * sw * s14v * vdw;
                const ereal he = ereal(0.5) * sw * s14e * ele;
                accVdw += hv;
                accEle += he;
                // The half-pair split double counts a pair with both ends in
                // the changed set, so accumulate those separately and let the
                // host correct with 2*V1 - V2.
                if (shC[base + k]) { accVdw2 += hv; accEle2 += he; }
            }
        }
        __syncwarp();
    }

    if (!active) {
        eVdw[s] = eEle[s] = ereal(0);
        if (eVdw2) eVdw2[s] = ereal(0);
        if (eEle2) eEle2[s] = ereal(0);
        if (chunk == 0) eSolvP[s] = eSolvN[s] = eSolvS[s] = ereal(0);
        return;
    }

    eVdw[s] = p.vdwScale  * accVdw;
    eEle[s] = p.elecScale * accEle;
    if (eVdw2) eVdw2[s] = p.vdwScale  * accVdw2;
    if (eEle2) eEle2[s] = p.elecScale * accEle2;

    if (chunk != 0) return;

    // --- solvation, one evaluation per atom ---
    // The original folded this into the i==j diagonal of an N^2 launch, which
    // silently skipped exactly one atom.  Here every atom is covered by
    // construction.
    ereal w = si.waters;
    if (w > ereal(0))
    {
        ereal bornDenom = (ri + p.waterRadius) * si.eps;
        ereal wPolar = p.bornNormalize
                     ? (w / si.capacity) * ereal(p.bornReferenceCapacity)
                     : w;
        eSolvP[s] = p.eSolvationFactor *
                    (-(c_kc * ereal(0.5)) * qi * qi / bornDenom) * wPolar;

        // vdW contact with ordered shell water: sqrt(epsI*epsWater) per water.
        eSolvN[s] = p.hSolvationFactor * (-(ei * esqrt(p.waterEpsilon))) * w;

        // Entropy cost of ordering shell water.  -kT*ln(0.5^w) = kT*ln(2)*w.
        // Writing it as a multiply avoids a pow and a log per atom and removes
        // the underflow that would appear for large w.
        eSolvS[s] = p.entropyFactor * (c_kb * c_T * ereal(0.6931471805599453)) * w;
    }
    else
    {
        eSolvP[s] = eSolvN[s] = eSolvS[s] = ereal(0);
    }
}

// Sum the per-chunk pair-term slices produced by kEnergy into the final
// per-atom arrays.  Each thread owns one (candidate, atom) slot and walks the
// chunks in increasing index order, so the summation order is fixed and the
// result is reproducible.  With jSplit == 1 this degenerates to a copy and the
// values are bit-identical to an unsplit kernel.
__global__ void kReduceOcc(int nCand, int nPad, int jSplit,
                           const ereal* __restrict__ part,
                           ereal* __restrict__ occ)
{
    const size_t total = (size_t)nCand * nPad;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    ereal a = ereal(0);
    for (int c = 0; c < jSplit; ++c) a += part[(size_t)c * total + idx];
    occ[idx] = a;
}

// --- frozen dielectric ---------------------------------------------------
//
// Scatter a freshly computed occupancy field from sorted order into the
// original-order snapshot buffer.  Padding slots carry sorig == -1 and are
// skipped.
__global__ void kSnapshotOcc(int nPad, int N,
                             const int* __restrict__ sorig,
                             const ereal* __restrict__ occ,
                             ereal* __restrict__ occFrozen)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nPad) return;
    const int o = sorig[s];
    if (o < 0 || o >= N) return;
    occFrozen[o] = occ[s];
}

// Gather the frozen occupancy back over the freshly computed one, except for
// the atoms flagged for recomputation.  Those keep their live value, so the
// near shell around a moved sidechain stays exact while the far field is held.
__global__ void kApplyFrozenOcc(int nCand, int nPad, int N,
                                const int* __restrict__ sorig,
                                const ereal* __restrict__ occFrozen,
                                const unsigned char* __restrict__ thaw,
                                ereal* __restrict__ occ)
{
    const size_t total = (size_t)nCand * nPad;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int s = (int)(idx % (size_t)nPad);   // candidates share one sort
    const int o = sorig[s];
    if (o < 0 || o >= N) return;
    if (thaw[o]) return;
    occ[idx] = occFrozen[o];
}

// Overwrite the frozen far field in place.  Returns nothing: a failure here is
// reported by the caller's next CUDA_OK.
static void applyFreeze(energyContext* ctx, int nCand, ereal* occ)
{
    if (!ctx->freezeActive) return;
    const size_t total = (size_t)nCand * ctx->nPad;
    kApplyFrozenOcc<<<(int)((total + 255) / 256), 256>>>(
        nCand, ctx->nPad, ctx->N, ctx->d_sorig,
        ctx->d_occFrozen, ctx->d_thaw, occ);
}

__global__ void kReduceParts(int nCand, int nPad, int jSplit,
                             const ereal* __restrict__ part,
                             ereal* __restrict__ eVdw,
                             ereal* __restrict__ eEle)
{
    const size_t total = (size_t)nCand * nPad;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const ereal* pv = part;
    const ereal* pe = part + (size_t)jSplit * total;

    ereal sv = ereal(0), se = ereal(0);
    for (int c = 0; c < jSplit; ++c)
    {
        sv += pv[(size_t)c * total + idx];
        se += pe[(size_t)c * total + idx];
    }
    eVdw[idx] = sv; eEle[idx] = se;
}

// ---------------------------------------------------------------------------
// Coordinate bounds
// ---------------------------------------------------------------------------

// Single-block min/max reduction over the resident coordinates.  Keeping this
// on the device is what allows a whole minimisation trajectory to run without
// the coordinates ever touching host memory.
__global__ void kBounds(int N, const ereal* __restrict__ x,
                        const ereal* __restrict__ y,
                        const ereal* __restrict__ z,
                        ereal* __restrict__ out)
{
    __shared__ ereal sLo[3][256], sHi[3][256];
    const int t = threadIdx.x;
    ereal lo[3] = { ereal(1e30),  ereal(1e30),  ereal(1e30) };
    ereal hi[3] = { ereal(-1e30), ereal(-1e30), ereal(-1e30) };
    for (int i = t; i < N; i += blockDim.x) {
        ereal v[3] = { x[i], y[i], z[i] };
        for (int d = 0; d < 3; ++d) {
            if (v[d] < lo[d]) lo[d] = v[d];
            if (v[d] > hi[d]) hi[d] = v[d];
        }
    }
    for (int d = 0; d < 3; ++d) { sLo[d][t] = lo[d]; sHi[d][t] = hi[d]; }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s)
            for (int d = 0; d < 3; ++d) {
                if (sLo[d][t + s] < sLo[d][t]) sLo[d][t] = sLo[d][t + s];
                if (sHi[d][t + s] > sHi[d][t]) sHi[d][t] = sHi[d][t + s];
            }
        __syncthreads();
    }
    if (t == 0)
        for (int d = 0; d < 3; ++d) { out[d] = sLo[d][0]; out[3 + d] = sHi[d][0]; }
}

// ---------------------------------------------------------------------------
// Shell state export
// ---------------------------------------------------------------------------

// Export the per-atom local dielectric and shell water count.  This reads the
// same occupancy field and the same shellFromOcclusion() helper the energy pass
// uses, so an exported dielectric can never disagree with the one the energy
// was computed with.
__global__ void kShellExport(int nTiles,
                             const ereal* __restrict__ srad,
                             const unsigned char* __restrict__ ssilent,
                             const ereal* __restrict__ occ,
                             energyParams p, ereal v43,
                             ereal* __restrict__ outEps,
                             ereal* __restrict__ outWaters)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nTiles * TILE) return;
    if (ssilent[s]) { outEps[s] = ereal(0); outWaters[s] = ereal(0); return; }
    shellState st = shellFromOcclusion(occ[s], srad[s], p, v43);
    outEps[s]    = st.eps;
    outWaters[s] = st.waters;
}

// ---------------------------------------------------------------------------
// Clash counting
// ---------------------------------------------------------------------------

__global__ void kClash(int nTiles,
                       const ereal* __restrict__ sx,
                       const ereal* __restrict__ sy,
                       const ereal* __restrict__ sz,
                       const ereal* __restrict__ srad,
                       const int*   __restrict__ sresIndex,
                       const int*   __restrict__ sorig,
                       const unsigned char* __restrict__ ssilent,
                       const ereal* __restrict__ tileLo,
                       const ereal* __restrict__ tileHi,
                       const int* __restrict__ exclCount,
                       const int* __restrict__ exclList, int exclStride,
                       const int* __restrict__ exclSpan,
                       int clashMode, ereal clashTol, ereal maxRadSum,
                       int* __restrict__ clashOut)
{
    __shared__ ereal shX[BLOCK], shY[BLOCK], shZ[BLOCK], shR[BLOCK];
    __shared__ int   shRes[BLOCK], shOrig[BLOCK];

    const int warp = threadIdx.x / TILE;
    const int lane = threadIdx.x % TILE;
    const int iTile = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (iTile >= nTiles) return;

    const int s = iTile * TILE + lane;
    const bool active = !ssilent[s];

    const ereal xi = sx[s], yi = sy[s], zi = sz[s], ri = srad[s];
    const int resI = sresIndex[s], origI = sorig[s];

    const ereal aLoX = tileLo[iTile * 3 + 0], aLoY = tileLo[iTile * 3 + 1], aLoZ = tileLo[iTile * 3 + 2];
    const ereal aHiX = tileHi[iTile * 3 + 0], aHiY = tileHi[iTile * 3 + 1], aHiZ = tileHi[iTile * 3 + 2];

    int acc = 0;
    const int base = warp * TILE;

    for (int jTile = 0; jTile < nTiles; ++jTile)
    {
        ereal d2 = boxBoxDist2(aLoX, aLoY, aLoZ, aHiX, aHiY, aHiZ,
                               tileLo[jTile * 3 + 0], tileLo[jTile * 3 + 1], tileLo[jTile * 3 + 2],
                               tileHi[jTile * 3 + 0], tileHi[jTile * 3 + 1], tileHi[jTile * 3 + 2]);
        if (d2 > maxRadSum * maxRadSum) continue;

        int js = jTile * TILE + lane;
        shX[base + lane] = sx[js]; shY[base + lane] = sy[js]; shZ[base + lane] = sz[js];
        shR[base + lane] = srad[js];
        shRes[base + lane] = sresIndex[js]; shOrig[base + lane] = sorig[js];
        if (ssilent[js]) shX[base + lane] = ereal(1e9);
        __syncwarp();

        if (active)
        {
            for (int k = 0; k < TILE; ++k)
            {
                // Count each unordered pair once, using the original atom index
                // so the result does not depend on the spatial sort.
                if (shOrig[base + k] == origI) continue;   // no self pair

                ereal dx = xi - shX[base + k];
                ereal dy = yi - shY[base + k];
                ereal dz = zi - shZ[base + k];
                ereal rsum = ri + shR[base + k];

                bool hit;
                if (clashMode == CLASH_INSCRIBED_CUBE) {
                    // Matches residue::isClash on the CPU path: the cube
                    // inscribed in the contact sphere, side (radI+radJ)/sqrt(2).
                    ereal c = rsum * ereal(0.7071067811865476);
                    hit = (eabs(dx) < c) && (eabs(dy) < c) && (eabs(dz) < c);
                } else {
                    ereal t = rsum * clashTol;
                    hit = (dx * dx + dy * dy + dz * dz) < t * t;
                }
                if (!hit) continue;

                // Any bonded relationship suppresses a clash, 1-4 included.
                // Three-bond neighbours are held close by the geometry itself
                // and were never clash candidates; damping their nonbonded
                // energy does not make them ones.
                if (exclusionCode(origI, shOrig[base + k], resI, shRes[base + k],
                                  exclSpan, exclCount, exclList, exclStride))
                    continue;
                ++acc;
            }
        }
        __syncwarp();
    }

    clashOut[s] = active ? acc : 0;
}

} // namespace

// ---------------------------------------------------------------------------
// Host: construction
// ---------------------------------------------------------------------------

template <typename T>
static bool devAlloc(T** p, size_t n)
{
    return cudaMalloc((void**)p, n * sizeof(T)) == cudaSuccess;
}

energyContext* energyCreate(const energyTopology& topo, const energyParams& params)
{
    if (topo.numAtoms <= 0) return 0;

    energyContext* ctx = new energyContext();
    // Zero every device pointer explicitly.  A memset over the struct would be
    // undefined behaviour here because the context holds a std::vector and a
    // std::string.
    ctx->d_rad = ctx->d_sqrtEps = ctx->d_chg = ctx->d_selfVol = 0;
    ctx->maxRadius = ereal(0);
    ctx->d_cmask = 0; ctx->d_tileList = 0; ctx->d_tileCount = 0;
    ctx->d_inv = 0; ctx->d_atoms = 0;
    ctx->d_eVdw2 = ctx->d_eEle2 = ctx->d_gather = 0;
    ctx->d_dpart = 0; ctx->dSplitCap = 0;
    ctx->d_stage = 0; ctx->h_stage = 0;
    ctx->d_torList = 0; ctx->d_torSub = 0; ctx->h_torSub = 0; ctx->torPrimed = 0;
    ctx->deltaSetValid = 0; ctx->deltaNList = 0;
    ctx->d_x = ctx->d_y = ctx->d_z = 0;
    ctx->d_sx = ctx->d_sy = ctx->d_sz = 0;
    ctx->d_srad = ctx->d_ssqrtEps = ctx->d_schg = ctx->d_sselfVol = 0;
    ctx->d_tileLo = ctx->d_tileHi = ctx->d_occ = ctx->d_occFrozen = 0;
    ctx->d_thaw = 0;
    ctx->freezeActive = 0;
    ctx->d_eVdw = ctx->d_eEle = ctx->d_eSolvP = ctx->d_eSolvN = ctx->d_eSolvS = 0;
    ctx->d_resIndex = ctx->d_exclCount = ctx->d_exclList = ctx->d_exclSpan = 0;
    ctx->d_order = ctx->d_sresIndex = ctx->d_sorig = ctx->d_clash = 0;
    ctx->d_cellOf = ctx->d_cellCount = ctx->d_cellStart = ctx->d_cellCursor = 0;
    ctx->d_silent = ctx->d_ssilent = 0;
    ctx->h_x = ctx->h_y = ctx->h_z = ctx->h_terms = 0;
    ctx->h_order = 0;
    ctx->nTor = 0; ctx->torCap = 0;
    ctx->d_torAtoms = 0;
    ctx->d_torBarrier = ctx->d_torPhase = ctx->d_torPeriod = ctx->d_torE = 0;

    ctx->N      = topo.numAtoms;
    ctx->nTiles = (ctx->N + TILE - 1) / TILE;
    ctx->nPad   = ctx->nTiles * TILE;
    ctx->p      = params;
    ctx->exclStride = topo.exclusionStride > 0 ? topo.exclusionStride : 1;

    const int N = ctx->N, nPad = ctx->nPad, nT = ctx->nTiles;

    // Grid arrays are sized generously; the cell count is recomputed per call
    // but never exceeds this capacity because cellSize is raised if it would.
    ctx->cellCapacity = std::max(64, 4 * N);
    ctx->numGroups = 0; ctx->angleCap = 0;
    ctx->d_axisA = ctx->d_axisB = ctx->d_memberStart = ctx->d_members = 0;
    ctx->d_angles = 0;
    ctx->replCap = 0;
    ctx->d_rx = ctx->d_ry = ctx->d_rz = 0;
    ctx->d_groupBegin = ctx->d_nGroups = ctx->d_accept = 0;

    ctx->batchCap = 0;
    ctx->d_part = 0; ctx->partCap = 0; ctx->jSplit = 0;
    ctx->d_bx = ctx->d_by = ctx->d_bz = 0;
    ctx->d_bsx = ctx->d_bsy = ctx->d_bsz = 0;
    ctx->d_bocc = ctx->d_btileLo = ctx->d_btileHi = ctx->d_bterms = 0;
    ctx->d_bdpart = ctx->d_bterms2 = ctx->d_bstage = ctx->d_bgather = 0;
    ctx->d_moved = 0; ctx->movedCap = 0;
    ctx->d_torBase = 0; ctx->d_torPart = 0;
    ctx->torBaseCap = ctx->torPartCap = 0;
    ctx->d_bpart = 0; ctx->bdPartCap = 0;
    ctx->d_btgat = 0;
    ctx->bdSplitCap = ctx->bdCandCap = 0;
    ctx->bdStageCap = ctx->bdGatherCap = ctx->btgatCap = 0;

    bool ok = true;
    ok &= devAlloc(&ctx->d_cmask, (size_t)ctx->nPad);
    ok &= devAlloc(&ctx->d_tileList, (size_t)ctx->nTiles + 1);
    ok &= devAlloc(&ctx->d_tileCount, 1);
    ok &= devAlloc(&ctx->d_inv, (size_t)ctx->nPad);
    ok &= devAlloc(&ctx->d_atoms, (size_t)ctx->nPad);
    ok &= devAlloc(&ctx->d_eVdw2, (size_t)ctx->nPad);
    ok &= devAlloc(&ctx->d_eEle2, (size_t)ctx->nPad);
    ok &= devAlloc(&ctx->d_gather, (size_t)ctx->nPad * 7);
    ok &= devAlloc(&ctx->d_stage, (size_t)ctx->nPad * 3);
    if (ok && cudaMallocHost((void**)&ctx->h_stage,
                             (size_t)ctx->nPad * 3 * sizeof(ereal)) != cudaSuccess)
    {   ctx->h_stage = 0; ok = false; }
    ok &= devAlloc(&ctx->d_rad, N)      && devAlloc(&ctx->d_sqrtEps, N);
    ok &= devAlloc(&ctx->d_chg, N)      && devAlloc(&ctx->d_selfVol, N);
    ok &= devAlloc(&ctx->d_resIndex, N) && devAlloc(&ctx->d_silent, N);
    ok &= devAlloc(&ctx->d_exclCount, N);
    ok &= devAlloc(&ctx->d_exclSpan, N);
    ok &= devAlloc(&ctx->d_exclList, (size_t)N * ctx->exclStride);
    ok &= devAlloc(&ctx->d_x, N) && devAlloc(&ctx->d_y, N) && devAlloc(&ctx->d_z, N);
    ok &= devAlloc(&ctx->d_order, nPad);
    ok &= devAlloc(&ctx->d_sx, nPad) && devAlloc(&ctx->d_sy, nPad) && devAlloc(&ctx->d_sz, nPad);
    ok &= devAlloc(&ctx->d_srad, nPad) && devAlloc(&ctx->d_ssqrtEps, nPad);
    ok &= devAlloc(&ctx->d_schg, nPad) && devAlloc(&ctx->d_sselfVol, nPad);
    ok &= devAlloc(&ctx->d_sresIndex, nPad) && devAlloc(&ctx->d_sorig, nPad);
    ok &= devAlloc(&ctx->d_ssilent, nPad);
    ok &= devAlloc(&ctx->d_tileLo, 3 * nT) && devAlloc(&ctx->d_tileHi, 3 * nT);
    ok &= devAlloc(&ctx->d_occ, nPad);
    ok &= devAlloc(&ctx->d_occFrozen, nPad);
    ok &= devAlloc(&ctx->d_thaw, nPad);
    ok &= devAlloc(&ctx->d_eVdw, nPad) && devAlloc(&ctx->d_eEle, nPad);
    ok &= devAlloc(&ctx->d_eSolvP, nPad) && devAlloc(&ctx->d_eSolvN, nPad);
    ok &= devAlloc(&ctx->d_eSolvS, nPad);
    ok &= devAlloc(&ctx->d_clash, nPad);
    ok &= devAlloc(&ctx->d_cellOf, N);
    ok &= devAlloc(&ctx->d_cellCount, ctx->cellCapacity);
    ok &= devAlloc(&ctx->d_cellStart, ctx->cellCapacity);
    ok &= devAlloc(&ctx->d_cellCursor, ctx->cellCapacity);
    ok &= devAlloc(&ctx->d_xSave, N);
    ok &= devAlloc(&ctx->d_ySave, N);
    ok &= devAlloc(&ctx->d_zSave, N);
    ok &= devAlloc(&ctx->d_bounds, 6);

    // Dihedrals.  Absent from a topology means simply no bonded term, not an
    // error: callers that have not been taught to build them still work.
    if (topo.torsionCount > 0 && topo.torsionAtoms && topo.torsionParams)
    {
        const int nTor = topo.torsionCount;
        ctx->nTor = nTor;
        ok &= devAlloc(&ctx->d_torAtoms, (size_t)4 * nTor);
        ok &= devAlloc(&ctx->d_torBarrier, nTor);
        ok &= devAlloc(&ctx->d_torPhase, nTor);
        ok &= devAlloc(&ctx->d_torPeriod, nTor);
        if (ok) {
            std::vector<ereal> bb(nTor), ph(nTor), pn(nTor);
            for (int t = 0; t < nTor; ++t) {
                bb[t] = ereal(topo.torsionParams[3 * t + 0]);
                ph[t] = ereal(topo.torsionParams[3 * t + 1]);
                pn[t] = ereal(topo.torsionParams[3 * t + 2]);
            }
            cudaMemcpy(ctx->d_torAtoms, topo.torsionAtoms,
                       (size_t)4 * nTor * sizeof(int), cudaMemcpyHostToDevice);
            ctx->h_torAtoms.assign(topo.torsionAtoms,
                                   topo.torsionAtoms + (size_t)4 * nTor);
            cudaMemcpy(ctx->d_torBarrier, &bb[0], nTor * sizeof(ereal), cudaMemcpyHostToDevice);
            cudaMemcpy(ctx->d_torPhase,   &ph[0], nTor * sizeof(ereal), cudaMemcpyHostToDevice);
            cudaMemcpy(ctx->d_torPeriod,  &pn[0], nTor * sizeof(ereal), cudaMemcpyHostToDevice);
        }
    }

    ctx->haveSnapshot = false;
    ctx->coordsValid  = false;

    ok &= cudaMallocHost((void**)&ctx->h_x, N * sizeof(ereal)) == cudaSuccess;
    ok &= cudaMallocHost((void**)&ctx->h_y, N * sizeof(ereal)) == cudaSuccess;
    ok &= cudaMallocHost((void**)&ctx->h_z, N * sizeof(ereal)) == cudaSuccess;
    ok &= cudaMallocHost((void**)&ctx->h_terms, 5 * nPad * sizeof(ereal)) == cudaSuccess;
    ok &= cudaMallocHost((void**)&ctx->h_order, nPad * sizeof(int)) == cudaSuccess;

    if (!ok) {
        setError(ctx, "energyCreate allocation", cudaGetErrorString(cudaGetLastError()),
                 __FILE__, __LINE__);
        energyDestroy(ctx);
        return 0;
    }

    // Precompute sqrt(epsilon) and atomic volumes once instead of every pair.
    ereal v43 = (params.occupancy == OCCUPANCY_LEGACY_FULLVOLUME)
              ? ereal(4.188) : ereal(4.1887902048);
    std::vector<ereal> rad(N), sqrtEps(N), chg(N), selfVol(N);
    for (int i = 0; i < N; ++i) {
        rad[i]     = ereal(topo.radius[i]);
        ereal e    = ereal(topo.epsilon[i]);
        sqrtEps[i] = e > ereal(0) ? ereal(sqrt(double(e))) : ereal(0);
        chg[i]     = ereal(topo.charge[i]);
        selfVol[i] = v43 * rad[i] * rad[i] * rad[i];
    }

    std::vector<unsigned char> silent(N, 0);
    if (topo.silent) memcpy(&silent[0], topo.silent, N);
    std::vector<int> resIdx(N, 0);
    if (topo.residueIndex) memcpy(&resIdx[0], topo.residueIndex, N * sizeof(int));

    ctx->maxRadius = ereal(0);
    for (int a = 0; a < N; a++)
        if (rad[a] > ctx->maxRadius) ctx->maxRadius = rad[a];

    cudaMemcpy(ctx->d_rad, &rad[0], N * sizeof(ereal), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_sqrtEps, &sqrtEps[0], N * sizeof(ereal), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_chg, &chg[0], N * sizeof(ereal), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_selfVol, &selfVol[0], N * sizeof(ereal), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_resIndex, &resIdx[0], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_silent, &silent[0], N, cudaMemcpyHostToDevice);

    if (topo.exclusionCount)
        cudaMemcpy(ctx->d_exclCount, topo.exclusionCount, N * sizeof(int), cudaMemcpyHostToDevice);
    else
        cudaMemset(ctx->d_exclCount, 0, N * sizeof(int));

    if (topo.exclusionList)
        cudaMemcpy(ctx->d_exclList, topo.exclusionList,
                   (size_t)N * ctx->exclStride * sizeof(int), cudaMemcpyHostToDevice);
    else
        cudaMemset(ctx->d_exclList, 0, (size_t)N * ctx->exclStride * sizeof(int));

    // Exact per-atom exclusion reach, in residues.  -1 means "no exclusions",
    // which makes the kernel's fast reject unconditional for that atom.
    {
        std::vector<int> span(N, -1);
        if (topo.exclusionCount && topo.exclusionList && topo.residueIndex)
        {
            for (int i = 0; i < N; ++i) {
                int best = -1;
                const int* list = topo.exclusionList + (size_t)i * ctx->exclStride;
                for (int k = 0; k < topo.exclusionCount[i]; ++k) {
                    // 1-4 partners are stored as -(j+1); decode before use, or
                    // the span is computed against a negative index and the
                    // fast reject starts discarding real exclusions.
                    int j = list[k] < 0 ? -list[k] - 1 : list[k];
                    int dr = topo.residueIndex[i] - topo.residueIndex[j];
                    if (dr < 0) dr = -dr;
                    if (dr > best) best = dr;
                }
                span[i] = best;
            }
        }
        cudaMemcpy(ctx->d_exclSpan, &span[0], N * sizeof(int), cudaMemcpyHostToDevice);
    }

    ctx->perAtom.resize(nPad);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        setError(ctx, "energyCreate upload", cudaGetErrorString(e), __FILE__, __LINE__);
        energyDestroy(ctx);
        return 0;
    }
    return ctx;
}

void energyDestroy(energyContext* ctx)
{
    if (!ctx) return;
    void* ptrs[] = {
        ctx->d_rad, ctx->d_sqrtEps, ctx->d_chg, ctx->d_selfVol, ctx->d_resIndex,
        ctx->d_silent, ctx->d_exclCount, ctx->d_exclList, ctx->d_exclSpan,
        ctx->d_x, ctx->d_y, ctx->d_z, ctx->d_order,
        ctx->d_sx, ctx->d_sy, ctx->d_sz, ctx->d_srad, ctx->d_ssqrtEps,
        ctx->d_schg, ctx->d_sselfVol, ctx->d_sresIndex, ctx->d_sorig,
        ctx->d_ssilent, ctx->d_tileLo, ctx->d_tileHi, ctx->d_occ,
        ctx->d_occFrozen, ctx->d_thaw,
        ctx->d_eVdw, ctx->d_eEle, ctx->d_eSolvP, ctx->d_eSolvN, ctx->d_eSolvS,
        ctx->d_clash, ctx->d_cellOf, ctx->d_cellCount, ctx->d_cellStart,
        ctx->d_cellCursor,
        ctx->d_xSave, ctx->d_ySave, ctx->d_zSave, ctx->d_bounds,
        ctx->d_bx, ctx->d_by, ctx->d_bz, ctx->d_bsx, ctx->d_bsy, ctx->d_bsz,
        ctx->d_bocc, ctx->d_btileLo, ctx->d_btileHi, ctx->d_bterms,
        ctx->d_part,
        ctx->d_torAtoms, ctx->d_torBarrier, ctx->d_torPhase, ctx->d_torPeriod,
        ctx->d_torE,
        ctx->d_axisA, ctx->d_axisB, ctx->d_memberStart, ctx->d_members,
        ctx->d_angles,
        ctx->d_rx, ctx->d_ry, ctx->d_rz,
        ctx->d_groupBegin, ctx->d_nGroups, ctx->d_accept,
        ctx->d_cmask, ctx->d_tileList, ctx->d_tileCount, ctx->d_inv,
        ctx->d_atoms, ctx->d_eVdw2, ctx->d_eEle2, ctx->d_gather,
        ctx->d_dpart, ctx->d_stage, ctx->d_torSub,
        ctx->d_bdpart, ctx->d_bterms2, ctx->d_bstage, ctx->d_bgather, ctx->d_bpart,
        ctx->d_torBase, ctx->d_torPart,
        ctx->d_moved,
        ctx->d_btgat
    };
    for (size_t i = 0; i < sizeof(ptrs) / sizeof(ptrs[0]); ++i)
        if (ptrs[i]) cudaFree(ptrs[i]);

    if (ctx->h_stage) cudaFreeHost(ctx->h_stage);
    if (ctx->h_torSub) cudaFreeHost(ctx->h_torSub);
    if (ctx->d_torList) cudaFree(ctx->d_torList);
    if (ctx->h_x) cudaFreeHost(ctx->h_x);
    if (ctx->h_y) cudaFreeHost(ctx->h_y);
    if (ctx->h_z) cudaFreeHost(ctx->h_z);
    if (ctx->h_terms) cudaFreeHost(ctx->h_terms);
    if (ctx->h_order) cudaFreeHost(ctx->h_order);
    delete ctx;
}

void energySetParams(energyContext* ctx, const energyParams& params)
{
    if (ctx) ctx->p = params;
}

const energyParams& energyGetParams(energyContext* ctx)
{
    static energyParams fallback = defaultEnergyParams();
    if (!ctx) return fallback;
    return ctx->p;
}


void energySetSilent(energyContext* ctx, const unsigned char* silent)
{
    if (!ctx || !silent) return;
    cudaMemcpy(ctx->d_silent, silent, ctx->N, cudaMemcpyHostToDevice);
}

// ---------------------------------------------------------------------------
// Host: shared spatial sort
// ---------------------------------------------------------------------------

// Uploads coordinates, bins atoms, sorts them into tile order and computes tile
// bounding boxes.  Returns 0 on success.
// Upload host coordinates into the resident device state.
static int uploadCoords(energyContext* ctx,
                        const double* x, const double* y, const double* z)
{
    const int N = ctx->N;
    for (int i = 0; i < N; ++i) {
        ctx->h_x[i] = ereal(x[i]);
        ctx->h_y[i] = ereal(y[i]);
        ctx->h_z[i] = ereal(z[i]);
    }
    CUDA_OK(ctx, cudaMemcpy(ctx->d_x, ctx->h_x, N * sizeof(ereal), cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_y, ctx->h_y, N * sizeof(ereal), cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_z, ctx->h_z, N * sizeof(ereal), cudaMemcpyHostToDevice));
    ctx->coordsValid = true;
    return 0;
}

// Rebuild the spatial ordering from whatever is currently in the resident
// device coordinates.  No host coordinate access at all: the bounds are
// reduced on the device, and only the six scalars come back.
static int buildOrder(energyContext* ctx)
{
    const int N = ctx->N, nPad = ctx->nPad, nT = ctx->nTiles;

    if (!ctx->coordsValid) {
        setError(ctx, "buildOrder", "no coordinates have been uploaded",
                 __FILE__, __LINE__);
        return -1;
    }

    kBounds<<<1, 256>>>(N, ctx->d_x, ctx->d_y, ctx->d_z, ctx->d_bounds);
    CUDA_OK(ctx, cudaPeekAtLastError());

    ereal hb[6];
    CUDA_OK(ctx, cudaMemcpy(hb, ctx->d_bounds, 6 * sizeof(ereal), cudaMemcpyDeviceToHost));
    ereal loX = hb[0], loY = hb[1], loZ = hb[2];
    ereal hiX = hb[3], hiY = hb[4], hiZ = hb[5];

    double ex = double(hiX - loX) + 1e-3;
    double ey = double(hiY - loY) + 1e-3;
    double ez = double(hiZ - loZ) + 1e-3;

    // Pick a cell edge that puts roughly TILE atoms in a cell, so tiles come
    // out compact and cubic.  Then raise it if that would blow the cell budget.
    double cell = pow(ATOMS_PER_CELL_TARGET * ex * ey * ez / std::max(N, 1), 1.0 / 3.0);
    if (!(cell > 0.5)) cell = 0.5;

    int nx, ny, nz, nCells;
    for (;;) {
        nx = std::max(1, int(ex / cell) + 1);
        ny = std::max(1, int(ey / cell) + 1);
        nz = std::max(1, int(ez / cell) + 1);
        nCells = nx * ny * nz;
        if (nCells <= ctx->cellCapacity) break;
        cell *= 1.26;   // ~2x the cell volume per step
    }

    CUDA_OK(ctx, cudaMemset(ctx->d_cellCount, 0, nCells * sizeof(int)));

    int gridN = (N + 255) / 256;
    kCellOf<<<gridN, 256>>>(ctx->d_x, ctx->d_y, ctx->d_z, N,
                            loX, loY, loZ, ereal(1.0 / cell), nx, ny, nz,
                            ctx->d_cellOf, ctx->d_cellCount);
    CUDA_OK(ctx, cudaPeekAtLastError());

    // The cell count is O(N/32), a few hundred values, so the exclusive scan is
    // cheaper on the host than any device scan plus its launch overhead.
    std::vector<int> counts(nCells), starts(nCells);
    CUDA_OK(ctx, cudaMemcpy(&counts[0], ctx->d_cellCount, nCells * sizeof(int),
                            cudaMemcpyDeviceToHost));
    int run = 0;
    for (int c = 0; c < nCells; ++c) { starts[c] = run; run += counts[c]; }

    CUDA_OK(ctx, cudaMemcpy(ctx->d_cellStart, &starts[0], nCells * sizeof(int),
                            cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_cellCursor, &starts[0], nCells * sizeof(int),
                            cudaMemcpyHostToDevice));

    kPlace<<<gridN, 256>>>(ctx->d_cellOf, N, ctx->d_cellCursor, ctx->d_order);
    CUDA_OK(ctx, cudaPeekAtLastError());

    kSortCells<<<nCells, 128>>>(ctx->d_cellStart, ctx->d_cellCount,
                                nCells, ctx->d_order);
    CUDA_OK(ctx, cudaPeekAtLastError());

    kGather<<<(nPad + 255) / 256, 256>>>(
        nPad, N, ctx->d_order,
        ctx->d_x, ctx->d_y, ctx->d_z, ctx->d_rad, ctx->d_sqrtEps, ctx->d_chg,
        ctx->d_selfVol, ctx->d_resIndex, ctx->d_silent,
        ctx->d_sx, ctx->d_sy, ctx->d_sz, ctx->d_srad, ctx->d_ssqrtEps,
        ctx->d_schg, ctx->d_sselfVol, ctx->d_sresIndex, ctx->d_sorig,
        ctx->d_ssilent);
    CUDA_OK(ctx, cudaPeekAtLastError());

    kTileBounds<<<(nT + 127) / 128, 128>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz,
                                           ctx->d_srad, ctx->d_ssilent,
                                           ctx->d_tileLo, ctx->d_tileHi);
    CUDA_OK(ctx, cudaPeekAtLastError());
    ctx->deltaSetValid = 0;   // d_sorig moved, so cmask/inv/tileList are stale
    return 0;
}

// ---------------------------------------------------------------------------
// Host: energy
// ---------------------------------------------------------------------------

// Choose how finely to split the j-tile loop.
//
// One warp per i-tile means the energy pass offers only nTiles * nCand warps
// of parallelism.  That fills a GPU when nCand is large, but minimisation
// wants a small population -- energy at a fixed wall budget is monotone in
// sweeps-per-chain, so the default is four replicas -- and four candidates on
// a 44-tile protein produce 176 warps, which does not.  Splitting j recovers
// the parallelism without touching the sampler.
//
// Measured on the P2200 at the current default of one replica (b2e66f9).  Arms
// were run round robin and each is reported as a paired ratio to jSplit=8
// within its own round, because the card slides from 1518 to 999 MHz as it
// heats and blocking by arm charges that drift to whichever arm ran late: the
// same split read 563 us/candidate in one block and 394 in another.  Paired,
// the spread falls to about 1%.  Ratios of whole-evaluation us/candidate,
// 7 rounds:
//
//   1crn  nTiles 21   js 4 1.421   8 1.000   12 0.866   16 0.838   21 0.897
//   1ubq  nTiles 44   js 4 1.061   8 1.000   12 0.998   16 0.996   24 0.988   32 1.019   44 1.076
//   1lyz  nTiles 71   js 2 1.429   4 1.018   8 1.000   12 0.945   16 0.937   24 0.973
//   2lzm  nTiles 94   js 4 1.039   8 1.000   12 0.955   16 0.997   24 1.018   32 1.032
//
// The optima sit at 336, 1056, 1136 and 1128 warps (nTiles * jSplit * nCand).
// For everything at or above nTiles 44 that is a constant near 1100, which
// reproduces the measured optimum exactly in all three cases (25, 16, 12).
// This constant is per-card -- see targetWarps below, where the 3090's own
// measured value is 3400.  The old value of 800 was fitted at nCand=4 and then
// capped at 8, which made it inert; with one replica the launch has a quarter
// the warps and the target has to rise to compensate.
//
// 1crn is the exception: its optimum is 16 but the formula asks for 53, so the
// clamp to nTiles returns 21 and gives up about 7% against its own best.  That
// residual is real and left in place -- a target low enough to land on 16 at
// nTiles 21 would ask for 4 at nTiles 94, which costs 4%.  The clamp still beats
// the old cap of 8 by 10% there, which is the largest gain of the four; the
// split matters most on small proteins, which are the ones that cannot fill the
// card without it, and is worth 1-6% elsewhere.
//
// Ending energy is bit-identical across every split above at a fixed seed, so
// the reduction's index-ordered chunk walk does hold re-association.  These
// absolutes were taken thermally saturated at 999 MHz; compare them with each
// other, not across commits.
//
// The factor depends on nTiles and on the device, but deliberately not on
// nCand.  That last part is what batchTest and replicaTest exist to check: if
// it varied with nCand, a batched evaluation and a single resident one would
// group their float sums differently and stop agreeing bit for bit.  Varying
// by device is safe for the same reason the split can be tuned at all --
// ending energy is bit-identical across splits -- but it does mean the chosen
// jSplit is a property of the card, so quote it from the profile line rather
// than assuming the value another machine reported.
//
// Geometry the split is derived from, recorded for the profile report.  Without
// it the profile shows what a launch cost but not the shape that produced it,
// so a chosen jSplit could only be inferred from the source, not read off a run.
static int g_geomN = 0, g_geomTiles = 0, g_geomSplit = 0;

// Warp target for the split, resolved once per process and cached: chooseSplit
// runs on every evaluation and cudaGetDeviceProperties is far too slow to sit
// on that path.
//
// The two entries below are each measured on their own card, by the round-robin
// paired sweep described above.  They are NOT related by any capacity ratio,
// and the first attempt at this function assumed they were:
//
//     card    SMs  warps/SM  resident  measured optimum  optimum/resident
//     P2200    10        64       640              1100            1.72x
//     3090     82        48      3936              3400            0.86x
//
// Expressing the P2200 number as 1.72x resident and scaling it to the 3090
// predicts 6765, which is 2x the measured optimum and pushes every protein to
// the nTiles clamp -- 8-9% slower than 3400 on 1lyz and 2lzm.  Occupancy alone
// does not determine this, so the constant stays a measurement, not a formula.
//
// 3400 was read off the 3090 sweep, where the unclamped optima were 3408 warps
// (1lyz, nTiles 71, jSplit 48) and 3384 (2lzm, nTiles 94, jSplit 36).  It
// reproduces the measured best jSplit on all three: 1ubq 44 (clamped), 1lyz 47
// against a measured 48, 2lzm 36 exactly.
//
// An unrecognised card gets 1.0x its own resident capacity.  That is a guess,
// but a bracketed one -- it sits between the two measured ratios -- and the
// optimum is broad enough (24/36/48 are within 2.5% of each other on 2lzm)
// that being somewhat off costs little.  Measure before trusting it on a new
// card; PROTCAD_ENERGY_JSPLIT overrides the whole rule for exactly that.
static int targetWarps()
{
    static int cached = 0;
    if (cached == 0)
    {
        cached = 1100;                       // P2200 value if the query fails
        int dev = 0;
        cudaDeviceProp prop;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        {
            const int resident =
                prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / 32);
            if (prop.major == 6)      cached = 1100;   // Pascal, measured
            else if (prop.major == 8) cached = 3400;   // Ampere, measured
            else if (resident > 0)    cached = resident;
        }
    }
    return cached;
}

static int chooseSplit(energyContext* ctx)
{
    if (ctx->jSplit > 0) return ctx->jSplit;
    int s = 0;
    if (const char* e = getenv("PROTCAD_ENERGY_JSPLIT")) s = atoi(e);
    if (s <= 0)
    {
        // Per-card measured target; see targetWarps above.  Exact no-op on the
        // P2200, so yesterday's sweep on that card still holds.
        s = (targetWarps() + ctx->nTiles - 1) / std::max(1, ctx->nTiles);
    }
    // Splitting past nTiles only launches empty chunks.  There is no cap below
    // this: the override has to be able to reach the values the rule is being
    // measured against, and the rule self-limits because s falls as nTiles grows.
    if (s > ctx->nTiles) s = ctx->nTiles;
    if (s < 1) s = 1;
    ctx->jSplit = s;
    g_geomN = ctx->N; g_geomTiles = ctx->nTiles; g_geomSplit = s;
    return s;
}

// Rebuild invOrder from the freshly copied h_order.  Call once per evaluation,
// after h_order has been read back and before any term is reduced.
static void buildInvOrder(energyContext* ctx, int nPad)
{
    ctx->invOrder.assign(ctx->N, -1);
    for (int s = 0; s < nPad; ++s) {
        int oi = ctx->h_order[s];
        if (oi >= 0) ctx->invOrder[oi] = s;
    }
}

// Dihedral energies for nCand conformations held in original atom order at
// (x,y,z).  Returns one total per candidate in tot[].
//
// The reduction is on the host in term-index order with Kahan compensation,
// which is what every other term in this file does and for the same reason:
// the working precision is single, and the order has to be a property of the
// topology rather than of the launch geometry or two evaluations of the same
// structure stop agreeing bit for bit.
// Queue the torsion kernel and its device-to-host copy on the current stream
// without synchronising.  Split out from the reduction so the batch path can
// issue it alongside kEnergy and let the single existing sync cover both: run
// as its own launch-then-blocking-copy step it cost 23 us/candidate on the
// 3090, almost all of it a second round trip rather than arithmetic.
__global__ void kGatherTor(int nList, const int* __restrict__ torList,
                           const ereal* __restrict__ torE, ereal* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nList) out[i] = torE[torList[i]];
}

static int torsionLaunch(energyContext* ctx, const ereal* x, const ereal* y,
                         const ereal* z, int nCand)
{
    const int nTor = ctx->nTor;
    if (nTor <= 0 || ctx->p.torsionScale == 0.0) return 0;

    if (nCand > ctx->torCap) {
        if (ctx->d_torE) { cudaFree(ctx->d_torE); ctx->d_torE = 0; }
        if (!devAlloc(&ctx->d_torE, (size_t)nTor * nCand)) {
            setError(ctx, "torsion", "out of memory", __FILE__, __LINE__);
            return -1;
        }
        ctx->torCap = nCand;
        ctx->h_torE.resize((size_t)nTor * nCand);
    }

    dim3 g((nTor + 255) / 256, nCand);
    kTorsion<<<g, 256>>>(x, y, z, ctx->N, nCand, nTor,
                         ctx->d_torAtoms, ctx->d_torBarrier,
                         ctx->d_torPhase, ctx->d_torPeriod, ctx->d_silent,
                         ereal(ctx->p.torsionScale), ctx->d_torE);
    CUDA_OK(ctx, cudaPeekAtLastError());
    CUDA_OK(ctx, cudaMemcpyAsync(&ctx->h_torE[0], ctx->d_torE,
                                 (size_t)nTor * nCand * sizeof(ereal),
                                 cudaMemcpyDeviceToHost, 0));
    return 0;
}

// Host side of the torsion term.  Requires that the copy queued by
// torsionLaunch has completed.
static int torsionReduce(energyContext* ctx, int nCand, double* tot,
                         double* perAtomOut = 0)
{
    const int nTor = ctx->nTor;
    for (int k = 0; k < nCand; ++k) tot[k] = 0.0;
    if (nTor <= 0 || ctx->p.torsionScale == 0.0) return 0;

    for (int k = 0; k < nCand; ++k) {
        const ereal* src = &ctx->h_torE[(size_t)k * nTor];
        double sum = 0.0, c = 0.0;
        for (int t = 0; t < nTor; ++t) {
            double yv = double(src[t]) - c;
            double tt = sum + yv;
            c = (tt - sum) - yv;
            sum = tt;
        }
        tot[k] = sum;
    }

    // Attribute each torsion evenly to its four atoms, matching the convention
    // the nonbonded terms use (a pair interaction is split evenly between its
    // two atoms).  Without this the per-atom export omits the torsion term
    // entirely while the reported total includes it, so the documented
    // "per-atom values sum to the total" identity fails -- on 1crn by 516 of a
    // 511 kcal/mol total, which is the whole quantity, not a rounding error.
    if (perAtomOut && nCand == 1 && !ctx->h_torAtoms.empty()) {
        const ereal* src = &ctx->h_torE[0];
        for (int t = 0; t < nTor; ++t) {
            double q = 0.25 * double(src[t]);
            for (int a = 0; a < 4; ++a) perAtomOut[ctx->h_torAtoms[4 * t + a]] += q;
        }
    }
    return 0;
}

static int torsionListBuild(energyContext* ctx, const int* atoms, int count,
                            bool rebuildList)
{
    const int nTor = ctx->nTor;
    if (ctx->torStart.empty()) {
        ctx->torStart.assign(ctx->N + 1, 0);
        for (int t = 0; t < 4 * nTor; ++t) ++ctx->torStart[ctx->h_torAtoms[t] + 1];
        for (int a = 0; a < ctx->N; ++a) ctx->torStart[a + 1] += ctx->torStart[a];
        std::vector<int> cur(ctx->torStart.begin(), ctx->torStart.end() - 1);
        ctx->torIdx.assign(4 * nTor, 0);
        for (int t = 0; t < nTor; ++t)
            for (int j = 0; j < 4; ++j) ctx->torIdx[cur[ctx->h_torAtoms[4 * t + j]]++] = t;
    }

    if (rebuildList || ctx->h_torList.empty()) {
        std::vector<unsigned char> seen(nTor, 0);
        ctx->h_torList.clear();
        for (int i = 0; i < count; ++i) {
            const int a = atoms[i];
            if (a < 0 || a >= ctx->N) continue;
            for (int j = ctx->torStart[a]; j < ctx->torStart[a + 1]; ++j) {
                const int t = ctx->torIdx[j];
                if (!seen[t]) { seen[t] = 1; ctx->h_torList.push_back(t); }
            }
        }
        std::sort(ctx->h_torList.begin(), ctx->h_torList.end());
        if (!ctx->d_torList) {
            if (!devAlloc(&ctx->d_torList, nTor)) return -1;
            if (!ctx->d_torSub && !devAlloc(&ctx->d_torSub, nTor)) return -1;
            if (!ctx->h_torSub &&
                cudaMallocHost((void**)&ctx->h_torSub, (size_t)nTor * sizeof(ereal))
                    != cudaSuccess) { ctx->h_torSub = 0; return -1; }
        }
        CUDA_OK(ctx, cudaMemcpy(ctx->d_torList, &ctx->h_torList[0],
                                ctx->h_torList.size() * sizeof(int),
                                cudaMemcpyHostToDevice));
    }
    return 0;
}

// A move can only change torsions that contain a moved atom.  Recompute just
// those, refresh their cached per-torsion energies, and re-sum on the host:
// the host sum stays over every torsion in ascending order, so the result is
// bit-identical to a full evaluation rather than a drifting running total.
static int torsionDelta(energyContext* ctx, const int* atoms, int count,
                        bool rebuildList, double* tot)
{
    const int nTor = ctx->nTor;
    if (nTor <= 0 || ctx->p.torsionScale == 0.0) { if (tot) *tot = 0.0; return 0; }
    if (!ctx->torPrimed || ctx->h_torAtoms.empty()) return 1;   // caller falls back

    if (torsionListBuild(ctx, atoms, count, rebuildList) != 0) return -1;

    const int nL = (int)ctx->h_torList.size();
    if (nL > 0) {
        kTorsion<<<dim3((nL + 255) / 256, 1), 256>>>(
            ctx->d_x, ctx->d_y, ctx->d_z, ctx->N, 1, nTor,
            ctx->d_torAtoms, ctx->d_torBarrier, ctx->d_torPhase, ctx->d_torPeriod,
            ctx->d_silent, ereal(ctx->p.torsionScale), ctx->d_torE,
            ctx->d_torList, nL);
        CUDA_OK(ctx, cudaPeekAtLastError());
        kGatherTor<<<(nL + 255) / 256, 256>>>(nL, ctx->d_torList, ctx->d_torE,
                                              ctx->d_torSub);
        CUDA_OK(ctx, cudaMemcpyAsync(ctx->h_torSub, ctx->d_torSub,
                                     (size_t)nL * sizeof(ereal),
                                     cudaMemcpyDeviceToHost, 0));
        CUDA_OK(ctx, cudaStreamSynchronize(0));
        for (int i = 0; i < nL; ++i) ctx->h_torE[ctx->h_torList[i]] = ctx->h_torSub[i];
    }

    double sum = 0.0, c = 0.0;
    for (int t = 0; t < nTor; ++t) {
        double yv = double(ctx->h_torE[t]) - c;
        double tt = sum + yv;
        c = (tt - sum) - yv;
        sum = tt;
    }
    if (tot) *tot = sum;
    return 0;
}

__global__ void kGatherTorBatch(int nList, int nCand, int nTor,
                                const int* __restrict__ torList,
                                const ereal* __restrict__ torE,
                                ereal* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (i >= nList || k >= nCand) return;
    out[(size_t)k * nList + i] = torE[(size_t)k * nTor + torList[i]];
}

// Torsion term for a batch of candidates that all perturb the same atoms.  The
// affected torsion list is shared, so one launch covers every candidate, and
// each candidate's total is re-summed over all nTor in ascending order from the
// primed cache with its own listed values substituted in -- the same identity
// torsionDelta uses, so a batched total is bit-identical to the single-move one
// for the same conformation.
// The batch torsion total used to be finished on the host: every candidate got
// a full copy of the primed per-torsion energies, had its changed entries
// overwritten, and was then Kahan-summed over every torsion in the structure.
// That is O(nCand * nTor) of host arithmetic per trial behind an nL * nCand
// readback, and it made the torsion half 51% of the batch delta on 1crn and
// 66% on 1ake -- the part of this path that had been assumed cheap and never
// measured.
//
// The full ascending sum is not the mistake; it is what makes the delta
// bit-comparable to a full evaluation instead of a drifting running total, and
// dropping it for a base-plus-correction would trade that away for arithmetic
// that is only cheap because it is wrong about cancellation.  The mistake is
// doing it on the host, once per candidate.  So the primed baseline is
// broadcast into the candidate rows on the device, the listed torsions
// overwrite it exactly as before, and the ascending sum happens there.
//
// Summation order is fixed by torsion index alone -- contiguous per-thread
// ranges, in-order Kahan within a thread and across threads -- so a
// candidate's torsion total does not depend on the batch it shared or on how
// many torsions the move touched, which is the same guarantee the nonbonded
// half carries.
__global__ void kBroadcastTorBase(int nTor, int nCand,
                                  const ereal* __restrict__ base,
                                  ereal* __restrict__ rows)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (t >= nTor || k >= nCand) return;
    rows[(size_t)k * nTor + t] = base[t];
}

__global__ void kReduceTorRows(int nTor, const ereal* __restrict__ rows,
                               double* __restrict__ out)
{
    const int k = blockIdx.x;
    const ereal* r = rows + (size_t)k * nTor;
    __shared__ double part[256];
    const int nt = blockDim.x, tid = threadIdx.x;
    const int chunk = (nTor + nt - 1) / nt;
    const int lo = tid * chunk;
    int hi = lo + chunk; if (hi > nTor) hi = nTor;

    double sum = 0.0, c = 0.0;
    for (int t = lo; t < hi; ++t) {
        const double y = double(r[t]) - c, tt = sum + y;
        c = (tt - sum) - y; sum = tt;
    }
    part[tid] = (lo < nTor) ? sum : 0.0;
    __syncthreads();
    if (tid == 0) {
        double sv = 0.0, cc = 0.0;
        for (int i = 0; i < nt; ++i) {
            const double y = part[i] - cc, tt = sv + y;
            cc = (tt - sv) - y; sv = tt;
        }
        out[k] = sv;
    }
}

static int torsionBatchDelta(energyContext* ctx, int nCand, const int* atoms,
                             int count, bool rebuildList, double* tot)
{
    const int nTor = ctx->nTor;
    if (nTor <= 0 || ctx->p.torsionScale == 0.0) {
        for (int k = 0; k < nCand; ++k) tot[k] = 0.0;
        return 0;
    }
    if (!ctx->torPrimed || ctx->h_torAtoms.empty()) return 1;   // caller falls back

    if (torsionListBuild(ctx, atoms, count, rebuildList) != 0) return -1;
    const int nL = (int)ctx->h_torList.size();

    // d_torE is the scratch the launch writes into; the primed per-torsion
    // energies live in h_torE and are never disturbed here.
    if (nCand > ctx->torCap) {
        if (ctx->d_torE) { cudaFree(ctx->d_torE); ctx->d_torE = 0; }
        if (!devAlloc(&ctx->d_torE, (size_t)nTor * nCand)) {
            setError(ctx, "torsionBatchDelta", "out of memory", __FILE__, __LINE__);
            return -1;
        }
        ctx->torCap = nCand;
        ctx->h_torE.resize((size_t)nTor * nCand);
    }

    static const bool torHostReduce = (getenv("PROTCAD_TORSION_HOSTREDUCE") != 0);

    if (!torHostReduce) {
        // Baseline is refreshed every call: the single-move delta writes
        // accepted torsions straight into h_torE, so the host copy is the only
        // authority on what "unchanged" currently means.
        if (!ctx->d_torBase || ctx->torBaseCap < (size_t)nTor) {
            if (ctx->d_torBase) cudaFree(ctx->d_torBase);
            ctx->d_torBase = 0; ctx->torBaseCap = 0;
            if (!devAlloc(&ctx->d_torBase, (size_t)nTor)) {
                setError(ctx, "torsionBatchDelta", "out of memory for torsion base",
                         __FILE__, __LINE__);
                return -1;
            }
            ctx->torBaseCap = (size_t)nTor;
        }
        if (!ctx->d_torPart || ctx->torPartCap < (size_t)nCand) {
            if (ctx->d_torPart) cudaFree(ctx->d_torPart);
            ctx->d_torPart = 0; ctx->torPartCap = 0;
            if (!devAlloc(&ctx->d_torPart, (size_t)nCand)) {
                setError(ctx, "torsionBatchDelta", "out of memory for torsion totals",
                         __FILE__, __LINE__);
                return -1;
            }
            ctx->torPartCap = (size_t)nCand;
        }
        CUDA_OK(ctx, cudaMemcpy(ctx->d_torBase, &ctx->h_torE[0],
                                (size_t)nTor * sizeof(ereal),
                                cudaMemcpyHostToDevice));
        kBroadcastTorBase<<<dim3((nTor + 255) / 256, nCand), 256>>>(
            nTor, nCand, ctx->d_torBase, ctx->d_torE);
        CUDA_OK(ctx, cudaPeekAtLastError());
        if (nL > 0) {
            kTorsion<<<dim3((nL + 255) / 256, nCand), 256>>>(
                ctx->d_bx, ctx->d_by, ctx->d_bz, ctx->N, nCand, nTor,
                ctx->d_torAtoms, ctx->d_torBarrier, ctx->d_torPhase, ctx->d_torPeriod,
                ctx->d_silent, ereal(ctx->p.torsionScale), ctx->d_torE,
                ctx->d_torList, nL);
            CUDA_OK(ctx, cudaPeekAtLastError());
        }
        kReduceTorRows<<<nCand, 256>>>(nTor, ctx->d_torE, ctx->d_torPart);
        CUDA_OK(ctx, cudaPeekAtLastError());
        CUDA_OK(ctx, cudaMemcpy(tot, ctx->d_torPart, (size_t)nCand * sizeof(double),
                                cudaMemcpyDeviceToHost));
        return 0;
    }

    std::vector<ereal> sub((size_t)nL * nCand, ereal(0));
    if (nL > 0) {
        if (!ctx->d_btgat || ctx->btgatCap < (size_t)nL * nCand) {
            if (ctx->d_btgat) cudaFree(ctx->d_btgat);
            ctx->d_btgat = 0; ctx->btgatCap = 0;
            if (!devAlloc(&ctx->d_btgat, (size_t)nL * nCand)) {
                setError(ctx, "torsionBatchDelta", "out of memory for torsion gather",
                         __FILE__, __LINE__);
                return -1;
            }
            ctx->btgatCap = (size_t)nL * nCand;
        }
        kTorsion<<<dim3((nL + 255) / 256, nCand), 256>>>(
            ctx->d_bx, ctx->d_by, ctx->d_bz, ctx->N, nCand, nTor,
            ctx->d_torAtoms, ctx->d_torBarrier, ctx->d_torPhase, ctx->d_torPeriod,
            ctx->d_silent, ereal(ctx->p.torsionScale), ctx->d_torE,
            ctx->d_torList, nL);
        CUDA_OK(ctx, cudaPeekAtLastError());
        kGatherTorBatch<<<dim3((nL + 255) / 256, nCand), 256>>>(
            nL, nCand, nTor, ctx->d_torList, ctx->d_torE, ctx->d_btgat);
        CUDA_OK(ctx, cudaPeekAtLastError());
        CUDA_OK(ctx, cudaMemcpy(&sub[0], ctx->d_btgat,
                                (size_t)nL * nCand * sizeof(ereal),
                                cudaMemcpyDeviceToHost));
    }

    std::vector<ereal> row(nTor);
    for (int k = 0; k < nCand; ++k) {
        std::copy(ctx->h_torE.begin(), ctx->h_torE.begin() + nTor, row.begin());
        for (int i = 0; i < nL; ++i) row[ctx->h_torList[i]] = sub[(size_t)k * nL + i];
        double sum = 0.0, c = 0.0;
        for (int t = 0; t < nTor; ++t) {
            double yv = double(row[t]) - c;
            double tt = sum + yv;
            c = (tt - sum) - yv;
            sum = tt;
        }
        tot[k] = sum;
    }
    return 0;
}

// Convenience wrapper for the non-batched paths, where the extra round trip is// paid once per evaluation rather than once per candidate batch.
static int torsionTotals(energyContext* ctx, const ereal* x, const ereal* y,
                         const ereal* z, int nCand, double* tot,
                         double* perAtomOut = 0)
{
    if (torsionLaunch(ctx, x, y, z, nCand) != 0) return -1;
    CUDA_OK(ctx, cudaStreamSynchronize(0));
    if (nCand == 1) ctx->torPrimed = 1;
    return torsionReduce(ctx, nCand, tot, perAtomOut);
}

static bool ensurePart(energyContext* ctx, int nCand)
{
    const int jSplit = chooseSplit(ctx);
    if (ctx->d_part && nCand <= ctx->partCap) return true;
    if (ctx->d_part) { cudaFree(ctx->d_part); ctx->d_part = 0; ctx->partCap = 0; }
    const size_t need = (size_t)2 * jSplit * (size_t)nCand * ctx->nPad;
    if (!devAlloc(&ctx->d_part, need)) return false;
    ctx->partCap = nCand;
    return true;
}

// ---------------------------------------------------------------------------
// Frozen dielectric
// ---------------------------------------------------------------------------
//
// Measured motivation, from projects/rotamerRank.cc: rotating one sidechain
// moves the dielectric of the rest of the structure by 0.19-0.71% on average,
// but moves the worst-affected environment atom by 38-46%.  So a global freeze
// is not safe -- it would be wrong exactly at the contacts that decide a
// packing -- while a freeze that exempts the near shell is.  The number of
// environment atoms that move more than 1% is about 65 and does not grow with
// the protein (51 on 1CRN, 65 on 1UBQ, 66 on 2LZM), so the exempt set is a
// constant, not a fraction of N.

int energyFreezeDielectric(energyContext* ctx,
                           const double* x, const double* y, const double* z)
{
    if (!ctx) return -1;
    const int nPad = ctx->nPad, nT = ctx->nTiles;

    // Snapshot the true field, never a held one: freezing twice in a row must
    // not laminate an old far field into the new snapshot.
    ctx->freezeActive = 0;

    if (x && uploadCoords(ctx, x, y, z) != 0) return -1;
    if (buildOrder(ctx) != 0) return -1;

    const ereal v43 = (ctx->p.occupancy == OCCUPANCY_LEGACY_FULLVOLUME)
                    ? ereal(4.188) : ereal(4.1887902048);

    CUDA_OK(ctx, cudaMemset(ctx->d_occ, 0, nPad * sizeof(ereal)));
    if (!ensurePart(ctx, 1)) {
        setError(ctx, "energyFreezeDielectric", "out of memory for j-split partials",
                 __FILE__, __LINE__);
        return -1;
    }
    const int jSplit = ctx->jSplit;
    const int blocks = (nT + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 gOcc(blocks, 1, jSplit);
    kOccupancy<<<gOcc, BLOCK>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz,
                                ctx->d_srad, ctx->d_sselfVol, ctx->d_ssilent,
                                ctx->d_tileLo, ctx->d_tileHi,
                                ctx->p, v43, ctx->p.occupancy, 1, jSplit,
                                ctx->d_part, 0, 0);
    CUDA_OK(ctx, cudaPeekAtLastError());
    kReduceOcc<<<(int)((nPad + 255) / 256), 256>>>(1, nPad, jSplit,
                                                   ctx->d_part, ctx->d_occ);
    CUDA_OK(ctx, cudaPeekAtLastError());
    kSnapshotOcc<<<(int)((nPad + 255) / 256), 256>>>(nPad, ctx->N, ctx->d_sorig,
                                                     ctx->d_occ, ctx->d_occFrozen);
    CUDA_OK(ctx, cudaPeekAtLastError());

    // Default to thawing nothing.  A caller that wants the fully coupled model
    // back should release rather than thaw everything, but thawing everything
    // is supported and must reproduce the coupled energy exactly -- that
    // equivalence is the regression test for this whole path.
    CUDA_OK(ctx, cudaMemset(ctx->d_thaw, 0, nPad * sizeof(unsigned char)));
    ctx->h_thaw.assign(nPad, 0);
    ctx->freezeActive = 1;
    return 0;
}

// Re-snapshot the frozen field from the occupancy already resident on the
// device, without recomputing it.
//
// After a delta the resident field is exactly the new conformation's: the
// thawed atoms were recomputed and applyFreeze wrote the held values back over
// everything else, and the exemption radius makes those held values provably
// unchanged rather than approximately so.  Recomputing here instead would cost
// a full occupancy pass per accepted move, which is most of a full evaluation
// and would cap a delta-driven minimisation at about 2x no matter how good the
// delta itself is.
int energyRefreezeDielectric(energyContext* ctx)
{
    if (!ctx) return -1;
    if (!ctx->freezeActive) {
        setError(ctx, "energyRefreezeDielectric", "no frozen field to refresh",
                 __FILE__, __LINE__);
        return -1;
    }
    const int nPad = ctx->nPad;
    kSnapshotOcc<<<(int)((nPad + 255) / 256), 256>>>(nPad, ctx->N, ctx->d_sorig,
                                                     ctx->d_occ, ctx->d_occFrozen);
    CUDA_OK(ctx, cudaPeekAtLastError());
    CUDA_OK(ctx, cudaMemset(ctx->d_thaw, 0, nPad * sizeof(unsigned char)));
    ctx->h_thaw.assign(nPad, 0);
    ctx->deltaSetValid = 0;
    return 0;
}

int energySetDielectricThaw(energyContext* ctx, const int* atoms, int count,
                            int accumulate)
{
    if (!ctx) return -1;
    ctx->deltaSetValid = 0;   // a new thaw set means a new changed set
    if (ctx->h_thaw.size() != (size_t)ctx->nPad) ctx->h_thaw.assign(ctx->nPad, 0);
    std::vector<unsigned char>& mask = ctx->h_thaw;
    if (!accumulate) std::fill(mask.begin(), mask.end(), (unsigned char)0);
    if (!atoms || count < 0)
    {
        std::fill(mask.begin(), mask.begin() + ctx->N, (unsigned char)1);
    }
    else
    {
        for (int k = 0; k < count; ++k)
        {
            const int a = atoms[k];
            if (a < 0 || a >= ctx->N)
            {
                setError(ctx, "energySetDielectricThaw", "atom index out of range",
                         __FILE__, __LINE__);
                return -1;
            }
            mask[a] = 1;
        }
    }
    CUDA_OK(ctx, cudaMemcpy(ctx->d_thaw, &mask[0],
                            ctx->nPad * sizeof(unsigned char),
                            cudaMemcpyHostToDevice));
    return 0;
}

double energyDielectricInfluenceRadius(energyContext* ctx)
{
    if (!ctx || ctx->N == 0) return 0.0;
    // kOccupancy accumulates neighbour j into atom i when
    //     |xi - xj| < (r_i + effectiveWaterDiameter) + r_j.
    // So moving an atom can only disturb the occupancy of atoms within
    // 2 * maxRadius + effectiveWaterDiameter of where it was or where it went.
    // Exempting that ball is sufficient, and it is the smallest radius for
    // which sufficiency is guaranteed rather than merely observed.
    return (double)(ereal(2) * ctx->maxRadius + ctx->p.effectiveWaterDiameter);
}

int energyDielectricThawCount(const energyContext* ctx)
{
    if (!ctx) return -1;
    if (ctx->h_thaw.size() != (size_t)ctx->nPad) return 0;
    int n = 0;
    for (int i = 0; i < ctx->N; ++i) if (ctx->h_thaw[i]) ++n;
    return n;
}

int energyReleaseDielectric(energyContext* ctx)
{
    if (!ctx) return -1;
    ctx->freezeActive = 0;
    return 0;
}

int energyDielectricFrozen(const energyContext* ctx)
{
    return ctx ? ctx->freezeActive : 0;
}

// ---------------------------------------------------------------------------
// Delta evaluation
//
// A move changes only the pair terms with at least one end in the changed set
// C, plus the per-atom solvation of atoms in C.  Everything else is bit for bit
// what it was, so tracking
//
//     E_new = E_old - P(C, old) + P(C, new)
//
// is exact rather than approximate, provided C really does contain every atom
// whose position or dielectric moved.  With the dielectric frozen, C is the
// thaw set, whose size is independent of N.
//
// kEnergy splits each pair half and half between its endpoints, so summing the
// per-atom partials over C undercounts the pairs that cross out of C.  V1 is
// that half sum over all j, V2 the same restricted to j in C, and
// P = 2*V1 - V2 recovers each changed pair exactly once.
// ---------------------------------------------------------------------------

__global__ void kBuildCmask(int nPad, const int* __restrict__ sorig,
                            const unsigned char* __restrict__ thawOrig,
                            unsigned char* __restrict__ cmask)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nPad) return;
    const int o = sorig[s];
    cmask[s] = (o >= 0 && thawOrig[o]) ? (unsigned char)1 : (unsigned char)0;
}

// Only the atoms of the changed set can have moved, so a delta never needs to
// re-upload N coordinates.  Writing d_s* through the inverse permutation as
// well is what lets the delta skip the spatial re-sort entirely.
__global__ void kScatterCoords(int count, const int* __restrict__ atoms,
                               const ereal* __restrict__ stage,
                               const int* __restrict__ inv,
                               ereal* x, ereal* y, ereal* z,
                               ereal* sx, ereal* sy, ereal* sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int a = atoms[i];
    const ereal vx = stage[i], vy = stage[count + i], vz = stage[2 * count + i];
    x[a] = vx; y[a] = vy; z[a] = vz;
    const int sIdx = inv[a];
    sx[sIdx] = vx; sy[sIdx] = vy; sz[sIdx] = vz;
}

__global__ void kBuildInv(int nPad, const int* __restrict__ sorig,
                          int* __restrict__ inv)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nPad) return;
    const int o = sorig[s];
    if (o >= 0) inv[o] = s;
}

__global__ void kSelectTiles(int nTiles, const unsigned char* __restrict__ cmask,
                             int* __restrict__ list, int* __restrict__ count)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nTiles) return;
    bool any = false;
    for (int k = 0; k < TILE && !any; ++k) if (cmask[t * TILE + k]) any = true;
    // Launch order does not affect the result: every tile writes only its own
    // atoms, and the host sums in original atom order.
    if (any) list[atomicAdd(count, 1)] = t;
}

__global__ void kGatherTerms(int count, const int* __restrict__ atoms,
                             const int* __restrict__ inv, int nPad,
                             const ereal* __restrict__ eVdw,
                             const ereal* __restrict__ eEle,
                             const ereal* __restrict__ eVdw2,
                             const ereal* __restrict__ eEle2,
                             const ereal* __restrict__ eSolvP,
                             const ereal* __restrict__ eSolvN,
                             const ereal* __restrict__ eSolvS,
                             ereal* __restrict__ out)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    const int s = inv[atoms[k]];
    out[0 * count + k] = eVdw[s];
    out[1 * count + k] = eEle[s];
    out[2 * count + k] = eVdw2[s];
    out[3 * count + k] = eEle2[s];
    out[4 * count + k] = eSolvP[s];
    out[5 * count + k] = eSolvN[s];
    out[6 * count + k] = eSolvS[s];
}

// Every candidate in a batched delta starts from the same resident
// conformation and differs from it only on the changed set, so seeding is a
// broadcast and the upload is a scatter.  Doing it this way keeps the
// per-candidate host cost proportional to the changed set rather than to N,
// which is the whole point of the delta.
__global__ void kSeedBatch(int nCand, int N, int nPad,
                           const ereal* __restrict__ x0,
                           const ereal* __restrict__ y0,
                           const ereal* __restrict__ z0,
                           const ereal* __restrict__ sx0,
                           const ereal* __restrict__ sy0,
                           const ereal* __restrict__ sz0,
                           ereal* bx, ereal* by, ereal* bz,
                           ereal* bsx, ereal* bsy, ereal* bsz)
{
    const int k = blockIdx.y;
    if (k >= nCand) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const size_t o = (size_t)k * N + i;
        bx[o] = x0[i]; by[o] = y0[i]; bz[o] = z0[i];
    }
    if (i < nPad) {
        const size_t o = (size_t)k * nPad + i;
        bsx[o] = sx0[i]; bsy[o] = sy0[i]; bsz[o] = sz0[i];
    }
}

__global__ void kScatterCoordsBatch(int count, int nCand, int N, int nPad,
                                    const int* __restrict__ atoms,
                                    const ereal* __restrict__ stage,
                                    const int* __restrict__ inv,
                                    ereal* bx, ereal* by, ereal* bz,
                                    ereal* bsx, ereal* bsy, ereal* bsz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (i >= count || k >= nCand) return;
    const int a = atoms[i];
    const ereal* c = stage + (size_t)k * 3 * count;
    const ereal vx = c[i], vy = c[count + i], vz = c[2 * count + i];
    bx[(size_t)k * N + a] = vx;
    by[(size_t)k * N + a] = vy;
    bz[(size_t)k * N + a] = vz;
    const int s = inv[a];
    bsx[(size_t)k * nPad + s] = vx;
    bsy[(size_t)k * nPad + s] = vy;
    bsz[(size_t)k * nPad + s] = vz;
}

__global__ void kGatherTermsBatch(int count, int nCand, int nPad,
                                  const int* __restrict__ atoms,
                                  const int* __restrict__ inv,
                                  const ereal* __restrict__ eVdw,
                                  const ereal* __restrict__ eEle,
                                  const ereal* __restrict__ eVdw2,
                                  const ereal* __restrict__ eEle2,
                                  const ereal* __restrict__ eSolvP,
                                  const ereal* __restrict__ eSolvN,
                                  const ereal* __restrict__ eSolvS,
                                  ereal* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y;
    if (i >= count || k >= nCand) return;
    const size_t s = (size_t)k * nPad + inv[atoms[i]];
    ereal* o = out + (size_t)k * 7 * count;
    o[0 * count + i] = eVdw[s];
    o[1 * count + i] = eEle[s];
    o[2 * count + i] = eVdw2[s];
    o[3 * count + i] = eEle2[s];
    o[4 * count + i] = eSolvP[s];
    o[5 * count + i] = eSolvN[s];
    o[6 * count + i] = eSolvS[s];
}

// Stage timing, enabled with PROTCAD_DELTA_TIMING=1.  The delta is meant to be// dominated by fixed overhead rather than arithmetic, so it is worth being able
// to see which stage that overhead is in.
struct deltaTiming {
    double upload, order, masks, count, occ, energy, gather, torsion;
    long   calls;
    deltaTiming() : upload(0), order(0), masks(0), count(0), occ(0),
                    energy(0), gather(0), torsion(0), calls(0) {}
    ~deltaTiming() {
        if (!calls || !getenv("PROTCAD_DELTA_TIMING")) return;
        fprintf(stderr,
            "delta timing over %ld calls, us/call:\n"
            "  uploadCoords %7.1f\n  buildOrder   %7.1f\n  masks+tiles  %7.1f\n"
            "  tileCount    %7.1f\n  occupancy    %7.1f\n  energy       %7.1f\n"
            "  gather+D2H   %7.1f\n  torsion      %7.1f\n",
            calls, 1e3 * upload / calls, 1e3 * order / calls, 1e3 * masks / calls,
            1e3 * count / calls, 1e3 * occ / calls, 1e3 * energy / calls,
            1e3 * gather / calls, 1e3 * torsion / calls);
    }
};
static deltaTiming g_dt;
static bool deltaTimingOn()
{
    static int on = -1;
    if (on < 0) on = getenv("PROTCAD_DELTA_TIMING") ? 1 : 0;
    return on != 0;
}
static double dtNow()
{
    cudaDeviceSynchronize();
    struct timeval tv; gettimeofday(&tv, 0);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#define DT_MARK(field) do { if (timing) { const double _n = dtNow(); \
    g_dt.field += _n - _t; _t = _n; } } while (0)

// The changed set, its tile list and the inverse permutation depend only on
// the thaw set and the spatial sort, and a move changes neither between the
// before and after evaluation.  Build them once per move, not twice.
static int buildDeltaSet(energyContext* ctx, const int* atoms, int count,
                         bool* rebuiltOut)
{
    const int tb = 256, nPad = ctx->nPad, nT = ctx->nTiles;
    const bool rebuilt = !ctx->deltaSetValid;
    if (rebuiltOut) *rebuiltOut = rebuilt;
    if (!rebuilt) return 0;
    CUDA_OK(ctx, cudaMemcpy(ctx->d_atoms, atoms, count * sizeof(int),
                            cudaMemcpyHostToDevice));
    kBuildCmask<<<(nPad + tb - 1) / tb, tb>>>(nPad, ctx->d_sorig, ctx->d_thaw,
                                              ctx->d_cmask);
    kBuildInv<<<(nPad + tb - 1) / tb, tb>>>(nPad, ctx->d_sorig, ctx->d_inv);
    CUDA_OK(ctx, cudaMemset(ctx->d_tileCount, 0, sizeof(int)));
    kSelectTiles<<<(nT + tb - 1) / tb, tb>>>(nT, ctx->d_cmask,
                                             ctx->d_tileList, ctx->d_tileCount);
    CUDA_OK(ctx, cudaPeekAtLastError());
    CUDA_OK(ctx, cudaMemcpy(&ctx->deltaNList, ctx->d_tileCount, sizeof(int),
                            cudaMemcpyDeviceToHost));
    ctx->deltaSetValid = 1;
    return 0;
}

int energyComputeDelta(energyContext* ctx,
                       const double* x, const double* y, const double* z,
                       const int* atoms, int count,
                       double* partOut, double* torsionOut)
{
    const bool timing = deltaTimingOn();
    double _t = timing ? dtNow() : 0.0;
    if (timing) g_dt.calls++;
    if (!ctx || count <= 0 || count > ctx->N) return -1;
    if (!ctx->freezeActive) {
        setError(ctx, "energyComputeDelta",
                 "delta evaluation requires a frozen dielectric",
                 __FILE__, __LINE__);
        return -1;
    }
    const int nPad = ctx->nPad, nT = ctx->nTiles;

    ereal v43 = (ctx->p.occupancy == OCCUPANCY_LEGACY_FULLVOLUME)
              ? ereal(4.188) : ereal(4.1887902048);
    const int tb = 256;

    if (!ctx->coordsValid) {
        setError(ctx, "energyComputeDelta", "no coordinates have been uploaded",
                 __FILE__, __LINE__);
        return -1;
    }

    // The changed set, its tile list and the inverse permutation depend only on
    // the thaw set and the spatial sort, and a move changes neither between the
    // before and after evaluation.  Build them once per move, not twice.
    bool rebuiltSet = false;
    if (buildDeltaSet(ctx, atoms, count, &rebuiltSet) != 0) return -1;
    DT_MARK(masks);
    const int nList = ctx->deltaNList;

    // Scatter, rather than re-upload.  Nothing outside the changed set can have
    // moved, so the rest of d_x and d_sx are already correct.
    if (x) {
        for (int i = 0; i < count; ++i) {
            const int a = atoms[i];
            ctx->h_stage[i]             = ereal(x[a]);
            ctx->h_stage[count + i]     = ereal(y[a]);
            ctx->h_stage[2 * count + i] = ereal(z[a]);
        }
        CUDA_OK(ctx, cudaMemcpy(ctx->d_stage, ctx->h_stage,
                                3 * count * sizeof(ereal), cudaMemcpyHostToDevice));
        kScatterCoords<<<(count + tb - 1) / tb, tb>>>(
            count, ctx->d_atoms, ctx->d_stage, ctx->d_inv,
            ctx->d_x, ctx->d_y, ctx->d_z, ctx->d_sx, ctx->d_sy, ctx->d_sz);
        CUDA_OK(ctx, cudaPeekAtLastError());
    }
    DT_MARK(upload);

    // Keeping the existing sort costs nothing in correctness -- the sort is a
    // locality heuristic, and the box-box rejection stays exact because the
    // tile bounds are recomputed.  Over nT tiles this is a rounding error next
    // to buildOrder's bin/scan/gather round trip.
    kTileBounds<<<(nT + 127) / 128, 128>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz,
                                           ctx->d_srad, ctx->d_ssilent,
                                           ctx->d_tileLo, ctx->d_tileHi);
    CUDA_OK(ctx, cudaPeekAtLastError());
    DT_MARK(order);
    DT_MARK(count);
    if (nList == 0) { if (partOut) *partOut = 0.0; }

    // Restricting the i rows removes work, but it also removes parallelism:
    // 300 atoms is about ten tiles, so ten warps, on a card that wants
    // hundreds.  Split the j loop hard enough to fill it again.  This is the
    // difference between the delta being slower than a full pass and faster.
    const int DSPLIT_MAX = 64;
    int dSplit = nList > 0 ? (targetWarps() + nList - 1) / nList : 1;
    if (dSplit > nT) dSplit = nT;
    if (dSplit > DSPLIT_MAX) dSplit = DSPLIT_MAX;
    if (dSplit < 1) dSplit = 1;

    if (!ctx->d_dpart || ctx->dSplitCap < dSplit) {
        if (ctx->d_dpart) cudaFree(ctx->d_dpart);
        ctx->d_dpart = 0; ctx->dSplitCap = 0;
        if (!devAlloc(&ctx->d_dpart, (size_t)4 * DSPLIT_MAX * ctx->nPad)) {
            setError(ctx, "energyComputeDelta", "out of memory for delta partials",
                     __FILE__, __LINE__);
            return -1;
        }
        ctx->dSplitCap = DSPLIT_MAX;
    }
    const size_t blk = (size_t)dSplit * nPad;
    const int blocks = (nList + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (nList > 0)
    {
        // Occupancy for the thawed atoms only.  The frozen field supplies
        // every other atom, so the untouched slots of d_occ are overwritten by
        // applyFreeze and their stale partials are never read.
        kOccupancy<<<dim3(blocks, 1, dSplit), BLOCK>>>(
            nT, ctx->d_sx, ctx->d_sy, ctx->d_sz, ctx->d_srad, ctx->d_sselfVol,
            ctx->d_ssilent, ctx->d_tileLo, ctx->d_tileHi, ctx->p, v43,
            ctx->p.occupancy, 1, dSplit, ctx->d_dpart, ctx->d_tileList, nList);
        CUDA_OK(ctx, cudaPeekAtLastError());
        kReduceOcc<<<(int)((nPad + 255) / 256), 256>>>(1, nPad, dSplit,
                                                       ctx->d_dpart, ctx->d_occ);
        applyFreeze(ctx, 1, ctx->d_occ);
        CUDA_OK(ctx, cudaPeekAtLastError());
        DT_MARK(occ);

        kEnergy<<<dim3(blocks, 1, dSplit), BLOCK>>>(
            nT, ctx->d_sx, ctx->d_sy, ctx->d_sz, ctx->d_srad, ctx->d_ssqrtEps,
            ctx->d_schg, ctx->d_sresIndex, ctx->d_sorig, ctx->d_ssilent,
            ctx->d_occ, ctx->d_tileLo, ctx->d_tileHi,
            ctx->d_exclCount, ctx->d_exclList, ctx->exclStride, ctx->d_exclSpan,
            ctx->p, v43, 1, dSplit,
            ctx->d_dpart, ctx->d_dpart + blk,
            ctx->d_eSolvP, ctx->d_eSolvN, ctx->d_eSolvS,
            ctx->d_tileList, nList, ctx->d_cmask,
            ctx->d_dpart + 2 * blk, ctx->d_dpart + 3 * blk);
        CUDA_OK(ctx, cudaPeekAtLastError());
        {
            const int rb = (int)((nPad + 255) / 256);
            kReduceParts<<<rb, 256>>>(1, nPad, dSplit, ctx->d_dpart,
                                      ctx->d_eVdw, ctx->d_eEle);
            kReduceParts<<<rb, 256>>>(1, nPad, dSplit, ctx->d_dpart + 2 * blk,
                                      ctx->d_eVdw2, ctx->d_eEle2);
        }
        CUDA_OK(ctx, cudaPeekAtLastError());
        DT_MARK(energy);

        kGatherTerms<<<(count + tb - 1) / tb, tb>>>(
            count, ctx->d_atoms, ctx->d_inv, nPad,
            ctx->d_eVdw, ctx->d_eEle, ctx->d_eVdw2, ctx->d_eEle2,
            ctx->d_eSolvP, ctx->d_eSolvN, ctx->d_eSolvS, ctx->d_gather);
        CUDA_OK(ctx, cudaPeekAtLastError());

        std::vector<ereal> g((size_t)count * 7);
        CUDA_OK(ctx, cudaMemcpy(&g[0], ctx->d_gather,
                                (size_t)count * 7 * sizeof(ereal),
                                cudaMemcpyDeviceToHost));

        // Kahan, in the caller's atom order, so the result does not depend on
        // the spatial sort.
        double term[7];
        for (int t = 0; t < 7; ++t) {
            double sum = 0.0, c = 0.0;
            const ereal* src = &g[(size_t)t * count];
            for (int k = 0; k < count; ++k) {
                double yv = double(src[k]) - c;
                double tt = sum + yv;
                c = (tt - sum) - yv;
                sum = tt;
            }
            term[t] = sum;
        }
        const double pVdw = 2.0 * term[0] - term[2];
        const double pEle = 2.0 * term[1] - term[3];
        if (partOut) *partOut = pVdw + pEle + term[4] + term[5] + term[6];
        DT_MARK(gather);
    }

    if (torsionOut) {
        double eTor = 0.0;
        const int rc = torsionDelta(ctx, atoms, count, rebuiltSet, &eTor);
        if (rc < 0) return -1;
        if (rc > 0 &&
            torsionTotals(ctx, ctx->d_x, ctx->d_y, ctx->d_z, 1, &eTor, 0) != 0)
            return -1;
        *torsionOut = eTor;
        DT_MARK(torsion);
    }
    return 0;
}

static int energyComputeImpl(energyContext* ctx,
                             const double* x, const double* y, const double* z,
                             double* totalOut, energyBreakdown* breakdown,
                             double* perAtomOut)
{
    if (!ctx) return -1;
    const int nPad = ctx->nPad, nT = ctx->nTiles;

    if (x && uploadCoords(ctx, x, y, z) != 0) return -1;
    if (buildOrder(ctx) != 0) return -1;

    ereal v43 = (ctx->p.occupancy == OCCUPANCY_LEGACY_FULLVOLUME)
              ? ereal(4.188) : ereal(4.1887902048);

    CUDA_OK(ctx, cudaMemset(ctx->d_occ, 0, nPad * sizeof(ereal)));

    int blocks = (nT + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (!ensurePart(ctx, 1)) {
        setError(ctx, "occupancy", "out of memory for j-split partials",
                 __FILE__, __LINE__);
        return -1;
    }
    {
        // The occlusion partials borrow the energy partial buffer.  Ordering
        // makes that safe: the occupancy reduction completes on this stream
        // before the energy pass writes anything, and the energy pass reads
        // only the reduced occ array.
        const int jSplit = ctx->jSplit;
        dim3 gOcc(blocks, 1, jSplit);
        kOccupancy<<<gOcc, BLOCK>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz,
                                  ctx->d_srad, ctx->d_sselfVol, ctx->d_ssilent,
                                  ctx->d_tileLo, ctx->d_tileHi,
                                  ctx->p, v43, ctx->p.occupancy, 1, jSplit,
                                  ctx->d_part, 0, 0);
        CUDA_OK(ctx, cudaPeekAtLastError());
        kReduceOcc<<<(int)((nPad + 255) / 256), 256>>>(1, nPad, jSplit,
                                                       ctx->d_part, ctx->d_occ);
        CUDA_OK(ctx, cudaPeekAtLastError());
        applyFreeze(ctx, 1, ctx->d_occ);
        CUDA_OK(ctx, cudaPeekAtLastError());
    }

    if (!ensurePart(ctx, 1)) {
        setError(ctx, "energyComputeImpl", "out of memory for j-split partials",
                 __FILE__, __LINE__);
        return -1;
    }
    const int jSplit = ctx->jSplit;
    dim3 gEnergy(blocks, 1, jSplit);
    kEnergy<<<gEnergy, BLOCK>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz,
                               ctx->d_srad, ctx->d_ssqrtEps, ctx->d_schg,
                               ctx->d_sresIndex, ctx->d_sorig, ctx->d_ssilent,
                               ctx->d_occ, ctx->d_tileLo, ctx->d_tileHi,
                               ctx->d_exclCount, ctx->d_exclList, ctx->exclStride,
                               ctx->d_exclSpan,
                               ctx->p, v43, 1, jSplit,
                               ctx->d_part,
                               ctx->d_part + (size_t)jSplit * nPad,
                               ctx->d_eSolvP,
                               ctx->d_eSolvN, ctx->d_eSolvS, 0, 0, 0, 0, 0);
    CUDA_OK(ctx, cudaPeekAtLastError());
    {
        int rb = (int)((nPad + 255) / 256);
        kReduceParts<<<rb, 256>>>(1, nPad, jSplit, ctx->d_part,
                                  ctx->d_eVdw, ctx->d_eEle);
        CUDA_OK(ctx, cudaPeekAtLastError());
    }

    ereal* h = ctx->h_terms;
    CUDA_OK(ctx, cudaMemcpy(h + 0 * nPad, ctx->d_eVdw,   nPad * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(h + 1 * nPad, ctx->d_eEle,   nPad * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(h + 2 * nPad, ctx->d_eSolvP, nPad * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(h + 3 * nPad, ctx->d_eSolvN, nPad * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(h + 4 * nPad, ctx->d_eSolvS, nPad * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(ctx->h_order, ctx->d_sorig,  nPad * sizeof(int),   cudaMemcpyDeviceToHost));

    // Reduce on the host, in original atom order, with Kahan compensation.
    // Summation order is therefore independent of the spatial sort, which is
    // what makes repeated evaluations of the same conformation bit-identical.
    if (perAtomOut) std::fill(perAtomOut, perAtomOut + ctx->N, 0.0);

    buildInvOrder(ctx, nPad);
    const int* inv = &ctx->invOrder[0];

    double out[5];
    for (int t = 0; t < 5; ++t) {
        const ereal* src = h + (size_t)t * nPad;
        double sum = 0.0, c = 0.0;
        for (int i = 0; i < ctx->N; ++i) {
            int s = inv[i];
            double v = (s >= 0 ? double(src[s]) : 0.0);
            if (perAtomOut) perAtomOut[i] += v;
            double yv = v - c;
            double tt = sum + yv;
            c = (tt - sum) - yv;
            sum = tt;
        }
        out[t] = sum;
    }

    double eTor = 0.0;
    if (torsionTotals(ctx, ctx->d_x, ctx->d_y, ctx->d_z, 1, &eTor, perAtomOut) != 0) return -1;

    double total = out[0] + out[1] + out[2] + out[3] + out[4] + eTor;
    if (totalOut) *totalOut = total;
    if (breakdown) {
        breakdown->vdw               = out[0];
        breakdown->torsion           = eTor;
        breakdown->electrostatic     = out[1];
        breakdown->solvationPolar    = out[2];
        breakdown->solvationNonpolar = out[3];
        breakdown->solvationEntropy  = out[4];
        breakdown->total             = total;
    }
    return 0;
}

int energyCompute(energyContext* ctx,
                  const double* x, const double* y, const double* z,
                  double* totalOut, energyBreakdown* breakdown)
{
    return energyComputeImpl(ctx, x, y, z, totalOut, breakdown, 0);
}

// ---------------------------------------------------------------------------
// Host: batched candidate evaluation
// ---------------------------------------------------------------------------
//
// A protein-sized system leaves the GPU almost entirely idle -- a 600-atom
// protein is ~19 warps on 1280 cores, about 5% occupancy -- so evaluating one
// candidate conformation costs very nearly what evaluating many costs.  This
// turns that idle capacity into throughput by giving every kernel a candidate
// dimension on blockIdx.y.
//
// All candidates share the spatial order built from the context's current
// resident coordinates.  That is exact (the order only drives AABB culling) and
// it keeps radii, charges, residue indices, exclusions and silent flags
// batch-invariant.  Callers should set the base conformation with
// energySetCoords before calling, so the order reflects a representative
// geometry; candidates that differ by a single sidechain will cull as well
// against the base order as against their own.
//
// The host reduction is the same Kahan sum in original atom order used by the
// single-candidate path, applied per candidate, so a batched energy is
// bit-identical to the same conformation evaluated alone.
// Build candidate conformations on the device.
//
// One block per candidate.  Each block copies the resident coordinates into its
// own slice, then walks the requested rotation groups in order, applying the
// candidate's angle delta to each.  The walk must be sequential because the
// sidechain is a kinematic chain: the axis of chi_2 has already been moved by
// chi_1.  A single barrier at the top of each group is sufficient, since a
// group's axis atoms are never among its own members -- the host builds members
// from the subtree strictly distal to axisB.
//
// Angles are deltas in degrees, matching residue::setChiByDelta.  The
// degrees-to-radians constant is deliberately the truncated 0.017453293 used by
// CMath::rotationMatrix, not M_PI/180: the point of this kernel is to reproduce
// the host transform, and the more accurate constant would put a small
// systematic rotation between the two.
__global__ void kBuildRotamers(int N, int groupBegin, int nGroups,
                               const ereal* __restrict__ x0,
                               const ereal* __restrict__ y0,
                               const ereal* __restrict__ z0,
                               const int* __restrict__ axisA,
                               const int* __restrict__ axisB,
                               const int* __restrict__ memberStart,
                               const int* __restrict__ members,
                               const ereal* __restrict__ angles,
                               ereal* ox, ereal* oy, ereal* oz)
{
    const int k   = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t off = (size_t)k * N;

    for (int i = tid; i < N; i += blockDim.x)
    {
        ox[off + i] = x0[i]; oy[off + i] = y0[i]; oz[off + i] = z0[i];
    }

    for (int g = 0; g < nGroups; ++g)
    {
        __syncthreads();

        const int gi = groupBegin + g;
        const int i1 = axisA[gi], i2 = axisB[gi];

        const ereal a1x = ox[off + i1], a1y = oy[off + i1], a1z = oz[off + i1];
        const ereal dx  = ox[off + i2] - a1x;
        const ereal dy  = oy[off + i2] - a1y;
        const ereal dz  = oz[off + i2] - a1z;

        const ereal theta = (ereal)0.017453293 * angles[(size_t)k * nGroups + g];
        const ereal st = sin(theta), ct = cos(theta);
        const ereal nrm = sqrt(dx * dx + dy * dy + dz * dz);
        if (nrm <= (ereal)0) continue;
        const ereal n1 = dx / nrm, n2 = dy / nrm, n3 = dz / nrm;
        const ereal n11 = n1 * n1, n12 = n1 * n2, n13 = n1 * n3;
        const ereal n22 = n2 * n2, n23 = n2 * n3, n33 = n3 * n3;
        const ereal omc = (ereal)1 - ct;

        const ereal r00 = n11 + ((ereal)1 - n11) * ct;
        const ereal r01 = n12 * omc - n3 * st;
        const ereal r02 = n13 * omc + n2 * st;
        const ereal r10 = n12 * omc + n3 * st;
        const ereal r11 = n22 + ((ereal)1 - n22) * ct;
        const ereal r12 = n23 * omc - n1 * st;
        const ereal r20 = n13 * omc - n2 * st;
        const ereal r21 = n23 * omc + n1 * st;
        const ereal r22 = n33 + ((ereal)1 - n33) * ct;

        const int s = memberStart[gi], e = memberStart[gi + 1];
        for (int m = s + tid; m < e; m += blockDim.x)
        {
            const int a = members[m];
            const ereal px = ox[off + a] - a1x;
            const ereal py = oy[off + a] - a1y;
            const ereal pz = oz[off + a] - a1z;
            ox[off + a] = r00 * px + r01 * py + r02 * pz + a1x;
            oy[off + a] = r10 * px + r11 * py + r12 * pz + a1y;
            oz[off + a] = r20 * px + r21 * py + r22 * pz + a1z;
        }
    }
}

// Population Monte Carlo proposal.  Identical rotation mathematics to
// kBuildRotamers, but each replica reads its own accepted state rather than a
// single shared base, and applies its own rotation-group range, so replicas may
// be moving different residues on the same launch.  Angles are indexed with a
// fixed stride so a ragged set of chi counts needs no prefix sum.
__global__ void kBuildReplicas(int N,
                               const int* __restrict__ groupBegin,
                               const int* __restrict__ nGroupsArr,
                               const ereal* __restrict__ rx,
                               const ereal* __restrict__ ry,
                               const ereal* __restrict__ rz,
                               const int* __restrict__ axisA,
                               const int* __restrict__ axisB,
                               const int* __restrict__ memberStart,
                               const int* __restrict__ members,
                               const ereal* __restrict__ angles,
                               int angleStride,
                               ereal* ox, ereal* oy, ereal* oz)
{
    const int k   = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t off = (size_t)k * N;

    for (int i = tid; i < N; i += blockDim.x)
    {
        ox[off + i] = rx[off + i]; oy[off + i] = ry[off + i]; oz[off + i] = rz[off + i];
    }

    const int gb = groupBegin[k], ng = nGroupsArr[k];

    for (int g = 0; g < ng; ++g)
    {
        __syncthreads();

        const int gi = gb + g;
        const int i1 = axisA[gi], i2 = axisB[gi];

        const ereal a1x = ox[off + i1], a1y = oy[off + i1], a1z = oz[off + i1];
        const ereal dx  = ox[off + i2] - a1x;
        const ereal dy  = oy[off + i2] - a1y;
        const ereal dz  = oz[off + i2] - a1z;

        const ereal theta = (ereal)0.017453293 * angles[(size_t)k * angleStride + g];
        const ereal st = sin(theta), ct = cos(theta);
        const ereal nrm = sqrt(dx * dx + dy * dy + dz * dz);
        if (nrm <= (ereal)0) continue;
        const ereal n1 = dx / nrm, n2 = dy / nrm, n3 = dz / nrm;
        const ereal n11 = n1 * n1, n12 = n1 * n2, n13 = n1 * n3;
        const ereal n22 = n2 * n2, n23 = n2 * n3, n33 = n3 * n3;
        const ereal omc = (ereal)1 - ct;

        const ereal r00 = n11 + ((ereal)1 - n11) * ct;
        const ereal r01 = n12 * omc - n3 * st;
        const ereal r02 = n13 * omc + n2 * st;
        const ereal r10 = n12 * omc + n3 * st;
        const ereal r11 = n22 + ((ereal)1 - n22) * ct;
        const ereal r12 = n23 * omc - n1 * st;
        const ereal r20 = n13 * omc - n2 * st;
        const ereal r21 = n23 * omc + n1 * st;
        const ereal r22 = n33 + ((ereal)1 - n33) * ct;

        const int s = memberStart[gi], e = memberStart[gi + 1];
        for (int m = s + tid; m < e; m += blockDim.x)
        {
            const int a = members[m];
            const ereal px = ox[off + a] - a1x;
            const ereal py = oy[off + a] - a1y;
            const ereal pz = oz[off + a] - a1z;
            ox[off + a] = r00 * px + r01 * py + r02 * pz + a1x;
            oy[off + a] = r10 * px + r11 * py + r12 * pz + a1y;
            oz[off + a] = r20 * px + r21 * py + r22 * pz + a1z;
        }
    }
}

// Commit accepted proposals.  Rejected replicas keep their previous state, so
// a rejection costs nothing beyond the evaluation already performed.
__global__ void kCommitReplicas(int N, const int* __restrict__ accept,
                                const ereal* __restrict__ bx,
                                const ereal* __restrict__ by,
                                const ereal* __restrict__ bz,
                                ereal* rx, ereal* ry, ereal* rz)
{
    const int k = blockIdx.x;
    if (!accept[k]) return;
    const size_t off = (size_t)k * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        rx[off + i] = bx[off + i]; ry[off + i] = by[off + i]; rz[off + i] = bz[off + i];
    }
}

__global__ void kSeedReplicas(int N, int nRepl,
                              const ereal* __restrict__ x0,
                              const ereal* __restrict__ y0,
                              const ereal* __restrict__ z0,
                              ereal* rx, ereal* ry, ereal* rz)
{
    const int k = blockIdx.x;
    if (k >= nRepl) return;
    const size_t off = (size_t)k * N;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
    {
        rx[off + i] = x0[i]; ry[off + i] = y0[i]; rz[off + i] = z0[i];
    }
}

static bool ensureBatch(energyContext* ctx, int nCand)
{
    if (nCand <= ctx->batchCap) return true;

    void* old[] = { ctx->d_bx, ctx->d_by, ctx->d_bz,
                    ctx->d_bsx, ctx->d_bsy, ctx->d_bsz,
                    ctx->d_bocc, ctx->d_btileLo, ctx->d_btileHi, ctx->d_bterms };
    for (size_t i = 0; i < sizeof(old) / sizeof(old[0]); ++i)
        if (old[i]) cudaFree(old[i]);
    ctx->d_bx = ctx->d_by = ctx->d_bz = 0;
    ctx->d_bsx = ctx->d_bsy = ctx->d_bsz = 0;
    ctx->d_bocc = ctx->d_btileLo = ctx->d_btileHi = ctx->d_bterms = 0;
    ctx->batchCap = 0;

    const size_t K = (size_t)nCand, N = ctx->N, nPad = ctx->nPad, nT = ctx->nTiles;
    bool ok = true;
    ok &= devAlloc(&ctx->d_bx,  K * N)    && devAlloc(&ctx->d_by, K * N)
       && devAlloc(&ctx->d_bz,  K * N);
    ok &= devAlloc(&ctx->d_bsx, K * nPad) && devAlloc(&ctx->d_bsy, K * nPad)
       && devAlloc(&ctx->d_bsz, K * nPad);
    ok &= devAlloc(&ctx->d_bocc, K * nPad);
    ok &= devAlloc(&ctx->d_btileLo, K * nT * 3) && devAlloc(&ctx->d_btileHi, K * nT * 3);
    ok &= devAlloc(&ctx->d_bterms, K * nPad * 5);
    if (!ok) return false;

    ctx->h_bx.resize(K * N); ctx->h_by.resize(K * N); ctx->h_bz.resize(K * N);
    ctx->h_bterms.resize(K * nPad * 5);
    ctx->batchCap = nCand;
    return true;
}

// Shared tail of every batched evaluation: candidate coordinates are already in
// d_bx/d_by/d_bz and the spatial order is already built.
// Phase timing for evalBatch, enabled with PROTCAD_ENERGY_PROFILE=1.
//
// The host reduction here is O(nCand * nPad * 5) and single threaded, and it
// has been guessed at more than once without being measured. Moving the
// reduction onto the device is only worth doing if it is actually a meaningful
// share of the batch cost, so measure before optimising.
struct batchProfile
{
    double gather, tile, occ, energy, copy, torsion, reduce; long calls; long cands;
    batchProfile() : gather(0), tile(0), occ(0), energy(0), copy(0), torsion(0),
                     reduce(0), calls(0), cands(0) {}
};
static batchProfile g_bprof;
static int g_bprofOn = -1;

static double profWall()
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void profReport()
{
    if (!g_bprof.calls) return;
    double tot = g_bprof.gather + g_bprof.tile + g_bprof.occ + g_bprof.energy
               + g_bprof.copy + g_bprof.torsion + g_bprof.reduce;
    if (tot <= 0) return;
    const long nc = std::max(1L, g_bprof.cands);
    fprintf(stderr, "\n[energy] evalBatch profile: %ld calls, %ld candidates"
            " (N=%d, nTiles=%d, jSplit=%d)\n",
            g_bprof.calls, g_bprof.cands, g_geomN, g_geomTiles, g_geomSplit);
    // "torsion" is the whole torsionTotals call -- kernel launch, its own D2H,
    // and its Kahan sum.  It used to be billed to "host Kahan", which made that
    // bucket look like pure host reduction cost when it was not.
    const char* nm[7] = {"kGatherCoords", "kTileBounds", "kOccupancy",
                         "kEnergy", "D2H copy", "torsion", "host Kahan"};
    double vv[7] = {g_bprof.gather, g_bprof.tile, g_bprof.occ,
                    g_bprof.energy, g_bprof.copy, g_bprof.torsion,
                    g_bprof.reduce};
    for (int i = 0; i < 7; ++i)
        fprintf(stderr, "  %-14s %8.3f s  %5.1f%%   %8.1f us/cand\n",
                nm[i], vv[i], 100.0 * vv[i] / tot, 1e6 * vv[i] / nc);
    fprintf(stderr, "  %-14s %8.3f s          %8.1f us/cand\n",
            "total", tot, 1e6 * tot / nc);
}

static bool profEnabled()
{
    if (g_bprofOn < 0)
    {
        const char* e = getenv("PROTCAD_ENERGY_PROFILE");
        g_bprofOn = (e && atoi(e)) ? 1 : 0;
        if (g_bprofOn) atexit(profReport);
    }
    return g_bprofOn == 1;
}

static int evalBatch(energyContext* ctx, int nCand, double* totals)
{
    const bool prof = profEnabled();
    double tA = prof ? (cudaDeviceSynchronize(), profWall()) : 0.0;
    const int nPad = ctx->nPad, nT = ctx->nTiles, N = ctx->N;
    ereal v43 = (ctx->p.occupancy == OCCUPANCY_LEGACY_FULLVOLUME)
              ? ereal(4.188) : ereal(4.1887902048);

    dim3 gGather((nPad + 255) / 256, nCand);
    kGatherCoords<<<gGather, 256>>>(nPad, N, ctx->d_order,
                                    ctx->d_bx, ctx->d_by, ctx->d_bz,
                                    ctx->d_bsx, ctx->d_bsy, ctx->d_bsz);
    CUDA_OK(ctx, cudaPeekAtLastError());

    double tGather = prof ? (cudaDeviceSynchronize(), profWall()) : 0.0;

    dim3 gTile((nT + 127) / 128, nCand);
    kTileBounds<<<gTile, 128>>>(nT, ctx->d_bsx, ctx->d_bsy, ctx->d_bsz,
                                ctx->d_srad, ctx->d_ssilent,
                                ctx->d_btileLo, ctx->d_btileHi);
    CUDA_OK(ctx, cudaPeekAtLastError());

    double tTile = prof ? (cudaDeviceSynchronize(), profWall()) : 0.0;

    CUDA_OK(ctx, cudaMemset(ctx->d_bocc, 0, (size_t)nCand * nPad * sizeof(ereal)));

    if (!ensurePart(ctx, nCand)) {
        setError(ctx, "evalBatch", "out of memory for j-split partials",
                 __FILE__, __LINE__);
        return -1;
    }
    dim3 gWork((nT + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, nCand);
    {
        const int jSplit = ctx->jSplit;
        dim3 gOcc((nT + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, nCand, jSplit);
        kOccupancy<<<gOcc, BLOCK>>>(nT, ctx->d_bsx, ctx->d_bsy, ctx->d_bsz,
                                 ctx->d_srad, ctx->d_sselfVol, ctx->d_ssilent,
                                 ctx->d_btileLo, ctx->d_btileHi,
                                 ctx->p, v43, ctx->p.occupancy, nCand, jSplit,
                                 ctx->d_part, 0, 0);
        CUDA_OK(ctx, cudaPeekAtLastError());
        const size_t tot = (size_t)nCand * nPad;
        kReduceOcc<<<(int)((tot + 255) / 256), 256>>>(nCand, nPad, jSplit,
                                                      ctx->d_part, ctx->d_bocc);
        CUDA_OK(ctx, cudaPeekAtLastError());
        applyFreeze(ctx, nCand, ctx->d_bocc);
        CUDA_OK(ctx, cudaPeekAtLastError());
    }

    double tOcc = prof ? (cudaDeviceSynchronize(), profWall()) : 0.0;

    const size_t stride = (size_t)nCand * nPad;
    ereal* t0 = ctx->d_bterms;
    if (!ensurePart(ctx, nCand)) {
        setError(ctx, "evalBatch", "out of memory for j-split partials",
                 __FILE__, __LINE__);
        return -1;
    }
    const int jSplit = ctx->jSplit;
    dim3 gEnergy((nT + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, nCand, jSplit);
    kEnergy<<<gEnergy, BLOCK>>>(nT, ctx->d_bsx, ctx->d_bsy, ctx->d_bsz,
                              ctx->d_srad, ctx->d_ssqrtEps, ctx->d_schg,
                              ctx->d_sresIndex, ctx->d_sorig, ctx->d_ssilent,
                              ctx->d_bocc, ctx->d_btileLo, ctx->d_btileHi,
                              ctx->d_exclCount, ctx->d_exclList, ctx->exclStride,
                              ctx->d_exclSpan, ctx->p, v43, nCand, jSplit,
                              ctx->d_part,
                              ctx->d_part + (size_t)jSplit * stride,
                              t0 + 2 * stride, t0 + 3 * stride, t0 + 4 * stride,
                              0, 0, 0, 0, 0);
    CUDA_OK(ctx, cudaPeekAtLastError());

    {
        const size_t tot = stride;
        int rb = (int)((tot + 255) / 256);
        kReduceParts<<<rb, 256>>>(nCand, nPad, jSplit, ctx->d_part,
                                  t0, t0 + stride);
        CUDA_OK(ctx, cudaPeekAtLastError());
    }

    // Issue the torsion kernel and its copy here, before the single sync
    // below, so it overlaps kEnergy and shares the one round trip instead of
    // adding a second launch-and-block after the nonbonded copies.
    if (torsionLaunch(ctx, ctx->d_bx, ctx->d_by, ctx->d_bz, nCand) != 0)
        return -1;

    double tB = prof ? (cudaDeviceSynchronize(), profWall()) : 0.0;

    CUDA_OK(ctx, cudaMemcpy(&ctx->h_bterms[0], ctx->d_bterms,
                            stride * 5 * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(ctx->h_order, ctx->d_sorig, nPad * sizeof(int),
                            cudaMemcpyDeviceToHost));

    double tC = prof ? profWall() : 0.0;

    buildInvOrder(ctx, nPad);
    const int* inv = &ctx->invOrder[0];

    std::vector<double> eTor(nCand, 0.0);
    if (torsionReduce(ctx, nCand, &eTor[0]) != 0)
        return -1;

    double tT = prof ? profWall() : 0.0;

    for (int k = 0; k < nCand; ++k) {
        double total = eTor[k];
        for (int t = 0; t < 5; ++t) {
            const ereal* src = &ctx->h_bterms[(size_t)t * stride + (size_t)k * nPad];
            double sum = 0.0, c = 0.0;
            for (int i = 0; i < N; ++i) {
                int sIdx = inv[i];
                double yv = (sIdx >= 0 ? double(src[sIdx]) : 0.0) - c;
                double tt = sum + yv;
                c = (tt - sum) - yv;
                sum = tt;
            }
            total += sum;
        }
        totals[k] = total;
    }
    if (prof)
    {
        double tD = profWall();
        g_bprof.gather  += tGather - tA;
        g_bprof.tile    += tTile - tGather;
        g_bprof.occ     += tOcc - tTile;
        g_bprof.energy  += tB - tOcc;
        g_bprof.copy    += tC - tB;
        g_bprof.torsion += tT - tC;
        g_bprof.reduce  += tD - tT;
        g_bprof.calls   += 1;
        g_bprof.cands   += nCand;
    }
    return 0;
}

int energyComputeBatch(energyContext* ctx, int nCand,
                       const double* x, const double* y, const double* z,
                       double* totals)
{
    if (!ctx || nCand <= 0 || !x || !y || !z || !totals) return -1;
    const int N = ctx->N;

    if (buildOrder(ctx) != 0) return -1;
    if (!ensureBatch(ctx, nCand)) { ctx->lastError = "batch allocation failed"; return -1; }

    const size_t KN = (size_t)nCand * N;
    for (size_t i = 0; i < KN; ++i) {
        ctx->h_bx[i] = ereal(x[i]); ctx->h_by[i] = ereal(y[i]); ctx->h_bz[i] = ereal(z[i]);
    }
    CUDA_OK(ctx, cudaMemcpy(ctx->d_bx, &ctx->h_bx[0], KN * sizeof(ereal), cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_by, &ctx->h_by[0], KN * sizeof(ereal), cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_bz, &ctx->h_bz[0], KN * sizeof(ereal), cudaMemcpyHostToDevice));

    return evalBatch(ctx, nCand, totals);
}

// Batched restricted evaluation.  This is energyComputeDelta's restricted sum
// and evalBatch's candidate dimension in a single launch: nCand conformations
// that differ from the resident one only on `atoms`, each evaluated over the
// pairs that set can have changed.
//
// The candidates share a changed set, a tile list and a torsion list, so all of
// that is built once for the batch rather than once per candidate.  Candidate
// coordinates are seeded from the resident conformation on the device and only
// the changed atoms are uploaded, so host cost per candidate is |C| rather than
// N -- without which the batch's own bookkeeping would eat the delta's win, as
// it did for the single-move path before the scatter landed.

// Reduce a batch's changed-set terms to one number per candidate.
//
// This used to be done on the host: kGatherTermsBatch wrote 7 values per
// changed atom per candidate, all of it came home over PCIe, and the caller
// ran a Kahan sum over 7 * count * nCand doubles.  On 1crn that is 58k
// double-precision adds behind a 233 kB transfer for every trial, and it made
// energyComputeBatchDelta 77% of a minimisation step while the kernels it
// exists to run accounted for a fifth of it.  The changed set is the one thing
// in this path whose size grows with the structure, so shipping it per atom is
// exactly the wrong thing to move across the bus.
//
// The seven terms enter the total linearly -- 2*t0 + 2*t1 - t2 - t3 + t4 + t5
// + t6 -- so there is no reason to reduce them separately or to reduce them
// here at all: each atom's contribution collapses to one scalar and the whole
// changed set collapses to one number per candidate.
//
// The host version's determinism guarantee is preserved and it is the only
// real constraint on the schedule.  A candidate's value must not depend on the
// spatial sort or on how many candidates share the batch, so the summation
// order has to be fixed by atom index and nothing else.  Each thread therefore
// takes one contiguous index range and sums it in order, and thread zero
// combines the partials in thread order; the result is a function of `count`
// alone.  A conventional strided or tree reduction would be faster and would
// quietly make a candidate's energy depend on the batch it was evaluated in.
// Kahan is kept at both levels, so accuracy is no worse than the host loop's.
// Fusing the gather into that reduction.  Once the seven per-atom values are
// only ever consumed by the sum above, materialising them is pure cost: the
// gather wrote 7 * count * nCand elements to global memory and the reduction
// immediately read them back, for a quantity that collapses to one double per
// candidate.  This kernel reads the term arrays where they already live and
// combines each atom on the spot, so the intermediate buffer, its allocation
// and a kernel launch all disappear.
//
// The summation schedule is fixed by atom index and nothing else --
// contiguous per-thread index ranges, in-order Kahan at both levels -- which
// is the whole point: a candidate's value depends on `count` and nothing else, not on
// the spatial sort and not on the batch it happened to share.
__global__ void kFusedBatchPart(int count, int nPad,
                                const int* __restrict__ atoms,
                                const int* __restrict__ inv,
                                const ereal* __restrict__ eVdw,
                                const ereal* __restrict__ eEle,
                                const ereal* __restrict__ eVdw2,
                                const ereal* __restrict__ eEle2,
                                const ereal* __restrict__ eSolvP,
                                const ereal* __restrict__ eSolvN,
                                const ereal* __restrict__ eSolvS,
                                double* __restrict__ out)
{
    const int k = blockIdx.x;
    __shared__ double part[256];
    const int nt = blockDim.x, tid = threadIdx.x;
    const int chunk = (count + nt - 1) / nt;
    const int lo = tid * chunk;
    int hi = lo + chunk; if (hi > count) hi = count;

    double sum = 0.0, c = 0.0;
    for (int i = lo; i < hi; ++i) {
        const size_t s = (size_t)k * nPad + inv[atoms[i]];
        const double v = 2.0 * (double(eVdw[s]) + double(eEle[s]))
                       - double(eVdw2[s]) - double(eEle2[s])
                       + double(eSolvP[s]) + double(eSolvN[s])
                       + double(eSolvS[s]);
        const double y = v - c, t = sum + y;
        c = (t - sum) - y; sum = t;
    }
    part[tid] = (lo < count) ? sum : 0.0;
    __syncthreads();
    if (tid == 0) {
        double s = 0.0, cc = 0.0;
        for (int i = 0; i < nt; ++i) {
            const double y = part[i] - cc, t = s + y;
            cc = (t - s) - y; s = t;
        }
        out[k] = s;
    }
}

// Sub-phase timing for the batch delta, enabled with PROTCAD_PROFILE=1.  The
// caller-level profile only says this call dominates a trial; it cannot say
// which half of it, and two plausible answers in a row turned out to be wrong.
// Each mark synchronises, so the total here runs slightly above the unprofiled
// call -- the distribution is the point, not the absolute.
struct batchProf {
    double stageFill, stageCopy, scatter, bounds, occ, energy, reduce, torsion;
    long calls;
    batchProf() : stageFill(0), stageCopy(0), scatter(0), bounds(0), occ(0),
                  energy(0), reduce(0), torsion(0), calls(0) {}
    ~batchProf() {
        if (!calls || !getenv("PROTCAD_PROFILE")) return;
        const double us = 1e6 / double(calls);
        double tot = stageFill + stageCopy + scatter + bounds + occ + energy
                   + reduce + torsion;
        fprintf(stderr,
            "  batchDelta over %ld calls, us/call:\n"
            "    stage fill (host)  %8.1f  %5.1f %%\n"
            "    stage H2D+scatter  %8.1f  %5.1f %%\n"
            "    tile bounds        %8.1f  %5.1f %%\n"
            "    occupancy          %8.1f  %5.1f %%\n"
            "    energy             %8.1f  %5.1f %%\n"
            "    reduce + D2H       %8.1f  %5.1f %%\n"
            "    torsion            %8.1f  %5.1f %%\n"
            "    sum                %8.1f\n",
            calls,
            stageFill * us, 100 * stageFill / tot,
            (stageCopy + scatter) * us, 100 * (stageCopy + scatter) / tot,
            bounds * us, 100 * bounds / tot,
            occ * us, 100 * occ / tot,
            energy * us, 100 * energy / tot,
            reduce * us, 100 * reduce / tot,
            torsion * us, 100 * torsion / tot,
            tot * us);
    }
};
static batchProf g_bp;
static bool batchProfOn()
{
    static int on = -1;
    if (on < 0) on = getenv("PROTCAD_PROFILE") ? 1 : 0;
    return on != 0;
}
static double bpNow()
{
    cudaDeviceSynchronize();
    struct timeval tv; gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec / 1e6;
}
#define BP_MARK(field) do { if (bprof) { const double _n = bpNow(); \
    g_bp.field += _n - _bt; _bt = _n; } } while (0)

int energyComputeBatchDelta(energyContext* ctx, int nCand,
                            const double* x, const double* y, const double* z,
                            const int* atoms, int count,
                            double* partOut, double* torsionOut,
                            const int* moved, int nMoved)
{
    if (!ctx || nCand <= 0 || count <= 0 || count > ctx->N || !atoms) return -1;
    if (!ctx->freezeActive) {
        setError(ctx, "energyComputeBatchDelta",
                 "delta evaluation requires a frozen dielectric",
                 __FILE__, __LINE__);
        return -1;
    }
    if (!ctx->coordsValid) {
        setError(ctx, "energyComputeBatchDelta",
                 "no coordinates have been uploaded", __FILE__, __LINE__);
        return -1;
    }
    if (!ensureBatch(ctx, nCand)) { ctx->lastError = "batch allocation failed"; return -1; }

    const int nPad = ctx->nPad, nT = ctx->nTiles, N = ctx->N, tb = 256;
    const ereal v43 = (ctx->p.occupancy == OCCUPANCY_LEGACY_FULLVOLUME)
                    ? ereal(4.188) : ereal(4.1887902048);

    bool rebuiltSet = false;
    if (buildDeltaSet(ctx, atoms, count, &rebuiltSet) != 0) return -1;
    const int nList = ctx->deltaNList;

    // Seed every candidate from the resident conformation, then overwrite only
    // the changed atoms.  Both the original-order and sorted copies are seeded,
    // so the spatial order is inherited rather than rebuilt.
    {
        const int span = nPad > N ? nPad : N;
        dim3 g((span + tb - 1) / tb, nCand);
        kSeedBatch<<<g, tb>>>(nCand, N, nPad,
                              ctx->d_x, ctx->d_y, ctx->d_z,
                              ctx->d_sx, ctx->d_sy, ctx->d_sz,
                              ctx->d_bx, ctx->d_by, ctx->d_bz,
                              ctx->d_bsx, ctx->d_bsy, ctx->d_bsz);
        CUDA_OK(ctx, cudaPeekAtLastError());
    }

    const bool bprof = batchProfOn();
    double _bt = bprof ? bpNow() : 0.0;
    if (bprof) ++g_bp.calls;

    // kSeedBatch has already given every candidate the resident conformation,
    // so only atoms that actually differ from it need staging.  The changed set
    // is the union of what the move displaced and everything whose dielectric
    // environment it disturbed, and the second part is by construction
    // unchanged -- shipping it was sending the GPU values it already had.
    const int*  sAtoms = (moved && nMoved > 0) ? moved  : atoms;
    const int   sCount = (moved && nMoved > 0) ? nMoved : count;
    const int*  dStage = ctx->d_atoms;
    if (moved && nMoved > 0) {
        if (!ctx->d_moved || ctx->movedCap < (size_t)nMoved) {
            if (ctx->d_moved) cudaFree(ctx->d_moved);
            ctx->d_moved = 0; ctx->movedCap = 0;
            if (!devAlloc(&ctx->d_moved, (size_t)nMoved)) {
                setError(ctx, "energyComputeBatchDelta", "out of memory for moved set",
                         __FILE__, __LINE__);
                return -1;
            }
            ctx->movedCap = (size_t)nMoved;
        }
        CUDA_OK(ctx, cudaMemcpy(ctx->d_moved, moved, (size_t)nMoved * sizeof(int),
                                cudaMemcpyHostToDevice));
        dStage = ctx->d_moved;
    }

    if (x) {
        const size_t need = (size_t)3 * sCount * nCand;
        if (!ctx->d_bstage || ctx->bdStageCap < need) {
            if (ctx->d_bstage) cudaFree(ctx->d_bstage);
            ctx->d_bstage = 0; ctx->bdStageCap = 0;
            if (!devAlloc(&ctx->d_bstage, need)) {
                setError(ctx, "energyComputeBatchDelta", "out of memory for staging",
                         __FILE__, __LINE__);
                return -1;
            }
            ctx->bdStageCap = need;
        }
        ctx->h_bstage.resize(need);
        for (int k = 0; k < nCand; ++k) {
            const double* cx = x + (size_t)k * N;
            const double* cy = y + (size_t)k * N;
            const double* cz = z + (size_t)k * N;
            ereal* s = &ctx->h_bstage[(size_t)k * 3 * sCount];
            for (int i = 0; i < sCount; ++i) {
                const int a = sAtoms[i];
                s[i]              = ereal(cx[a]);
                s[sCount + i]     = ereal(cy[a]);
                s[2 * sCount + i] = ereal(cz[a]);
            }
        }
        BP_MARK(stageFill);
        CUDA_OK(ctx, cudaMemcpy(ctx->d_bstage, &ctx->h_bstage[0],
                                need * sizeof(ereal), cudaMemcpyHostToDevice));
        dim3 g((sCount + tb - 1) / tb, nCand);
        kScatterCoordsBatch<<<g, tb>>>(sCount, nCand, N, nPad,
                                       dStage, ctx->d_bstage, ctx->d_inv,
                                       ctx->d_bx, ctx->d_by, ctx->d_bz,
                                       ctx->d_bsx, ctx->d_bsy, ctx->d_bsz);
        CUDA_OK(ctx, cudaPeekAtLastError());
        BP_MARK(scatter);
    }

    // Bounds are per candidate; the sort itself is shared and stays exact
    // because box-box rejection only needs the bounds to be current.
    {
        dim3 g((nT + 127) / 128, nCand);
        kTileBounds<<<g, 128>>>(nT, ctx->d_bsx, ctx->d_bsy, ctx->d_bsz,
                                ctx->d_srad, ctx->d_ssilent,
                                ctx->d_btileLo, ctx->d_btileHi);
        CUDA_OK(ctx, cudaPeekAtLastError());
        BP_MARK(bounds);
    }

    if (nList == 0) {
        for (int k = 0; k < nCand; ++k) if (partOut) partOut[k] = 0.0;
    }

    // The batch already supplies nCand times the parallelism the single-move
    // delta had to manufacture by splitting j, so the split is sized against
    // nList * nCand rather than nList alone.
    const int DSPLIT_MAX = 64;
    const long rows = (long)nList * nCand;
    int dSplit = rows > 0 ? (int)((targetWarps() + rows - 1) / rows) : 1;
    if (dSplit > nT) dSplit = nT;
    if (dSplit > DSPLIT_MAX) dSplit = DSPLIT_MAX;
    if (dSplit < 1) dSplit = 1;

    if (!ctx->d_bdpart || ctx->bdSplitCap < dSplit || ctx->bdCandCap < nCand) {
        if (ctx->d_bdpart) cudaFree(ctx->d_bdpart);
        ctx->d_bdpart = 0; ctx->bdSplitCap = 0;
        const int sc = DSPLIT_MAX, cc = nCand > ctx->bdCandCap ? nCand : ctx->bdCandCap;
        if (!devAlloc(&ctx->d_bdpart, (size_t)4 * sc * cc * nPad)) {
            setError(ctx, "energyComputeBatchDelta", "out of memory for delta partials",
                     __FILE__, __LINE__);
            return -1;
        }
        ctx->bdSplitCap = sc; ctx->bdCandCap = cc;
        if (ctx->d_bterms2) { cudaFree(ctx->d_bterms2); ctx->d_bterms2 = 0; }
        if (!devAlloc(&ctx->d_bterms2, (size_t)2 * cc * nPad)) {
            setError(ctx, "energyComputeBatchDelta", "out of memory for pair terms",
                     __FILE__, __LINE__);
            return -1;
        }
    }

    const size_t stride = (size_t)nCand * nPad;
    const size_t blk = (size_t)dSplit * stride;
    const int blocks = (nList + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (nList > 0)
    {
        kOccupancy<<<dim3(blocks, nCand, dSplit), BLOCK>>>(
            nT, ctx->d_bsx, ctx->d_bsy, ctx->d_bsz, ctx->d_srad, ctx->d_sselfVol,
            ctx->d_ssilent, ctx->d_btileLo, ctx->d_btileHi, ctx->p, v43,
            ctx->p.occupancy, nCand, dSplit, ctx->d_bdpart, ctx->d_tileList, nList);
        CUDA_OK(ctx, cudaPeekAtLastError());
        kReduceOcc<<<(int)((stride + 255) / 256), 256>>>(nCand, nPad, dSplit,
                                                         ctx->d_bdpart, ctx->d_bocc);
        BP_MARK(occ);
        CUDA_OK(ctx, cudaPeekAtLastError());
        applyFreeze(ctx, nCand, ctx->d_bocc);
        CUDA_OK(ctx, cudaPeekAtLastError());

        ereal* t0 = ctx->d_bterms;
        kEnergy<<<dim3(blocks, nCand, dSplit), BLOCK>>>(
            nT, ctx->d_bsx, ctx->d_bsy, ctx->d_bsz, ctx->d_srad, ctx->d_ssqrtEps,
            ctx->d_schg, ctx->d_sresIndex, ctx->d_sorig, ctx->d_ssilent,
            ctx->d_bocc, ctx->d_btileLo, ctx->d_btileHi,
            ctx->d_exclCount, ctx->d_exclList, ctx->exclStride, ctx->d_exclSpan,
            ctx->p, v43, nCand, dSplit,
            ctx->d_bdpart, ctx->d_bdpart + blk,
            t0 + 2 * stride, t0 + 3 * stride, t0 + 4 * stride,
            ctx->d_tileList, nList, ctx->d_cmask,
            ctx->d_bdpart + 2 * blk, ctx->d_bdpart + 3 * blk);
        CUDA_OK(ctx, cudaPeekAtLastError());
        {
            const int rb = (int)((stride + 255) / 256);
            kReduceParts<<<rb, 256>>>(nCand, nPad, dSplit, ctx->d_bdpart,
                                      t0, t0 + stride);
            kReduceParts<<<rb, 256>>>(nCand, nPad, dSplit, ctx->d_bdpart + 2 * blk,
                                      ctx->d_bterms2, ctx->d_bterms2 + stride);
            CUDA_OK(ctx, cudaPeekAtLastError());
        }
        BP_MARK(energy);

        // Only the host reference path needs the per-atom gather materialised;
        // the device path fuses it into the reduction below.
        static const bool hostReduce = (getenv("PROTCAD_BATCH_HOSTREDUCE") != 0);
        const size_t need = (size_t)7 * count * nCand;
        if (hostReduce) {
        if (!ctx->d_bgather || ctx->bdGatherCap < need) {
            if (ctx->d_bgather) cudaFree(ctx->d_bgather);
            ctx->d_bgather = 0; ctx->bdGatherCap = 0;
            if (!devAlloc(&ctx->d_bgather, need)) {
                setError(ctx, "energyComputeBatchDelta", "out of memory for gather",
                         __FILE__, __LINE__);
                return -1;
            }
            ctx->bdGatherCap = need;
        }
        {
            dim3 g((count + tb - 1) / tb, nCand);
            kGatherTermsBatch<<<g, tb>>>(count, nCand, nPad, ctx->d_atoms, ctx->d_inv,
                                         t0, t0 + stride,
                                         ctx->d_bterms2, ctx->d_bterms2 + stride,
                                         t0 + 2 * stride, t0 + 3 * stride,
                                         t0 + 4 * stride, ctx->d_bgather);
            CUDA_OK(ctx, cudaPeekAtLastError());
        }
        }
        // Reduce on the device and bring home one number per candidate.  The
        // host path is kept behind PROTCAD_BATCH_HOSTREDUCE because it is the
        // reference the device schedule was validated against.
        if (hostReduce) {
            ctx->h_bgather.resize(need);
            CUDA_OK(ctx, cudaMemcpy(&ctx->h_bgather[0], ctx->d_bgather,
                                    need * sizeof(ereal), cudaMemcpyDeviceToHost));

            // Kahan in the caller's atom order, per candidate, so a candidate's
            // value does not depend on the spatial sort or on the batch size.
            for (int k = 0; k < nCand; ++k) {
                const ereal* g = &ctx->h_bgather[(size_t)k * 7 * count];
                double term[7];
                for (int t = 0; t < 7; ++t) {
                    double sum = 0.0, c = 0.0;
                    const ereal* src = g + (size_t)t * count;
                    for (int i = 0; i < count; ++i) {
                        double yv = double(src[i]) - c;
                        double tt = sum + yv;
                        c = (tt - sum) - yv;
                        sum = tt;
                    }
                    term[t] = sum;
                }
                const double pVdw = 2.0 * term[0] - term[2];
                const double pEle = 2.0 * term[1] - term[3];
                if (partOut) partOut[k] = pVdw + pEle + term[4] + term[5] + term[6];
            }
        }
        else if (partOut) {
            if (!ctx->d_bpart || ctx->bdPartCap < (size_t)nCand) {
                if (ctx->d_bpart) cudaFree(ctx->d_bpart);
                ctx->d_bpart = 0; ctx->bdPartCap = 0;
                if (!devAlloc(&ctx->d_bpart, (size_t)nCand)) {
                    setError(ctx, "energyComputeBatchDelta", "out of memory for parts",
                             __FILE__, __LINE__);
                    return -1;
                }
                ctx->bdPartCap = (size_t)nCand;
            }
            kFusedBatchPart<<<nCand, 256>>>(count, nPad, ctx->d_atoms, ctx->d_inv,
                                            t0, t0 + stride,
                                            ctx->d_bterms2, ctx->d_bterms2 + stride,
                                            t0 + 2 * stride, t0 + 3 * stride,
                                            t0 + 4 * stride, ctx->d_bpart);
            CUDA_OK(ctx, cudaPeekAtLastError());
            CUDA_OK(ctx, cudaMemcpy(partOut, ctx->d_bpart,
                                    (size_t)nCand * sizeof(double),
                                    cudaMemcpyDeviceToHost));
        }
        BP_MARK(reduce);
    }

    if (torsionOut) {
        const int rc = torsionBatchDelta(ctx, nCand, atoms, count, rebuiltSet,
                                         torsionOut);
        if (rc < 0) return -1;
        if (rc > 0 &&
            torsionTotals(ctx, ctx->d_bx, ctx->d_by, ctx->d_bz, nCand, torsionOut, 0) != 0)
            return -1;
    }
    BP_MARK(torsion);
    return 0;
}

int energyGetBatchCoords(energyContext* ctx, int k, double* x, double* y, double* z){
    if (!ctx || k < 0 || k >= ctx->batchCap || !x || !y || !z) return -1;
    const int N = ctx->N;
    std::vector<ereal> t(N);
    const size_t off = (size_t)k * N;
    CUDA_OK(ctx, cudaMemcpy(&t[0], ctx->d_bx + off, N * sizeof(ereal), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) x[i] = double(t[i]);
    CUDA_OK(ctx, cudaMemcpy(&t[0], ctx->d_by + off, N * sizeof(ereal), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) y[i] = double(t[i]);
    CUDA_OK(ctx, cudaMemcpy(&t[0], ctx->d_bz + off, N * sizeof(ereal), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) z[i] = double(t[i]);
    return 0;
}

int energySetRotationGroups(energyContext* ctx, int numGroups,
                            const int* axisA, const int* axisB,
                            const int* memberStart, const int* members)
{
    if (!ctx || numGroups < 0) return -1;
    void* old[] = { ctx->d_axisA, ctx->d_axisB, ctx->d_memberStart, ctx->d_members };
    for (size_t i = 0; i < 4; ++i) if (old[i]) cudaFree(old[i]);
    ctx->d_axisA = ctx->d_axisB = ctx->d_memberStart = ctx->d_members = 0;
    ctx->numGroups = 0;
    if (numGroups == 0) return 0;
    if (!axisA || !axisB || !memberStart || !members) return -1;

    const int nMem = memberStart[numGroups];
    for (int g = 0; g < numGroups; ++g)
        if (axisA[g] < 0 || axisA[g] >= ctx->N || axisB[g] < 0 || axisB[g] >= ctx->N)
            { ctx->lastError = "rotation group axis out of range"; return -1; }
    for (int m = 0; m < nMem; ++m)
        if (members[m] < 0 || members[m] >= ctx->N)
            { ctx->lastError = "rotation group member out of range"; return -1; }

    bool ok = devAlloc(&ctx->d_axisA, (size_t)numGroups)
           && devAlloc(&ctx->d_axisB, (size_t)numGroups)
           && devAlloc(&ctx->d_memberStart, (size_t)numGroups + 1)
           && devAlloc(&ctx->d_members, (size_t)(nMem > 0 ? nMem : 1));
    if (!ok) { ctx->lastError = "rotation group allocation failed"; return -1; }

    CUDA_OK(ctx, cudaMemcpy(ctx->d_axisA, axisA, numGroups * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_axisB, axisB, numGroups * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_memberStart, memberStart, (numGroups + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (nMem > 0)
        CUDA_OK(ctx, cudaMemcpy(ctx->d_members, members, nMem * sizeof(int), cudaMemcpyHostToDevice));
    ctx->numGroups = numGroups;
    return 0;
}

int energyComputeRotamerBatch(energyContext* ctx, int nCand,
                              int groupBegin, int nGroups,
                              const double* anglesDeg, double* totals)
{
    if (!ctx || nCand <= 0 || nGroups < 0 || !totals) return -1;
    if (!ctx->coordsValid) { ctx->lastError = "no resident coordinates"; return -1; }
    if (groupBegin < 0 || groupBegin + nGroups > ctx->numGroups)
        { ctx->lastError = "rotation group range out of bounds"; return -1; }
    if (nGroups > 0 && !anglesDeg) return -1;

    const int N = ctx->N;
    if (buildOrder(ctx) != 0) return -1;
    if (!ensureBatch(ctx, nCand)) { ctx->lastError = "batch allocation failed"; return -1; }

    const size_t nAng = (size_t)nCand * (nGroups > 0 ? nGroups : 1);
    if ((int)nAng > ctx->angleCap) {
        if (ctx->d_angles) cudaFree(ctx->d_angles);
        ctx->d_angles = 0; ctx->angleCap = 0;
        if (!devAlloc(&ctx->d_angles, nAng)) { ctx->lastError = "angle allocation failed"; return -1; }
        ctx->angleCap = (int)nAng;
    }
    if (nGroups > 0) {
        std::vector<ereal> h(nAng);
        for (size_t i = 0; i < (size_t)nCand * nGroups; ++i) h[i] = ereal(anglesDeg[i]);
        CUDA_OK(ctx, cudaMemcpy(ctx->d_angles, &h[0],
                                (size_t)nCand * nGroups * sizeof(ereal), cudaMemcpyHostToDevice));
    }

    kBuildRotamers<<<nCand, 256>>>(N, groupBegin, nGroups,
                                   ctx->d_x, ctx->d_y, ctx->d_z,
                                   ctx->d_axisA, ctx->d_axisB,
                                   ctx->d_memberStart, ctx->d_members,
                                   ctx->d_angles,
                                   ctx->d_bx, ctx->d_by, ctx->d_bz);
    CUDA_OK(ctx, cudaPeekAtLastError());

    return evalBatch(ctx, nCand, totals);
}

// ---------------------------------------------------------------------------
// Population Monte Carlo over replica states
// ---------------------------------------------------------------------------
//
// The batch kernel does not care whether its candidates are competing proposals
// for one walker or one proposal each from many independent walkers.
// Best-of-K spends K evaluations to advance a single chain by one step;
// replicas spend the same K evaluations to advance K chains by one step each --
// identical GPU cost for K times the search.  Only the bookkeeping differs, and
// it lives here.
//
// The spatial order is still seeded from the resident coordinates rather than
// from any individual replica.  That stays correct as replicas diverge, since
// per-candidate tile bounds are recomputed every launch and the cull is
// conservative, but it does get slower as tile membership loses spatial
// coherence.  With a frozen backbone the drift is bounded to sidechain motion;
// if the backbone is ever released, reseed the order periodically.

int energySetReplicas(energyContext* ctx, int nRepl)
{
    if (!ctx || nRepl <= 0) return -1;
    if (!ctx->coordsValid) { ctx->lastError = "no resident coordinates"; return -1; }

    const size_t N = ctx->N;
    if (nRepl > ctx->replCap)
    {
        void* old[] = { ctx->d_rx, ctx->d_ry, ctx->d_rz,
                        ctx->d_groupBegin, ctx->d_nGroups, ctx->d_accept };
        for (size_t i = 0; i < sizeof(old) / sizeof(old[0]); ++i)
            if (old[i]) cudaFree(old[i]);
        ctx->d_rx = ctx->d_ry = ctx->d_rz = 0;
        ctx->d_groupBegin = ctx->d_nGroups = ctx->d_accept = 0;
        ctx->replCap = 0;

        bool ok = devAlloc(&ctx->d_rx, (size_t)nRepl * N)
               && devAlloc(&ctx->d_ry, (size_t)nRepl * N)
               && devAlloc(&ctx->d_rz, (size_t)nRepl * N)
               && devAlloc(&ctx->d_groupBegin, (size_t)nRepl)
               && devAlloc(&ctx->d_nGroups, (size_t)nRepl)
               && devAlloc(&ctx->d_accept, (size_t)nRepl);
        if (!ok) { ctx->lastError = "replica allocation failed"; return -1; }
        ctx->replCap = nRepl;
    }

    kSeedReplicas<<<nRepl, 256>>>((int)N, nRepl, ctx->d_x, ctx->d_y, ctx->d_z,
                                  ctx->d_rx, ctx->d_ry, ctx->d_rz);
    CUDA_OK(ctx, cudaPeekAtLastError());
    CUDA_OK(ctx, cudaDeviceSynchronize());
    return 0;
}

int energyComputeReplicaBatch(energyContext* ctx, int nRepl,
                              const int* groupBegin, const int* nGroups,
                              const double* anglesDeg, int angleStride,
                              double* totals)
{
    if (!ctx || nRepl <= 0 || !totals || !groupBegin || !nGroups) return -1;
    if (nRepl > ctx->replCap) { ctx->lastError = "replicas not initialised"; return -1; }
    if (angleStride < 0) return -1;
    if (angleStride > 0 && !anglesDeg) return -1;

    for (int k = 0; k < nRepl; ++k)
    {
        if (nGroups[k] < 0 || nGroups[k] > angleStride)
            { ctx->lastError = "replica group count exceeds angle stride"; return -1; }
        if (groupBegin[k] < 0 || groupBegin[k] + nGroups[k] > ctx->numGroups)
            { ctx->lastError = "replica rotation group range out of bounds"; return -1; }
    }

    const int N = ctx->N;
    if (buildOrder(ctx) != 0) return -1;
    if (!ensureBatch(ctx, nRepl)) { ctx->lastError = "batch allocation failed"; return -1; }

    const int stride = angleStride > 0 ? angleStride : 1;
    const size_t nAng = (size_t)nRepl * stride;
    if ((int)nAng > ctx->angleCap) {
        if (ctx->d_angles) cudaFree(ctx->d_angles);
        ctx->d_angles = 0; ctx->angleCap = 0;
        if (!devAlloc(&ctx->d_angles, nAng)) { ctx->lastError = "angle allocation failed"; return -1; }
        ctx->angleCap = (int)nAng;
    }
    if (angleStride > 0) {
        std::vector<ereal> h(nAng);
        for (size_t i = 0; i < nAng; ++i) h[i] = ereal(anglesDeg[i]);
        CUDA_OK(ctx, cudaMemcpy(ctx->d_angles, &h[0], nAng * sizeof(ereal), cudaMemcpyHostToDevice));
    }

    CUDA_OK(ctx, cudaMemcpy(ctx->d_groupBegin, groupBegin, nRepl * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_nGroups,    nGroups,    nRepl * sizeof(int), cudaMemcpyHostToDevice));

    kBuildReplicas<<<nRepl, 256>>>(N, ctx->d_groupBegin, ctx->d_nGroups,
                                   ctx->d_rx, ctx->d_ry, ctx->d_rz,
                                   ctx->d_axisA, ctx->d_axisB,
                                   ctx->d_memberStart, ctx->d_members,
                                   ctx->d_angles, stride,
                                   ctx->d_bx, ctx->d_by, ctx->d_bz);
    CUDA_OK(ctx, cudaPeekAtLastError());

    return evalBatch(ctx, nRepl, totals);
}

int energyCommitReplicas(energyContext* ctx, int nRepl, const int* accept)
{
    if (!ctx || nRepl <= 0 || !accept) return -1;
    if (nRepl > ctx->replCap) { ctx->lastError = "replicas not initialised"; return -1; }
    CUDA_OK(ctx, cudaMemcpy(ctx->d_accept, accept, nRepl * sizeof(int), cudaMemcpyHostToDevice));
    kCommitReplicas<<<nRepl, 256>>>(ctx->N, ctx->d_accept,
                                    ctx->d_bx, ctx->d_by, ctx->d_bz,
                                    ctx->d_rx, ctx->d_ry, ctx->d_rz);
    CUDA_OK(ctx, cudaPeekAtLastError());
    CUDA_OK(ctx, cudaDeviceSynchronize());
    return 0;
}

int energyGetReplicaCoords(energyContext* ctx, int k, double* x, double* y, double* z)
{
    if (!ctx || k < 0 || k >= ctx->replCap || !x || !y || !z) return -1;
    const int N = ctx->N;
    std::vector<ereal> t(N);
    const size_t off = (size_t)k * N;
    CUDA_OK(ctx, cudaMemcpy(&t[0], ctx->d_rx + off, N * sizeof(ereal), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) x[i] = double(t[i]);
    CUDA_OK(ctx, cudaMemcpy(&t[0], ctx->d_ry + off, N * sizeof(ereal), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) y[i] = double(t[i]);
    CUDA_OK(ctx, cudaMemcpy(&t[0], ctx->d_rz + off, N * sizeof(ereal), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) z[i] = double(t[i]);
    return 0;
}

// ---------------------------------------------------------------------------
// Residency
// ---------------------------------------------------------------------------

static int clashComputeImpl(energyContext* ctx,
                            const double* x, const double* y, const double* z,
                            int* clashCountOut, int* perAtomOut);

int energySetCoords(energyContext* ctx,
                    const double* x, const double* y, const double* z)
{
    if (!ctx || !x || !y || !z) return -1;
    return uploadCoords(ctx, x, y, z);
}

int energyGetCoords(energyContext* ctx, double* x, double* y, double* z)
{
    if (!ctx || !ctx->coordsValid) return -1;
    const int N = ctx->N;
    CUDA_OK(ctx, cudaMemcpy(ctx->h_x, ctx->d_x, N * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(ctx->h_y, ctx->d_y, N * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(ctx->h_z, ctx->d_z, N * sizeof(ereal), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        if (x) x[i] = double(ctx->h_x[i]);
        if (y) y[i] = double(ctx->h_y[i]);
        if (z) z[i] = double(ctx->h_z[i]);
    }
    return 0;
}

int energySnapshot(energyContext* ctx)
{
    if (!ctx || !ctx->coordsValid) return -1;
    const int N = ctx->N;
    CUDA_OK(ctx, cudaMemcpy(ctx->d_xSave, ctx->d_x, N * sizeof(ereal), cudaMemcpyDeviceToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_ySave, ctx->d_y, N * sizeof(ereal), cudaMemcpyDeviceToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_zSave, ctx->d_z, N * sizeof(ereal), cudaMemcpyDeviceToDevice));
    ctx->haveSnapshot = true;
    return 0;
}

int energyRestore(energyContext* ctx)
{
    if (!ctx || !ctx->haveSnapshot) return -1;
    const int N = ctx->N;
    CUDA_OK(ctx, cudaMemcpy(ctx->d_x, ctx->d_xSave, N * sizeof(ereal), cudaMemcpyDeviceToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_y, ctx->d_ySave, N * sizeof(ereal), cudaMemcpyDeviceToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_z, ctx->d_zSave, N * sizeof(ereal), cudaMemcpyDeviceToDevice));
    return 0;
}

int energyComputeResident(energyContext* ctx, double* totalOut,
                          energyBreakdown* breakdown, double* perAtomOut)
{
    return energyComputeImpl(ctx, 0, 0, 0, totalOut, breakdown, perAtomOut);
}

int clashComputeResident(energyContext* ctx, int* clashCountOut, int* perAtomOut)
{
    return clashComputeImpl(ctx, 0, 0, 0, clashCountOut, perAtomOut);
}

int shellComputeResident(energyContext* ctx, double* dielectricOut, double* watersOut)
{
    return shellCompute(ctx, 0, 0, 0, dielectricOut, watersOut);
}

int energyComputeAtoms(energyContext* ctx,
                       const double* x, const double* y, const double* z,
                       double* totalOut, energyBreakdown* breakdown,
                       double* perAtomOut)
{
    return energyComputeImpl(ctx, x, y, z, totalOut, breakdown, perAtomOut);
}

// ---------------------------------------------------------------------------
// Host: shell state
// ---------------------------------------------------------------------------

int shellCompute(energyContext* ctx,
                 const double* x, const double* y, const double* z,
                 double* dielectricOut, double* watersOut)
{
    if (!ctx) return -1;
    const int nPad = ctx->nPad, nT = ctx->nTiles;

    if (x && uploadCoords(ctx, x, y, z) != 0) return -1;
    if (buildOrder(ctx) != 0) return -1;

    ereal v43 = (ctx->p.occupancy == OCCUPANCY_LEGACY_FULLVOLUME)
              ? ereal(4.188) : ereal(4.1887902048);

    CUDA_OK(ctx, cudaMemset(ctx->d_occ, 0, nPad * sizeof(ereal)));

    int blocks = (nT + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (!ensurePart(ctx, 1)) {
        setError(ctx, "occupancy", "out of memory for j-split partials",
                 __FILE__, __LINE__);
        return -1;
    }
    {
        // The occlusion partials borrow the energy partial buffer.  Ordering
        // makes that safe: the occupancy reduction completes on this stream
        // before the energy pass writes anything, and the energy pass reads
        // only the reduced occ array.
        const int jSplit = ctx->jSplit;
        dim3 gOcc(blocks, 1, jSplit);
        kOccupancy<<<gOcc, BLOCK>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz,
                                  ctx->d_srad, ctx->d_sselfVol, ctx->d_ssilent,
                                  ctx->d_tileLo, ctx->d_tileHi,
                                  ctx->p, v43, ctx->p.occupancy, 1, jSplit,
                                  ctx->d_part, 0, 0);
        CUDA_OK(ctx, cudaPeekAtLastError());
        kReduceOcc<<<(int)((nPad + 255) / 256), 256>>>(1, nPad, jSplit,
                                                       ctx->d_part, ctx->d_occ);
        CUDA_OK(ctx, cudaPeekAtLastError());
        applyFreeze(ctx, 1, ctx->d_occ);
        CUDA_OK(ctx, cudaPeekAtLastError());
    }

    // Reuse the energy term scratch; the energy pass has no live data here.
    kShellExport<<<(nPad + 255) / 256, 256>>>(nT, ctx->d_srad, ctx->d_ssilent,
                                              ctx->d_occ, ctx->p, v43,
                                              ctx->d_eVdw, ctx->d_eEle);
    CUDA_OK(ctx, cudaPeekAtLastError());

    ereal* h = ctx->h_terms;
    CUDA_OK(ctx, cudaMemcpy(h + 0 * nPad, ctx->d_eVdw, nPad * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(h + 1 * nPad, ctx->d_eEle, nPad * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(ctx->h_order, ctx->d_sorig, nPad * sizeof(int), cudaMemcpyDeviceToHost));

    if (dielectricOut) std::fill(dielectricOut, dielectricOut + ctx->N, 0.0);
    if (watersOut)     std::fill(watersOut,     watersOut     + ctx->N, 0.0);
    for (int s = 0; s < nPad; ++s) {
        int oi = ctx->h_order[s];
        if (oi < 0) continue;
        if (dielectricOut) dielectricOut[oi] = double(h[0 * nPad + s]);
        if (watersOut)     watersOut[oi]     = double(h[1 * nPad + s]);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Host: clashes
// ---------------------------------------------------------------------------

static int clashComputeImpl(energyContext* ctx,
                            const double* x, const double* y, const double* z,
                            int* clashCountOut, int* perAtomOut)
{
    if (!ctx) return -1;
    const int nPad = ctx->nPad, nT = ctx->nTiles;

    if (x && uploadCoords(ctx, x, y, z) != 0) return -1;
    if (buildOrder(ctx) != 0) return -1;

    // Bound the tile test by the largest possible contact distance.
    static ereal maxRadSum = ereal(0);
    if (maxRadSum == ereal(0)) maxRadSum = ereal(8.0);   // 2 * a generous vdW radius

    int blocks = (nT + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    kClash<<<blocks, BLOCK>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz, ctx->d_srad,
                              ctx->d_sresIndex, ctx->d_sorig, ctx->d_ssilent,
                              ctx->d_tileLo, ctx->d_tileHi,
                              ctx->d_exclCount, ctx->d_exclList, ctx->exclStride,
                              ctx->d_exclSpan, ctx->p.clash,
                              ereal(ctx->p.clashTolerance), maxRadSum, ctx->d_clash);
    CUDA_OK(ctx, cudaPeekAtLastError());

    std::vector<int> counts(nPad);
    CUDA_OK(ctx, cudaMemcpy(&counts[0], ctx->d_clash, nPad * sizeof(int),
                            cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(ctx->h_order, ctx->d_sorig, nPad * sizeof(int),
                            cudaMemcpyDeviceToHost));

    // Every pair is recorded by both of its atoms, so the raw sum is exactly
    // twice the number of distinct clashing pairs and is always even.
    long long total = 0;
    for (int s = 0; s < nPad; ++s) total += counts[s];
    if (clashCountOut) *clashCountOut = int(total / 2);

    if (perAtomOut) {
        std::fill(perAtomOut, perAtomOut + ctx->N, 0);
        for (int s = 0; s < nPad; ++s) {
            int oi = ctx->h_order[s];
            if (oi >= 0) perAtomOut[oi] = counts[s];
        }
    }
    return 0;
}

int clashCompute(energyContext* ctx,
                 const double* x, const double* y, const double* z,
                 int* clashCountOut)
{
    return clashComputeImpl(ctx, x, y, z, clashCountOut, 0);
}

int clashComputeAtoms(energyContext* ctx,
                      const double* x, const double* y, const double* z,
                      int* clashCountOut, int* perAtomOut)
{
    return clashComputeImpl(ctx, x, y, z, clashCountOut, perAtomOut);
}
