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
    int   *d_resIndex;
    unsigned char *d_silent;
    int   *d_exclCount, *d_exclList, *d_exclSpan;
    int    exclStride;

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

    // Per-chunk pair-term slices for the j-split energy launch, and the split
    // factor itself.  jSplit is a function of the tile count alone, never of
    // the candidate count, so a batched evaluation and a resident one group
    // their sums identically and stay bit-comparable.
    ereal *d_part;
    int    partCap;      // candidates the partial buffer is sized for
    int    jSplit;
    std::vector<ereal> h_bx, h_by, h_bz, h_bterms;

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
#else
__device__ __forceinline__ ereal esqrt(ereal a)  { return sqrtf(a); }
__device__ __forceinline__ ereal emax(ereal a, ereal b) { return fmaxf(a, b); }
__device__ __forceinline__ ereal emin(ereal a, ereal b) { return fminf(a, b); }
__device__ __forceinline__ ereal eabs(ereal a)   { return fabsf(a); }
__device__ __forceinline__ ereal etrunc(ereal a) { return truncf(a); }
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

__device__ __forceinline__ bool isExcluded(
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
    if (dr > exclSpan[origI]) return false;

    int n = exclCount[origI];
    const int* list = exclList + (size_t)origI * stride;
    for (int k = 0; k < n; ++k)
        if (list[k] == origJ) return true;
    return false;
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
                           ereal* __restrict__ occ)
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
    const int iTile = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (iTile >= nTiles) return;

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
                        ereal* __restrict__ eSolvS)
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
        eSolvP += k * nPad; eSolvN += k * nPad; eSolvS += k * nPad;
    }
    __shared__ ereal shX[BLOCK], shY[BLOCK], shZ[BLOCK];
    __shared__ ereal shR[BLOCK], shE[BLOCK], shQ[BLOCK], shD[BLOCK];
    __shared__ int   shRes[BLOCK], shOrig[BLOCK];

    const int warp = threadIdx.x / TILE;
    const int lane = threadIdx.x % TILE;
    const int iTile = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (iTile >= nTiles) return;

    const int s = iTile * TILE + lane;
    const bool active = !ssilent[s];

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

                if (isExcluded(origI, shOrig[base + k], resI, shRes[base + k],
                               exclSpan, exclCount, exclList, exclStride))
                    continue;

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
                accVdw += ereal(0.5) * sw * vdw;
                accEle += ereal(0.5) * sw * ele;
            }
        }
        __syncwarp();
    }

    if (!active) {
        eVdw[s] = eEle[s] = ereal(0);
        if (chunk == 0) eSolvP[s] = eSolvN[s] = eSolvS[s] = ereal(0);
        return;
    }

    eVdw[s] = p.vdwScale  * accVdw;
    eEle[s] = p.elecScale * accEle;

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

                if (isExcluded(origI, shOrig[base + k], resI, shRes[base + k],
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
    ctx->d_x = ctx->d_y = ctx->d_z = 0;
    ctx->d_sx = ctx->d_sy = ctx->d_sz = 0;
    ctx->d_srad = ctx->d_ssqrtEps = ctx->d_schg = ctx->d_sselfVol = 0;
    ctx->d_tileLo = ctx->d_tileHi = ctx->d_occ = 0;
    ctx->d_eVdw = ctx->d_eEle = ctx->d_eSolvP = ctx->d_eSolvN = ctx->d_eSolvS = 0;
    ctx->d_resIndex = ctx->d_exclCount = ctx->d_exclList = ctx->d_exclSpan = 0;
    ctx->d_order = ctx->d_sresIndex = ctx->d_sorig = ctx->d_clash = 0;
    ctx->d_cellOf = ctx->d_cellCount = ctx->d_cellStart = ctx->d_cellCursor = 0;
    ctx->d_silent = ctx->d_ssilent = 0;
    ctx->h_x = ctx->h_y = ctx->h_z = ctx->h_terms = 0;
    ctx->h_order = 0;

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

    bool ok = true;
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
                    int dr = topo.residueIndex[i] - topo.residueIndex[list[k]];
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
        ctx->d_eVdw, ctx->d_eEle, ctx->d_eSolvP, ctx->d_eSolvN, ctx->d_eSolvS,
        ctx->d_clash, ctx->d_cellOf, ctx->d_cellCount, ctx->d_cellStart,
        ctx->d_cellCursor,
        ctx->d_xSave, ctx->d_ySave, ctx->d_zSave, ctx->d_bounds,
        ctx->d_bx, ctx->d_by, ctx->d_bz, ctx->d_bsx, ctx->d_bsy, ctx->d_bsz,
        ctx->d_bocc, ctx->d_btileLo, ctx->d_btileHi, ctx->d_bterms,
        ctx->d_part,
        ctx->d_axisA, ctx->d_axisB, ctx->d_memberStart, ctx->d_members,
        ctx->d_angles,
        ctx->d_rx, ctx->d_ry, ctx->d_rz,
        ctx->d_groupBegin, ctx->d_nGroups, ctx->d_accept
    };
    for (size_t i = 0; i < sizeof(ptrs) / sizeof(ptrs[0]); ++i)
        if (ptrs[i]) cudaFree(ptrs[i]);

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
// Measured on the P2200, warm clocks, kEnergy us/candidate:
//
//   1ubq (nTiles 44, nCand 4)    split 1: 255.8   4: 222.9   8: 224.5   32: 249.9
//   1crn (nTiles 10, nCand 4)    split 1: 304.0   4: 129.6   8: 106.3   32: 119.8
//
// The gain is large where it was most needed and the optimum sits near 8 in
// both cases, so the rule caps there; past that the duplicated i-side setup
// (including the occupancy shell lookup, repeated once per chunk) eats the
// benefit.  For reference 1ubq at nCand=64 costs 183.3 us/candidate, so the
// small-population penalty was 1.40x and the split removes about half of it.
// Note that these numbers are only meaningful warm: the first launch after an
// idle GPU runs at reduced clocks and reads ~45% slow, which briefly made this
// deficit look like 2.2x.
//
// The factor depends only on nTiles.  That matters: if it varied with nCand,
// a batched evaluation and a single resident one would group their float sums
// differently and stop agreeing bit for bit, which is what batchTest and
// replicaTest exist to check.
static int chooseSplit(energyContext* ctx)
{
    if (ctx->jSplit > 0) return ctx->jSplit;
    int s = 0;
    if (const char* e = getenv("PROTCAD_ENERGY_JSPLIT")) s = atoi(e);
    if (s <= 0)
    {
        const int TARGET_WARPS = 800;
        s = (TARGET_WARPS + ctx->nTiles - 1) / std::max(1, ctx->nTiles);
    }
    if (s > ctx->nTiles) s = ctx->nTiles;
    if (s > 8) s = 8;
    if (s < 1) s = 1;
    ctx->jSplit = s;
    return s;
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
                                  ctx->d_part);
        CUDA_OK(ctx, cudaPeekAtLastError());
        kReduceOcc<<<(int)((nPad + 255) / 256), 256>>>(1, nPad, jSplit,
                                                       ctx->d_part, ctx->d_occ);
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
                               ctx->d_eSolvN, ctx->d_eSolvS);
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

    double out[5];
    for (int t = 0; t < 5; ++t) {
        std::fill(ctx->perAtom.begin(), ctx->perAtom.end(), 0.0);
        const ereal* src = h + (size_t)t * nPad;
        for (int s = 0; s < nPad; ++s) {
            int oi = ctx->h_order[s];
            if (oi >= 0) ctx->perAtom[oi] = double(src[s]);
        }
        if (perAtomOut)
            for (int i = 0; i < ctx->N; ++i) perAtomOut[i] += ctx->perAtom[i];

        double sum = 0.0, c = 0.0;
        for (int i = 0; i < ctx->N; ++i) {
            double yv = ctx->perAtom[i] - c;
            double tt = sum + yv;
            c = (tt - sum) - yv;
            sum = tt;
        }
        out[t] = sum;
    }

    double total = out[0] + out[1] + out[2] + out[3] + out[4];
    if (totalOut) *totalOut = total;
    if (breakdown) {
        breakdown->vdw               = out[0];
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
    double gather, tile, occ, energy, copy, reduce; long calls; long cands;
    batchProfile() : gather(0), tile(0), occ(0), energy(0), copy(0), reduce(0),
                     calls(0), cands(0) {}
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
               + g_bprof.copy + g_bprof.reduce;
    if (tot <= 0) return;
    const long nc = std::max(1L, g_bprof.cands);
    fprintf(stderr, "\n[energy] evalBatch profile: %ld calls, %ld candidates\n",
            g_bprof.calls, g_bprof.cands);
    const char* nm[6] = {"kGatherCoords", "kTileBounds", "kOccupancy",
                         "kEnergy", "D2H copy", "host Kahan"};
    double vv[6] = {g_bprof.gather, g_bprof.tile, g_bprof.occ,
                    g_bprof.energy, g_bprof.copy, g_bprof.reduce};
    for (int i = 0; i < 6; ++i)
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
                                 ctx->d_part);
        CUDA_OK(ctx, cudaPeekAtLastError());
        const size_t tot = (size_t)nCand * nPad;
        kReduceOcc<<<(int)((tot + 255) / 256), 256>>>(nCand, nPad, jSplit,
                                                      ctx->d_part, ctx->d_bocc);
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
                              t0 + 2 * stride, t0 + 3 * stride, t0 + 4 * stride);
    CUDA_OK(ctx, cudaPeekAtLastError());

    {
        const size_t tot = stride;
        int rb = (int)((tot + 255) / 256);
        kReduceParts<<<rb, 256>>>(nCand, nPad, jSplit, ctx->d_part,
                                  t0, t0 + stride);
        CUDA_OK(ctx, cudaPeekAtLastError());
    }

    double tB = prof ? (cudaDeviceSynchronize(), profWall()) : 0.0;

    CUDA_OK(ctx, cudaMemcpy(&ctx->h_bterms[0], ctx->d_bterms,
                            stride * 5 * sizeof(ereal), cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(ctx->h_order, ctx->d_sorig, nPad * sizeof(int),
                            cudaMemcpyDeviceToHost));

    double tC = prof ? profWall() : 0.0;

    for (int k = 0; k < nCand; ++k) {
        double total = 0.0;
        for (int t = 0; t < 5; ++t) {
            std::fill(ctx->perAtom.begin(), ctx->perAtom.end(), 0.0);
            const ereal* src = &ctx->h_bterms[(size_t)t * stride + (size_t)k * nPad];
            for (int sIdx = 0; sIdx < nPad; ++sIdx) {
                int oi = ctx->h_order[sIdx];
                if (oi >= 0) ctx->perAtom[oi] = double(src[sIdx]);
            }
            double sum = 0.0, c = 0.0;
            for (int i = 0; i < N; ++i) {
                double yv = ctx->perAtom[i] - c;
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
        g_bprof.reduce  += tD - tC;
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

int energyGetBatchCoords(energyContext* ctx, int k, double* x, double* y, double* z)
{
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
                                  ctx->d_part);
        CUDA_OK(ctx, cudaPeekAtLastError());
        kReduceOcc<<<(int)((nPad + 255) / 256), 256>>>(1, nPad, jSplit,
                                                       ctx->d_part, ctx->d_occ);
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
