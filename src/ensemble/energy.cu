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
    int   *d_exclCount, *d_exclList;
    int    exclStride;

    // Coordinates, original order.
    ereal *d_x, *d_y, *d_z;

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

    p.bornNormalize    = 0;   // preserves the existing energy scale; see notes
    p.eSolvationFactor = ereal(1.0);
    p.hSolvationFactor = ereal(1.0);
    p.entropyFactor    = ereal(1.0);

    p.vdwScale  = ereal(1.0);
    p.elecScale = ereal(1.0);

    p.clash = CLASH_INSCRIBED_CUBE;

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
    p.bornNormalize  = 0;
    p.clash          = CLASH_SPHERE;
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
    int origI, int origJ, int resI, int resJ, int span,
    const int* exclCount, const int* exclList, int stride)
{
    // Exclusions in a protein never reach beyond a couple of residues, so this
    // test rejects essentially every pair before touching global memory.
    int dr = resI - resJ;
    if (dr < 0) dr = -dr;
    if (dr > span) return false;

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

// Sort each cell's members by original atom index.  Cells hold ~32 atoms, so an
// insertion sort is a few hundred operations.  This is what makes the whole
// pipeline deterministic despite the atomicAdd placement above.
__global__ void kSortCells(const int* cellStart, const int* cellCount,
                           int nCells, int* order)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCells) return;
    int s = cellStart[c], n = cellCount[c];
    for (int a = 1; a < n; ++a) {
        int key = order[s + a];
        int b = a - 1;
        while (b >= 0 && order[s + b] > key) { order[s + b + 1] = order[s + b]; --b; }
        order[s + b + 1] = key;
    }
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
                           ereal* __restrict__ occ)
{
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

    for (int jTile = 0; jTile < nTiles; ++jTile)
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
                        energyParams p, ereal v43,
                        ereal* __restrict__ eVdw, ereal* __restrict__ eEle,
                        ereal* __restrict__ eSolvP, ereal* __restrict__ eSolvN,
                        ereal* __restrict__ eSolvS)
{
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

    for (int jTile = 0; jTile < nTiles; ++jTile)
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
                               p.exclusionResidueSpan, exclCount, exclList, exclStride))
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
        eVdw[s] = eEle[s] = eSolvP[s] = eSolvN[s] = eSolvS[s] = ereal(0);
        return;
    }

    eVdw[s] = p.vdwScale  * accVdw;
    eEle[s] = p.elecScale * accEle;

    // --- solvation, one evaluation per atom ---
    // The original folded this into the i==j diagonal of an N^2 launch, which
    // silently skipped exactly one atom.  Here every atom is covered by
    // construction.
    ereal w = si.waters;
    if (w > ereal(0))
    {
        ereal bornDenom = (ri + p.waterRadius) * si.eps;
        ereal wPolar = p.bornNormalize ? (w / si.capacity) : w;
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
                       int span, int clashMode, ereal maxRadSum,
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
                if (shOrig[base + k] <= origI) continue;

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
                    hit = (dx * dx + dy * dy + dz * dz) < rsum * rsum;
                }
                if (!hit) continue;

                if (isExcluded(origI, shOrig[base + k], resI, shRes[base + k],
                               span, exclCount, exclList, exclStride))
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
    ctx->d_resIndex = ctx->d_exclCount = ctx->d_exclList = 0;
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

    bool ok = true;
    ok &= devAlloc(&ctx->d_rad, N)      && devAlloc(&ctx->d_sqrtEps, N);
    ok &= devAlloc(&ctx->d_chg, N)      && devAlloc(&ctx->d_selfVol, N);
    ok &= devAlloc(&ctx->d_resIndex, N) && devAlloc(&ctx->d_silent, N);
    ok &= devAlloc(&ctx->d_exclCount, N);
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
        ctx->d_silent, ctx->d_exclCount, ctx->d_exclList,
        ctx->d_x, ctx->d_y, ctx->d_z, ctx->d_order,
        ctx->d_sx, ctx->d_sy, ctx->d_sz, ctx->d_srad, ctx->d_ssqrtEps,
        ctx->d_schg, ctx->d_sselfVol, ctx->d_sresIndex, ctx->d_sorig,
        ctx->d_ssilent, ctx->d_tileLo, ctx->d_tileHi, ctx->d_occ,
        ctx->d_eVdw, ctx->d_eEle, ctx->d_eSolvP, ctx->d_eSolvN, ctx->d_eSolvS,
        ctx->d_clash, ctx->d_cellOf, ctx->d_cellCount, ctx->d_cellStart,
        ctx->d_cellCursor
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
static int buildOrder(energyContext* ctx,
                      const double* x, const double* y, const double* z)
{
    const int N = ctx->N, nPad = ctx->nPad, nT = ctx->nTiles;

    ereal loX = ereal(1e30), loY = ereal(1e30), loZ = ereal(1e30);
    ereal hiX = ereal(-1e30), hiY = ereal(-1e30), hiZ = ereal(-1e30);
    for (int i = 0; i < N; ++i) {
        ereal xi = ereal(x[i]), yi = ereal(y[i]), zi = ereal(z[i]);
        ctx->h_x[i] = xi; ctx->h_y[i] = yi; ctx->h_z[i] = zi;
        loX = std::min(loX, xi); hiX = std::max(hiX, xi);
        loY = std::min(loY, yi); hiY = std::max(hiY, yi);
        loZ = std::min(loZ, zi); hiZ = std::max(hiZ, zi);
    }

    CUDA_OK(ctx, cudaMemcpy(ctx->d_x, ctx->h_x, N * sizeof(ereal), cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_y, ctx->h_y, N * sizeof(ereal), cudaMemcpyHostToDevice));
    CUDA_OK(ctx, cudaMemcpy(ctx->d_z, ctx->h_z, N * sizeof(ereal), cudaMemcpyHostToDevice));

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

    kSortCells<<<(nCells + 127) / 128, 128>>>(ctx->d_cellStart, ctx->d_cellCount,
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

int energyCompute(energyContext* ctx,
                  const double* x, const double* y, const double* z,
                  double* totalOut, energyBreakdown* breakdown)
{
    if (!ctx) return -1;
    const int nPad = ctx->nPad, nT = ctx->nTiles;

    if (buildOrder(ctx, x, y, z) != 0) return -1;

    ereal v43 = (ctx->p.occupancy == OCCUPANCY_LEGACY_FULLVOLUME)
              ? ereal(4.188) : ereal(4.1887902048);

    CUDA_OK(ctx, cudaMemset(ctx->d_occ, 0, nPad * sizeof(ereal)));

    int blocks = (nT + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    kOccupancy<<<blocks, BLOCK>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz,
                                  ctx->d_srad, ctx->d_sselfVol, ctx->d_ssilent,
                                  ctx->d_tileLo, ctx->d_tileHi,
                                  ctx->p, v43, ctx->p.occupancy, ctx->d_occ);
    CUDA_OK(ctx, cudaPeekAtLastError());

    kEnergy<<<blocks, BLOCK>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz,
                               ctx->d_srad, ctx->d_ssqrtEps, ctx->d_schg,
                               ctx->d_sresIndex, ctx->d_sorig, ctx->d_ssilent,
                               ctx->d_occ, ctx->d_tileLo, ctx->d_tileHi,
                               ctx->d_exclCount, ctx->d_exclList, ctx->exclStride,
                               ctx->p, v43,
                               ctx->d_eVdw, ctx->d_eEle, ctx->d_eSolvP,
                               ctx->d_eSolvN, ctx->d_eSolvS);
    CUDA_OK(ctx, cudaPeekAtLastError());

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
    double out[5];
    for (int t = 0; t < 5; ++t) {
        std::fill(ctx->perAtom.begin(), ctx->perAtom.end(), 0.0);
        const ereal* src = h + (size_t)t * nPad;
        for (int s = 0; s < nPad; ++s) {
            int oi = ctx->h_order[s];
            if (oi >= 0) ctx->perAtom[oi] = double(src[s]);
        }
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

// ---------------------------------------------------------------------------
// Host: clashes
// ---------------------------------------------------------------------------

int clashCompute(energyContext* ctx,
                 const double* x, const double* y, const double* z,
                 int* clashCountOut)
{
    if (!ctx) return -1;
    const int nPad = ctx->nPad, nT = ctx->nTiles;

    if (buildOrder(ctx, x, y, z) != 0) return -1;

    // Bound the tile test by the largest possible contact distance.
    static ereal maxRadSum = ereal(0);
    if (maxRadSum == ereal(0)) maxRadSum = ereal(8.0);   // 2 * a generous vdW radius

    int blocks = (nT + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    kClash<<<blocks, BLOCK>>>(nT, ctx->d_sx, ctx->d_sy, ctx->d_sz, ctx->d_srad,
                              ctx->d_sresIndex, ctx->d_sorig, ctx->d_ssilent,
                              ctx->d_tileLo, ctx->d_tileHi,
                              ctx->d_exclCount, ctx->d_exclList, ctx->exclStride,
                              ctx->p.exclusionResidueSpan, ctx->p.clash,
                              maxRadSum, ctx->d_clash);
    CUDA_OK(ctx, cudaPeekAtLastError());

    std::vector<int> counts(nPad);
    CUDA_OK(ctx, cudaMemcpy(&counts[0], ctx->d_clash, nPad * sizeof(int),
                            cudaMemcpyDeviceToHost));
    CUDA_OK(ctx, cudaMemcpy(ctx->h_order, ctx->d_sorig, nPad * sizeof(int),
                            cudaMemcpyDeviceToHost));

    long long total = 0;
    for (int s = 0; s < nPad; ++s) total += counts[s];
    if (clashCountOut) *clashCountOut = int(total);
    return 0;
}
