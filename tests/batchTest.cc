// Batched candidate evaluation must agree with evaluating each candidate alone.
// The batch exists to use idle GPU capacity, not to approximate anything.
//
// The agreement is exact in exact arithmetic but not bitwise in FP32.  Every
// candidate in a batch shares the spatial order built from the base
// conformation, while a lone evaluation sorts on its own coordinates, so pairs
// accumulate into each atom in a different order.  That is floating-point
// reassociation, and it shows up as ~1e-9 relative in FP32 and vanishes
// entirely in FP64 -- which is what this test checks, in both precisions, to
// distinguish rounding from a logic error.  On an energy of ~1e3 kcal/mol that
// is ~1e-6 kcal/mol absolute, six orders of magnitude below KT, so it cannot
// change an acceptance decision.
//
// What must hold exactly: the batch is deterministic, and all candidates in a
// batch are ranked against each other on one consistent order.
//
// Build: nvcc -O3 -arch=sm_61 -Isrc/ensemble -x cu tests/batchTest.cc \
//            src/ensemble/energy.cu -o batchTest

#include "energy.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ctime>

static int failures = 0;

// FP32 leaves ~1e-9 relative from reassociation; FP64 reproduces bitwise.
// The tolerance is set just above the observed noise so that a real traversal
// or indexing bug, which would move the energy by far more, still fails.
#ifdef PROTCAD_ENERGY_FP64
static const double kTol = 0.0;
#else
static const double kTol = 1e-6;
#endif

struct Sys
{
    int N, stride;
    std::vector<double> x, y, z, rad, eps, chg;
    std::vector<int> resIndex, exclCount, exclList;
    std::vector<unsigned char> silent;
};

static unsigned rngState = 12345u;
static double rnd()
{
    rngState = rngState * 1664525u + 1013904223u;
    return double(rngState >> 8) / double(1u << 24);
}

// Jittered lattice, residues of 8, short-range exclusions inside each residue.
// Same construction as energyTest: uniform random placement makes overlapping
// pairs whose r^-12 swamps everything and hides real differences.
static Sys makeSys(int N)
{
    Sys s;
    s.N = N; s.stride = 8;
    s.x.resize(N); s.y.resize(N); s.z.resize(N);
    s.rad.resize(N); s.eps.resize(N); s.chg.resize(N);
    s.resIndex.resize(N); s.silent.assign(N, 0);
    s.exclCount.assign(N, 0); s.exclList.assign((size_t)N * s.stride, 0);

    int side = int(ceil(pow(double(N), 1.0 / 3.0)));
    for (int i = 0; i < N; ++i) {
        int ix = i % side, iy = (i / side) % side, iz = i / (side * side);
        s.x[i] = ix * 3.4 + (rnd() - 0.5) * 0.8;
        s.y[i] = iy * 3.4 + (rnd() - 0.5) * 0.8;
        s.z[i] = iz * 3.4 + (rnd() - 0.5) * 0.8;
        s.rad[i] = 1.2 + rnd() * 0.9;
        s.eps[i] = 0.05 + rnd() * 0.15;
        s.chg[i] = (rnd() - 0.5) * 0.8;
        s.resIndex[i] = i / 8;
    }
    for (int i = 0; i < N; ++i) {
        int r = s.resIndex[i], c = 0;
        for (int j = i - 3; j <= i + 3 && c < s.stride; ++j) {
            if (j < 0 || j >= N || j == i) continue;
            if (s.resIndex[j] != r) continue;
            s.exclList[(size_t)i * s.stride + c] = j;
            ++c;
        }
        s.exclCount[i] = c;
    }
    return s;
}

static void runCase(const char* label, int N, int K, energyParams p)
{
    Sys s = makeSys(N);

    energyTopology t;
    t.numAtoms = N; t.radius = &s.rad[0]; t.epsilon = &s.eps[0];
    t.charge = &s.chg[0]; t.residueIndex = &s.resIndex[0];
    t.silent = &s.silent[0];
    t.exclusionCount = &s.exclCount[0]; t.exclusionList = &s.exclList[0];
    t.exclusionStride = s.stride;

    printf("[%s, N=%d, K=%d]\n", label, N, K);

    energyContext* ctx = energyCreate(t, p);
    if (!ctx) { printf("  energyCreate FAILED\n\n"); ++failures; return; }

    // Candidate 0 is the base conformation; the rest perturb a single
    // "residue", which is the move protMin actually makes.
    std::vector<double> bx((size_t)K * N), by((size_t)K * N), bz((size_t)K * N);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < N; ++i) {
            bx[(size_t)k * N + i] = s.x[i];
            by[(size_t)k * N + i] = s.y[i];
            bz[(size_t)k * N + i] = s.z[i];
        }
        if (k == 0) continue;
        int res = k % (N / 8);
        for (int i = res * 8; i < res * 8 + 8 && i < N; ++i) {
            bx[(size_t)k * N + i] += (rnd() - 0.5) * 1.5;
            by[(size_t)k * N + i] += (rnd() - 0.5) * 1.5;
            bz[(size_t)k * N + i] += (rnd() - 0.5) * 1.5;
        }
    }

    // The batch builds its spatial order from the resident coordinates, so set
    // the base conformation first -- exactly as a caller would.
    if (energySetCoords(ctx, &s.x[0], &s.y[0], &s.z[0]) != 0) {
        printf("  energySetCoords FAILED: %s\n\n", energyLastError(ctx));
        ++failures; energyDestroy(ctx); return;
    }

    std::vector<double> batched(K, 0.0);
    if (energyComputeBatch(ctx, K, &bx[0], &by[0], &bz[0], &batched[0]) != 0) {
        printf("  energyComputeBatch FAILED: %s\n\n", energyLastError(ctx));
        ++failures; energyDestroy(ctx); return;
    }

    // Reference: each candidate alone, through the ordinary path.
    int mismatches = 0;
    double worst = 0.0;
    for (int k = 0; k < K; ++k) {
        double one = 0.0;
        if (energyCompute(ctx, &bx[(size_t)k * N], &by[(size_t)k * N],
                          &bz[(size_t)k * N], &one, 0) != 0) {
            printf("  energyCompute FAILED: %s\n\n", energyLastError(ctx));
            ++failures; energyDestroy(ctx); return;
        }
        double d = fabs(one - batched[k]);
        double rel = d / (fabs(one) > 1e-30 ? fabs(one) : 1.0);
        if (rel > worst) worst = rel;
        if (rel > kTol) {
            ++mismatches;
            if (mismatches <= 3)
                printf("    k=%-4d single=%.17g batch=%.17g rel=%.2e\n",
                       k, one, batched[k], rel);
        }
    }

    if (mismatches == 0)
        printf("  agrees with single      %d/%d candidates, worst rel=%.2e (tol %.0e)  ok\n",
               K, K, worst, kTol);
    else {
        printf("  agrees with single      %d/%d EXCEED tol (worst rel=%.2e)  FAIL\n",
               K - mismatches, K, worst);
        ++failures;
    }

    // Candidates must not be identical to each other, or the test proves nothing.
    int distinct = 0;
    for (int k = 1; k < K; ++k) if (batched[k] != batched[0]) ++distinct;
    if (distinct == K - 1)
        printf("  candidates distinct     %d/%d  ok\n", distinct, K - 1);
    else {
        printf("  candidates distinct     %d/%d  FAIL\n", distinct, K - 1);
        ++failures;
    }

    // Repeating the batch must reproduce it exactly.
    std::vector<double> again(K, 0.0);
    energySetCoords(ctx, &s.x[0], &s.y[0], &s.z[0]);
    energyComputeBatch(ctx, K, &bx[0], &by[0], &bz[0], &again[0]);
    int drift = 0;
    for (int k = 0; k < K; ++k) if (again[k] != batched[k]) ++drift;
    printf("  batch determinism       %d/%d deviations  %s\n\n",
           drift, K, drift ? "FAIL" : "ok");
    if (drift) ++failures;

    energyDestroy(ctx);
}

int main()
{
    energyParams p = defaultEnergyParams();

    runCase("batch", 512, 8, p);
    runCase("batch", 1024, 32, p);
    runCase("batch", 2400, 64, p);

    // Growing the batch must reallocate correctly rather than reuse stale sizes.
    runCase("batch-regrow", 1024, 4, p);

    energyParams lp = legacyEnergyParams();
    runCase("batch-legacy", 1024, 16, lp);

    printf("RESULT: %s (%d failures)\n",
           failures ? "FAILURES" : "all checks passed", failures);
    return failures ? 1 : 0;
}
