// Standalone validation for the rewritten CUDA energy kernels.
//
// The tiled, cell-sorted traversal is the part of this rewrite most likely to
// go quietly wrong: a bad bounding-box test drops interactions without
// crashing, and the total energy just comes out a little different.  So the
// reference here is a brute-force O(N^2) CPU evaluation of the *identical*
// model, not a stored number.  Any pruning bug shows up immediately.
//
// Also covers:
//   * small N (the padded-block out-of-bounds case in the original kernel)
//   * every atom receiving a solvation term (the original skipped exactly one)
//   * bitwise determinism across repeated evaluations
//   * both clash criteria
//
// Build:  nvcc -O3 -arch=sm_61 tests/energyTest.cc src/ensemble/energy.cu \
//              -Isrc/ensemble -o bin/energyTest

#include "energy.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

static unsigned long long rngState = 88172645463325252ULL;
static double rnd()
{
    rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17;
    return double(rngState % 1000000007ULL) / 1000000007.0;
}

struct System
{
    int N;
    std::vector<double> x, y, z, rad, eps, chg;
    std::vector<int> resIndex, exclCount, exclList;
    std::vector<unsigned char> silent;
    int stride;
};

// Build a protein-like system: atoms on a jittered lattice at realistic
// separation, grouped into residues of 8, with short-range exclusions inside
// each residue.
//
// The lattice matters.  Uniform random placement produces overlapping atoms
// whose r^-12 terms reach 1e15 and swamp the total, so a traversal bug at
// ordinary bonding distances would hide inside the rounding of a few singular
// pairs.  Jittered lattice placement keeps every term in a physical range and
// makes the comparison against brute force actually sensitive.
static System makeSystem(int N)
{
    System s;
    s.N = N; s.stride = 8;
    s.x.resize(N); s.y.resize(N); s.z.resize(N);
    s.rad.resize(N); s.eps.resize(N); s.chg.resize(N);
    s.resIndex.resize(N); s.silent.assign(N, 0);
    s.exclCount.assign(N, 0); s.exclList.assign((size_t)N * s.stride, 0);

    const double spacing = 3.0;
    int side = int(ceil(pow(double(N), 1.0 / 3.0)));
    for (int i = 0; i < N; ++i) {
        int cx = i % side, cy = (i / side) % side, cz = i / (side * side);
        s.x[i] = cx * spacing + (rnd() - 0.5) * 0.8;
        s.y[i] = cy * spacing + (rnd() - 0.5) * 0.8;
        s.z[i] = cz * spacing + (rnd() - 0.5) * 0.8;
        s.rad[i] = 1.4 + rnd() * 0.6;
        s.eps[i] = 0.05 + rnd() * 0.15;
        s.chg[i] = (rnd() - 0.5) * 0.8;
        s.resIndex[i] = i / 8;
    }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            if (s.resIndex[i] != s.resIndex[j]) continue;
            if (abs(i - j) > 2) continue;
            s.exclList[(size_t)i * s.stride + s.exclCount[i]++] = j;
        }
    return s;
}

// --- CPU reference, mirroring the device model exactly -----------------------

struct Shell { double waters, fraction, eps, capacity; };

static double lensVol(double d, double R, double r, double v43)
{
    if (d >= R + r) return 0.0;
    if (d <= R - r) return v43 * r * r * r;
    if (d <= r - R) return v43 * R * R * R;
    if (d < 1e-6)   return v43 * pow(R < r ? R : r, 3);
    double t = R + r - d;
    double poly = d*d + 2*d*r - 3*r*r + 2*d*R + 6*r*R - 3*R*R;
    double v = 3.1415926535 * t * t * poly / (12.0 * d);
    return v > 0 ? v : 0;
}

static Shell shellOf(double occ, double rad, const energyParams& p, double v43)
{
    Shell s;
    double shellVol = v43 * pow(rad + p.effectiveWaterDiameter, 3);
    double selfVol  = v43 * rad * rad * rad;
    double envVol = (occ + selfVol) * p.occlusionScale;
    double freeVol = shellVol - envVol;
    if (freeVol < 0) freeVol = 0;
    if (freeVol > shellVol) freeVol = shellVol;
    s.capacity = shellVol / p.waterVolume;
    s.waters   = freeVol / p.waterVolume;
    if (p.quantizeWaters) s.waters = trunc(s.waters);
    s.fraction = freeVol / shellVol;
    double pol = s.waters * p.waterPolarizability;
    if (p.dielectric == DIELECTRIC_LEGACY_LINEAR)
        s.eps = p.epsProtein + (4.0 * 3.1415926535 / 3.0) * pol;
    else if (p.dielectric == DIELECTRIC_OCCUPANCY)
        s.eps = p.epsProtein + (p.epsWater - p.epsProtein) * s.fraction;
    else {
        double n = s.waters / shellVol;
        double yy = (4.0 * 3.1415926535 / 3.0) * n * p.waterPolarizability;
        if (yy > 0.95) yy = 0.95;
        s.eps = p.epsProtein * (1 + 2 * yy) / (1 - yy);
    }
    if (s.eps < 1) s.eps = 1;
    return s;
}

static double switchFnH(double d2, double d0sq, double d1sq)
{
    if (d2 <= d0sq) return 1.0;
    if (d2 >= d1sq) return 0.0;
    double u = (d1sq - d2) / (d1sq - d0sq);
    return u * u * (3 - 2 * u);
}

static bool excludedH(const System& s, int i, int j, int span)
{
    int dr = s.resIndex[i] - s.resIndex[j]; if (dr < 0) dr = -dr;
    if (dr > span) return false;
    for (int k = 0; k < s.exclCount[i]; ++k)
        if (s.exclList[(size_t)i * s.stride + k] == j) return true;
    return false;
}

static void referenceEnergy(const System& s, const energyParams& p,
                            energyBreakdown& out)
{
    const int N = s.N;
    double v43 = (p.occupancy == OCCUPANCY_LEGACY_FULLVOLUME) ? 4.188 : 4.1887902048;
    const double kc = 332.0636, kb = 0.0019872041, T = 300.0;

    std::vector<double> occ(N, 0.0);
    for (int i = 0; i < N; ++i) {
        if (s.silent[i]) continue;
        double shellI = s.rad[i] + p.effectiveWaterDiameter;
        for (int j = 0; j < N; ++j) {
            if (i == j || s.silent[j]) continue;
            double dx = s.x[i]-s.x[j], dy = s.y[i]-s.y[j], dz = s.z[i]-s.z[j];
            double r2 = dx*dx + dy*dy + dz*dz;
            if (p.occupancy == OCCUPANCY_LEGACY_FULLVOLUME) {
                if (r2 < shellI * shellI) occ[i] += v43 * pow(s.rad[j], 3);
            } else {
                double rj = s.rad[j];
                if (r2 < (shellI + rj) * (shellI + rj))
                    occ[i] += lensVol(sqrt(r2), shellI, rj, v43);
            }
        }
    }

    std::vector<Shell> sh(N);
    for (int i = 0; i < N; ++i) sh[i] = shellOf(occ[i], s.rad[i], p, v43);

    double vdwT = 0, eleT = 0, spT = 0, snT = 0, ssT = 0;
    double cutSq = p.cutoff * p.cutoff, swSq = p.switchStart * p.switchStart;

    for (int i = 0; i < N; ++i) {
        if (s.silent[i]) continue;
        for (int j = i + 1; j < N; ++j) {
            if (s.silent[j]) continue;
            double dx = s.x[i]-s.x[j], dy = s.y[i]-s.y[j], dz = s.z[i]-s.z[j];
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 >= cutSq) continue;
            if (excludedH(s, i, j, p.exclusionResidueSpan)) continue;
            double d = sqrt(r2);
            if (d < p.minSeparation) { d = p.minSeparation; r2 = d * d; }
            double sw = p.useSwitching ? switchFnH(r2, swSq, cutSq) : 1.0;
            double rsum = s.rad[i] + s.rad[j];
            double ratio2 = (rsum * rsum) / r2;
            double r6 = ratio2 * ratio2 * ratio2;
            vdwT += p.vdwScale * sw * sqrt(s.eps[i]) * sqrt(s.eps[j]) * (r6*r6 - 2*r6);
            double ep = (p.pairMixing == PAIRMIX_HARMONIC)
                      ? 2*sh[i].eps*sh[j].eps/(sh[i].eps+sh[j].eps)
                      : (sh[i].eps + sh[j].eps) * 0.5;
            eleT += p.elecScale * sw * kc * s.chg[i] * s.chg[j] / (d * ep);
        }
    }

    for (int i = 0; i < N; ++i) {
        if (s.silent[i]) continue;
        double w = sh[i].waters;
        if (w <= 0) continue;
        double wp = p.bornNormalize ? (w / sh[i].capacity) : w;
        spT += p.eSolvationFactor * (-(kc*0.5) * s.chg[i]*s.chg[i]
                / ((s.rad[i] + p.waterRadius) * sh[i].eps)) * wp;
        snT += p.hSolvationFactor * (-(sqrt(s.eps[i]) * sqrt(p.waterEpsilon))) * w;
        ssT += p.entropyFactor * (kb * T * 0.6931471805599453) * w;
    }

    out.vdw = vdwT; out.electrostatic = eleT;
    out.solvationPolar = spT; out.solvationNonpolar = snT; out.solvationEntropy = ssT;
    out.total = vdwT + eleT + spT + snT + ssT;
}

static int referenceClash(const System& s, const energyParams& p)
{
    int n = 0;
    for (int i = 0; i < s.N; ++i) {
        if (s.silent[i]) continue;
        for (int j = i + 1; j < s.N; ++j) {
            if (s.silent[j]) continue;
            double dx = s.x[i]-s.x[j], dy = s.y[i]-s.y[j], dz = s.z[i]-s.z[j];
            double rsum = s.rad[i] + s.rad[j];
            bool hit;
            if (p.clash == CLASH_INSCRIBED_CUBE) {
                double c = rsum * 0.7071067811865476;
                hit = fabs(dx) < c && fabs(dy) < c && fabs(dz) < c;
            } else hit = (dx*dx+dy*dy+dz*dz) < rsum*rsum;
            if (!hit) continue;
            if (excludedH(s, i, j, p.exclusionResidueSpan)) continue;
            ++n;
        }
    }
    return n;
}

// --- harness -----------------------------------------------------------------

static int failures = 0;

static void report(const char* what, double got, double want, double tol)
{
    double denom = fabs(want) > 1.0 ? fabs(want) : 1.0;
    double rel = fabs(got - want) / denom;
    bool ok = rel <= tol;
    if (!ok) ++failures;
    printf("  %-24s gpu=%15.6f  cpu=%15.6f  rel=%8.1e  %s\n",
           what, got, want, rel, ok ? "ok" : "FAIL");
}

static void runCase(int N, const energyParams& p, const char* label, double tol)
{
    System s = makeSystem(N);
    energyTopology t;
    t.numAtoms = N; t.radius = &s.rad[0]; t.epsilon = &s.eps[0];
    t.charge = &s.chg[0]; t.residueIndex = &s.resIndex[0];
    t.silent = &s.silent[0];
    t.exclusionCount = &s.exclCount[0]; t.exclusionList = &s.exclList[0];
    t.exclusionStride = s.stride;

    printf("[%s, N=%d]\n", label, N);

    energyContext* ctx = energyCreate(t, p);
    if (!ctx) { printf("  energyCreate FAILED\n\n"); ++failures; return; }

    energyBreakdown g;
    double total = 0;
    if (energyCompute(ctx, &s.x[0], &s.y[0], &s.z[0], &total, &g) != 0) {
        printf("  energyCompute FAILED: %s\n\n", energyLastError(ctx));
        ++failures; energyDestroy(ctx); return;
    }

    energyBreakdown r;
    referenceEnergy(s, p, r);

    report("vdw",                g.vdw,               r.vdw,               tol);
    report("electrostatic",      g.electrostatic,     r.electrostatic,     tol);
    report("solvation polar",    g.solvationPolar,    r.solvationPolar,    tol);
    report("solvation nonpolar", g.solvationNonpolar, r.solvationNonpolar, tol);
    report("solvation entropy",  g.solvationEntropy,  r.solvationEntropy,  tol);
    report("total",              g.total,             r.total,             tol);

    int gc = 0; clashCompute(ctx, &s.x[0], &s.y[0], &s.z[0], &gc);
    int rc = referenceClash(s, p);
    bool cok = (gc == rc); if (!cok) ++failures;
    printf("  %-24s gpu=%15d  cpu=%15d  %s\n", "clashes", gc, rc, cok ? "ok" : "FAIL");

    double first = total; int distinct = 0;
    for (int k = 0; k < 40; ++k) {
        double e = 0; energyCompute(ctx, &s.x[0], &s.y[0], &s.z[0], &e, 0);
        if (e != first) ++distinct;
    }
    if (distinct) ++failures;
    printf("  %-24s %d/40 deviations  %s\n\n", "determinism (bitwise)",
           distinct, distinct ? "FAIL" : "ok");

    energyDestroy(ctx);
}

int main()
{
#ifdef PROTCAD_ENERGY_FP64
    const double tol = 1e-9;
    printf("precision: FP64\n\n");
#else
    const double tol = 3e-4;
    printf("precision: FP32\n\n");
#endif

    energyParams leg = legacyEnergyParams();
    energyParams def = defaultEnergyParams();

    // Small N is where the original kernel indexed out of bounds.
    runCase(3,    leg, "legacy", tol);
    runCase(20,   leg, "legacy", tol);
    runCase(33,   leg, "legacy", tol);
    runCase(255,  leg, "legacy", tol);
    runCase(1000, leg, "legacy", tol);
    runCase(3000, leg, "legacy", tol);

    runCase(20,   def, "default", tol);
    runCase(1000, def, "default", tol);
    runCase(3000, def, "default", tol);

    energyParams cm = def; cm.dielectric = DIELECTRIC_CLAUSIUS_MOSSOTTI;
    runCase(1000, cm, "clausius-mossotti", tol);

    energyParams bn = def; bn.bornNormalize = 1;
    runCase(1000, bn, "born-normalized", tol);

    printf(failures ? "RESULT: %d FAILURES\n" : "RESULT: all checks passed (%d failures)\n",
           failures);
    return failures ? 1 : 0;
}
