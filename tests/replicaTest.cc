// replicaTest -- correctness of the population Monte Carlo replica API.
//
// The replica path differs from the rotamer-batch path in exactly one way: each
// candidate is built from its *own* resident state rather than from the single
// shared resident conformation.  Everything downstream -- spatial ordering,
// tiling, occupancy, the energy kernels, the host reduction -- is shared.  So
// the test strategy is to pin the one thing that differs and use the existing,
// already-validated batch path as the oracle.
//
// Immediately after seeding, every replica holds the resident conformation.  In
// that state the two paths must agree term for term, because they are computing
// the same thing by different routes.  Once replicas have diverged through a
// commit, that oracle no longer applies and the test switches to a per-replica
// comparison against a direct energyCompute on coordinates read back from the
// device.
//
// What actually needs guarding here, in order of how easy it is to get wrong:
//
//   1. Angle indexing.  kBuildReplicas strides by angleStride, not by nGroups,
//      so a ragged set of residue types shares one rectangular array.  An
//      off-by-one in that stride silently rotates the wrong chi.
//   2. Per-replica base offsets.  Every buffer is indexed by k*N; one missing
//      offset makes replica k read replica 0's coordinates.
//   3. Commit masking.  A rejected replica must be left *bit* unchanged, not
//      merely close.  Metropolis correctness depends on the rejected state
//      being the previous state exactly, since it will be reused as the base
//      for the next proposal and errors would accumulate over a long run.
//
// Build:
//   nvcc -O3 -arch=sm_61 [-DPROTCAD_ENERGY_FP64] -Isrc/ensemble \
//        -x cu tests/replicaTest.cc src/ensemble/energy.cu -o /tmp/replicaTest

#include "energy.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(const char* what, bool ok)
{
    printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

static void checkClose(const char* what, double a, double b, double tol)
{
    double d = fabs(a - b);
    double s = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    bool ok = (d <= tol * (s > 1.0 ? s : 1.0));
    printf("  %-52s %14.6g vs %14.6g  %s\n", what, a, b, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

// A jittered lattice with synthetic rotation groups.  Each group of four atoms
// acts like a residue: atoms 4g and 4g+1 define a rotation axis and atoms 4g+2,
// 4g+3 are the movable distal set.  That reproduces the shape of the real
// chi-group data without needing a protein loaded.
struct Sys
{
    int N, nGroups;
    std::vector<double> x, y, z, rad, eps, chg;
    std::vector<int> res, ec, el;
    std::vector<unsigned char> sil;
    std::vector<int> axisA, axisB, memberStart, members;
};

static Sys makeSys(int side)
{
    Sys s;
    s.N = side * side * side;
    s.x.resize(s.N); s.y.resize(s.N); s.z.resize(s.N);
    s.rad.resize(s.N); s.eps.resize(s.N); s.chg.resize(s.N);
    s.res.resize(s.N); s.ec.assign(s.N, 0); s.el.assign(s.N, 0);
    s.sil.assign(s.N, 0);

    unsigned seed = 987654321u;
    for (int i = 0; i < s.N; ++i)
    {
        int ix = i % side, iy = (i / side) % side, iz = i / (side * side);
        double j[3];
        for (int c = 0; c < 3; ++c) {
            seed = seed * 1103515245u + 12345u;
            j[c] = ((seed >> 16) & 0x7fff) / 32767.0 - 0.5;
        }
        s.x[i] = ix * 3.6 + 0.4 * j[0];
        s.y[i] = iy * 3.6 + 0.4 * j[1];
        s.z[i] = iz * 3.6 + 0.4 * j[2];
        s.rad[i] = 1.8; s.eps[i] = 0.12;
        s.chg[i] = ((i % 3) - 1) * 0.4;
        s.res[i] = i / 4;
    }

    s.nGroups = s.N / 4;
    s.memberStart.resize(s.nGroups + 1);
    for (int g = 0; g < s.nGroups; ++g)
    {
        s.axisA.push_back(4 * g);
        s.axisB.push_back(4 * g + 1);
        s.memberStart[g] = (int)s.members.size();
        s.members.push_back(4 * g + 2);
        s.members.push_back(4 * g + 3);
    }
    s.memberStart[s.nGroups] = (int)s.members.size();
    return s;
}

static energyContext* build(Sys& s)
{
    energyTopology t;
    t.numAtoms = s.N; t.radius = &s.rad[0]; t.epsilon = &s.eps[0];
    t.charge = &s.chg[0]; t.residueIndex = &s.res[0]; t.silent = &s.sil[0];
    t.exclusionCount = &s.ec[0]; t.exclusionList = &s.el[0]; t.exclusionStride = 1;

    energyParams p = defaultEnergyParams();
    energyContext* ctx = energyCreate(t, p);
    if (!ctx) return 0;
    if (energySetCoords(ctx, &s.x[0], &s.y[0], &s.z[0]) != 0) { energyDestroy(ctx); return 0; }
    if (energySetRotationGroups(ctx, s.nGroups, &s.axisA[0], &s.axisB[0],
                                &s.memberStart[0], &s.members[0]) != 0)
        { energyDestroy(ctx); return 0; }
    return ctx;
}

int main()
{
#ifdef PROTCAD_ENERGY_FP64
    const double tol = 1e-12; const char* prec = "FP64";
#else
    const double tol = 1e-5;  const char* prec = "FP32";
#endif
    printf("replicaTest (%s)\n", prec);

    Sys s = makeSys(4);                       // 64 atoms, 16 rotation groups
    energyContext* ctx = build(s);
    if (!ctx) { printf("  setup FAILED\n"); return 1; }
    const int N = s.N;
    const int P = 8;                          // replicas
    const int NG = 2;                         // groups moved per replica

    // ---- 1. seeding -------------------------------------------------------
    printf("\n[1] seeding\n");
    if (energySetReplicas(ctx, P) != 0) {
        printf("  energySetReplicas FAILED: %s\n", energyLastError(ctx));
        energyDestroy(ctx); return 1;
    }
    std::vector<double> rx(N), ry(N), rz(N);
    // Compare against the resident coordinates read back through the same
    // path, not against the original host doubles.  In the FP32 build the
    // device stores float, so a double -> float -> double round trip is not
    // bit-equal to the literal it came from.  That is rounding, not a copy
    // error, and testing against the raw host array would flag it falsely.
    std::vector<double> ox(N), oy(N), oz(N);
    energyGetCoords(ctx, &ox[0], &oy[0], &oz[0]);

    bool seedOk = true;
    for (int k = 0; k < P && seedOk; ++k) {
        if (energyGetReplicaCoords(ctx, k, &rx[0], &ry[0], &rz[0]) != 0) { seedOk = false; break; }
        for (int i = 0; i < N; ++i)
            // Seeding is a straight device copy, so this must be bit exact.
            if (rx[i] != ox[i] || ry[i] != oy[i] || rz[i] != oz[i]) { seedOk = false; break; }
    }
    check("every replica bit-equals resident coords", seedOk);

    // ---- 2. agreement with the rotamer batch oracle ------------------------
    // All replicas still hold the resident conformation, so replica k applying
    // angles[k] must equal rotamer candidate k applying the same angles.
    printf("\n[2] undiverged replicas vs energyComputeRotamerBatch\n");
    const int gb = 3;
    std::vector<double> ang((size_t)P * NG);
    for (int k = 0; k < P; ++k) {
        ang[(size_t)k * NG + 0] = -140.0 + 31.0 * k;
        ang[(size_t)k * NG + 1] =   75.0 - 19.0 * k;
    }
    std::vector<double> tRot(P, 0.0), tRep(P, 0.0);
    std::vector<int> gbv(P, gb), ngv(P, NG);

    if (energyComputeRotamerBatch(ctx, P, gb, NG, &ang[0], &tRot[0]) != 0)
        { printf("  rotamerBatch FAILED: %s\n", energyLastError(ctx)); ++failures; }
    if (energyComputeReplicaBatch(ctx, P, &gbv[0], &ngv[0], &ang[0], NG, &tRep[0]) != 0)
        { printf("  replicaBatch FAILED: %s\n", energyLastError(ctx)); ++failures; }

    int bitEq = 0;
    for (int k = 0; k < P; ++k) {
        if (tRot[k] == tRep[k]) ++bitEq;
        char buf[80]; snprintf(buf, sizeof buf, "replica %d total", k);
        checkClose(buf, tRep[k], tRot[k], tol);
    }
    printf("  (%d of %d bit-identical to the rotamer path)\n", bitEq, P);

    // Distinct angles must give distinct energies, or the angle stride is being
    // ignored and every replica is silently evaluating the same conformation.
    int distinct = 0;
    for (int k = 1; k < P; ++k) if (tRep[k] != tRep[0]) ++distinct;
    check("replica energies are distinct (angle stride honoured)", distinct == P - 1);

    // ---- 3. ragged group counts -------------------------------------------
    // Mixed nGroups per replica with a rectangular angle array is the case a
    // real run hits constantly, since residues differ in chi count.
    printf("\n[3] ragged nGroups against per-replica oracle\n");
    const int STRIDE = 3;
    std::vector<double> ang3((size_t)P * STRIDE, 0.0);
    std::vector<int> gb3(P), ng3(P);
    for (int k = 0; k < P; ++k) {
        ng3[k] = 1 + (k % 3);                 // 1, 2 or 3 groups
        gb3[k] = 2 + k;                       // different start per replica
        for (int g = 0; g < ng3[k]; ++g)
            ang3[(size_t)k * STRIDE + g] = 20.0 * (g + 1) + 7.0 * k;
    }
    std::vector<double> tRag(P, 0.0);
    if (energyComputeReplicaBatch(ctx, P, &gb3[0], &ng3[0], &ang3[0], STRIDE, &tRag[0]) != 0)
        { printf("  ragged replicaBatch FAILED: %s\n", energyLastError(ctx)); ++failures; }

    for (int k = 0; k < P; ++k) {
        double one = 0.0;
        std::vector<double> a1(ng3[k]);
        for (int g = 0; g < ng3[k]; ++g) a1[g] = ang3[(size_t)k * STRIDE + g];
        if (energyComputeRotamerBatch(ctx, 1, gb3[k], ng3[k], &a1[0], &one) != 0)
            { printf("  oracle FAILED: %s\n", energyLastError(ctx)); ++failures; continue; }
        char buf[80]; snprintf(buf, sizeof buf, "ragged replica %d (ng=%d, gb=%d)", k, ng3[k], gb3[k]);
        checkClose(buf, tRag[k], one, tol);
    }

    // ---- 4. commit masking -------------------------------------------------
    // Re-run the uniform proposal so the batch buffer holds known coordinates,
    // then commit only the even replicas.
    printf("\n[4] accept-masked commit\n");
    if (energyComputeReplicaBatch(ctx, P, &gbv[0], &ngv[0], &ang[0], NG, &tRep[0]) != 0)
        { printf("  replicaBatch FAILED\n"); ++failures; }

    std::vector< std::vector<double> > prop(P);
    for (int k = 0; k < P; ++k) {
        prop[k].resize(3 * N);
        energyGetBatchCoords(ctx, k, &prop[k][0], &prop[k][N], &prop[k][2 * N]);
    }

    std::vector<int> accept(P);
    for (int k = 0; k < P; ++k) accept[k] = (k % 2 == 0) ? 1 : 0;

    // Capture the pre-commit state so the rejected-replica check compares
    // against what the device actually held, not against host doubles.
    std::vector< std::vector<double> > before(P);
    for (int k = 0; k < P; ++k) {
        before[k].resize(3 * N);
        energyGetReplicaCoords(ctx, k, &before[k][0], &before[k][N], &before[k][2 * N]);
    }

    if (energyCommitReplicas(ctx, P, &accept[0]) != 0)
        { printf("  commit FAILED: %s\n", energyLastError(ctx)); ++failures; }

    bool acc = true, rej = true;
    for (int k = 0; k < P; ++k) {
        energyGetReplicaCoords(ctx, k, &rx[0], &ry[0], &rz[0]);
        for (int i = 0; i < N; ++i) {
            if (accept[k]) {
                if (rx[i] != prop[k][i] || ry[i] != prop[k][N + i] || rz[i] != prop[k][2 * N + i])
                    { acc = false; break; }
            } else {
                if (rx[i] != before[k][i] || ry[i] != before[k][N + i] || rz[i] != before[k][2 * N + i])
                    { rej = false; break; }
            }
        }
    }
    check("accepted replicas bit-equal their proposal", acc);
    check("rejected replicas bit-unchanged from previous state", rej);

    // ---- 5. diverged replicas ---------------------------------------------
    // Now half the population has moved.  A zero-angle proposal must reproduce
    // each replica's own current energy -- this is what proves kBuildReplicas
    // reads per-replica bases rather than a single shared conformation.
    printf("\n[5] zero-angle identity on a diverged population\n");
    std::vector<double> zero((size_t)P * NG, 0.0), tZero(P, 0.0);
    if (energyComputeReplicaBatch(ctx, P, &gbv[0], &ngv[0], &zero[0], NG, &tZero[0]) != 0)
        { printf("  zero-angle FAILED\n"); ++failures; }

    for (int k = 0; k < P; ++k) {
        // Independent route: read the replica back and evaluate it directly.
        energyGetReplicaCoords(ctx, k, &rx[0], &ry[0], &rz[0]);
        double direct = 0.0; energyBreakdown b;
        energyContext* c2 = build(s);
        energySetCoords(c2, &rx[0], &ry[0], &rz[0]);
        energyCompute(c2, &rx[0], &ry[0], &rz[0], &direct, &b);
        energyDestroy(c2);
        char buf[80];
        snprintf(buf, sizeof buf, "replica %d (%s) zero-angle vs direct", k,
                 accept[k] ? "moved" : "held");
        checkClose(buf, tZero[k], direct, tol);
    }

    // Accepted and rejected replicas must now genuinely differ.
    check("population has actually diverged", tZero[0] != tZero[1]);

    energyDestroy(ctx);
    printf("\nRESULT: %d failure%s\n", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
