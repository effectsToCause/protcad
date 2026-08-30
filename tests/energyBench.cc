#ifdef PROTCAD_ENERGY_FP64
#define PROTCAD_EREAL_SIZE 8.0
#else
#define PROTCAD_EREAL_SIZE 4.0
#endif
// Throughput and memory scaling for the rewritten energy path.
//
// Reports wall time per evaluation against system size, plus the device memory
// the original O(N^2) implementation would have required for the same system.
// The original allocated an N(N-1)/2 int bonding matrix and an N(N-1)/2 double
// distance array, so its memory grew as 6*N^2 bytes regardless of cutoff.

#include "energy.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ctime>

static unsigned long long rs = 88172645463325252ULL;
static double rnd(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return double(rs%1000000007ULL)/1000000007.0; }

static double now()
{
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char** argv)
{
    int sizes[] = {500, 1000, 3000, 6000, 10000, 20000, 50000};
    int nSizes = sizeof(sizes)/sizeof(sizes[0]);
    int reps = (argc > 1) ? atoi(argv[1]) : 20;

#ifdef PROTCAD_ENERGY_FP64
    printf("precision: FP64\n");
#else
    printf("precision: FP32\n");
#endif
    printf("%8s %10s %10s %12s %12s %12s\n",
           "N", "ms/eval", "clash ms", "new MB", "old MB", "old/new");

    for (int si = 0; si < nSizes; ++si)
    {
        int N = sizes[si];
        std::vector<double> x(N), y(N), z(N), rad(N), eps(N), chg(N);
        std::vector<int> res(N), ec(N, 0), el((size_t)N*8, 0);
        std::vector<unsigned char> sil(N, 0);

        const double spacing = 3.0;
        int side = int(ceil(pow(double(N), 1.0/3.0)));
        for (int i = 0; i < N; ++i) {
            int cx=i%side, cy=(i/side)%side, cz=i/(side*side);
            x[i]=cx*spacing+(rnd()-0.5)*0.8;
            y[i]=cy*spacing+(rnd()-0.5)*0.8;
            z[i]=cz*spacing+(rnd()-0.5)*0.8;
            rad[i]=1.4+rnd()*0.6; eps[i]=0.05+rnd()*0.15;
            chg[i]=(rnd()-0.5)*0.8; res[i]=i/8;
        }
        for (int i = 0; i < N; ++i)
            for (int j = (i/8)*8; j < (i/8)*8+8 && j < N; ++j)
                if (j != i && abs(i-j) <= 2) el[(size_t)i*8 + ec[i]++] = j;

        energyTopology t;
        t.numAtoms=N; t.radius=&rad[0]; t.epsilon=&eps[0]; t.charge=&chg[0];
        t.residueIndex=&res[0]; t.silent=&sil[0];
        t.exclusionCount=&ec[0]; t.exclusionList=&el[0]; t.exclusionStride=8;

        energyParams p = defaultEnergyParams();
        energyContext* ctx = energyCreate(t, p);
        if (!ctx) { printf("%8d  allocation failed\n", N); continue; }

        double e; int c;
        energyCompute(ctx, &x[0], &y[0], &z[0], &e, 0);   // warm up

        double t0 = now();
        for (int r = 0; r < reps; ++r) energyCompute(ctx, &x[0], &y[0], &z[0], &e, 0);
        double tE = (now() - t0) / reps * 1000.0;

        clashCompute(ctx, &x[0], &y[0], &z[0], &c);
        t0 = now();
        for (int r = 0; r < reps; ++r) clashCompute(ctx, &x[0], &y[0], &z[0], &c);
        double tC = (now() - t0) / reps * 1000.0;

        // New: ~20 arrays of N elements.  Old: N(N-1)/2 ints + N(N-1)/2 doubles.
        double newMB = N * (14.0 * PROTCAD_EREAL_SIZE + 8.0 * sizeof(int)) / 1048576.0;
        double oldMB = (double)N * (N - 1) / 2.0 * 12.0 / 1048576.0;

        printf("%8d %10.3f %10.3f %12.1f %12.1f %11.0fx\n",
               N, tE, tC, newMB, oldMB, oldMB / newMB);
        energyDestroy(ctx);
    }
    return 0;
}
