#ifdef PROTCAD_ENERGY_FP64
#define PROTCAD_EREAL_SIZE 8.0
#else
#define PROTCAD_EREAL_SIZE 4.0
#endif
// Head-to-head: original O(N^2) kernels vs the rewritten O(N) path.
//
// Both are driven with identical coordinates and identical exclusions, in the
// precision each natively uses (the original was FP64-only).  The original is
// linked from a preserved copy of the shipped energy.cu, so this is a genuine
// before/after and not an estimate.

#include "energy.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ctime>

// Original API, declared here so the legacy translation unit can be linked
// alongside the new header without an include-name collision.
extern void loadEnergyDeviceMem(double*, double*, double*, double*, double*,
                                double*, double*, int*, double*, int);
extern void freeEnergyDeviceMem();
extern void calcEnergies(double*, double*, double*, double*, double*, int);

static unsigned long long rs = 88172645463325252ULL;
static double rnd(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return double(rs%1000000007ULL)/1000000007.0; }
static double now(){ timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

int main(int argc, char** argv)
{
    int sizes[] = {500, 1000, 2000, 4000, 8000};
    int nS = sizeof(sizes)/sizeof(sizes[0]);
    int reps = (argc>1)?atoi(argv[1]):10;

    printf("%7s %12s %12s %10s %12s %12s %9s\n",
           "N", "orig ms", "new ms", "speedup", "orig MB", "new MB", "mem");

    for (int si = 0; si < nS; ++si)
    {
        int N = sizes[si];
        size_t pairs = (size_t)N*(N-1)/2;

        std::vector<double> x(N),y(N),z(N),rad(N),eps(N),chg(N),vol(N,0.0);
        std::vector<int> res(N), ec(N,0), el((size_t)N*8,0);
        std::vector<unsigned char> sil(N,0);
        const double sp = 3.0;
        int side = int(ceil(pow(double(N),1.0/3.0)));
        for (int i=0;i<N;++i){
            int cx=i%side, cy=(i/side)%side, cz=i/(side*side);
            x[i]=cx*sp+(rnd()-0.5)*0.8; y[i]=cy*sp+(rnd()-0.5)*0.8; z[i]=cz*sp+(rnd()-0.5)*0.8;
            rad[i]=1.4+rnd()*0.6; eps[i]=0.05+rnd()*0.15; chg[i]=(rnd()-0.5)*0.8; res[i]=i/8;
        }
        for (int i=0;i<N;++i)
            for (int j=(i/8)*8; j<(i/8)*8+8 && j<N; ++j)
                if (j!=i && abs(i-j)<=2) el[(size_t)i*8 + ec[i]++]=j;

        // Dense exclusion matrix, exactly as the original required.  Building
        // this on the host is itself O(N^2) and is not counted below.
        std::vector<int> bon(pairs, 0);
        for (int i=0;i<N;++i)
            for (int j=i+1;j<N;++j)
                if (res[i]==res[j] && abs(i-j)<=2)
                    bon[(size_t)i*(N-1)-(size_t)(i-1)*i/2+j-i-1] = 1;

        // --- original ---
        double E = 0.0;
        loadEnergyDeviceMem(&x[0],&y[0],&z[0],&rad[0],&eps[0],&chg[0],&vol[0],&bon[0],&E,N);
        std::vector<double> zero(N,0.0);
        E=0; std::vector<double> v=zero; calcEnergies(&x[0],&y[0],&z[0],&v[0],&E,N);
        double t0=now();
        for (int r=0;r<reps;++r){ E=0; v=zero; calcEnergies(&x[0],&y[0],&z[0],&v[0],&E,N); }
        double tOld=(now()-t0)/reps*1000.0;
        freeEnergyDeviceMem();

        // --- new ---
        energyTopology t;
        t.numAtoms=N; t.radius=&rad[0]; t.epsilon=&eps[0]; t.charge=&chg[0];
        t.residueIndex=&res[0]; t.silent=&sil[0];
        t.exclusionCount=&ec[0]; t.exclusionList=&el[0]; t.exclusionStride=8;
        energyParams p = legacyEnergyParams();
        energyContext* ctx = energyCreate(t,p);
        double e; energyCompute(ctx,&x[0],&y[0],&z[0],&e,0);
        t0=now();
        for (int r=0;r<reps;++r) energyCompute(ctx,&x[0],&y[0],&z[0],&e,0);
        double tNew=(now()-t0)/reps*1000.0;
        energyDestroy(ctx);

        double oldMB = pairs*12.0/1048576.0 + N*7.0*8.0/1048576.0;
        double newMB = N*(14.0*PROTCAD_EREAL_SIZE+8.0*sizeof(int))/1048576.0;

        printf("%7d %12.3f %12.3f %9.1fx %12.1f %12.2f %8.0fx\n",
               N, tOld, tNew, tOld/tNew, oldMB, newMB, oldMB/newMB);
    }
    return 0;
}
