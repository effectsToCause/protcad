// minsep_probe -- is the "fusion channel" a property of the potential, or of
// the distance clamp?
//
// fusionTest reported that the total energy fell on approach inside contact in
// a dense lattice, first at r = 0.74 A.  That looked like the best available
// mechanistic explanation for the multi-basin noise floor: a downhill channel
// at close approach would let the minimiser find seed-dependent collapse
// basins.  This probe kills that hypothesis.
//
// energy.cu pins the pair distance at energyParams::minSeparation to keep 1/r
// finite (energy.cu:900, "if (d < minSep) { d = minSep; r2 = d * d; }").  Below
// that floor the scanned pair contributes a constant, while the moving atom is
// still receding from all its other neighbours -- so the total falls for a
// purely geometric reason that has nothing to do with solvation compensation.
//
// Sweeping the floor shows the onset of the "channel" tracking it exactly:
//
//   minSep 0.80 A : first fall at r = 0.79 A, E = 8.27e+06 kcal/mol
//   minSep 0.60 A : first fall at r = 0.59 A, E = 2.61e+08 kcal/mol
//   minSep 0.40 A : first fall at r = 0.39 A, E = 3.39e+10 kcal/mol
//   minSep 0.20 A : first fall at r = 0.15 A, E = 1.39e+14 kcal/mol
//   minSep 0.10 A : none
//   minSep 0.05 A : none
//
// Always inside the floor, never outside it, and gone once the floor is below
// the scan.  It is an artefact.  It is also thermally irrelevant regardless:
// entry costs upwards of 10^6 kcal/mol, so Metropolis acceptance is zero at any
// temperature this code will ever run.
//
// Conclusion: the fusion channel does not explain the noise floor.  The clash
// pre-filter is not reproducing a missing physical constraint.  Look elsewhere.
//
// Build:
//   nvcc -O3 -arch=sm_61 -D__CUDA__ -Isrc/ensemble -x cu \
//        bench/fusion/minsep_probe.cc src/ensemble/energy.cu -o /tmp/minsep_probe

#include "energy.h"
#include <cstdio>
#include <vector>
#include <cmath>

// Dense-lattice approach scan, parameterised by the minSeparation floor, to
// test whether the reported fusion channel is a property of the potential or
// an artefact of the distance clamp.
static void probe(double minSep, double qScale)
{
    const int side = 4, N = side*side*side;
    std::vector<double> x(N),y(N),z(N),rad(N),eps(N),chg(N);
    std::vector<int> res(N), ec(N,0), el(N,1);
    std::vector<unsigned char> sil(N,0);
    unsigned seed = 12345;
    for (int i=0;i<N;++i){
        int ix=i%side, iy=(i/side)%side, iz=i/(side*side);
        seed=seed*1103515245u+12345u; double j1=((seed>>16)&0x7fff)/32767.0-0.5;
        seed=seed*1103515245u+12345u; double j2=((seed>>16)&0x7fff)/32767.0-0.5;
        seed=seed*1103515245u+12345u; double j3=((seed>>16)&0x7fff)/32767.0-0.5;
        x[i]=ix*3.3+0.3*j1; y[i]=iy*3.3+0.3*j2; z[i]=iz*3.3+0.3*j3;
        rad[i]=1.8; eps[i]=0.12; chg[i]=qScale*((i%2)?0.5:-0.5); res[i]=i/4;
    }
    energyTopology t;
    t.numAtoms=N; t.radius=&rad[0]; t.epsilon=&eps[0]; t.charge=&chg[0];
    t.residueIndex=&res[0]; t.silent=&sil[0];
    t.exclusionCount=&ec[0]; t.exclusionList=&el[0]; t.exclusionStride=1;
    energyParams p = defaultEnergyParams();
    p.minSeparation = minSep;
    energyContext* ctx = energyCreate(t,p);
    if(!ctx){printf("create failed\n");return;}

    double x0=x[0],y0=y[0],z0=z[0];
    double dx=x[1]-x0,dy=y[1]-y0,dz=z[1]-z0;
    double d0=sqrt(dx*dx+dy*dy+dz*dz); dx/=d0;dy/=d0;dz/=d0;

    double prev=0; bool have=false; int nonmono=0; double firstR=0, drop=0, eAtFirst=0;
    for(double r=d0;r>=0.10;r-=0.01){
        x[0]=x[1]-dx*r; y[0]=y[1]-dy*r; z[0]=z[1]-dz*r;
        double total=0; energyBreakdown b;
        if(energyCompute(ctx,&x[0],&y[0],&z[0],&total,&b)!=0){printf("compute failed\n");break;}
        if(have && r<2*rad[0] && total<prev-1e-9){
            if(!nonmono){firstR=r; eAtFirst=total;}
            drop+=prev-total; ++nonmono;
        }
        prev=total; have=true;
    }
    printf("  minSep %.2f A : %s", minSep, nonmono?"channel":"none   ");
    if(nonmono) printf("  first fall at r = %.2f A (%s minSep), E there = %.3g kcal/mol, drop %.4g",
                       firstR, firstR<minSep?"inside":"OUTSIDE", eAtFirst, drop);
    printf("\n");
    energyDestroy(ctx);
}

int main(){
    printf("Does the fusion channel track the minSeparation clamp?\n\n");
    printf("charged lattice:\n");
    for(double m : {0.80,0.60,0.40,0.20,0.10,0.05}) probe(m,1.0);
    printf("\nneutral lattice:\n");
    for(double m : {0.80,0.40,0.10}) probe(m,0.0);
    return 0;
}
