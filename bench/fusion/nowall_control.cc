#include "energy.h"
#include <cstdio>
#include <vector>
#include <cmath>
// Negative control: does the monotonicity criterion actually have teeth?
// Suppress the LJ wall (epsilon -> ~0) and leave a strong Coulomb attraction.
// The total must then be downhill to the floor, and the test must say so.
int main(){
    double rad[2]={1.908,1.6612}, eps[2]={1e-12,1e-12}, chg[2]={+2.0,-2.0};
    int res[2]={0,1}; unsigned char sil[2]={0,0}; int ec[2]={0,0};
    std::vector<int> el(2,0);
    energyTopology t; t.numAtoms=2; t.radius=rad; t.epsilon=eps; t.charge=chg;
    t.residueIndex=res; t.silent=sil; t.exclusionCount=ec;
    t.exclusionList=&el[0]; t.exclusionStride=1;
    energyParams p=defaultEnergyParams();
    energyContext* ctx=energyCreate(t,p);
    if(!ctx){printf("create failed\n");return 2;}
    std::vector<double> rs,es;
    for(double r=4.00;r>=p.minSeparation;r-=0.05){
        double x[2]={0,r},y[2]={0,0},z[2]={0,0},total=0; energyBreakdown b;
        energyCompute(ctx,x,y,z,&total,&b);
        rs.push_back(r); es.push_back(total);
    }
    size_t imin=0; for(size_t i=1;i<es.size();++i) if(es[i]<es[imin]) imin=i;
    printf("no-wall control: minimum at r = %.2f A (innermost sample r = %.2f A)\n",
           rs[imin], rs.back());
    printf("  E(4.00) = %.4g   E(floor) = %.4g\n", es.front(), es.back());
    bool caught = (imin==es.size()-1);
    printf("  criterion %s this as a fusion channel\n", caught?"CATCHES":"MISSES");
    energyDestroy(ctx);
    return caught?0:1;
}
