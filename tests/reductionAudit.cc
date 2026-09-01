// Audit of the host reduction: attribution correctness and the error cost of
// each summation strategy, measured on a real structure.
//
// Two questions:
//   1. Do the per-atom values sum to the reported total?  energy.h promises
//      they do, and protein::updateResidueEnergiesCU builds per-residue
//      energies on that promise.
//   2. What does the Kahan compensation actually buy?  The kernel emits
//      float32 per-atom terms, so the input error is already ~1e-7 relative;
//      the question is whether the double-precision compensated sum is
//      measurably better than a plain double sum over the same order, and how
//      much a float32 accumulator would cost.

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"
#include "chain.h"
#include "energy.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

static double kahan(const std::vector<double>& v)
{
    double s = 0.0, c = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        double y = v[i] - c, t = s + y;
        c = (t - s) - y; s = t;
    }
    return s;
}

static double naive(const std::vector<double>& v)
{
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) s += v[i];
    return s;
}

static double naive32(const std::vector<double>& v)
{
    float s = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) s += float(v[i]);
    return double(s);
}

// Pairwise (binary tree) sum in float32 -- the shape a device reduction has.
static float treeF(std::vector<float>& v, size_t lo, size_t hi)
{
    if (hi - lo == 1) return v[lo];
    size_t mid = lo + (hi - lo) / 2;
    return treeF(v, lo, mid) + treeF(v, mid, hi);
}

int main(int argc, char** argv)
{
    if (argc < 2) {printf("usage: torsionAudit <file.pdb>\n"); return 1;}
    PDBInterface* thePDB = new PDBInterface(argv[1]);
    ensemble* theEnsemble = thePDB->getEnsemblePointer();
    protein* prot = static_cast<protein*>(theEnsemble->getMoleculePointer(0));

    printf("\n%s  (%d atoms)\n", argv[1], prot->getNumAtoms());

    energyBreakdown b;
    if (!prot->protEnergyBreakdownCU(b)) {printf("breakdown FAILED\n"); return 1;}
    prot->updateResidueEnergiesCU();

    std::vector<double> res;
    for (UInt ci = 0; ci < prot->getNumChains(); ++ci) {
        chain* ch = prot->getChain(ci);
        for (UInt ri = 0; ri < ch->getNumResidues(); ++ri)
            res.push_back(ch->getEnergy(ri));
    }

    printf("\n-- attribution --\n");
    double s = kahan(res);
    printf("  breakdown.total        %14.4f\n", b.total);
    printf("  breakdown.torsion      %14.4f\n", b.torsion);
    printf("  sum(residue energy)    %14.4f\n", s);
    printf("  total - sum(residues)  %14.6f   (must be 0)\n", b.total - s);

    printf("\n-- error cost of the summation strategy (%d residue terms) --\n",
           (int)res.size());
    double magn = 0.0;
    for (size_t i = 0; i < res.size(); ++i) magn += std::fabs(res[i]);
    printf("  sum|term|              %14.4f\n", magn);
    printf("  |total|                %14.4f\n", std::fabs(s));
    printf("  cancellation ratio     %14.1f   (sum|x| / |sum x|)\n",
           magn / std::max(1e-30, std::fabs(s)));

    double kd = kahan(res), nd = naive(res), n32 = naive32(res);
    std::vector<float> vf(res.size());
    for (size_t i = 0; i < res.size(); ++i) vf[i] = float(res[i]);
    double tf = double(treeF(vf, 0, vf.size()));

    printf("  kahan double  (shipped)%18.10f      0 (reference)\n", kd);
    printf("  naive double           %18.10f   %+.3e\n", nd, nd - kd);
    printf("  naive float32          %18.10f   %+.3e\n", n32, n32 - kd);
    printf("  tree  float32 (device) %18.10f   %+.3e\n", tf, tf - kd);

    // Reverse order models what a differently-ordered reduction would give at
    // the same precision: this is the reproducibility cost, not an accuracy cost.
    std::vector<double> rev(res.rbegin(), res.rend());
    printf("  kahan double, reversed %18.10f   %+.3e  (order sensitivity)\n",
           kahan(rev), kahan(rev) - kd);
    printf("  naive double, reversed %18.10f   %+.3e  (order sensitivity)\n",
           naive(rev), naive(rev) - nd);
    return 0;
}
