// Per-term energy breakdown and one-lever-at-a-time ablation on a real PDB.
//
// The default parameter set changes several independent things at once, so a
// single default-vs-legacy total says nothing about which correction is
// responsible.  Each row below starts from legacy and flips exactly one lever.

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"
#include "energy.h"
#include <cstdio>
#include <cstring>
#include <string>

static void row(const char* label, protein* prot, energyParams p, double ref, bool isRef)
{
    prot->freeDeviceMemAll();
    prot->setEnergyParamsOverride(p);
    energyBreakdown b;
    if (!prot->protEnergyBreakdownCU(b)) {printf("%-26s FAILED\n", label); return;}
    int c = prot->getNumClashesCU();
    printf("%-26s %12.2f %11.2f %11.2f %11.2f %11.2f %11.2f %8d",
           label, b.vdw, b.electrostatic, b.solvationPolar,
           b.solvationNonpolar, b.solvationEntropy, b.total, c);
    if (!isRef) printf("  %+10.2f", b.total - ref);
    printf("\n");
}

int main(int argc, char** argv)
{
    if (argc < 2) {printf("usage: energyTerms <file.pdb>\n"); return 1;}
    PDBInterface* thePDB = new PDBInterface(argv[1]);
    ensemble* theEnsemble = thePDB->getEnsemblePointer();
    molecule* pMol = theEnsemble->getMoleculePointer(0);
    protein* prot = static_cast<protein*>(pMol);

    printf("\n%s   (%d atoms)\n", argv[1], prot->getNumAtoms());
    printf("%-26s %12s %11s %11s %11s %11s %11s %8s  %10s\n",
           "model", "vdW", "elec", "solv-pol", "solv-npol", "entropy", "TOTAL", "clashes", "d(total)");
    printf("--------------------------------------------------------------------------------------------------------\n");

    energyParams L = legacyEnergyParams();
    prot->freeDeviceMemAll(); prot->setEnergyParamsOverride(L);
    energyBreakdown b0; prot->protEnergyBreakdownCU(b0);
    double ref = b0.total;

    row("legacy (shipped model)", prot, L, ref, true);

    energyParams p;
    p = L; p.dielectric = DIELECTRIC_OCCUPANCY;   row("+ occupancy dielectric", prot, p, ref, false);
    p = L; p.occupancy  = OCCUPANCY_LENS;         row("+ lens shell volume", prot, p, ref, false);
    p = L; p.occlusionScale = 1.0;                row("+ no dbl-count comp (0.5->1)", prot, p, ref, false);
    p = L; p.occupancy = OCCUPANCY_LENS; p.occlusionScale = 1.0;
                                                  row("+ lens, no comp (both)", prot, p, ref, false);
    p = L; p.quantizeWaters = 0;                  row("+ continuous waters", prot, p, ref, false);
    p = L; p.useSwitching = 1;                    row("+ switched cutoff", prot, p, ref, false);
    p = L; p.pairMixing = PAIRMIX_HARMONIC;       row("+ harmonic pair mixing", prot, p, ref, false);
    p = L; p.clash = CLASH_INSCRIBED_CUBE;        row("+ inscribed-cube clash", prot, p, ref, false);
    p = L; p.bornNormalize = 1;                   row("+ born normalization", prot, p, ref, false);

    printf("--------------------------------------------------------------------------------------------------------\n");
    row("default (all corrections)", prot, defaultEnergyParams(), ref, false);
    printf("\n");
    return 0;
}
