// Cross-checks the CPU clash accounting against the GPU, which counts each
// unordered clashing pair exactly once and is independent of residue ordering
// and of which residues have moved.  Both use the inscribed-cube criterion.

#include "ensemble.h"
#include "PDBInterface.h"
#include <cstdio>

int main(int argc, char** argv)
{
    if (argc < 2) {printf("usage: clashCheck <file.pdb>\n"); return 1;}
    PDBInterface* thePDB = new PDBInterface(argv[1]);
    ensemble* theEnsemble = thePDB->getEnsemblePointer();
    protein* prot = static_cast<protein*>(theEnsemble->getMoleculePointer(0));

    UInt cpu = prot->getNumHardClashes();
    // Force every residue to look moved, so updateClashes visits every residue
    // pair.  This separates a criterion/exclusion difference from a failure of
    // the incremental moved-flag propagation.
    // The CPU reference implements the legacy anisotropic cube criterion, so
    // pin the GPU to that mode for the parity check.  The shipped default is
    // now CLASH_SPHERE, which the CPU has no equivalent for.
    energyParams cubeP = defaultEnergyParams();
    cubeP.clash = CLASH_INSCRIBED_CUBE;
    prot->setEnergyParamsOverride(cubeP);

    prot->setMoved(true, 1);
    UInt cpuAll = prot->getNumHardClashes();
    UInt gpu = prot->getNumClashesCU();
    printf("%-12s cpu=%6u  cpu(all-moved)=%6u  gpu=%6u  %s\n",
           argv[1], cpu, cpuAll, gpu,
           cpuAll == gpu ? "match" : "MISMATCH");
    return cpuAll == gpu ? 0 : 1;
}
