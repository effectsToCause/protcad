// Reports the CPU clash count against the GPU.
//
// These two no longer agree, and are not expected to.  The GPU topology takes
// atom radii from the AMBER energy-type table, because that same array is the
// radius used by the vdW and solvation terms and it has to match the AMBER well
// depths they are paired with.  The CPU's residue::isClash instead uses the
// element-level table in the atom.cc dataBase, which assigns every hydrogen a
// flat 1.090 A -- a C-H bond length rather than a van der Waals radius.  Both
// paths are self-consistent; the CPU one is built on a placeholder.
//
// What is still worth checking here, and what this program asserts, is that the
// CPU count does not depend on which residues are flagged as moved.  That was a
// real accounting bug.  The radius divergence is reported for information and
// disappears when the CPU path is retired.

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
    printf("%-12s cpu=%6u  cpu(all-moved)=%6u  gpu=%6u  (gpu uses AMBER radii, cpu uses element radii)\n",
           argv[1], cpu, cpuAll, gpu);
    if (cpu != cpuAll) {
        printf("  FAIL: cpu count depends on moved flags (%u vs %u)\n", cpu, cpuAll);
        return 1;
    }
    printf("  ok: cpu count is independent of moved flags\n");
    return 0;
}
