// Loading a structure must never place two atoms on top of each other.
//
// This is not hypothetical.  At load time every residue is rebuilt from its
// template by chain::rebuildResidue, and the coordinates the file supplied are
// restored onto the rebuilt residue.  The restore was done by atom *index*.
// That is only valid if the rebuilt residue has the same atom layout as the
// one the coordinates were captured from, and for the first and last residue
// of a chain it does not: mutateNew promotes them to the N- and C-terminal
// templates, which carry different atoms in a different order.
//
//   ALA  N CA C O CB H  HA  HB1 HB2 HB3
//   ALN  N CA C O CB HA HB1 HB2 HB3 H1 H2 H3
//
// Restoring positionally therefore wrote the amide H's coordinates onto HA and
// left the terminal hydrogens at raw mol.lib template coordinates, i.e. in the
// template's own frame rather than the molecule's.  The result was atom pairs
// separated by 0.001 A.
//
// The damage was invisible under the legacy parameter set, which excludes 1-4
// pairs outright (vdw14Scale = 0), and catastrophic under the default set,
// which damps but keeps them: a two-residue peptide reported a vdW term of
// 306,296 kcal/mol, and 1crn passed through one load/write cycle went from
// -85.9 to +372,824.  A forcefield cannot be calibrated against a structure the
// loader has corrupted, so this is tested as a geometry invariant rather than
// an energy one.
//
// Any non-bonded pair closer than 0.5 A is unphysical; the shortest real bond
// in a protein is roughly 1.0 A (N-H).  Terminal residues are what this is
// about, so the peptide below is short enough to be almost entirely termini.

#include "PDBInterface.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

// A two-residue alanine peptide with all hydrogens present, which is what
// protcad itself writes.  Coordinates are an ordinary extended conformation.
const char* kPeptide =
"ATOM      1  N   ALA A   1       0.000   0.000   0.000\n"
"ATOM      2  CA  ALA A   1       1.458   0.000   0.000\n"
"ATOM      3  C   ALA A   1       2.009   1.420   0.000\n"
"ATOM      4  O   ALA A   1       1.251   2.394   0.000\n"
"ATOM      5  CB  ALA A   1       1.988  -0.773  -1.199\n"
"ATOM      6  H   ALA A   1      -0.476   0.882   0.000\n"
"ATOM      7  HA  ALA A   1       1.804  -0.507   0.901\n"
"ATOM      8  HB1 ALA A   1       3.076  -0.752  -1.226\n"
"ATOM      9  HB2 ALA A   1       1.647  -1.807  -1.144\n"
"ATOM     10  HB3 ALA A   1       1.631  -0.320  -2.125\n"
"ATOM     11  N   ALA A   2       3.332   1.549   0.000\n"
"ATOM     12  CA  ALA A   2       3.986   2.851   0.000\n"
"ATOM     13  C   ALA A   2       5.497   2.679   0.000\n"
"ATOM     14  O   ALA A   2       6.008   1.557   0.000\n"
"ATOM     15  CB  ALA A   2       3.566   3.659  -1.223\n"
"ATOM     16  H   ALA A   2       3.902   0.716   0.000\n"
"ATOM     17  HA  ALA A   2       3.681   3.400   0.892\n"
"ATOM     18  HB1 ALA A   2       4.055   4.635  -1.204\n"
"ATOM     19  HB2 ALA A   2       2.487   3.815  -1.222\n"
"ATOM     20  HB3 ALA A   2       3.845   3.146  -2.145\n"
"TER\n"
"END\n";

// Returns the shortest distance between any two atoms, and names the pair.
double closestPair(const char* _file, std::string& _what)
{
    PDBInterface* thePDB = new PDBInterface(_file);
    ensemble* theEnsemble = thePDB->getEnsemblePointer();
    molecule* pMol = theEnsemble->getMoleculePointer(0);
    protein* pProt = static_cast<protein*>(pMol);

    std::vector<dblVec> xyz;
    std::vector<std::string> tag;
    for (UInt c = 0; c < pProt->getNumChains(); c++)
    {
        for (UInt r = 0; r < pProt->getNumResidues(c); r++)
        {
            residue* pRes = pProt->getChain(c)->getResidue(r);
            for (UInt a = 0; a < pRes->getNumAtoms(); a++)
            {
                xyz.push_back(pRes->getCoords(a));
                tag.push_back(pRes->getAtom(a)->getName() + "/" +
                              std::to_string(r + 1));
            }
        }
    }

    double best = 1.0e30;
    for (UInt i = 0; i < xyz.size(); i++)
    {
        for (UInt j = i + 1; j < xyz.size(); j++)
        {
            double s = 0.0;
            for (UInt k = 0; k < 3; k++)
            {
                double d = xyz[i][k] - xyz[j][k];
                s += d * d;
            }
            s = sqrt(s);
            if (s < best) { best = s; _what = tag[i] + " - " + tag[j]; }
        }
    }
    return best;
}

}

int main(int argc, char** argv)
{
    const double kMinSeparation = 0.5;
    int failures = 0;

    // 1. A hydrogen-bearing peptide dominated by its termini.
    const char* tmp = "terminusTest_peptide.pdb";
    {
        std::ofstream out(tmp);
        out << kPeptide;
    }

    std::string what;
    double d = closestPair(tmp, what);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "two-residue peptide: closest pair " << what
              << " = " << d << " A" << std::endl;
    if (d < kMinSeparation)
    {
        std::cout << "FAIL: atoms superimposed after load; terminal residue "
                  << "rebuild is not matching atoms by name." << std::endl;
        failures++;
    }
    std::remove(tmp);

    // 2. The same invariant on a real structure, if one was supplied.
    if (argc > 1)
    {
        double dRef = closestPair(argv[1], what);
        std::cout << argv[1] << ": closest pair " << what
                  << " = " << dRef << " A" << std::endl;
        if (dRef < kMinSeparation)
        {
            std::cout << "FAIL: atoms superimposed after load." << std::endl;
            failures++;
        }
    }

    if (failures == 0) { std::cout << "PASS" << std::endl; }
    return failures == 0 ? 0 : 1;
}
