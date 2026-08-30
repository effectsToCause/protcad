// protMutate -- apply one point mutation and write the structure out.
//
//   protMutate <in.pdb> <chain> <resIndex> <aaIndex> <out.pdb>
//
// protMutator already exists but drives a whole design protocol from an input
// file.  The cancellation measurement needs the mutation on its own, with no
// optimisation attached, so that the minimisation that follows is the same
// protMinRep run used everywhere else and the two arms differ only in where
// they start.
//
// resIndex is protcad's internal 0-based index within the chain, not the PDB
// residue number.  aaIndex follows data/mol.lib ordering, so ALA is 0.

#include <iostream>
#include <string>
#include <cstdlib>
#include "ensemble.h"
#include "PDBInterface.h"

int main(int argc, char* argv[])
{
	if (argc != 6)
	{
		cout << "protMutate <in.pdb> <chain> <resIndex> <aaIndex> <out.pdb>" << endl;
		cout << "  resIndex is the internal 0-based index, aaIndex follows mol.lib (ALA=0)" << endl;
		return 1;
	}

	string infile  = argv[1];
	UInt   chain   = (UInt)atoi(argv[2]);
	UInt   resIdx  = (UInt)atoi(argv[3]);
	UInt   aaIdx   = (UInt)atoi(argv[4]);
	string outfile = argv[5];

	PDBInterface* thePDB = new PDBInterface(infile);
	ensemble* theEnsemble = thePDB->getEnsemblePointer();
	molecule* pMol = theEnsemble->getMoleculePointer(0);
	protein* prot = static_cast<protein*>(pMol);

	prot->mutate(chain, resIdx, aaIdx);
	pdbWriter(prot, outfile);
	return 0;
}
