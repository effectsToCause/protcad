//*******************************************************************************************************
//*******************************************************************************************************
//*****************************                         *************************************************
//*****************************    protBindingEnergy    *************************************************
//*****************************                         *************************************************
//*******************************************************************************************************
//*******************************************************************************************************

#include <iostream>
#include <string>
#include "ensemble.h"
#include "PDBInterface.h"

//--Program setup-------------------------------------------------------------
int main (int argc, char* argv[])
{	
	if (argc !=4)
	{
		cout << "protBindingEnergy <inFile.pdb> <receptor chain index> <ligand chain index>" << endl;
		exit(1);
	}
	string infile = argv[1];
	// These are zero-based chain indices, not the chain letters in the file.
	// They used to be left uninitialised when the argument did not parse, and
	// the resulting out-of-range energy of 0 was silently reported as a binding
	// energy equal to the whole complex energy.
	int ligandChainID = -1, receptorChainID = -1;
	if (sscanf(argv[2], "%d", &receptorChainID) != 1 || receptorChainID < 0 ||
	    sscanf(argv[3], "%d", &ligandChainID) != 1 || ligandChainID < 0)
	{
		cout << "protBindingEnergy: chain arguments must be non-negative integer indices" << endl;
		exit(1);
	}
	PDBInterface* thePDB = new PDBInterface(infile);
	ensemble* theEnsemble = thePDB->getEnsemblePointer();
	molecule* pMol = theEnsemble->getMoleculePointer(0);
	protein* prot = static_cast<protein*>(pMol);

	residue::setElectroSolvationScaleFactor(1.0);
	residue::setHydroSolvationScaleFactor(1.0);
	residue::setPolarizableElec(false);
	amberElec::setScaleFactor(1.0);
	amberVDW::setScaleFactor(1.0);
	residue::setEntropyFactor(0.0);
	residue::setTemperature(300);

	double complexE = prot->protEnergy();
	double receptorE = prot->protEnergy((UInt)receptorChainID);
	//residue::setEntropyFactor(0.0); //assuming free ligand if molecule, peptide or small unfolded protein, is not conformationally restrained, if protein this should be commented out
	double ligandE = prot->protEnergy((UInt)ligandChainID);
	
	// calculate binding energys
	double bindingEnergy = complexE-(ligandE+receptorE);
	cout << infile << " " << complexE << " " << bindingEnergy << endl;
	//pdbWriter(prot,infile);
	return 0;
}
