// protMinRep -- fixed-budget population Monte Carlo sidechain minimisation.
//
//   protMinRep <sweeps> <replicas> <inFile.pdb> <outFile.pdb>
//
// Unlike protMin there is no plateau detection: the cost is chosen up front.
// Total energy evaluations is sweeps * replicas, so runs are directly
// comparable across settings by holding that product fixed.

#include "ensemble.h"
#include "PDBInterface.h"
#include <sstream>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

static double wall()
{
	struct timeval tv; gettimeofday(&tv, 0);
	return tv.tv_sec + 1e-6 * tv.tv_usec;
}

int main (int argc, char* argv[])
{
	if (argc != 5)
	{   cout << "protMinRep <sweeps> <replicas> <inFile.pdb> <outFile.pdb>" << endl;
		exit(1); }

	UInt sweeps   = (UInt)atoi(argv[1]);
	UInt replicas = (UInt)atoi(argv[2]);
	string infile = argv[3];
	string outFile = argv[4];

	PDBInterface* thePDB = new PDBInterface(infile);
	ensemble* theEnsemble = thePDB->getEnsemblePointer();
	molecule* pMol = theEnsemble->getMoleculePointer(0);
	protein* _prot = static_cast<protein*>(pMol);

	residue::setElectroSolvationScaleFactor(1.0);
	residue::setHydroSolvationScaleFactor(1.0);
	residue::setPolarizableElec(false);
	amberElec::setScaleFactor(1.0);
	amberVDW::setScaleFactor(1.0);
	residue::setTemperature(300);
	residue::setEntropyFactor(0.0);

#ifdef __CUDA__
	_prot->loadDeviceMemAll();
	double startE = _prot->protEnergyCU();
	cout << "Starting Energy: " << startE << " kcal/mol" << endl;
	cout << "sweeps " << sweeps << "  replicas " << replicas
	     << "  evaluations " << (double)sweeps * replicas << endl;

	double t0 = wall();
	_prot->protMinReplicaCU(sweeps, replicas);
	double t1 = wall();

	double endE = _prot->protEnergyCU();
	cout << "Ending Energy: " << endE << " kcal/mol" << endl;
	cout << "Delta: " << endE - startE << " kcal/mol" << endl;
	cout << "Wall: " << (t1 - t0) << " s" << endl;
	if (sweeps && replicas)
		cout << "Per-candidate: "
		     << 1e6 * (t1 - t0) / ((double)sweeps * replicas) << " us" << endl;
#else
	cout << "protMinRep requires a CUDA build" << endl;
#endif

	pdbWriter(_prot, outFile);
	return 0;
}
