// protMinRep -- fixed-budget population Monte Carlo sidechain minimisation.
//
//   protMinRep <sweeps> [replicas] <inFile.pdb> <outFile.pdb>
//
// Unlike protMin there is no plateau detection: the cost is chosen up front.
// Total energy evaluations is sweeps * replicas, so runs are directly
// comparable across settings by holding that product fixed.
//
// Replicas defaults to 1. Energy at a fixed wall budget is monotone in
// sweeps-per-chain, and a matched-wall test resolves what the earlier sweep
// could not: a population is not merely unnecessary for minimisation, it is
// mildly harmful. Raise it to sample an ensemble, not to minimise faster.
// See protMinReplicaCU in protein.cc for the measurements.

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
	if (argc != 4 && argc != 5)
	{   cout << "protMinRep <sweeps> [replicas] <inFile.pdb> <outFile.pdb>" << endl;
		cout << "  replicas defaults to 1; raise it to sample an ensemble," << endl;
		cout << "  not to minimise faster -- see protMinReplicaCU" << endl;
		exit(1); }

	UInt sweeps   = (UInt)atoi(argv[1]);
	UInt replicas = (argc == 5) ? (UInt)atoi(argv[2]) : 1;
	string infile  = argv[argc - 2];
	string outFile = argv[argc - 1];

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
	if (getenv("PROTCAD_TOPO_DEBUG"))
	{
		energyBreakdown bd;
		if (_prot->protEnergyBreakdownCU(bd))
		{
			cout << "[breakdown] vdw " << bd.vdw
			     << "  torsion " << bd.torsion
			     << "  elec " << bd.electrostatic
			     << "  solvP " << bd.solvationPolar
			     << "  solvN " << bd.solvationNonpolar
			     << "  solvE " << bd.solvationEntropy << endl;
		}
	}
	cout << "sweeps " << sweeps << "  replicas " << replicas
	     << "  evaluations " << (double)sweeps * replicas << endl;

	double t0 = wall();
	_prot->protMinReplicaCU(sweeps, replicas);
	double t1 = wall();

	double endE = _prot->protEnergyCU();
	cout << "Ending Energy: " << endE << " kcal/mol" << endl;
	cout << "Delta: " << endE - startE << " kcal/mol" << endl;

	// The ensemble block is the estimator that matters for a reference state.
	// "Ending Energy" above is the best conformation seen, which is a biased
	// summary of a canonical trajectory and gets worse the more torsions the
	// residue has.
	const protein::ensembleStats& es = _prot->getEnsembleStats();
	if (es.valid)
	{
		cout << "--- ensemble (post burn-in) ---" << endl;
		cout << "samples " << es.samples << "  torsions " << es.torsions
		     << "  accept " << 100.0 * es.acceptRate << " %" << endl;
		cout << "<E>: " << es.meanEnergy << " +/- " << es.sdEnergy << " kcal/mol" << endl;
		cout << "minE: " << es.minEnergy << " kcal/mol" << endl;
		cout << "S_conf: " << es.conformEntropy << " kcal/(mol K)" << endl;
		cout << "T*S_conf: " << protein::Temperature() * es.conformEntropy << " kcal/mol" << endl;
		cout << "A = <E> - T*S_conf: " << es.freeEnergy << " kcal/mol" << endl;
	}
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
