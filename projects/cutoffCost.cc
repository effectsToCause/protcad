// cutoffCost -- what does the 12 A nonbonded cutoff actually buy?
//
// Both GPU passes are tile-based and visit every tile pair, rejecting a pair
// with a box-box distance test before entering the 32-iteration inner loop.
// So the cutoff never removes the O(N^2 / TILE^2) tile sweep, only the inner
// work.  On a small protein every tile is within 12 A of every other tile and
// the test can only cost; on a large one it should dominate.  Measure where
// the crossover is, and what the cutoff costs in accuracy.
//
//   cutoffCost <in.pdb> [reps]

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"
#include "energy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <vector>

using namespace std;

static double nowMs()
{
	struct timeval tv; gettimeofday(&tv, 0);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char** argv)
{
	if (argc < 2) {printf("cutoffCost <in.pdb> [reps]\n"); return 1;}
	const int reps = (argc > 2) ? atoi(argv[2]) : 200;

#ifndef __CUDA__
	printf("cutoffCost requires a CUDA build\n"); return 1;
#else
	PDBInterface* thePDB = new PDBInterface(argv[1]);
	protein* prot = static_cast<protein*>(
		thePDB->getEnsemblePointer()->getMoleculePointer(0));

	residue::setElectroSolvationScaleFactor(1.0);
	residue::setHydroSolvationScaleFactor(1.0);
	residue::setPolarizableElec(false);
	amberElec::setScaleFactor(1.0);
	amberVDW::setScaleFactor(1.0);
	residue::setTemperature(300);
	residue::setEntropyFactor(0.0);
	prot->loadDeviceMemAll();

	const int N = prot->getNumAtoms();
	const double cuts[] = {8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 1.0e4};
	const int nC = (int)(sizeof(cuts) / sizeof(cuts[0]));

	printf("\n%s  %d atoms  %d reps\n", argv[1], N, reps);
	printf("  cutoff      energy        d(E) vs 12 A     ms/eval   vs 12 A\n");

	double e12 = 0, t12 = 0;
	vector<double> es(nC), ts(nC);
	for (int q = 0; q < nC; q++)
	{
		prot->protSetCutoffCU(cuts[q]);
		prot->protEnergyCU();                       // warm up, and force the sort
		const double t0 = nowMs();
		double e = 0;
		for (int k = 0; k < reps; k++) {e = prot->protEnergyCU();}
		const double dt = (nowMs() - t0) / reps;
		es[q] = e; ts[q] = dt;
		if (cuts[q] == 12.0) {e12 = e; t12 = dt;}
	}
	for (int q = 0; q < nC; q++)
	{
		if (cuts[q] > 1000.0) printf("   none ");
		else                  printf("  %5.1f ", cuts[q]);
		printf("  %14.4f   %+12.4f   %8.4f   %6.2fx\n",
		       es[q], es[q] - e12, ts[q], t12 > 0 ? ts[q] / t12 : 0.0);
	}
	printf("\n");
	prot->protSetCutoffCU(12.0);
	return 0;
#endif
}
