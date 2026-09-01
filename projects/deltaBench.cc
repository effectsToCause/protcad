// deltaBench -- is the incremental evaluation exact, and is it faster?
//
// A move changes only the pair terms with an end in the thawed set, plus the
// per-atom solvation of those atoms.  So
//
//     E_new = E_old - P(C, old) + P(C, new) + torsion_new
//
// is an identity, not an approximation, as long as C contains every atom whose
// position or dielectric moved.  This checks that against a full evaluation at
// the same coordinates, and times both.
//
//   deltaBench <in.pdb> [moves] [seed]

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
	if (argc < 2) {printf("deltaBench <in.pdb> [moves] [seed]\n"); return 1;}
	const int nMove = (argc > 2) ? atoi(argv[2]) : 60;
	srand((argc > 3) ? (unsigned)atoi(argv[3]) : 1u);

#ifndef __CUDA__
	printf("deltaBench requires a CUDA build\n"); return 1;
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

	vector< pair<UInt,UInt> > flex;
	for (UInt c = 0; c < prot->getNumChains(); c++)
		for (UInt r = 0; r < prot->getNumResidues(c); r++)
		{
			vector< vector<double> > d = prot->getSidechainDihedrals(c, r);
			if (!d.empty() && !d[0].empty()) {flex.push_back(make_pair(c, r));}
		}
	if (flex.empty()) {printf("no flexible residues\n"); return 1;}

	printf("\n%s  %d atoms  %d flexible residues  %d moves\n",
	       argv[1], N, (int)flex.size(), nMove);

	double tFull = 0, tDelta = 0, worst = 0, sumC = 0;
	int done = 0;

	for (int m = 0; m < nMove; m++)
	{
		const pair<UInt,UInt> pick = flex[rand() % flex.size()];
		vector< vector<double> > base = prot->getSidechainDihedrals(pick.first, pick.second);
		if (base.empty() || base[0].empty()) {continue;}
		vector< vector<double> > turned = base;
		const size_t k = rand() % turned[0].size();
		turned[0][k] = base[0][k] + 30.0 + 120.0 * (rand() / (double)RAND_MAX);

		vector<double> ox, oy, oz;
		prot->getDeviceCoordsCU(ox, oy, oz);

		// Anchor: the full energy at the pre-move conformation, split so the
		// torsion term can be replaced wholesale rather than differenced.
		prot->protReleaseDielectricCU();
		energyBreakdown b0;
		if (!prot->protEnergyBreakdownCU(b0)) {continue;}
		const double nbOld = b0.total - b0.torsion;

		if (!prot->protFreezeDielectricCU()) {continue;}

		// Thaw set for this move, and P at the pre-move geometry.
		prot->setSidechainDihedralAngles(pick.first, pick.second, turned);
		int nMoved = 0;
		const int nC = prot->protThawDielectricForMoveCU(ox, oy, oz, 0.0, &nMoved);
		prot->setSidechainDihedralAngles(pick.first, pick.second, base);
		if (nC <= 0) {prot->protReleaseDielectricCU(); continue;}

		double pOld = 0, torOld = 0;
		prot->protEnergyDeltaCU(pOld, torOld);

		// The move itself, timed both ways.
		prot->setSidechainDihedralAngles(pick.first, pick.second, turned);

		double pNew = 0, torNew = 0;
		prot->protEnergyDeltaCU(pNew, torNew);          // warm
		const double t0 = nowMs();
		for (int rep = 0; rep < 20; rep++) {prot->protEnergyDeltaCU(pNew, torNew);}
		tDelta += (nowMs() - t0) / 20.0;

		const double eDelta = nbOld - pOld + pNew + torNew;

		prot->protReleaseDielectricCU();
		double eFull = prot->protEnergyCU();            // warm
		const double t1 = nowMs();
		for (int rep = 0; rep < 20; rep++) {eFull = prot->protEnergyCU();}
		tFull += (nowMs() - t1) / 20.0;

		prot->setSidechainDihedralAngles(pick.first, pick.second, base);

		const double rel = fabs(eDelta - eFull) /
		                   (fabs(eFull) > 1.0 ? fabs(eFull) : 1.0);
		if (rel > worst) {worst = rel;}
		sumC += nC;
		done++;
	}

	if (!done) {printf("no moves evaluated\n"); return 1;}
	printf("  changed set          %6.1f atoms  (%.1f%% of N)\n",
	       sumC / done, 100.0 * sumC / done / N);
	printf("  full evaluation      %8.4f ms/move\n", tFull / done);
	printf("  delta evaluation     %8.4f ms/move   %6.2fx faster\n",
	       tDelta / done, tDelta > 0 ? tFull / tDelta : 0.0);
	printf("  worst relative error %.3e over %d moves\n", worst, done);
	printf("\n");
	return 0;
#endif
}
