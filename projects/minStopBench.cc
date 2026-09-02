// minStopBench -- what the descent curve actually looks like
//
// The old plateau counter failed because it measured a property of the proposal
// mechanism rather than of the trajectory: "trials since an improvement better
// than KT" is a hitting time, and a best-of-32 move keeps the per-trial success
// probability bounded away from zero indefinitely, so the required run of
// consecutive failures never occurs.  Replacing it needs a criterion on the
// trajectory, and choosing one needs the trajectory's shape.
//
// protMinReplicaCU records the shape as roughly log-linear -- about 59 kcal/mol
// per e-fold of budget on 1ubq -- which if it held forever would mean there is
// no plateau to detect at all, and any threshold on absolute improvement per
// fixed window is measuring a quantity that decays like 1/B by construction.
// This dumps best-so-far energy at geometric checkpoints so the gain per
// doubling can be read directly, including whether and where it decays.
//
// The move is protMinCU's: a random movable residue, best of 32 candidates,
// Metropolis acceptance, on the delta chain.

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"
#include "energy.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;
using clk = chrono::steady_clock;

int main(int argc, char** argv)
{
	const char* path = (argc > 1) ? argv[1] : "tests/data/1crn.pdb";
	const int trials = (argc > 2) ? atoi(argv[2]) : 4096;
	const int seed   = (argc > 3) ? atoi(argv[3]) : 20260902;
	const UInt nCand = getenv("PROTCAD_CANDIDATES") ? (UInt)atoi(getenv("PROTCAD_CANDIDATES")) : 32;
	const int anchorEvery = 64;

	PDBInterface* thePDB = new PDBInterface(path);
	protein* prot = static_cast<protein*>(
	                  thePDB->getEnsemblePointer()->getMoleculePointer(0));
	if (!prot) {printf("could not read %s\n", path); return 1;}
	prot->silenceMessages();
	prot->loadDeviceMemAll();

	vector< pair<UInt, UInt> > movable;
	for (UInt c = 0; c < prot->getNumChains(); c++)
		for (UInt r = 0; r < prot->getNumResidues(c); r++)
		{
			vector< vector<double> > d = prot->getSidechainDihedrals(c, r);
			if (!d.empty() && !d[0].empty()) {movable.push_back(make_pair(c, r));}
		}
	if (movable.empty()) {printf("no movable sidechains\n"); return 1;}

	// A sweep is one trial per movable sidechain, so the unit of work scales
	// with the structure rather than with an arbitrary constant.
	const int sweep = (int)movable.size();
	printf("minStopBench: %s, %d atoms, %d movable, sweep = %d trials\n",
	       path, (int)prot->getNumAtoms(), (int)movable.size(), sweep);

	srand(seed);
	double nbCurrent = 0.0, current = 0.0;
	if (!prot->minDeltaAnchorCU(nbCurrent, current)) {printf("no anchor\n"); return 1;}
	const double start = current;
	double best = current;
	int sinceAnchor = 0;

	printf("  %10s %10s %14s %14s %10s\n",
	       "trials", "sweeps", "best", "gain/doubling", "wall s");

	int nextCheck = sweep;
	double bestAtHalf = start;
	clk::time_point t0 = clk::now();

	for (int i = 1; i <= trials; i++)
	{
		const int m = rand() % (int)movable.size();
		const UInt c = movable[m].first, r = movable[m].second;

		if (sinceAnchor >= anchorEvery)
		{
			double fresh = 0.0;
			if (!prot->minDeltaAnchorCU(nbCurrent, fresh)) {break;}
			current = fresh; sinceAnchor = 0;
		}

		vector< vector<double> > entry = prot->getSidechainDihedrals(c, r);
		vector< vector<double> > conf;
		double nbBest = 0.0, torBest = 0.0;
		const double e = prot->bestSidechainCandidateDeltaCU(c, r, nCand, conf,
		                                                     nbCurrent, nbBest, torBest);
		if (e < 1E29 && prot->boltzmannEnergyCriteria(e - current))
		{
			prot->setSidechainDihedralAngles(c, r, conf);
			if (prot->minDeltaCommitCU(nbBest, nbCurrent))
			{	current = e; sinceAnchor++;
				if (current < best) {best = current;} }
			else {prot->setSidechainDihedralAngles(c, r, entry); break;}
		}

		if (i >= nextCheck)
		{
			const double wall = chrono::duration<double>(clk::now() - t0).count();
			printf("  %10d %10.1f %14.4f %14.4f %10.2f\n",
			       i, (double)i / sweep, best, bestAtHalf - best, wall);
			bestAtHalf = best;
			nextCheck *= 2;
		}
	}

	prot->protReleaseDielectricCU();
	printf("  start %.4f  best %.4f  total gain %.4f\n", start, best, start - best);
	return 0;
}
