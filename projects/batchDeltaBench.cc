// batchDeltaBench -- what the batched delta is worth per candidate batch
//
// The single-move delta measured 1.73-4.72x against a full evaluation.  That
// number does not carry over to the minimiser, because the minimiser's move is
// best-of-32 and its thaw set is therefore the union over 32 random sidechain
// conformations, not the narrow shell one move perturbs.  This measures the
// thing that actually decides whether wiring the delta into the sweep loop is
// worth doing: one bestSidechainCandidateDeltaCU call against one
// bestSidechainCandidateCU call, on the same residues, with the same batch
// size.
//
// The union changed set is reported alongside, because it is the mechanism.  If
// the union covers most of the structure there is no restricted sum to speak
// of and the delta can only lose, which is what a small protein should show.

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

static double secs(clk::time_point a, clk::time_point b)
{	return chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv)
{
	const char* path = (argc > 1) ? argv[1] : "tests/data/1crn.pdb";
	const UInt nCand = (argc > 2) ? (UInt)atoi(argv[2]) : 32;
	const int reps   = (argc > 3) ? atoi(argv[3]) : 10;
	// Both paths read their results back to the host, so each call already
	// synchronises and the timings need no explicit barrier.

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

	printf("batchDeltaBench: %s, %d atoms, %u candidates, %d reps\n",
	       path, (int)prot->getNumAtoms(), nCand, reps);

	const int nRes = (int)movable.size() < 8 ? (int)movable.size() : 8;
	const int stride = (int)movable.size() / nRes;

	double tDelta = 0.0, tFull = 0.0, fracSum = 0.0;
	int done = 0;

	for (int t = 0; t < nRes; t++)
	{
		const UInt c = movable[t * stride].first;
		const UInt r = movable[t * stride].second;
		vector< vector<double> > entry = prot->getSidechainDihedrals(c, r);

		// Full path.  Warm once, then time; the first call pays for context
		// allocation that no later call repeats.
		vector< vector<double> > conf;
		prot->protReleaseDielectricCU();
		prot->bestSidechainCandidateCU(c, r, nCand, conf);
		prot->setSidechainDihedralAngles(c, r, entry);
		clk::time_point a = clk::now();
		for (int i = 0; i < reps; i++)
		{
			prot->bestSidechainCandidateCU(c, r, nCand, conf);
			prot->setSidechainDihedralAngles(c, r, entry);
		}
		tFull += secs(a, clk::now());

		// Delta path.  The anchor and the freeze are charged to the caller once
		// per accepted move in the minimiser, not once per candidate batch, so
		// they sit outside the timed region -- but the freeze is re-established
		// per move, so it is timed separately below.
		energyBreakdown b0;
		if (!prot->protEnergyBreakdownCU(b0)) {continue;}
		const double nbEntry = b0.total - b0.torsion;
		if (!prot->protFreezeDielectricCU()) {continue;}

		double nbBest = 0.0, torBest = 0.0;
		prot->bestSidechainCandidateDeltaCU(c, r, nCand, conf, nbEntry, nbBest, torBest);
		prot->setSidechainDihedralAngles(c, r, entry);
		const double frac = (double)prot->protDielectricThawCountCU()
		                  / (double)prot->getNumAtoms();

		a = clk::now();
		for (int i = 0; i < reps; i++)
		{
			prot->bestSidechainCandidateDeltaCU(c, r, nCand, conf, nbEntry, nbBest, torBest);
			prot->setSidechainDihedralAngles(c, r, entry);
		}
		tDelta += secs(a, clk::now());

		prot->protReleaseDielectricCU();
		fracSum += frac;
		done++;
		printf("  chain %u res %-4u changed %5.1f%%  full %7.2f ms  delta %7.2f ms\n",
		       c, r, 100.0 * frac, 1000.0 * tFull / (done * reps),
		       1000.0 * tDelta / (done * reps));
	}

	if (!done) {printf("nothing measured\n"); return 1;}
	printf("\n  union changed set  %.1f%% of structure\n", 100.0 * fracSum / done);
	printf("  full  %8.3f ms/batch\n", 1000.0 * tFull / (done * reps));
	printf("  delta %8.3f ms/batch\n", 1000.0 * tDelta / (done * reps));
	printf("  speedup %.2fx\n", tFull / tDelta);
	return 0;
}
