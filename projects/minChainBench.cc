// minChainBench -- the delta chain over a fixed number of accepted moves
//
// protMin cannot answer this.  Its plateau counter does not survive a batched
// move -- steepest descent over 32 candidates almost always finds some
// improvement, so the counter resets on nearly every trial and the loop runs
// far past where a single-candidate move would have stopped.  Timing protMin
// end to end therefore measures the termination criterion, not the evaluation,
// and the two arms would not even do the same amount of work.
//
// So the move loop is reproduced here with a fixed move count instead, driving
// exactly the path protMinCU now drives: anchor, batch delta, commit, and a
// periodic re-anchor.  Both arms take the same residues in the same order from
// the same seed, so the only difference is how the candidate energies are
// obtained.
//
// The drift readout is the point of the exercise as much as the timing.  The
// delta identity is exact in exact arithmetic but accumulates rounding across a
// chain of accepted moves, and the only way that fails is quietly.  At every
// re-anchor the carried energy is compared against a fresh coupled evaluation
// and the gap is reported.

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
	const int moves  = (argc > 2) ? atoi(argv[2]) : 200;
	const UInt nCand = 32;
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

	printf("minChainBench: %s, %d atoms, %d moves of %u candidates\n",
	       path, (int)prot->getNumAtoms(), moves, nCand);

	// Same residue order for both arms.
	vector<int> order(moves);
	srand(20260902);
	for (int i = 0; i < moves; i++) {order[i] = rand() % (int)movable.size();}

	// ---- full-evaluation arm -------------------------------------------
	prot->protReleaseDielectricCU();
	const double e0 = prot->protEnergyCU();
	vector< vector< vector<double> > > fullPath(moves);
	double eFull = e0;
	clk::time_point a = clk::now();
	for (int i = 0; i < moves; i++)
	{
		const UInt c = movable[order[i]].first, r = movable[order[i]].second;
		vector< vector<double> > conf;
		const double e = prot->bestSidechainCandidateCU(c, r, nCand, conf);
		if (e > 1E29) {continue;}
		prot->setSidechainDihedralAngles(c, r, conf);
		fullPath[i] = conf;
		eFull = e;
	}
	const double tFull = secs(a, clk::now());
	printf("  full   %8.3f s   %.4f -> %.4f kcal/mol\n", tFull, e0, eFull);

	// Rewind by replaying the inverse is not possible, so reload.
	PDBInterface* pdb2 = new PDBInterface(path);
	protein* p2 = static_cast<protein*>(
	                pdb2->getEnsemblePointer()->getMoleculePointer(0));
	p2->loadDeviceMemAll();

	// ---- delta-chain arm -----------------------------------------------
	p2->protReleaseDielectricCU();
	double nbCurrent = 0.0, carried = 0.0, worstDrift = 0.0;
	int sinceAnchor = 0, anchors = 0, committed = 0;
	if (!p2->minDeltaAnchorCU(nbCurrent, carried)) {printf("no anchor\n"); return 1;}
	double eDelta = carried;
	a = clk::now();
	for (int i = 0; i < moves; i++)
	{
		const UInt c = movable[order[i]].first, r = movable[order[i]].second;
		if (sinceAnchor >= anchorEvery)
		{
			double fresh = 0.0;
			if (!p2->minDeltaAnchorCU(nbCurrent, fresh)) {break;}
			const double drift = fabs(fresh - carried);
			if (drift > worstDrift) {worstDrift = drift;}
			carried = fresh; sinceAnchor = 0; anchors++;
		}
		vector< vector<double> > conf;
		double nbBest = 0.0, torBest = 0.0;
		const double e = p2->bestSidechainCandidateDeltaCU(c, r, nCand, conf,
		                                                   nbCurrent, nbBest, torBest);
		if (e > 1E29) {continue;}
		p2->setSidechainDihedralAngles(c, r, conf);
		if (!p2->minDeltaCommitCU(nbBest, nbCurrent)) {break;}
		carried = e; eDelta = e; sinceAnchor++; committed++;
	}
	const double tDelta = secs(a, clk::now());

	// Final audit against the coupled model, which is the only number that is
	// not itself produced by the machinery under test.
	p2->protReleaseDielectricCU();
	const double eFinal = p2->protEnergyCU();
	printf("  delta  %8.3f s   %.4f -> %.4f kcal/mol\n", tDelta, e0, eDelta);
	printf("  speedup %.2fx over %d committed moves, %d re-anchors\n",
	       tFull / tDelta, committed, anchors);
	printf("  worst re-anchor drift %.3e kcal/mol\n", worstDrift);
	printf("  carried %.6f vs coupled %.6f  (gap %.3e)\n",
	       eDelta, eFinal, fabs(eDelta - eFinal));
	return 0;
}
