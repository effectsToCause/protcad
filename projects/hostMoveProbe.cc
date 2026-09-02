// hostMoveProbe -- what does a trial cost with the energy model removed?
//
// nsys says the GPU is resident for about 19% of a minimisation on 1crn, and
// that the rest is split between transfer/launch overhead and plain host CPU.
// Deciding whether to make the coordinates device-resident needs those two
// separated: keeping coordinates on the device removes transfers, but if the
// time is going into host-side torsion-to-coordinate math then residency only
// helps if the move generation moves with it.
//
// So this runs the exact move-generation path bestSidechainCandidateDeltaCU
// uses -- 32 random sidechain conformations, each applied to the host tree and
// pushed to the device -- with no energy evaluation anywhere, and separately
// times the per-trial full-structure bookkeeping that surrounds it.

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"

#include <chrono>
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
	const int trials = (argc > 2) ? atoi(argv[2]) : 512;
	const UInt K = 32;

	PDBInterface* thePDB = new PDBInterface(path);
	protein* prot = static_cast<protein*>(
	                  thePDB->getEnsemblePointer()->getMoleculePointer(0));
	if (!prot) {printf("could not read %s\n", path); return 1;}
	prot->silenceMessages();
	prot->loadDeviceMemAll();
	const int N = prot->updateDeviceCoords();

	vector< pair<UInt, UInt> > movable;
	for (UInt c = 0; c < prot->getNumChains(); c++)
		for (UInt r = 0; r < prot->getNumResidues(c); r++)
		{
			vector< vector<double> > d = prot->getSidechainDihedrals(c, r);
			if (!d.empty() && !d[0].empty()) {movable.push_back(make_pair(c, r));}
		}
	if (movable.empty()) {printf("no movable sidechains\n"); return 1;}

	srand(20260902);
	double tUpdate = 0, tSnap = 0, tAlloc = 0, tGen = 0, tApply = 0, tPush = 0;
	const clk::time_point tAll0 = clk::now();

	for (int t = 0; t < trials; t++)
	{
		const UInt c = movable[rand() % movable.size()].first;
		const UInt r = movable[rand() % movable.size()].second % prot->getNumResidues(c);

		clk::time_point a = clk::now();
		prot->updateDeviceCoords();                       // full N refresh, per trial
		clk::time_point b = clk::now(); tUpdate += secs(a, b);

		vector<double> oldX(N, 0.0), oldY(N, 0.0), oldZ(N, 0.0); // pre-move snapshot
		clk::time_point d0 = clk::now(); tSnap += secs(b, d0);

		vector<double> bx((size_t)K * N, 0.0),            // K*N zeroed, per trial
		               by((size_t)K * N, 0.0),
		               bz((size_t)K * N, 0.0);
		clk::time_point d1 = clk::now(); tAlloc += secs(d0, d1);

		vector<int> resAtoms;
		prot->residueAtomIndicesCU(c, r, resAtoms);
		if (resAtoms.empty()) {continue;}
		vector< vector<double> > entry = prot->getSidechainDihedrals(c, r);

		for (UInt k = 0; k < K; k++)
		{
			clk::time_point g0 = clk::now();
			vector< vector<double> > conf = prot->randContinuousSidechainConformation(c, r);
			clk::time_point g1 = clk::now(); tGen += secs(g0, g1);

			prot->setSidechainDihedralAngles(c, r, conf);
			clk::time_point g2 = clk::now(); tApply += secs(g1, g2);

			prot->refreshDeviceCoords(resAtoms);
			clk::time_point g3 = clk::now(); tPush += secs(g2, g3);
		}
		prot->setSidechainDihedralAngles(c, r, entry);
		prot->refreshDeviceCoords(resAtoms);
	}
	const double wall = secs(tAll0, clk::now());

	printf("hostMoveProbe: %s, N = %d atoms, %d trials, K = %u, NO energy evaluation\n",
	       path, N, trials, K);
	printf("  %-34s %10s %12s\n", "phase", "total s", "us/trial");
	const char* nm[6] = {"updateDeviceCoords (full N)",
	                     "pre-move snapshot (3 x N copy)",
	                     "candidate arrays (3 x K*N zero)",
	                     "randContinuousSidechainConf x K",
	                     "setSidechainDihedralAngles x K",
	                     "refreshDeviceCoords x K"};
	const double v[6] = {tUpdate, tSnap, tAlloc, tGen, tApply, tPush};
	for (int i = 0; i < 6; i++)
	{	printf("  %-34s %10.3f %12.1f\n", nm[i], v[i], v[i] * 1e6 / trials); }
	printf("  %-34s %10.3f %12.1f\n", "TOTAL (host move generation)", wall, wall * 1e6 / trials);
	return 0;
}
