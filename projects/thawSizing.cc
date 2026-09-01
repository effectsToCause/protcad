// thawSizing -- how big is the exact dielectric exemption for a real move?
//
// The frozen dielectric is exact provided every atom whose occupancy changed is
// thawed.  The cost of a delta evaluation is set by the size of that set, so it
// decides whether an incremental minimiser is worth 8x or 30x.
//
// tests/frozenDielectricTest.cc thaws around every atom of the moved residue,
// which is correct but pessimistic: a chi rotation leaves the backbone and the
// proximal sidechain atoms exactly where they were, and only the atoms that
// actually moved can perturb anybody's hydration shell.  This measures the
// difference, and checks that the tighter set is still exact.
//
//   thawSizing <in.pdb> [radius] [movesPerProtein] [seed]

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"
#include "energy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

int main(int argc, char** argv)
{
	if (argc < 2) {printf("thawSizing <in.pdb> [radius] [moves] [seed]\n"); return 1;}
	const double radius = (argc > 2) ? atof(argv[2]) : 6.5;
	const int nMove = (argc > 3) ? atoi(argv[3]) : 40;
	srand((argc > 4) ? (unsigned)atoi(argv[4]) : 1u);

#ifndef __CUDA__
	printf("thawSizing requires a CUDA build\n"); return 1;
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

	// Collect the flexible residues once.
	vector< pair<UInt,UInt> > flex;
	for (UInt c = 0; c < prot->getNumChains(); c++)
		for (UInt r = 0; r < prot->getNumResidues(c); r++)
		{
			vector< vector<double> > d = prot->getSidechainDihedrals(c, r);
			if (!d.empty() && !d[0].empty()) {flex.push_back(make_pair(c, r));}
		}
	if (flex.empty()) {printf("no flexible residues\n"); return 1;}

	printf("\n%s  %d atoms  %d flexible residues\n", argv[1], N, (int)flex.size());

	// Sweep radius: the occupancy kernel sums neighbours within r_i + 4.35 A,
	// so the exemption is provably sufficient at max(r_i) + 4.35 and the 8 A
	// used earlier was over-insurance.  Check where exactness actually starts.
	printf("  provable exemption radius 2*maxR + waterDiam = %.3f A\n",
	       prot->protDielectricInfluenceRadiusCU());
	const double rExact = prot->protDielectricInfluenceRadiusCU();
	const double radii[] = {6.0, 7.0, rExact - 0.5, rExact, rExact + 1.0};
	const int nR = (int)(sizeof(radii) / sizeof(radii[0]));
	vector<double> sumTight(nR, 0.0), worstErr(nR, 0.0);
	vector<int> exactTight(nR, 0);
	double sumMoved = 0, sumResidue = 0;
	int done = 0;

	for (int m = 0; m < nMove; m++)
	{
		const pair<UInt,UInt> pick = flex[rand() % flex.size()];
		vector< vector<double> > base = prot->getSidechainDihedrals(pick.first, pick.second);
		if (base.empty() || base[0].empty()) {continue;}

		vector< vector<double> > turned = base;
		const size_t k = rand() % turned[0].size();
		turned[0][k] = base[0][k] + 30.0 + 120.0 * (rand() / (double)RAND_MAX);

		// Pessimistic set, for comparison only: the whole residue, both ends.
		prot->protReleaseDielectricCU();
		prot->protFreezeDielectricCU();
		prot->protThawDielectricNearCU(pick.first, pick.second, radius, false);
		prot->setSidechainDihedralAngles(pick.first, pick.second, turned);
		sumResidue += prot->protThawDielectricNearCU(pick.first, pick.second, radius, true);
		prot->setSidechainDihedralAngles(pick.first, pick.second, base);
		prot->protReleaseDielectricCU();

		int nMoved = 0;
		for (int q = 0; q < nR; q++)
		{
			vector<double> ox, oy, oz;
			prot->getDeviceCoordsCU(ox, oy, oz);
			if (!prot->protFreezeDielectricCU()) {continue;}
			prot->setSidechainDihedralAngles(pick.first, pick.second, turned);

			const int nTight = prot->protThawDielectricForMoveCU(ox, oy, oz, radii[q], &nMoved);
			const double eTight = prot->protEnergyCU();
			// Reference at exactly these coordinates, with no intervening
			// dihedral round trip: setSidechainDihedralAngles is lossy at the
			// 1e-5 level and would swamp what we are trying to measure.
			prot->protReleaseDielectricCU();
			const double eCoupled = prot->protEnergyCU();
			prot->setSidechainDihedralAngles(pick.first, pick.second, base);

			const double rel = fabs(eTight - eCoupled) /
			                   (fabs(eCoupled) > 1.0 ? fabs(eCoupled) : 1.0);
			if (rel > worstErr[q]) {worstErr[q] = rel;}
			if (rel <= 1e-9) {exactTight[q]++;}
			sumTight[q] += nTight;
		}
		sumMoved += nMoved;
		done++;
	}

	if (!done) {printf("no moves evaluated\n"); return 1;}
	printf("  atoms actually displaced per move   %6.1f\n", sumMoved / done);
	printf("  thaw set, whole residue at %.1f A   %6.1f  (%.1f%% of N)\n",
	       radius, sumResidue / done, 100.0 * sumResidue / done / N);
	printf("  radius   thawed   %%of N   exact/%d   worst rel err\n", done);
	for (int q = 0; q < nR; q++)
		printf("  %5.2f   %6.1f  %5.1f%%   %5d      %.2e\n",
		       radii[q], sumTight[q] / done, 100.0 * sumTight[q] / done / N,
		       exactTight[q], worstErr[q]);
	printf("\n");
	return 0;
#endif
}
