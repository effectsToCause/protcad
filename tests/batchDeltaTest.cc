// batchDeltaTest -- the batched delta must rank candidates like a full pass
//
// bestSidechainCandidateDeltaCU replaces N full energy evaluations of a
// candidate batch with one restricted sum per candidate, on the identity
//
//     E_k = E_entry - P(old) + P_k(new) + torsion_k
//
// where P is the sum over the changed set.  That identity is exact in exact
// arithmetic, so the test is not "is the delta close enough to be useful" but
// "does it reproduce the number the full path would have produced".  Anything
// beyond floating-point association is a bug in the changed set, the tile list,
// the double-count subtraction or the torsion substitution, and each of those
// fails in a way that looks like a small plausible energy shift rather than
// like a crash.
//
// The reference is the full evaluator run with the same held dielectric and the
// same thaw set still installed.  Comparing against the coupled model instead
// would fold in the frozen-field approximation, which is a separate and already
// tested question, and would set a tolerance loose enough to hide real errors.
//
// Every candidate is checked, not only the winner.  A delta that gets the
// minimum right by luck while mis-ranking the rest would pass a winner-only
// test and then quietly degrade the search.
//
// The union changed set is also reported as a fraction of the structure.  A
// batch of random sidechain conformations perturbs the whole rotamer sweep, so
// this set is inherently wider than a single move's, and how much wider decides
// whether the batched delta is worth anything at all.

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"
#include "energy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

static int failures = 0;

// Association noise only.  The single-move delta lands near 1.6e-7 relative on
// totals of a few hundred kcal/mol, and the batch sums the same terms in the
// same ascending order.  Measured: 2.4e-7 on 1crn rising to 1.5e-6 on 1ake,
// because the delta is a difference of two large restricted sums and the
// cancellation gets worse as the changed set grows.  The tolerance is set above
// that trend rather than at it, so the test fails on a broken changed set
// rather than on a larger structure.
static const double kRelTol = 5e-6;

static void check(const char* what, int k, double got, double want)
{
	const double scale = (fabs(want) > 1.0 ? fabs(want) : 1.0);
	const double rel = fabs(got - want) / scale;
	if (rel <= kRelTol) {return;}
	printf("  FAIL  %s candidate %d: delta %.9f  full %.9f  (rel %.2e)\n",
	       what, k, got, want, rel);
	failures++;
}

int main(int argc, char** argv)
{
	const char* path = (argc > 1) ? argv[1] : "tests/data/1crn.pdb";
	PDBInterface* thePDB = new PDBInterface(path);
	ensemble* theEnsemble = thePDB->getEnsemblePointer();
	molecule* pMol = theEnsemble->getMoleculePointer(0);
	protein* prot = static_cast<protein*>(pMol);
	if (!prot) {printf("could not read %s\n", path); return 1;}

	prot->silenceMessages();
	prot->loadDeviceMemAll();
	printf("batchDeltaTest: %s, %d atoms\n", path, (int)prot->getNumAtoms());

	// Residues with something to rotate.  Glycine and alanine have no
	// candidates to rank, so a batch there is not a test of anything.
	vector< pair<UInt, UInt> > movable;
	for (UInt c = 0; c < prot->getNumChains(); c++)
	{
		for (UInt r = 0; r < prot->getNumResidues(c); r++)
		{
			vector< vector<double> > d = prot->getSidechainDihedrals(c, r);
			if (!d.empty() && !d[0].empty()) {movable.push_back(make_pair(c, r));}
		}
	}
	if (movable.empty()) {printf("no movable sidechains\n"); return 1;}
	printf("  %d movable sidechains\n", (int)movable.size());

	const UInt nCand = 32;
	const int nTrials = (int)movable.size() < 12 ? (int)movable.size() : 12;
	const int stride = (int)movable.size() / nTrials;

	double worstRel = 0.0;
	double changedFracSum = 0.0;
	int trials = 0;

	for (int t = 0; t < nTrials; t++)
	{
		const UInt c = movable[t * stride].first;
		const UInt r = movable[t * stride].second;
		vector< vector<double> > entry = prot->getSidechainDihedrals(c, r);

		// Anchor at the entry conformation with the coupled model, split so the
		// torsion term is replaced wholesale rather than differenced, and so
		// the per-torsion cache the delta substitutes into is primed.
		prot->protReleaseDielectricCU();
		energyBreakdown b0;
		if (!prot->protEnergyBreakdownCU(b0)) {continue;}
		const double nbEntry = b0.total - b0.torsion;

		if (!prot->protFreezeDielectricCU()) {continue;}

		double nbBest = 0.0, torBest = 0.0;
		vector<double> deltaE;
		vector< vector< vector<double> > > confs;
		const double best = prot->bestSidechainCandidateDeltaCU(c, r, nCand, entry,
		                                                        nbEntry, nbBest,
		                                                        torBest, &deltaE,
		                                                        &confs);
		// entry was overwritten with the winning conformation by the call above.
		vector< vector<double> > winner = entry;
		if (best > 1E29 || deltaE.size() != nCand)
		{	printf("  skip  chain %u residue %u: no batch delta\n", c, r);
			prot->protReleaseDielectricCU();
			continue; }

		const int nChanged = prot->protDielectricThawCountCU();
		changedFracSum += (double)nChanged / (double)prot->getNumAtoms();

		// Reference: the same full evaluator, with the held field and the thaw
		// set the delta used still in place.
		for (UInt k = 0; k < nCand; k++)
		{
			// Reach every candidate from the entry conformation, which is how
			// the minimiser adopts the winner.  Chaining one candidate off the
			// last leaves each rotation to be measured from a geometry that has
			// already been rotated 30 times, and the drift that accumulates is
			// the reference's, not the delta's.
			prot->setSidechainDihedralAngles(c, r, entry);
			prot->setSidechainDihedralAngles(c, r, confs[k]);
			const double full = prot->protEnergyCU();
			check("batch delta", (int)k, deltaE[k], full);
			const double scale = (fabs(full) > 1.0 ? fabs(full) : 1.0);
			const double rel = fabs(deltaE[k] - full) / scale;
			if (rel > worstRel) {worstRel = rel;}
		}

		// The winner the delta chose must be the winner the full pass would
		// choose.  Ranking is what the minimiser consumes, not the energies.
		UInt argDelta = 0;
		for (UInt k = 1; k < nCand; k++) {if (deltaE[k] < deltaE[argDelta]) {argDelta = k;}}
		UInt argFull = 0; double bestFull = 1E30;
		for (UInt k = 0; k < nCand; k++)
		{
			prot->setSidechainDihedralAngles(c, r, entry);
			prot->setSidechainDihedralAngles(c, r, confs[k]);
			const double full = prot->protEnergyCU();
			if (k == 0 || full < bestFull) {bestFull = full; argFull = k;}
		}
		if (argDelta != argFull)
		{
			// Only a real failure if the two are not a numerical tie.
			const double spread = fabs(deltaE[argDelta] - deltaE[argFull]);
			if (spread > kRelTol * (fabs(bestFull) > 1.0 ? fabs(bestFull) : 1.0))
			{	printf("  FAIL  chain %u residue %u: delta picked %u, full picked %u\n",
				       c, r, argDelta, argFull);
				failures++; }
		}
		if (fabs(best - deltaE[argDelta]) > 1e-12)
		{	printf("  FAIL  chain %u residue %u: returned best %.9f != min %.9f\n",
			       c, r, best, deltaE[argDelta]);
			failures++; }

		// The carried anchor has to be self-consistent, or a chain of these
		// moves drifts even when every individual move is right.
		if (fabs((nbBest + torBest) - best) > 1e-9 * (fabs(best) > 1.0 ? fabs(best) : 1.0))
		{	printf("  FAIL  chain %u residue %u: nbBest + torBest %.9f != best %.9f\n",
			       c, r, nbBest + torBest, best);
			failures++; }

		prot->setSidechainDihedralAngles(c, r, winner);
		prot->protReleaseDielectricCU();
		trials++;
	}

	prot->protReleaseDielectricCU();
	printf("  %d residues tested, worst relative error %.2e\n", trials, worstRel);
	if (trials > 0)
	{	printf("  union changed set averages %.1f%% of the structure\n",
		       100.0 * changedFracSum / trials); }

	if (failures) {printf("batchDeltaTest: %d failure(s)\n", failures); return 1;}
	printf("batchDeltaTest: ok\n");
	return 0;
}
