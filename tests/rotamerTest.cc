// Round-trip test for on-device sidechain generation.
//
// energyComputeRotamerBatch generates candidate conformations on the GPU by
// applying chi deltas to the resident coordinates. That is only useful if it
// reproduces residue::setChiByDelta exactly, so this test applies the same
// deltas both ways and compares both the coordinates and the energies.
//
// Coordinates are the primary assertion. The energy is checked too, but it is a
// badly conditioned probe of a transform: its r^-12 term amplifies a 1e-6 A
// coordinate difference into a visible energy difference on clashed
// conformations, which is a property of the potential, not a transform error.
//
// Neither precision agrees bitwise with the host, and neither can. CUDA's double
// sin/cos differ from glibc's by up to 1 ulp, so the two rotation matrices are
// already different before any coordinate is touched. Measured agreement is
// ~6e-6 A in FP32 and ~1e-14 A in FP64 -- rounding in both cases.
#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"
#include "residue.h"
#include "atomIterator.h"
#include "energy.h"
#include <cstdio>
#include <cmath>
#include <vector>
using namespace std;

static int failures = 0;
static unsigned rs = 12345u;
static double rnd() {rs = rs * 1664525u + 1013904223u; return double(rs >> 8) / double(1u << 24);}

#ifdef PROTCAD_ENERGY_FP64
// Not bitwise, and cannot be: CUDA's double cos differs from glibc's by 1 ulp
// on some inputs, so the rotation matrices are not identical to begin with.
// These tolerances are a few ulp of the coordinate magnitude, which is what a
// correct transform is entitled to.
static const double kTol = 1e-10;
static const double kCoordTol = 1e-11;
static const char* kPrec = "FP64";
#else
static const double kTol = 1e-4;
static const double kCoordTol = 1e-4;
static const char* kPrec = "FP32";
#endif

int main(int argc, char** argv)
{
	if (argc < 2) {printf("usage: rotamerTest <file.pdb>\n"); return 2;}
	PDBInterface* thePDB = new PDBInterface(argv[1]);
	ensemble* theEnsemble = thePDB->getEnsemblePointer();
	molecule* pMol = theEnsemble->getMoleculePointer(0);
	protein* prot = static_cast<protein*>(pMol);

	int nG = prot->buildRotationGroups();
	printf("%s  rotation groups: %d\n", kPrec, nG);
	if (nG == 0) {printf("no chis found\n"); return 1;}

	// Snapshot every atom so the host arm can be undone exactly.
	vector<dblVec> save;
	for (atomIterator a(prot); !(a.last()); a++) {save.push_back(a.getAtomPointer()->getCoords());}

	const int K = 16;
	int tested = 0;
	double worst = 0.0, worstCoord = 0.0;
	const UInt nAtoms = prot->getNumAtoms();

	for (UInt c = 0; c < prot->getNumChains() && tested < 12; c++)
	{
		for (UInt r = 0; r < prot->getNumResidues(c) && tested < 12; r++)
		{
			int begin = 0, count = 0;
			if (!prot->getRotationGroupRange(c, r, begin, count)) {continue;}

			vector<double> ang((size_t)K * count);
			for (size_t i = 0; i < ang.size(); i++) {ang[i] = (rnd() - 0.5) * 240.0;}

			// Device arm: all K candidates in one launch from resident coords.
			prot->syncDeviceCoords();
			vector<double> dev(K, 0.0);
			if (prot->energyRotamerBatch(K, begin, count, &ang[0], &dev[0]) != 0)
			{	printf("  FAIL device batch c%u r%u\n", c, r); failures++; continue; }

			// Host arm: same deltas through the atom tree, one candidate at a time.
			for (int k = 0; k < K; k++)
			{
				residue* pRes = prot->getChain(c)->getResidue(r);
				for (int i = 0; i < count; i++)
				{	pRes->setChiByDelta(0, (UInt)i, ang[(size_t)k * count + i]); }
				double hostE = prot->protEnergyCU();

				// Compare coordinates first. This is the direct test of the
				// transform; the energy is a downstream consumer whose r^-12
				// term amplifies small coordinate differences enormously on
				// clashed conformations.
				vector<double> dx(nAtoms), dy(nAtoms), dz(nAtoms);
				if (prot->getBatchCoords(k, &dx[0], &dy[0], &dz[0]) == 0)
				{
					UInt ai = 0; double cworst = 0.0;
					for (atomIterator a(prot); !(a.last()); a++, ai++)
					{
						dblVec hc = a.getAtomPointer()->getCoords();
						double e0 = fabs(hc[0] - dx[ai]), e1 = fabs(hc[1] - dy[ai]), e2 = fabs(hc[2] - dz[ai]);
						double m = e0 > e1 ? e0 : e1; if (e2 > m) {m = e2;}
						if (m > cworst) {cworst = m;}
					}
					if (cworst > worstCoord) {worstCoord = cworst;}
					if (cworst > kCoordTol)
					{	printf("  FAIL coords c%u r%u k%d max dev %.3e A\n", c, r, k, cworst);
						failures++; }
				}

				double d = fabs(hostE - dev[k]);
				double rel = d / (fabs(hostE) > 1.0 ? fabs(hostE) : 1.0);
				if (rel > worst) {worst = rel;}
				if (rel > kTol)
				{	printf("  FAIL c%u r%u k%d host %.12f dev %.12f rel %.3e\n",
					       c, r, k, hostE, dev[k], rel); failures++; }

				// Restore exactly rather than by inverse rotation, so the next
				// candidate starts from bit-identical coordinates.
				UInt idx = 0;
				for (atomIterator a(prot); !(a.last()); a++, idx++)
				{	a.getAtomPointer()->setCoords(save[idx]); }
			}
			tested++;
		}
	}

	printf("worst coordinate deviation %.3e A (tol %.0e)\n", worstCoord, kCoordTol);
	printf("residues tested %d, candidates %d, worst rel %.3e (tol %.0e)\n",
	       tested, tested * K, worst, kTol);
	printf("\nRESULT: %s (%d failures)\n", failures ? "FAILURES" : "all checks passed", failures);
	return failures ? 1 : 0;
}
