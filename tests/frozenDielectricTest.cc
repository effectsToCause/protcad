// frozenDielectricTest -- the frozen dielectric must be a strict generalisation
//
// The packing acceleration in docs/packing-search.md rests on holding the
// occupancy field fixed while exempting a near shell around the moved
// sidechain.  That is only trustworthy if the frozen path is the coupled path
// plus a deliberate approximation, and nothing else: no reordering drift, no
// stale sort binding, no silent change to the terms that are supposed to be
// untouched.
//
// Three identities pin that down, all required to hold exactly, not
// approximately:
//
//   1. Freeze at a conformation, evaluate at that same conformation.  The
//      snapshot is by construction the field the coupled model would compute,
//      so the energy must be unchanged.
//   2. Freeze, then thaw every atom.  Nothing is held, so the frozen path must
//      reproduce the coupled path at any conformation, including one the
//      snapshot was not taken at.
//   3. Release.  The coupled model must come back exactly.
//
// Identity 2 is the load-bearing one.  It exercises the snapshot buffer, the
// original-order mapping and the thaw mask on a structure whose spatial sort
// has been rebuilt since the freeze, which is precisely where a sorted-order
// snapshot would go wrong and would otherwise show up only as a small,
// plausible-looking energy shift.
//
// Finally the approximation itself is reported, not asserted: freezing with no
// exemption at a moved conformation should be close to the coupled answer but
// is not required to equal it.  That is the error the design is trading for
// speed, and it is printed so a change in its size is visible.

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"
#include "energy.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace std;

static int failures = 0;

// Exact in exact arithmetic, which is the standard energyComputeBatch already
// documents.  A held occupancy is not bit-identical to a freshly computed one
// even for an atom whose neighbourhood did not change, because the spatial sort
// is rebuilt after a move and the neighbour sum then accumulates in a different
// order.  The discrepancy is pure floating-point association, around 1e-11
// relative, and it does not grow with the size of the held region -- which is
// the property that matters and that a bit-exact test would obscure by simply
// failing.
static void requireClose(const char* what, double got, double want)
{
	const double tol = 1e-9 * (fabs(want) > 1.0 ? fabs(want) : 1.0);
	const double diff = got - want;
	if (fabs(diff) <= tol)
	{	printf("  ok    %-46s %.6f  (rel %.1e)\n", what, got,
		       fabs(diff) / (fabs(want) > 1.0 ? fabs(want) : 1.0));
		return; }
	printf("  FAIL  %-46s got %.10f want %.10f (diff %.3e)\n",
	       what, got, want, diff);
	failures++;
}

static void requireExact(const char* what, double got, double want)
{
	// Bit-exact: same kernel, same inputs, same reduction order.  Anything else
	// means the frozen path is not passing the identical occupancy field
	// through, and a tolerance would hide exactly the bug this is looking for.
	if (got == want) {printf("  ok    %-46s %.10f\n", what, got); return;}
	printf("  FAIL  %-46s got %.10f want %.10f (diff %.3e)\n",
	       what, got, want, got - want);
	failures++;
}

int main(int argc, char** argv)
{
	if (argc < 2) {printf("frozenDielectricTest <in.pdb>\n"); return 1;}

#ifndef __CUDA__
	printf("frozenDielectricTest requires a CUDA build\n");
	return 1;
#else
	PDBInterface* thePDB = new PDBInterface(argv[1]);
	ensemble* theEnsemble = thePDB->getEnsemblePointer();
	molecule* pMol = theEnsemble->getMoleculePointer(0);
	protein* prot = static_cast<protein*>(pMol);

	residue::setElectroSolvationScaleFactor(1.0);
	residue::setHydroSolvationScaleFactor(1.0);
	residue::setPolarizableElec(false);
	amberElec::setScaleFactor(1.0);
	amberVDW::setScaleFactor(1.0);
	residue::setTemperature(300);
	residue::setEntropyFactor(0.0);

	prot->loadDeviceMemAll();

	const double eRef = prot->protEnergyCU();
	printf("\ncoupled reference %.10f\n\n", eRef);

	printf("identity 1: freeze at the reference conformation\n");
	if (!prot->protFreezeDielectricCU()) {printf("  FAIL  freeze failed\n"); return 1;}
	requireExact("frozen, nothing thawed, same coords", prot->protEnergyCU(), eRef);

	printf("\nidentity 2: freeze, thaw everything, then move\n");
	if (!prot->protThawDielectricAllCU()) {printf("  FAIL  thaw-all failed\n"); return 1;}
	requireExact("fully thawed, same coords", prot->protEnergyCU(), eRef);

	// Find a residue with a rotatable chi and move it well away from where the
	// snapshot was taken, so the sort is rebuilt and the true field differs.
	UInt mc = 0, mr = 0;
	bool moved = false;
	vector< vector<double> > base;
	for (UInt c = 0; c < prot->getNumChains() && !moved; c++)
	{
		for (UInt r = 0; r < prot->getNumResidues(c) && !moved; r++)
		{
			base = prot->getSidechainDihedrals(c, r);
			if (base.empty() || base[0].empty()) {continue;}
			mc = c; mr = r; moved = true;
		}
	}
	if (!moved) {printf("  FAIL  no rotatable sidechain in %s\n", argv[1]); return 1;}

	vector< vector<double> > turned = base;
	for (size_t k = 0; k < turned[0].size(); k++) {turned[0][k] = base[0][k] + 120.0;}
	prot->setSidechainDihedralAngles(mc, mr, turned);
	printf("  moved chain %u residue %u by 120 deg on every chi\n", mc, mr);

	const double eMovedThawed = prot->protEnergyCU();

	prot->protReleaseDielectricCU();
	const double eMovedCoupled = prot->protEnergyCU();
	requireExact("fully thawed at moved coords == coupled", eMovedThawed, eMovedCoupled);

	// Restoring the dihedrals does not restore the coordinates bit-exactly --
	// measuring an angle and reapplying it is a lossy round trip, worth about
	// 1e-5 kcal/mol here -- so the reference for identity 3 is a fresh coupled
	// evaluation at the current coordinates, not the energy from before the
	// move.  Comparing against eRef would be testing the dihedral round trip,
	// not the dielectric.
	printf("\nidentity 3: release restores the coupled model\n");
	prot->setSidechainDihedralAngles(mc, mr, base);
	const double eBackCoupled = prot->protEnergyCU();
	prot->protFreezeDielectricCU();
	prot->protThawDielectricAllCU();
	const double eBackThawed = prot->protEnergyCU();
	prot->protReleaseDielectricCU();
	requireExact("released == fully thawed, restored coords", eBackThawed, eBackCoupled);
	printf("  note  dihedral round trip moved the coupled energy by %+.3e\n",
	       eBackCoupled - eRef);

	// What the near-shell exemption actually buys.  An occupancy is a sum over
	// the atoms inside a hydration shell of radius (r_i + effectiveWaterDiameter),
	// so a move can only perturb atoms within about 6.4 A of an atom's old or
	// new position.  Cover that and the held far field is not an approximation
	// at all: the atoms outside kept the value the coupled model would have
	// computed for them, because nothing about their neighbourhood changed.
	printf("\nnear-shell exemption: error against the coupled model\n");
	prot->protFreezeDielectricCU();
	prot->setSidechainDihedralAngles(mc, mr, turned);

	const double eHeldAll = prot->protEnergyCU();
	printf("  frozen, no exemption            %.6f   error %+.6f\n",
	       eHeldAll, eHeldAll - eMovedCoupled);

	const double radii[] = {4.0, 6.0, 7.0, 8.0, 10.0};
	for (int i = 0; i < 5; i++)
	{
		// The thaw set is the union over both conformations, because an atom
		// matters if the move brings a neighbour into its shell or takes one
		// out of it.  Built from the reference geometry alone it would be
		// exact only for small moves.
		prot->setSidechainDihedralAngles(mc, mr, base);
		const int nBefore = prot->protThawDielectricNearCU(mc, mr, radii[i], false);
		prot->setSidechainDihedralAngles(mc, mr, turned);
		const int n = prot->protThawDielectricNearCU(mc, mr, radii[i], true);
		const double e = prot->protEnergyCU();
		printf("  %5.1f A, union %4d atoms (%d from one conformation)   %.6f   error %+.3e%s\n",
		       radii[i], n, nBefore, e, e - eMovedCoupled,
		       (fabs(e - eMovedCoupled) <= 1e-9 * fabs(eMovedCoupled)) ? "   exact" : "");
		if (radii[i] >= 8.0) {requireClose("union thaw at >= 8 A recovers coupled", e, eMovedCoupled);}
	}

	prot->protReleaseDielectricCU();
	prot->setSidechainDihedralAngles(mc, mr, base);

	printf("\n%s\n", failures ? "FAILED" : "all identities hold");
	return failures ? 1 : 0;
#endif
}
