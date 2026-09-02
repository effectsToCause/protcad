// minStopProbe -- does the stopping rule actually adapt?
//
// The claim minStopCU has to earn is that a structure which is already close to
// its minimum exits cheaply while one that still has far to fall keeps going.
// Testing that through a PDB round trip does not work: writing and re-reading a
// minimised structure reintroduces clashes and the starting energy comes back
// three orders of magnitude higher, so the second run is not given a good
// structure at all.  This runs protMinCU twice in the same process, so the
// second call starts from exactly the conformation the first one left.

#include "ensemble.h"
#include "PDBInterface.h"
#include "protein.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace std;
using clk = chrono::steady_clock;

int main(int argc, char** argv)
{
	const char* path = (argc > 1) ? argv[1] : "tests/data/1crn.pdb";
	const int passes = (argc > 2) ? atoi(argv[2]) : 3;

	PDBInterface* thePDB = new PDBInterface(path);
	protein* prot = static_cast<protein*>(
	                  thePDB->getEnsemblePointer()->getMoleculePointer(0));
	if (!prot) {printf("could not read %s\n", path); return 1;}
	prot->silenceMessages();
	prot->loadDeviceMemAll();

	printf("minStopProbe: %s, %d atoms\n", path, (int)prot->getNumAtoms());
	printf("  %5s %14s %14s %10s\n", "pass", "start", "end", "wall s");
	for (int p = 1; p <= passes; p++)
	{
		const double e0 = prot->protEnergyCU();
		const clk::time_point t = clk::now();
		prot->protMinCU(false);
		const double wall = chrono::duration<double>(clk::now() - t).count();
		printf("  %5d %14.3f %14.3f %10.2f\n", p, e0, prot->protEnergyCU(), wall);
		fflush(stdout);
	}
	return 0;
}
