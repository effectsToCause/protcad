// The total energy of a molecule is a function of its internal geometry alone,
// so translating or rotating a structure rigidly must leave every atom's
// relative position, and therefore the energy, unchanged.
//
// This is not a hypothetical.  Residues are rebuilt from templates at load time
// and their sidechain dihedrals are re-measured and restored.  Atoms absent from
// the input file keep raw template coordinates, so a dihedral measured across a
// mixture of file-derived and template-derived atoms compares two unrelated
// frames.  Its value tracked the absolute position of the molecule, which moved
// the hydroxyl hydrogens of Ser, Thr and Tyr by up to 1.8 A and shifted the
// total energy of 1crn by 10.4 kcal/mol under a rigid 60 A translation.  The
// error entered through the occupancy-derived dielectric, so it fell squarely on
// the electrostatic and solvation terms that dominate stability predictions.
//
// Both transforms below are exact in the three-decimal PDB coordinate format, so
// any surviving difference is a real defect and not a rounding artefact.

#include "PDBInterface.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Rewrites the coordinate columns of every ATOM/HETATM record.  mode 0 shifts x
// by 60 A, mode 1 rotates 90 degrees about z, which maps (x,y,z) to (-y,x,z).
bool writeTransformed(const std::string& in, const std::string& out, int mode)
{
	std::ifstream src(in.c_str());
	if (!src) { std::cout << "cannot read " << in << std::endl; return false; }
	std::ofstream dst(out.c_str());
	if (!dst) { std::cout << "cannot write " << out << std::endl; return false; }

	std::string line;
	while (std::getline(src, line))
	{
		if (line.size() >= 54 &&
		    (line.compare(0, 6, "ATOM  ") == 0 || line.compare(0, 6, "HETATM") == 0))
		{
			double x = atof(line.substr(30, 8).c_str());
			double y = atof(line.substr(38, 8).c_str());
			double z = atof(line.substr(46, 8).c_str());
			double nx = x, ny = y, nz = z;
			if (mode == 0) { nx = x + 60.0; }
			else           { nx = -y; ny = x; }

			std::ostringstream o;
			o << std::fixed << std::setprecision(3);
			o << std::setw(8) << nx << std::setw(8) << ny << std::setw(8) << nz;
			line = line.substr(0, 30) + o.str() + line.substr(54);
		}
		dst << line << "\n";
	}
	return true;
}

struct Snapshot
{
	std::vector<double> x, y, z;
	double energy;
	unsigned int clashes;
};

bool load(const std::string& file, Snapshot& s)
{
	PDBInterface* pdb = new PDBInterface(file);
	ensemble* ens = pdb->getEnsemblePointer();
	if (!ens) { std::cout << "no ensemble in " << file << std::endl; return false; }
	protein* prot = static_cast<protein*>(ens->getMoleculePointer(0));
	if (!prot) { std::cout << "no protein in " << file << std::endl; return false; }

	for (atomIterator it(prot); !(it.last()); it++)
	{
		atom* a = it.getResiduePointer()->getAtom(it.getAtomIndex());
		s.x.push_back(a->getX());
		s.y.push_back(a->getY());
		s.z.push_back(a->getZ());
	}
	s.energy  = prot->protEnergy();
	s.clashes = prot->getNumHardClashes();
	return !s.x.empty();
}

// Largest displacement between a reference structure and a transformed one after
// undoing the transform.  Returns -1.0 if the atom counts disagree.
double maxDeviation(const Snapshot& ref, const Snapshot& got, int mode)
{
	if (ref.x.size() != got.x.size()) return -1.0;
	double worst = 0.0;
	for (size_t i = 0; i < ref.x.size(); i++)
	{
		double ex, ey, ez;
		if (mode == 0) { ex = ref.x[i] + 60.0; ey = ref.y[i]; ez = ref.z[i]; }
		else           { ex = -ref.y[i];       ey = ref.x[i]; ez = ref.z[i]; }
		double dx = got.x[i] - ex, dy = got.y[i] - ey, dz = got.z[i] - ez;
		double d = sqrt(dx * dx + dy * dy + dz * dz);
		if (d > worst) worst = d;
	}
	return worst;
}

} // namespace

int main(int argc, char** argv)
{
	if (argc < 2) { std::cout << "usage: invarianceTest <structure.pdb>" << std::endl; return 1; }

	// The transforms are exact, so the only slack needed is single precision
	// reassociation in the energy reduction when the spatial grid re-tiles.
	const double coordTol  = 1.0e-4;
	const double energyTol = 0.05;

	const std::string base    = argv[1];
	const std::string shifted = "invarianceTest_shift.pdb";
	const std::string rotated = "invarianceTest_rot.pdb";

	if (!writeTransformed(base, shifted, 0)) return 1;
	if (!writeTransformed(base, rotated, 1)) return 1;

	Snapshot ref, shift, rot;
	if (!load(base, ref) || !load(shifted, shift) || !load(rotated, rot)) return 1;

	remove(shifted.c_str());
	remove(rotated.c_str());

	std::cout << std::fixed << std::setprecision(6);
	std::cout << "atoms                " << ref.x.size()  << std::endl;
	std::cout << "energy reference     " << ref.energy    << std::endl;
	std::cout << "energy translated    " << shift.energy  << std::endl;
	std::cout << "energy rotated       " << rot.energy    << std::endl;

	bool ok = true;
	const int modes[2] = {0, 1};
	const char* names[2] = {"translation 60 A in x", "rotation 90 deg about z"};
	const Snapshot* got[2] = {&shift, &rot};

	for (int t = 0; t < 2; t++)
	{
		double dev = maxDeviation(ref, *got[t], modes[t]);
		if (dev < 0.0)
		{
			std::cout << "FAIL " << names[t] << ": atom count changed, "
			          << ref.x.size() << " -> " << got[t]->x.size() << std::endl;
			ok = false;
			continue;
		}
		double dE = fabs(got[t]->energy - ref.energy);
		std::cout << names[t] << ": max atom deviation " << dev
		          << " A, energy change " << dE << " kcal/mol" << std::endl;
		if (dev > coordTol)
		{
			std::cout << "FAIL " << names[t] << ": structure is not rigid, "
			          << dev << " A > " << coordTol << " A" << std::endl;
			ok = false;
		}
		if (dE > energyTol)
		{
			std::cout << "FAIL " << names[t] << ": energy moved " << dE
			          << " kcal/mol > " << energyTol << std::endl;
			ok = false;
		}
	}

	// Pin the absolute 1crn energy as well.  Invariance alone would still be
	// satisfied by a model that had silently changed everywhere at once, and
	// nothing else in the suite scores a real structure -- the synthetic cases
	// in energyTest check the kernel against a reference implementation, not
	// the parameters or the build applied to a PDB.  The tolerance is loose
	// enough to absorb float reduction order but tight enough to catch a term
	// being dropped or double counted.
	// Updated when the PDB load stopped idealising the input.  The rebuild used
	// to replace every atom from the residue template, keeping only the
	// measured torsions, so this pinned the energy of an idealised 1crn rather
	// than of 1crn.  The structure now keeps its deposited coordinates, which
	// relieves the strain idealisation had introduced: 510.98 -> 492.79.  The
	// clash count rises because real close contacts survive that the template
	// rebuild had quietly relaxed away.
	const double refEnergy = 492.79;
	const double refTol    = 0.05;
	if (fabs(ref.energy - refEnergy) > refTol)
	{
		std::cout << "FAIL 1crn reference energy: " << ref.energy
		          << " kcal/mol, expected " << refEnergy
		          << " +/- " << refTol << std::endl;
		ok = false;
	}
	else
	{
		std::cout << "1crn reference energy " << ref.energy
		          << " kcal/mol, within " << refTol << " of " << refEnergy << std::endl;
	}

	const unsigned int refClashes = 457;
	if (ref.clashes != refClashes)
	{
		std::cout << "FAIL 1crn reference clashes: " << ref.clashes
		          << ", expected " << refClashes << std::endl;
		ok = false;
	}
	else
	{
		std::cout << "1crn reference clashes " << ref.clashes << std::endl;
	}

	if (!ok) { std::cout << "invarianceTest FAILED" << std::endl; return 1; }
	std::cout << "invarianceTest passed" << std::endl;
	return 0;
}
